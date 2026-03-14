"""
Complete Test Pipeline for AU-MIPGAN
=====================================

This script performs:
1. Inference: Generate 7T-like image from 3T input
2. Vessel Segmentation: Segment vessels from both real 7T and generated 7T-like images
3. Evaluation: Compute BBC, CNR, SNR for image quality, and DICE for vessel segmentation
"""

import os
import sys
import argparse
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from models.generator import UNet3DGenerator
from cerebrovascular_segmentation import (
    algorithm1_cerebrovascular_segmentation,
    algorithm2_non_vascular_segmentation,
    compute_bbc,
    compute_cnr,
    compute_snr
)


class TestPipeline:
    def __init__(self, config, checkpoint_path=None):
        self.config = config
        self.device = config.device
        
        self.model = UNet3DGenerator(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            base_features=config.base_features,
            num_layers=config.num_layers,
            output_uncertainty=True
        ).to(self.device)
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
    
    def load_checkpoint(self, path):
        print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES'))
        checkpoint = torch.load(path, map_location=self.device)
        
        if 'student_state_dict' in checkpoint:
            state_dict = checkpoint['student_state_dict']
        elif 'generator_state_dict' in checkpoint:
            state_dict = checkpoint['generator_state_dict']
        else:
            state_dict = checkpoint
        
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        
        print(f"Loaded checkpoint from: {path}")
    
    def load_volume(self, filepath):
        img = nib.load(filepath)
        data = img.get_fdata().astype(np.float32)
        affine = img.affine
        return data, affine
    
    def normalize(self, data):
        data_min = data.min()
        data_max = data.max()
        if data_max - data_min > 0:
            data = (data - data_min) / (data_max - data_min) * 255.0
        return data
    
    def extract_patches(self, volume, patch_size, stride):
        D, H, W = volume.shape
        patches = []
        locations = []
        
        for d in range(0, D - patch_size + 1, stride):
            for h in range(0, H - patch_size + 1, stride):
                for w in range(0, W - patch_size + 1, stride):
                    patch = volume[d:d+patch_size, h:h+patch_size, w:w+patch_size]
                    patches.append(patch)
                    locations.append((d, h, w))
        
        if D % stride != 0 or D < patch_size:
            d = max(0, D - patch_size)
            for h in range(0, H - patch_size + 1, stride):
                for w in range(0, W - patch_size + 1, stride):
                    patch = volume[d:d+patch_size, h:h+patch_size, w:w+patch_size]
                    patches.append(patch)
                    locations.append((d, h, w))
        
        if H % stride != 0 or H < patch_size:
            h = max(0, H - patch_size)
            for d in range(0, D - patch_size + 1, stride):
                for w in range(0, W - patch_size + 1, stride):
                    patch = volume[d:d+patch_size, h:h+patch_size, w:w+patch_size]
                    patches.append(patch)
                    locations.append((d, h, w))
        
        if W % stride != 0 or W < patch_size:
            w = max(0, W - patch_size)
            for d in range(0, D - patch_size + 1, stride):
                for h in range(0, H - patch_size + 1, stride):
                    patch = volume[d:d+patch_size, h:h+patch_size, w:w+patch_size]
                    patches.append(patch)
                    locations.append((d, h, w))
        
        return patches, locations
    
    def reconstruct_volume(self, patches, locations, original_shape, patch_size):
        D, H, W = original_shape
        
        output = np.zeros((D, H, W), dtype=np.float32)
        count = np.zeros((D, H, W), dtype=np.float32)
        
        for patch, (d, h, w) in zip(patches, locations):
            output[d:d+patch_size, h:h+patch_size, w:w+patch_size] += patch
            count[d:d+patch_size, h:h+patch_size, w:w+patch_size] += 1
        
        count[count == 0] = 1
        output = output / count
        
        return output
    
    @torch.no_grad()
    def infer_volume(self, volume, batch_size=8):
        D, H, W = volume.shape
        patch_size = self.config.patch_size
        stride = self.config.stride
        
        patches, locations = self.extract_patches(volume, patch_size, stride)
        
        output_patches = []
        uncertainty_patches = []
        
        for i in tqdm(range(0, len(patches), batch_size), desc='Processing patches'):
            batch_patches = patches[i:i+batch_size]
            
            batch_tensor = torch.from_numpy(np.stack(batch_patches)).unsqueeze(1)
            batch_tensor = batch_tensor.to(self.device)
            
            output, uncertainty = self.model(batch_tensor)
            
            output_patches.extend(output.cpu().numpy().squeeze(1))
            uncertainty_patches.extend(uncertainty.cpu().numpy().squeeze(1))
        
        output_volume = self.reconstruct_volume(output_patches, locations, (D, H, W), patch_size)
        uncertainty_volume = self.reconstruct_volume(uncertainty_patches, locations, (D, H, W), patch_size)
        
        return output_volume, uncertainty_volume
    
    def save_volume(self, data, affine, filepath):
        img = nib.Nifti1Image(data, affine)
        nib.save(img, filepath)
        print(f"Saved volume to: {filepath}")
    
    def segment_vessels(self, image, seg_params=None):
        """
        Perform vessel segmentation on the input image.
        
        Parameters:
        -----------
        image : ndarray
            Input MRA image
        seg_params : dict, optional
            Segmentation parameters
            
        Returns:
        --------
        dict
            Segmentation results including vessel mask, skeleton, etc.
        """
        if seg_params is None:
            seg_params = {
                'denoise': True,
                'denoise_window_size': 10,
                'denoise_max_iterations': 1,
                'sigmas': [1, 5],
                'tau': 0.5,
                'threshold': 0.5,
                'min_skeleton_area': 20,
                'region_threshold_low': 0.1,
                'region_threshold_high': 1.0,
                'region_max_iterations': 100000,
                'enhancement_method': 'improved',
                'verbose': True
            }
        
        result = algorithm1_cerebrovascular_segmentation(image, params=seg_params)
        return result
    
    def compute_image_quality_metrics(self, image, vessel_segmentation):
        """
        Compute image quality metrics (BBC, CNR, SNR).
        
        Parameters:
        -----------
        image : ndarray
            Input MRA image
        vessel_segmentation : ndarray
            Binary vessel segmentation mask
            
        Returns:
        --------
        dict
            Dictionary containing BBC, CNR, SNR values
        """
        algo2_result = algorithm2_non_vascular_segmentation(
            image, vessel_segmentation, params={'verbose': False}
        )
        
        vessel_region = vessel_segmentation
        non_vessel_tissue = algo2_result['non_vessel_brain_tissue']
        noise_region = algo2_result['noise_region']
        
        image_normalized = image.astype(np.float32)
        if np.max(image_normalized) > 1:
            image_normalized = (image_normalized - np.min(image_normalized)) / \
                              (np.max(image_normalized) - np.min(image_normalized))
        
        signal_vessel = image_normalized[vessel_region == 1]
        signal_non_vessel = image_normalized[non_vessel_tissue == 1]
        noise = image_normalized[noise_region == 1]
        
        if len(signal_vessel) > 0 and len(signal_non_vessel) > 0 and len(noise) > 0:
            bbc = compute_bbc(signal_vessel, signal_non_vessel)
            cnr = compute_cnr(signal_vessel, signal_non_vessel, noise)
            snr = compute_snr(signal_vessel, noise)
        else:
            bbc, cnr, snr = 0, 0, 0
        
        return {
            'bbc': bbc,
            'cnr': cnr,
            'snr': snr
        }
    
    def compute_dice(self, seg1, seg2):
        """
        Compute Dice similarity coefficient between two segmentations.
        
        DICE = 2 * |A ∩ B| / (|A| + |B|)
        
        Parameters:
        -----------
        seg1 : ndarray
            First binary segmentation
        seg2 : ndarray
            Second binary segmentation
            
        Returns:
        --------
        float
            Dice coefficient
        """
        intersection = np.sum(seg1 * seg2)
        sum_masks = np.sum(seg1) + np.sum(seg2)
        
        if sum_masks == 0:
            return 1.0 if np.sum(seg1) == 0 and np.sum(seg2) == 0 else 0.0
        
        return 2.0 * intersection / sum_masks
    
    def run_test(self, input_3t_path, gt_7t_path, output_dir, save_intermediate=True):
        """
        Run the complete test pipeline.
        
        Parameters:
        -----------
        input_3t_path : str
            Path to input 3T image
        gt_7t_path : str
            Path to ground truth 7T image
        output_dir : str
            Directory to save results
        save_intermediate : bool
            Whether to save intermediate results
            
        Returns:
        --------
        dict
            Complete test results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 70)
        print("AU-MIPGAN Complete Test Pipeline")
        print("=" * 70)
        
        print(f"\nInput 3T: {input_3t_path}")
        print(f"Ground Truth 7T: {gt_7t_path}")
        print(f"Output Directory: {output_dir}")
        
        print("\n" + "-" * 70)
        print("Step 1: Loading images...")
        print("-" * 70)
        
        volume_3t, affine_3t = self.load_volume(input_3t_path)
        volume_7t_gt, affine_7t = self.load_volume(gt_7t_path)
        
        print(f"3T volume shape: {volume_3t.shape}")
        print(f"7T GT volume shape: {volume_7t_gt.shape}")
        
        volume_3t_norm = self.normalize(volume_3t)
        
        print("\n" + "-" * 70)
        print("Step 2: Running inference (3T -> 7T-like)...")
        print("-" * 70)
        
        volume_7t_like, uncertainty = self.infer_volume(volume_3t_norm)
        
        base_name = os.path.splitext(os.path.basename(input_3t_path))[0]
        if base_name.endswith('.nii'):
            base_name = base_name[:-4]
        
        if save_intermediate:
            output_7t_like_path = os.path.join(output_dir, f"{base_name}_7t_like.nii.gz")
            output_uncertainty_path = os.path.join(output_dir, f"{base_name}_uncertainty.nii.gz")
            self.save_volume(volume_7t_like, affine_3t, output_7t_like_path)
            self.save_volume(uncertainty, affine_3t, output_uncertainty_path)
        
        print("\n" + "-" * 70)
        print("Step 3: Vessel segmentation on Ground Truth 7T...")
        print("-" * 70)
        
        seg_result_7t_gt = self.segment_vessels(volume_7t_gt)
        vessel_seg_7t_gt = seg_result_7t_gt['segmentation']
        
        if save_intermediate:
            vessel_seg_7t_gt_path = os.path.join(output_dir, f"{base_name}_7t_gt_vessel_seg.nii.gz")
            self.save_volume(vessel_seg_7t_gt.astype(np.float32), affine_7t, vessel_seg_7t_gt_path)
        
        print("\n" + "-" * 70)
        print("Step 4: Vessel segmentation on Generated 7T-like...")
        print("-" * 70)
        
        seg_result_7t_like = self.segment_vessels(volume_7t_like)
        vessel_seg_7t_like = seg_result_7t_like['segmentation']
        
        if save_intermediate:
            vessel_seg_7t_like_path = os.path.join(output_dir, f"{base_name}_7t_like_vessel_seg.nii.gz")
            self.save_volume(vessel_seg_7t_like.astype(np.float32), affine_3t, vessel_seg_7t_like_path)
        
        print("\n" + "-" * 70)
        print("Step 5: Computing image quality metrics...")
        print("-" * 70)
        
        print("\nComputing metrics for Ground Truth 7T...")
        metrics_7t_gt = self.compute_image_quality_metrics(volume_7t_gt, vessel_seg_7t_gt)
        
        print("\nComputing metrics for Generated 7T-like...")
        metrics_7t_like = self.compute_image_quality_metrics(volume_7t_like, vessel_seg_7t_like)
        
        print("\n" + "-" * 70)
        print("Step 6: Computing vessel segmentation DICE...")
        print("-" * 70)
        
        dice_score = self.compute_dice(vessel_seg_7t_gt, vessel_seg_7t_like)
        
        print(f"DICE Score: {dice_score:.4f}")
        
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        
        print("\n--- Ground Truth 7T ---")
        print(f"  BBC: {metrics_7t_gt['bbc']:.4f}")
        print(f"  CNR: {metrics_7t_gt['cnr']:.4f}")
        print(f"  SNR: {metrics_7t_gt['snr']:.4f}")
        
        print("\n--- Generated 7T-like ---")
        print(f"  BBC: {metrics_7t_like['bbc']:.4f}")
        print(f"  CNR: {metrics_7t_like['cnr']:.4f}")
        print(f"  SNR: {metrics_7t_like['snr']:.4f}")
        
        print("\n--- Vessel Segmentation Comparison ---")
        print(f"  DICE: {dice_score:.4f}")
        print(f"  GT 7T vessel voxels: {np.sum(vessel_seg_7t_gt)}")
        print(f"  7T-like vessel voxels: {np.sum(vessel_seg_7t_like)}")
        
        results = {
            'input_3t': input_3t_path,
            'gt_7t': gt_7t_path,
            'output_dir': output_dir,
            'timestamp': datetime.now().isoformat(),
            'metrics_7t_gt': metrics_7t_gt,
            'metrics_7t_like': metrics_7t_like,
            'dice_score': dice_score,
            'vessel_voxels_gt': int(np.sum(vessel_seg_7t_gt)),
            'vessel_voxels_like': int(np.sum(vessel_seg_7t_like))
        }
        
        results_path = os.path.join(output_dir, f"{base_name}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
        
        return results
    
    def run_batch_test(self, input_dir, output_dir, save_intermediate=True):
        """
        Run batch test on a directory of paired 3T/7T images.
        
        Expected directory structure:
        input_dir/
        ├── 3T/
        │   ├── subject_001.nii.gz
        │   └── ...
        └── 7T/
            ├── subject_001.nii.gz
            └── ...
        """
        dir_3t = os.path.join(input_dir, '3T')
        dir_7t = os.path.join(input_dir, '7T')
        
        if not os.path.exists(dir_3t) or not os.path.exists(dir_7t):
            raise ValueError(f"Expected directories '3T' and '7T' in {input_dir}")
        
        files_3t = sorted([f for f in os.listdir(dir_3t) if f.endswith('.nii') or f.endswith('.nii.gz')])
        
        all_results = []
        
        for file_3t in tqdm(files_3t, desc="Processing subjects"):
            file_7t = file_3t
            
            if not os.path.exists(os.path.join(dir_7t, file_7t)):
                print(f"Warning: No matching 7T file for {file_3t}, skipping...")
                continue
            
            input_3t_path = os.path.join(dir_3t, file_3t)
            gt_7t_path = os.path.join(dir_7t, file_7t)
            
            subject_output_dir = os.path.join(output_dir, file_3t.replace('.nii.gz', '').replace('.nii', ''))
            
            try:
                result = self.run_test(input_3t_path, gt_7t_path, subject_output_dir, save_intermediate)
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {file_3t}: {e}")
                continue
        
        if len(all_results) > 0:
            print("\n" + "=" * 70)
            print("BATCH RESULTS SUMMARY")
            print("=" * 70)
            
            avg_bbc_gt = np.mean([r['metrics_7t_gt']['bbc'] for r in all_results])
            avg_cnr_gt = np.mean([r['metrics_7t_gt']['cnr'] for r in all_results])
            avg_snr_gt = np.mean([r['metrics_7t_gt']['snr'] for r in all_results])
            
            avg_bbc_like = np.mean([r['metrics_7t_like']['bbc'] for r in all_results])
            avg_cnr_like = np.mean([r['metrics_7t_like']['cnr'] for r in all_results])
            avg_snr_like = np.mean([r['metrics_7t_like']['snr'] for r in all_results])
            
            avg_dice = np.mean([r['dice_score'] for r in all_results])
            
            print(f"\nNumber of subjects: {len(all_results)}")
            
            print("\n--- Ground Truth 7T (Average) ---")
            print(f"  BBC: {avg_bbc_gt:.4f}")
            print(f"  CNR: {avg_cnr_gt:.4f}")
            print(f"  SNR: {avg_snr_gt:.4f}")
            
            print("\n--- Generated 7T-like (Average) ---")
            print(f"  BBC: {avg_bbc_like:.4f}")
            print(f"  CNR: {avg_cnr_like:.4f}")
            print(f"  SNR: {avg_snr_like:.4f}")
            
            print("\n--- Vessel Segmentation DICE (Average) ---")
            print(f"  DICE: {avg_dice:.4f}")
            
            batch_summary = {
                'num_subjects': len(all_results),
                'timestamp': datetime.now().isoformat(),
                'average_metrics_7t_gt': {
                    'bbc': avg_bbc_gt,
                    'cnr': avg_cnr_gt,
                    'snr': avg_snr_gt
                },
                'average_metrics_7t_like': {
                    'bbc': avg_bbc_like,
                    'cnr': avg_cnr_like,
                    'snr': avg_snr_like
                },
                'average_dice': avg_dice,
                'individual_results': all_results
            }
            
            summary_path = os.path.join(output_dir, 'batch_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(batch_summary, f, indent=2)
            print(f"\nBatch summary saved to: {summary_path}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description='Complete Test Pipeline for AU-MIPGAN')
    parser.add_argument('--input', type=str, required=True,
                        help='Input 3T file path or directory containing 3T/7T subdirectories')
    parser.add_argument('--gt_7t', type=str, default=None,
                        help='Ground truth 7T file path (required for single file mode)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--mode', type=str, default='file', choices=['file', 'batch'],
                        help='Test mode: file (single file) or batch (directory)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID to use')
    parser.add_argument('--no_save_intermediate', action='store_true',
                        help='Do not save intermediate results')
    args = parser.parse_args()
    
    config = Config()
    config.set_gpu(args.gpu)
    
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_config", args.config)
        custom_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_config)
        for key, value in vars(custom_config).items():
            if not key.startswith('_'):
                setattr(config, key, value)
    
    pipeline = TestPipeline(config, args.checkpoint)
    
    if args.mode == 'file':
        if args.gt_7t is None:
            raise ValueError("--gt_7t is required for file mode")
        pipeline.run_test(
            args.input, 
            args.gt_7t, 
            args.output,
            save_intermediate=not args.no_save_intermediate
        )
    else:
        pipeline.run_batch_test(
            args.input, 
            args.output,
            save_intermediate=not args.no_save_intermediate
        )


if __name__ == '__main__':
    main()
