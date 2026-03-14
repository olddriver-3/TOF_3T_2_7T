import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import ants

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from models.generator import UNet3DGenerator


class InferenceEngine:
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
        img = ants.image_read(filepath)
        data = img.numpy().astype(np.float32)
        return data, img
    
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
        
        patches_array = np.stack(patches, axis=0)
        return patches_array, locations
    
    def reconstruct_volume(self, patches, locations, original_shape, patch_size):
        D, H, W = original_shape
        
        output = np.zeros((D, H, W), dtype=np.float32)
        count = np.zeros((D, H, W), dtype=np.float32)
        
        for i, (d, h, w) in enumerate(locations):
            output[d:d+patch_size, h:h+patch_size, w:w+patch_size] += patches[i]
            count[d:d+patch_size, h:h+patch_size, w:w+patch_size] += 1
        
        count[count == 0] = 1
        output = output / count
        
        return output
    
    @torch.inference_mode()
    def infer_volume(self, volume, batch_size=16):
        D, H, W = volume.shape
        patch_size = self.config.patch_size
        stride = self.config.stride
        
        patches_array, locations = self.extract_patches(volume, patch_size, stride)
        num_patches = len(locations)
        
        patches_tensor = torch.from_numpy(patches_array).unsqueeze(1).to(self.device)
        
        output_patches = []
        uncertainty_patches = []
        
        for i in tqdm(range(0, num_patches, batch_size), desc='Processing batches'):
            batch_tensor = patches_tensor[i:i+batch_size]
            
            output, uncertainty = self.model(batch_tensor)
            
            output_patches.append(output.cpu().numpy())
            uncertainty_patches.append(uncertainty.cpu().numpy())
        
        output_patches = np.concatenate(output_patches, axis=0).squeeze(1)
        uncertainty_patches = np.concatenate(uncertainty_patches, axis=0).squeeze(1)
        
        output_volume = self.reconstruct_volume(output_patches, locations, (D, H, W), patch_size)
        uncertainty_volume = self.reconstruct_volume(uncertainty_patches, locations, (D, H, W), patch_size)
        
        return output_volume, uncertainty_volume
    
    def save_volume(self, data, ref_img, filepath):
        output_img = ants.from_numpy(
            data.astype(np.float32),
            spacing=ref_img.spacing,
            origin=ref_img.origin,
            direction=ref_img.direction
        )
        ants.image_write(output_img, filepath)
        print(f"Saved volume to: {filepath}")
    
    def infer_file(self, input_path, output_path=None, uncertainty_path=None):
        print(f"Processing: {input_path}")
        
        volume, ref_img = self.load_volume(input_path)
        print(f"Volume shape: {volume.shape}")
        
        volume = self.normalize(volume)
        
        output, uncertainty = self.infer_volume(volume)
        
        if output_path is None:
            input_dir = os.path.dirname(input_path)
            input_name = os.path.splitext(os.path.basename(input_path))[0]
            if input_name.endswith('.nii'):
                input_name = input_name[:-4]
            output_path = os.path.join(input_dir, f'{input_name}_7t_like.nii.gz')
            uncertainty_path = os.path.join(input_dir, f'{input_name}_uncertainty.nii.gz')
        
        self.save_volume(output, ref_img, output_path)
        self.save_volume(uncertainty, ref_img, uncertainty_path)
        
        return output, uncertainty
    
    def infer_directory(self, input_dir, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(input_dir, 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        nii_files = [f for f in os.listdir(input_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
        
        print(f"Found {len(nii_files)} NIfTI files in {input_dir}")
        
        for nii_file in tqdm(nii_files, desc='Processing files'):
            input_path = os.path.join(input_dir, nii_file)
            
            name = nii_file
            if name.endswith('.nii.gz'):
                name = name[:-7]
            elif name.endswith('.nii'):
                name = name[:-4]
            
            output_path = os.path.join(output_dir, f'{name}_7t_like.nii.gz')
            uncertainty_path = os.path.join(output_dir, f'{name}_uncertainty.nii.gz')
            
            self.infer_file(input_path, output_path, uncertainty_path)


def main():
    parser = argparse.ArgumentParser(description='Inference for AU-MIPGAN')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file or directory path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file or directory path')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--mode', type=str, default='file', choices=['file', 'dir'],
                        help='Inference mode: file or directory')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID to use (e.g., 0, 1, 0,1 for multiple GPUs)')
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
    
    engine = InferenceEngine(config, args.checkpoint)
    
    if args.mode == 'file':
        engine.infer_file(args.input, args.output)
    else:
        engine.infer_directory(args.input, args.output)


if __name__ == '__main__':
    main()
