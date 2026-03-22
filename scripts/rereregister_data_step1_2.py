"""
数据重配准脚本 - 对已有的train/val/test数据进行血管增强和SyN配准

步骤：
1. 读取原始3T和7T图像
2. 对图像进行血管增强并保存
3. 用SyN方法将增强后的7T图像配准到增强后的3T图像
4. 将配准得到的变形场应用到原始7T图像
5. 保存配准后的结果
"""

import os
import sys
import argparse
import glob
import numpy as np
from tqdm import tqdm
import ants

from cerebrovascular_segmentation import nafsm_filter, multi_scale_vessel_enhancement


def enhance_image(image_data):
    """
    对图像进行血管增强
    参考 vessel_enhancement.py 中的处理流程
    """
    normalized = image_data.astype(np.float32)
    if np.max(normalized) > 1:
        normalized = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))
    
    denoised = nafsm_filter(normalized, window_size=1, max_iterations=0, use_gpu=False)
    
    enhanced = multi_scale_vessel_enhancement(
        denoised,
        sigmas=[1, 3],
        tau=0.5,
        method='improved',
        use_gpu=False
    )
    
    if np.iscomplexobj(enhanced):
        enhanced = np.abs(enhanced)
    
    return enhanced.astype(np.float32)


def syn_registration_with_warp(fixed, moving):
    """
    SyN配准，返回配准后的图像和变形场
    参考 train_preprocess_data_step1.py
    """
    result = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform='SyN',
        verbose=False
    )
    return result['warpedmovout'], result


def apply_warp_to_image(original_image, warp_result, fixed_image):
    """
    将配准变形场应用到原始图像
    """
    warped = ants.apply_transforms(
        fixed=fixed_image,
        moving=original_image,
        transformlist=warp_result['fwdtransforms'],
        interpolator='linear'
    )
    return warped


def process_split(split_dir, enhanced_dir, registered_dir, split_name):
    """
    处理一个数据集划分（train/val/test）
    """
    dir_3t = os.path.join(split_dir, '3T')
    dir_7t = os.path.join(split_dir, '7T')
    
    enhanced_3t_dir = os.path.join(enhanced_dir, split_name, '3T')
    enhanced_7t_dir = os.path.join(enhanced_dir, split_name, '7T')
    registered_3t_dir = os.path.join(registered_dir, split_name, '3T')
    registered_7t_dir = os.path.join(registered_dir, split_name, '7T')
    
    os.makedirs(enhanced_3t_dir, exist_ok=True)
    os.makedirs(enhanced_7t_dir, exist_ok=True)
    os.makedirs(registered_3t_dir, exist_ok=True)
    os.makedirs(registered_7t_dir, exist_ok=True)
    
    files_3t = sorted(glob.glob(os.path.join(dir_3t, '*.nii.gz')))
    files_7t = sorted(glob.glob(os.path.join(dir_7t, '*.nii.gz')))
    
    print(f"Processing {split_name}: {len(files_3t)} subjects")
    
    processed = 0
    skipped = 0
    failed = 0
    
    for file_3t, file_7t in tqdm(zip(files_3t, files_7t), total=len(files_3t), desc=split_name):
        try:
            subject_id = os.path.basename(file_3t).replace('.nii.gz', '')
            
            enhanced_3t_path = os.path.join(enhanced_3t_dir, f'{subject_id}.nii.gz')
            enhanced_7t_path = os.path.join(enhanced_7t_dir, f'{subject_id}.nii.gz')
            registered_3t_path = os.path.join(registered_3t_dir, f'{subject_id}.nii.gz')
            registered_7t_path = os.path.join(registered_7t_dir, f'{subject_id}.nii.gz')
            print(f"  Processing {subject_id}...")
            if os.path.exists(enhanced_3t_path) and os.path.exists(enhanced_7t_path) and \
               os.path.exists(registered_3t_path) and os.path.exists(registered_7t_path):
                skipped += 1
                continue
            
            img_3t = ants.image_read(file_3t)
            img_7t = ants.image_read(file_7t)
            
            data_3t = img_3t.numpy()
            data_7t = img_7t.numpy()
            
            enhanced_3t_data = enhance_image(data_3t)
            enhanced_7t_data = enhance_image(data_7t)
            
            enhanced_3t = ants.from_numpy(
                enhanced_3t_data,
                spacing=img_3t.spacing,
                origin=img_3t.origin,
                direction=img_3t.direction
            )
            enhanced_7t = ants.from_numpy(
                enhanced_7t_data,
                spacing=img_7t.spacing,
                origin=img_7t.origin,
                direction=img_7t.direction
            )
            
            ants.image_write(enhanced_3t, os.path.join(enhanced_3t_dir, f'{subject_id}.nii.gz'))
            ants.image_write(enhanced_7t, os.path.join(enhanced_7t_dir, f'{subject_id}.nii.gz'))
            
            warped_enhanced_7t, warp_result = syn_registration_with_warp(enhanced_3t, enhanced_7t)
            
            registered_7t = apply_warp_to_image(img_7t, warp_result, img_3t)
            
            ants.image_write(img_3t, os.path.join(registered_3t_dir, f'{subject_id}.nii.gz'))
            ants.image_write(registered_7t, os.path.join(registered_7t_dir, f'{subject_id}.nii.gz'))
            
            processed += 1
            
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"{split_name} done: processed={processed}, skipped={skipped}, failed={failed}")
    return processed, skipped, failed


def main():
    parser = argparse.ArgumentParser(description='Reregister data with vessel enhancement and SyN')
    parser.add_argument('--data_dir', type=str, default='d:\\project\\TOF_3T_2_7T_new\\data',
                        help='Directory containing train/val/test data')
    parser.add_argument('--enhanced_dir', type=str, default='d:\\project\\TOF_3T_2_7T_new\\data_enhanced',
                        help='Output directory for enhanced images')
    parser.add_argument('--registered_dir', type=str, default='d:\\project\\TOF_3T_2_7T_new\\data_registered',
                        help='Output directory for registered images')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Data Reregistration with Vessel Enhancement and SyN")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Enhanced output: {args.enhanced_dir}")
    print(f"Registered output: {args.registered_dir}")
    print("=" * 60)
    
    total_processed = 0
    total_skipped = 0
    total_failed = 0
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(args.data_dir, split)
        if os.path.exists(split_dir):
            p, s, f = process_split(split_dir, args.enhanced_dir, args.registered_dir, split)
            total_processed += p
            total_skipped += s
            total_failed += f
        else:
            print(f"Warning: {split_dir} does not exist, skipping")
    
    print("\n" + "=" * 60)
    print(f"All done! Total processed: {total_processed}, Total skipped: {total_skipped}, Total failed: {total_failed}")
    print("=" * 60)


if __name__ == '__main__':
    main()
