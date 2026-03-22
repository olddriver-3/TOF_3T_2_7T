"""
数据预处理脚本 - 将原始NRRD格式的3T和7T TOF-MRA数据预处理为训练所需的NIfTI格式

预处理步骤：
1. 读取NRRD文件 (ANTs直接读取，自动解析spacing/origin/direction)
2. 7T图像刚性配准到3T图像 (ANTs)
3. 偏置场校正 N4 (ANTs)
4. 颅骨去除 (可选)
5. 重采样到目标分辨率，然后pad/crop到指定大小
6. 强度归一化到 [0, 255]
7. 保存为NIfTI格式
8. 划分训练/验证/测试集
"""

import os
import glob
import argparse
import numpy as np
from tqdm import tqdm

import ants


def affine_registration(fixed, moving):
    result = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform='Affine',
        verbose=False
    )
    return result['warpedmovout']


def bias_field_correction(image):
    mask = image > image.mean()
    corrected = ants.n4_bias_field_correction(image, mask=mask, verbose=False)
    return corrected


def skull_stripping(image):
    threshold = np.percentile(image.numpy()[image.numpy() > 0], 10)
    mask = image > threshold
    
    mask_array = mask.numpy().astype(np.int32)
    from scipy.ndimage import binary_fill_holes, label
    mask_array = binary_fill_holes(mask_array)
    
    labeled, num_features = label(mask_array)
    if num_features > 0:
        sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]
        largest = np.argmax(sizes) + 1
        mask_array = (labeled == largest).astype(np.float32)
    
    mask_img = ants.from_numpy(
        mask_array,
        spacing=image.spacing,
        origin=image.origin,
        direction=image.direction
    )
    return image * mask_img


def resample_to_resolution(image, target_resolution, target_shape):
    target_res = np.array(target_resolution)
    resampled = ants.resample_image(
        image,
        target_res,
        use_voxels=False,
        interp_type=1
    )
    
    current_shape = np.array(resampled.shape)
    target_shape = np.array(target_shape)
    
    pad_or_crop = target_shape - current_shape
    
    if np.any(pad_or_crop > 0):
        pad_width = []
        for diff in pad_or_crop:
            if diff > 0:
                pad_before = int(diff // 2)
                pad_after = int(diff - pad_before)
                pad_width.append((pad_before, pad_after))
            else:
                pad_width.append((0, 0))
        
        data = resampled.numpy()
        data = np.pad(data, pad_width, mode='constant', constant_values=0)
        
        resampled = ants.from_numpy(
            data.astype(np.float32),
            spacing=resampled.spacing,
            origin=resampled.origin,
            direction=resampled.direction
        )
    
    if np.any(pad_or_crop < 0):
        crop_slices = []
        for i, (curr, target) in enumerate(zip(current_shape, target_shape)):
            if curr > target:
                start = int((curr - target) // 2)
                end = start + int(target)
                crop_slices.append(slice(start, end))
            else:
                crop_slices.append(slice(None))
        
        data = resampled.numpy()
        data = data[tuple(crop_slices)]
        
        new_origin = list(resampled.origin)
        for i, s in enumerate(crop_slices):
            if s.start is not None and s.start > 0:
                new_origin[i] += s.start * resampled.spacing[i]
        
        resampled = ants.from_numpy(
            data.astype(np.float32),
            spacing=resampled.spacing,
            origin=new_origin,
            direction=resampled.direction
        )
    
    return resampled


def normalize_intensity(image):
    data = image.numpy()
    data_min, data_max = data.min(), data.max()
    if data_max - data_min > 0:
        data = (data - data_min) / (data_max - data_min) * 255.0
    return ants.from_numpy(
        data.astype(np.float32),
        spacing=image.spacing,
        origin=image.origin,
        direction=image.direction
    )


def find_subjects(input_dir):
    subjects = []
    for item in sorted(os.listdir(input_dir)):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            nrrd_files = glob.glob(os.path.join(item_path, '*.nrrd'))
            if len(nrrd_files) >= 2:
                subjects.append(item)
    return subjects


def process_subject(input_dir, subject_id, output_dir, 
                    do_registration=True,
                    do_bias_correction=True,
                    do_skull_stripping=False,
                    target_shape=(512, 512, 320),
                    target_resolution=(0.4, 0.4, 0.4)):
    subject_dir = os.path.join(input_dir, subject_id)
    nrrd_files = glob.glob(os.path.join(subject_dir, '*.nrrd'))
    
    file_3t = file_7t = None
    for f in nrrd_files:
        basename = os.path.basename(f).lower()
        if basename.endswith('_3t.nrrd'):
            file_3t = f
        elif basename.endswith('_7t.nrrd'):
            file_7t = f
    
    if not file_3t or not file_7t:
        print(f"  Warning: Missing 3T/7T pair for {subject_id}")
        return False
    
    img_3t = ants.image_read(file_3t)
    img_7t = ants.image_read(file_7t)
    


    if do_registration:
        img_7t = affine_registration(img_3t, img_7t)
    
    if do_bias_correction:
        img_3t = bias_field_correction(img_3t)
        img_7t = bias_field_correction(img_7t)
    
    if do_skull_stripping:
        img_3t = skull_stripping(img_3t)
        img_7t = skull_stripping(img_7t)
    
    img_3t = resample_to_resolution(img_3t, target_resolution, target_shape)
    img_7t = resample_to_resolution(img_7t, target_resolution, target_shape)
    
    img_3t = normalize_intensity(img_3t)
    img_7t = normalize_intensity(img_7t)
    # 将 ANTsImage 转换为 numpy 数组后再做逻辑运算，最后再转回 ANTsImage
    mask_3t = img_3t.numpy() > 0
    mask_7t = img_7t.numpy() > 0
    available_mask_np = mask_3t & mask_7t

    img_3t_np = img_3t.numpy() * available_mask_np
    img_7t_np = img_7t.numpy() * available_mask_np

    img_3t = ants.from_numpy(
        img_3t_np.astype(np.float32),
        spacing=img_3t.spacing,
        origin=img_3t.origin,
        direction=img_3t.direction
    )
    img_7t = ants.from_numpy(
        img_7t_np.astype(np.float32),
        spacing=img_7t.spacing,
        origin=img_7t.origin,
        direction=img_7t.direction
    )

    os.makedirs(os.path.join(output_dir, '3T'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, '7T'), exist_ok=True)
    
    ants.image_write(img_3t, os.path.join(output_dir, '3T', f'{subject_id}.nii.gz'))
    ants.image_write(img_7t, os.path.join(output_dir, '7T', f'{subject_id}.nii.gz'))
    
    return True


def split_dataset(subjects, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    subjects = list(subjects)
    np.random.shuffle(subjects)
    
    n = len(subjects)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    return subjects[:n_train], subjects[n_train:n_train + n_val], subjects[n_train + n_val:]


def load_split_file(filepath):
    subjects = []
    if not os.path.exists(filepath):
        return subjects
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and ',' in line:
                subject_id = line.split(',')[0].strip()
                if subject_id:
                    subjects.append(subject_id)
    
    return subjects


def main():
    parser = argparse.ArgumentParser(description='Preprocess NRRD TOF-MRA data')
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument('--target_shape', type=int, nargs=3, default=[512, 512, 320])
    parser.add_argument('--target_resolution', type=float, nargs=3, default=[0.4, 0.4, 0.4])
    parser.add_argument('--no_registration', action='store_true')
    parser.add_argument('--no_bias_correction', action='store_true')
    parser.add_argument('--skull_stripping', action='store_true')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_dir', type=str, default=None,
                        help='Directory containing split_train.txt, split_val.txt, split_test.txt')
    
    args = parser.parse_args()
    ###

    print("=" * 60)
    print("TOF-MRA Data Preprocessing (ANTs)")
    print("=" * 60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Target shape: {args.target_shape}")
    print(f"Target resolution: {args.target_resolution}")
    print(f"Registration: {not args.no_registration}")
    print(f"Bias correction: {not args.no_bias_correction}")
    print(f"Skull stripping: {args.skull_stripping}")
    print("=" * 60)
    
    if args.split_dir:
        train = load_split_file(os.path.join(args.split_dir, 'split_train.txt'))
        val = load_split_file(os.path.join(args.split_dir, 'split_val.txt'))
        test = load_split_file(os.path.join(args.split_dir, 'split_test.txt'))
        print(f"Loaded from split files: train={len(train)}, val={len(val)}, test={len(test)}")
    else:
        subjects = find_subjects(args.input_dir)
        print(f"Found {len(subjects)} subjects")
        train, val, test = split_dataset(
            subjects,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
        print(f"Random split: train={len(train)}, val={len(val)}, test={len(test)}")
    
    processed, failed = 0, 0

    for split_name, split_subjects in [('train', train), ('val', val), ('test', test)]:
        split_dir = os.path.join(args.output_dir, split_name)
        
        for subject_id in tqdm(split_subjects, desc=split_name):
            try:
                if process_subject(
                    args.input_dir, subject_id, split_dir,
                    do_registration=not args.no_registration,
                    do_bias_correction=not args.no_bias_correction,
                    do_skull_stripping=args.skull_stripping,
                    target_shape=tuple(args.target_shape),
                    target_resolution=tuple(args.target_resolution)
                ):
                    processed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error {subject_id}: {e}")
                failed += 1
    
    print("\n" + "=" * 60)
    print(f"Done! Processed: {processed}, Failed: {failed}")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        for modality in ['3T', '7T']:
            dir_path = os.path.join(args.output_dir, split, modality)
            n_files = len(glob.glob(os.path.join(dir_path, '*.nii*'))) if os.path.exists(dir_path) else 0
            print(f"  {split}/{modality}: {n_files} files")


if __name__ == '__main__':
    main()
