import os
import sys
import argparse
import glob
import numpy as np
import ants

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config


def load_volume(filepath):
    img = ants.image_read(filepath)
    data = img.numpy()
    return data.astype(np.float32)


def save_volume(data, filepath):
    data = data.astype(np.float32)
    img = ants.from_numpy(data)
    ants.image_write(img, filepath)


def normalize(data):
    data_min = data.min()
    data_max = data.max()
    if data_max - data_min > 0:
        data = (data - data_min) / (data_max - data_min) * 255.0
    return data


def compute_mip_with_padding(volume, direction, mip_thickness):
    D, H, W = volume.shape
    pad_size = mip_thickness // 2
    
    if direction == 'axial':
        padded = np.pad(volume, ((pad_size, pad_size), (0, 0), (0, 0)), mode='constant', constant_values=0)
        num_slabs = D
        mip_slices = []
        for i in range(num_slabs):
            start = i
            end = min(start + mip_thickness, padded.shape[0])
            slab = padded[start:end, :, :]
            if slab.shape[0] < mip_thickness:
                slab = np.pad(slab, ((0, mip_thickness - slab.shape[0]), (0, 0), (0, 0)), mode='constant', constant_values=0)
            mip_slice = np.max(slab, axis=0)
            mip_slices.append(mip_slice)
        return np.stack(mip_slices, axis=0)
    
    elif direction == 'coronal':
        padded = np.pad(volume, ((0, 0), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)
        num_slabs = H
        mip_slices = []
        for i in range(num_slabs):
            start = i
            end = min(start + mip_thickness, padded.shape[1])
            slab = padded[:, start:end, :]
            if slab.shape[1] < mip_thickness:
                slab = np.pad(slab, ((0, 0), (0, mip_thickness - slab.shape[1]), (0, 0)), mode='constant', constant_values=0)
            mip_slice = np.max(slab, axis=1)
            mip_slices.append(mip_slice)
        return np.stack(mip_slices, axis=1)
    
    elif direction == 'sagittal':
        padded = np.pad(volume, ((0, 0), (0, 0), (pad_size, pad_size)), mode='constant', constant_values=0)
        num_slabs = W
        mip_slices = []
        for i in range(num_slabs):
            start = i
            end = min(start + mip_thickness, padded.shape[2])
            slab = padded[:, :, start:end]
            if slab.shape[2] < mip_thickness:
                slab = np.pad(slab, ((0, 0), (0, 0), (0, mip_thickness - slab.shape[2])), mode='constant', constant_values=0)
            mip_slice = np.max(slab, axis=2)
            mip_slices.append(mip_slice)
        return np.stack(mip_slices, axis=2)


def preprocess_sample(filepath_3t, filepath_7t, sample_idx, output_dirs, mip_thickness):
    base_name = f'sample_{sample_idx:04d}'
    
    volume_3t = load_volume(filepath_3t)
    volume_7t = load_volume(filepath_7t)
    
    volume_3t_norm = normalize(volume_3t)
    volume_7t_norm = normalize(volume_7t)
    
    save_volume(volume_3t_norm, os.path.join(output_dirs['volume'], f'{base_name}_3t.nii.gz'))
    save_volume(volume_7t_norm, os.path.join(output_dirs['volume'], f'{base_name}_7t.nii.gz'))
    
    for direction in ['axial', 'coronal', 'sagittal']:
        mip_3t = compute_mip_with_padding(volume_3t_norm, direction, mip_thickness)
        mip_7t = compute_mip_with_padding(volume_7t_norm, direction, mip_thickness)
        
        save_volume(mip_3t, os.path.join(output_dirs[direction], f'{base_name}_3t.nii.gz'))
        save_volume(mip_7t, os.path.join(output_dirs[direction], f'{base_name}_7t.nii.gz'))


def prepare_data(data_3t_dir, data_7t_dir, output_base_dir, mip_thickness, split_name):
    samples_3t = sorted(glob.glob(os.path.join(data_3t_dir, '*.nii.gz')))
    samples_7t = sorted(glob.glob(os.path.join(data_7t_dir, '*.nii.gz')))
    
    assert len(samples_3t) == len(samples_7t), \
        f"Number of 3T ({len(samples_3t)}) and 7T ({len(samples_7t)}) samples must match"
    
    output_dirs = {
        'volume': os.path.join(output_base_dir, split_name, 'volume'),
        'axial': os.path.join(output_base_dir, split_name, 'axial'),
        'coronal': os.path.join(output_base_dir, split_name, 'coronal'),
        'sagittal': os.path.join(output_base_dir, split_name, 'sagittal'),
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"Processing {split_name} data ({len(samples_3t)} samples)...")
    
    for sample_idx in range(len(samples_3t)):
        print(f"  Processing sample {sample_idx + 1}/{len(samples_3t)}...")
        preprocess_sample(
            samples_3t[sample_idx],
            samples_7t[sample_idx],
            sample_idx,
            output_dirs,
            mip_thickness
        )
    
    print(f"  Done! Output saved to:")
    for name, path in output_dirs.items():
        print(f"    {name}: {path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare data before training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output base directory (default: project_root/preprocessed)')
    parser.add_argument('--mip_thickness', type=int, default=None,
                        help='MIP thickness')
    args = parser.parse_args()
    
    config = Config()
    
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_config", args.config)
        custom_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_config)
        for key, value in vars(custom_config).items():
            if not key.startswith('_'):
                setattr(config, key, value)
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(config.project_root, 'preprocessed')
    
    mip_thickness = args.mip_thickness if args.mip_thickness else config.mip_thickness
    
    print("=" * 60)
    print("Data Preparation Script")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"MIP thickness: {mip_thickness}")
    print()
    
    print("Preparing training data...")
    prepare_data(
        config.train_3t_dir,
        config.train_7t_dir,
        output_dir,
        mip_thickness,
        'train'
    )
    
    print()
    print("Preparing validation data...")
    prepare_data(
        config.val_3t_dir,
        config.val_7t_dir,
        output_dir,
        mip_thickness,
        'val'
    )
    
    print()
    print("Preparing test data...")
    prepare_data(
        config.test_3t_dir,
        config.test_7t_dir,
        output_dir,
        mip_thickness,
        'test'
    )
    
    print()
    print("=" * 60)
    print("Data preparation completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
