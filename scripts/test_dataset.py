import os
import sys
import time
import glob
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import MIPDataset, MultiDirDataset


def test_mip_dataset():
    print("=" * 60)
    print("Testing MIPDataset")
    print("=" * 60)
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'preprocessed', 'train', 'axial')
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please run prepare_before_train.py first.")
        return
    
    num_samples = len(glob.glob(os.path.join(data_dir, '*_3t.nii.gz')))
    if num_samples == 0:
        print("No samples found in data directory")
        return
    
    print(f"Data directory: {data_dir}")
    print(f"Number of samples: {num_samples}")
    
    dataset = MIPDataset(
        data_dir=data_dir,
        num_samples=num_samples,
        patch_size=64,
        mode='train',
        batch_size=24,
        batch_per_sample=10,
        dataset_type='axial'
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    times = []
    for i in range(min(10, len(dataset))):
        start = time.time()
        item = dataset[i]
        elapsed = time.time() - start
        times.append(elapsed)

        if i == 0:
            print(f"\nFirst item keys: {list(item.keys())}")
            for key, value in item.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
        
        print(f"Item {i}: {elapsed*1000:.2f} ms")
    
    print(f"\nAverage time: {np.mean(times)*1000:.2f} ms")
    print(f"Min time: {np.min(times)*1000:.2f} ms")
    print(f"Max time: {np.max(times)*1000:.2f} ms")


def test_multi_dir_dataset():
    print("\n" + "=" * 60)
    print("Testing MultiDirDataset")
    print("=" * 60)
    
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'preprocessed', 'train')
    
    data_dirs = {
        'volume': os.path.join(base_dir, 'volume'),
        'axial': os.path.join(base_dir, 'axial'),
        'coronal': os.path.join(base_dir, 'coronal'),
        'sagittal': os.path.join(base_dir, 'sagittal'),
    }
    
    exists = all(os.path.exists(d) for d in data_dirs.values())
    if not exists:
        print(f"Data directories not found")
        print("Please run prepare_before_train.py first.")
        return
    
    num_samples = len(glob.glob(os.path.join(data_dirs['volume'], '*_3t.nii.gz')))
    if num_samples == 0:
        print("No samples found in data directories")
        return
    
    print(f"Data directories:")
    for name, path in data_dirs.items():
        print(f"  {name}: {path}")
    print(f"Number of samples: {num_samples}")
    
    dataset = MultiDirDataset(
        data_dirs=data_dirs,
        num_samples=num_samples,
        patch_size=64,
        mode='train',
        batch_size=24,
        batch_per_sample=10
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    times = []
    for i in range(min(10, len(dataset))):
        start = time.time()
        item = dataset[i]
        elapsed = time.time() - start
        times.append(elapsed)
        
        if i == 0:
            print(f"\nFirst item keys: {list(item.keys())}")
            for key, value in item.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
        
        print(f"Item {i}: {elapsed*1000:.2f} ms")
    
    print(f"\nAverage time: {np.mean(times)*1000:.2f} ms")
    print(f"Min time: {np.min(times)*1000:.2f} ms")
    print(f"Max time: {np.max(times)*1000:.2f} ms")


if __name__ == '__main__':
    test_mip_dataset()
    test_multi_dir_dataset()
