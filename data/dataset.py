import os
import torch
import numpy as np
from torch.utils.data import Dataset
import ants


class MIPDataset(Dataset):
    def __init__(self, data_dir, num_samples, patch_size=64, mode='train',
                 batch_size=4, batch_per_sample=10, dataset_type='volume'):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.mode = mode
        self.batch_size = batch_size
        self.batch_per_sample = batch_per_sample
        self.dataset_type = dataset_type
    
    def _get_file_paths(self, sample_idx):
        base_name = f'sample_{sample_idx:04d}'
        return {
            'vol_3t': os.path.join(self.data_dir, f'{base_name}_3t.nii.gz'),
            'vol_7t': os.path.join(self.data_dir, f'{base_name}_7t.nii.gz'),
        }
    
    def _load_volume(self, filepath):
        img = ants.image_read(filepath)
        data = img.numpy()
        return data.astype(np.float32)
    
    def _extract_patch(self, volume, center):
        d, h, w = center
        D, H, W = volume.shape
        half = self.patch_size // 2
        
        patch = np.zeros((self.patch_size, self.patch_size, self.patch_size), dtype=volume.dtype)
        
        d_start = max(0, d - half)
        h_start = max(0, h - half)
        w_start = max(0, w - half)
        d_end = min(D, d + half)
        h_end = min(H, h + half)
        w_end = min(W, w + half)
        
        patch_d_start = max(0, half - d)
        patch_h_start = max(0, half - h)
        patch_w_start = max(0, half - w)
        patch_d_end = patch_d_start + (d_end - d_start)
        patch_h_end = patch_h_start + (h_end - h_start)
        patch_w_end = patch_w_start + (w_end - w_start)
        
        patch[patch_d_start:patch_d_end, patch_h_start:patch_h_end, patch_w_start:patch_w_end] = \
            volume[d_start:d_end, h_start:h_end, w_start:w_end]
        
        return patch
    
    def _sample_centers(self, volume, sample_idx, batch_idx):
        D, H, W = volume.shape
        half = self.patch_size // 2
        
        if self.mode == 'train':
            np.random.seed()
        else:
            np.random.seed(sample_idx * 1000 + batch_idx)
        
        centers = []
        max_attempts = self.batch_size * 100
        
        d_range = (half, max(half + 1, D - half))
        h_range = (half, max(half + 1, H - half))
        w_range = (half, max(half + 1, W - half))
        
        attempts = 0
        while len(centers) < self.batch_size and attempts < max_attempts:
            d = np.random.randint(d_range[0], d_range[1])
            h = np.random.randint(h_range[0], h_range[1])
            w = np.random.randint(w_range[0], w_range[1])
            
            if volume[d, h, w] > 0:
                centers.append((d, h, w))
            
            attempts += 1
        
        while len(centers) < self.batch_size:
            d = np.random.randint(d_range[0], d_range[1])
            h = np.random.randint(h_range[0], h_range[1])
            w = np.random.randint(w_range[0], w_range[1])
            centers.append((d, h, w))
        
        return centers
    
    def __len__(self):
        return self.num_samples * self.batch_per_sample
    
    def __getitem__(self, idx):
        sample_idx = idx // self.batch_per_sample
        batch_idx = idx % self.batch_per_sample
        
        file_paths = self._get_file_paths(sample_idx)
        
        vol_3t = self._load_volume(file_paths['vol_3t'])
        vol_7t = self._load_volume(file_paths['vol_7t'])
        
        centers = self._sample_centers(vol_3t, sample_idx, batch_idx)
        
        patches_3t = []
        patches_7t = []
        
        for center in centers:
            patches_3t.append(torch.from_numpy(self._extract_patch(vol_3t, center)).unsqueeze(0))
            patches_7t.append(torch.from_numpy(self._extract_patch(vol_7t, center)).unsqueeze(0))
        
        if self.dataset_type == 'volume':
            key_3t = 'patch_3t'
            key_7t = 'patch_7t'
        else:
            key_3t = f'mip_3t_{self.dataset_type}'
            key_7t = f'mip_7t_{self.dataset_type}'
        
        return {
            key_3t: torch.stack(patches_3t, dim=0),
            key_7t: torch.stack(patches_7t, dim=0),
            'sample_idx': sample_idx,
            'batch_idx': batch_idx,
            'locations': centers
        }


class MultiDirDataset(Dataset):
    def __init__(self, data_dirs, num_samples, patch_size=64, mode='train',
                 batch_size=4, batch_per_sample=10):
        self.data_dirs = data_dirs
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.mode = mode
        self.batch_size = batch_size
        self.batch_per_sample = batch_per_sample
        
        self.dataset_types = list(data_dirs.keys())
    
    def _get_file_paths(self, sample_idx, data_type):
        base_name = f'sample_{sample_idx:04d}'
        data_dir = self.data_dirs[data_type]
        return {
            '3t': os.path.join(data_dir, f'{base_name}_3t.nii.gz'),
            '7t': os.path.join(data_dir, f'{base_name}_7t.nii.gz'),
        }
    
    def _load_volume(self, filepath):
        img = ants.image_read(filepath)
        data = img.numpy()
        return data.astype(np.float32)
    
    def _extract_patch(self, volume, center):
        d, h, w = center
        D, H, W = volume.shape
        half = self.patch_size // 2
        
        patch = np.zeros((self.patch_size, self.patch_size, self.patch_size), dtype=volume.dtype)
        
        d_start = max(0, d - half)
        h_start = max(0, h - half)
        w_start = max(0, w - half)
        d_end = min(D, d + half)
        h_end = min(H, h + half)
        w_end = min(W, w + half)
        
        patch_d_start = max(0, half - d)
        patch_h_start = max(0, half - h)
        patch_w_start = max(0, half - w)
        patch_d_end = patch_d_start + (d_end - d_start)
        patch_h_end = patch_h_start + (h_end - h_start)
        patch_w_end = patch_w_start + (w_end - w_start)
        
        patch[patch_d_start:patch_d_end, patch_h_start:patch_h_end, patch_w_start:patch_w_end] = \
            volume[d_start:d_end, h_start:h_end, w_start:w_end]
        
        return patch
    
    def _sample_centers(self, volume, sample_idx, batch_idx):
        D, H, W = volume.shape
        half = self.patch_size // 2
        
        if self.mode == 'train':
            np.random.seed()
        else:
            np.random.seed(sample_idx * 1000 + batch_idx)
        
        centers = []
        max_attempts = self.batch_size * 100
        
        d_range = (half, max(half + 1, D - half))
        h_range = (half, max(half + 1, H - half))
        w_range = (half, max(half + 1, W - half))
        
        attempts = 0
        while len(centers) < self.batch_size and attempts < max_attempts:
            d = np.random.randint(d_range[0], d_range[1])
            h = np.random.randint(h_range[0], h_range[1])
            w = np.random.randint(w_range[0], w_range[1])
            
            if volume[d, h, w] > 0:
                centers.append((d, h, w))
            
            attempts += 1
        
        while len(centers) < self.batch_size:
            d = np.random.randint(d_range[0], d_range[1])
            h = np.random.randint(h_range[0], h_range[1])
            w = np.random.randint(w_range[0], w_range[1])
            centers.append((d, h, w))
        
        return centers
    
    def __len__(self):
        return self.num_samples * self.batch_per_sample
    
    def __getitem__(self, idx):
        sample_idx = idx // self.batch_per_sample
        batch_idx = idx % self.batch_per_sample
        
        first_type = self.dataset_types[0]
        first_paths = self._get_file_paths(sample_idx, first_type)
        first_vol = self._load_volume(first_paths['3t'])
        
        centers = self._sample_centers(first_vol, sample_idx, batch_idx)
        
        result = {
            'sample_idx': sample_idx,
            'batch_idx': batch_idx,
            'locations': centers
        }
        
        for data_type in self.dataset_types:
            paths = self._get_file_paths(sample_idx, data_type)
            vol_3t = self._load_volume(paths['3t'])
            vol_7t = self._load_volume(paths['7t'])
            
            patches_3t = []
            patches_7t = []
            
            for center in centers:
                patches_3t.append(torch.from_numpy(self._extract_patch(vol_3t, center)).unsqueeze(0))
                patches_7t.append(torch.from_numpy(self._extract_patch(vol_7t, center)).unsqueeze(0))
            
            if data_type == 'volume':
                result['patch_3t'] = torch.stack(patches_3t, dim=0)
                result['patch_7t'] = torch.stack(patches_7t, dim=0)
            else:
                result[f'mip_3t_{data_type}'] = torch.stack(patches_3t, dim=0)
                result[f'mip_7t_{data_type}'] = torch.stack(patches_7t, dim=0)
        
        return result
