import torch
import torch.nn.functional as F
import numpy as np


def mip_1d(volume, direction='axial', thickness=50, stride=1):
    """
    Perform Maximum Intensity Projection (MIP) along a specified direction.
    
    Args:
        volume: Input 3D volume tensor of shape (D, H, W) or (C, D, H, W)
        direction: Projection direction, one of 'axial', 'coronal', 'sagittal'
        thickness: Thickness of the slab for MIP
        stride: Stride for sliding window
    
    Returns:
        mip_volume: MIP result volume
    """
    if volume.dim() == 3:
        volume = volume.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    C, D, H, W = volume.shape
    
    if direction == 'axial':
        num_slabs = max(1, (D - thickness) // stride + 1)
        mip_slices = []
        for i in range(num_slabs):
            start = i * stride
            end = min(start + thickness, D)
            slab = volume[:, start:end, :, :]
            mip_slice = slab.max(dim=1)[0]
            mip_slices.append(mip_slice)
        mip_volume = torch.stack(mip_slices, dim=1)
        
    elif direction == 'coronal':
        num_slabs = max(1, (H - thickness) // stride + 1)
        mip_slices = []
        for i in range(num_slabs):
            start = i * stride
            end = min(start + thickness, H)
            slab = volume[:, :, start:end, :]
            mip_slice = slab.max(dim=2)[0]
            mip_slices.append(mip_slice)
        mip_volume = torch.stack(mip_slices, dim=2)
        
    elif direction == 'sagittal':
        num_slabs = max(1, (W - thickness) // stride + 1)
        mip_slices = []
        for i in range(num_slabs):
            start = i * stride
            end = min(start + thickness, W)
            slab = volume[:, :, :, start:end]
            mip_slice = slab.max(dim=3)[0]
            mip_slices.append(mip_slice)
        mip_volume = torch.stack(mip_slices, dim=3)
    
    else:
        raise ValueError(f"Unknown direction: {direction}. Must be 'axial', 'coronal', or 'sagittal'")
    
    if squeeze_output:
        mip_volume = mip_volume.squeeze(0)
    
    return mip_volume


def mip_batch(volumes, direction='axial', thickness=50, stride=1):
    """
    Perform MIP on a batch of 3D volumes.
    
    Args:
        volumes: Input tensor of shape (B, C, D, H, W)
        direction: Projection direction
        thickness: Thickness of the slab for MIP
        stride: Stride for sliding window
    
    Returns:
        mip_volumes: MIP result volumes
    """
    batch_size = volumes.size(0)
    mip_volumes = []
    
    for i in range(batch_size):
        mip_vol = mip_1d(volumes[i], direction, thickness, stride)
        mip_volumes.append(mip_vol)
    
    return torch.stack(mip_volumes, dim=0)


class MIPTransform:
    def __init__(self, thickness=50, stride=1):
        self.thickness = thickness
        self.stride = stride
        self.directions = ['axial', 'coronal', 'sagittal']
    
    def __call__(self, volume):
        """
        Apply MIP in all three directions.
        
        Args:
            volume: Input 3D volume tensor
        
        Returns:
            Dictionary containing MIP results for each direction
        """
        mip_results = {}
        for direction in self.directions:
            mip_results[direction] = mip_1d(volume, direction, self.thickness, self.stride)
        return mip_results
    
    def get_mip_images(self, volume):
        """
        Get MIP images for all three directions.
        
        Args:
            volume: Input 3D volume tensor of shape (D, H, W) or (B, C, D, H, W)
        
        Returns:
            Tuple of (axial_mip, coronal_mip, sagittal_mip)
        """
        if volume.dim() == 3:
            axial = mip_1d(volume, 'axial', self.thickness, self.stride)
            coronal = mip_1d(volume, 'coronal', self.thickness, self.stride)
            sagittal = mip_1d(volume, 'sagittal', self.thickness, self.stride)
        else:
            axial = mip_batch(volume, 'axial', self.thickness, self.stride)
            coronal = mip_batch(volume, 'coronal', self.thickness, self.stride)
            sagittal = mip_batch(volume, 'sagittal', self.thickness, self.stride)
        
        return axial, coronal, sagittal


class MIPDataset:
    def __init__(self, thickness=50, stride=1):
        self.mip_transform = MIPTransform(thickness, stride)
    
    def create_mip_patches(self, volume_3t, volume_7t, patch_size=64, stride=32):
        """
        Create MIP patches from 3D volumes.
        
        Args:
            volume_3t: 3T volume tensor
            volume_7t: 7T volume tensor
            patch_size: Size of 3D patches
            stride: Stride for patch extraction
        
        Returns:
            List of dictionaries containing patches and MIP results
        """
        D, H, W = volume_3t.shape
        
        patches = []
        
        for d in range(0, D - patch_size + 1, stride):
            for h in range(0, H - patch_size + 1, stride):
                for w in range(0, W - patch_size + 1, stride):
                    patch_3t = volume_3t[d:d+patch_size, h:h+patch_size, w:w+patch_size]
                    patch_7t = volume_7t[d:d+patch_size, h:h+patch_size, w:w+patch_size]
                    
                    mip_3t = self.mip_transform(patch_3t)
                    mip_7t = self.mip_transform(patch_7t)
                    
                    patches.append({
                        'patch_3t': patch_3t,
                        'patch_7t': patch_7t,
                        'mip_3t': mip_3t,
                        'mip_7t': mip_7t,
                        'location': (d, h, w)
                    })
        
        return patches


def create_mip_volume(volume, thickness=50, stride=1, direction='axial'):
    """
    Create a MIP volume by performing MIP on sliding slabs.
    
    Args:
        volume: Input 3D volume tensor of shape (D, H, W)
        thickness: Thickness of each slab
        stride: Stride for sliding window
        direction: Projection direction
    
    Returns:
        MIP volume
    """
    return mip_1d(volume, direction, thickness, stride)


if __name__ == '__main__':
    volume = torch.randn(64, 64, 64)
    
    mip_transform = MIPTransform(thickness=50, stride=1)
    mip_results = mip_transform(volume)
    
    print("Original volume shape:", volume.shape)
    print("\nMIP results:")
    for direction, mip_vol in mip_results.items():
        print(f"  {direction}: {mip_vol.shape}")
    
    axial, coronal, sagittal = mip_transform.get_mip_images(volume)
    print(f"\nSeparate MIP images:")
    print(f"  Axial: {axial.shape}")
    print(f"  Coronal: {coronal.shape}")
    print(f"  Sagittal: {sagittal.shape}")
    
    batch_volume = torch.randn(4, 1, 64, 64, 64)
    batch_mip = mip_batch(batch_volume, 'axial', thickness=50, stride=1)
    print(f"\nBatch MIP shape: {batch_mip.shape}")
