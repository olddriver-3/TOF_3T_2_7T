"""
推理预处理Pipeline - 处理陌生的3T数据到指定格式用于生成7T图像

预处理步骤：
1. N4偏置场校正
2. 颅骨去除
3. 重采样到目标分辨率
4. 裁剪或填充到指定形状
5. 强度归一化到 [0, 255]

输入和输出都是numpy数组
"""

import numpy as np
from scipy.ndimage import binary_fill_holes, label
import ants


class InferencePreprocessPipeline:
    def __init__(self,
                 target_shape=(512, 512, 320),
                 target_resolution=(0.4, 0.4, 0.4),
                 do_bias_correction=True,
                 do_skull_stripping=True):
        self.target_shape = np.array(target_shape)
        self.target_resolution = np.array(target_resolution)
        self.do_bias_correction = do_bias_correction
        self.do_skull_stripping = do_skull_stripping

    def _numpy_to_ants(self, data, spacing=None, origin=None, direction=None):
        if spacing is None:
            spacing = [1.0] * data.ndim
        if origin is None:
            origin = [0.0] * data.ndim
        if direction is None:
            direction = np.eye(data.ndim).flatten()
        
        return ants.from_numpy(
            data.astype(np.float32),
            spacing=spacing,
            origin=origin,
            direction=direction
        )

    def _ants_to_numpy(self, image):
        return image.numpy()

    def bias_field_correction(self, image):
        mask = image > image.mean()
        corrected = ants.n4_bias_field_correction(image, mask=mask, verbose=False)
        return corrected

    def skull_stripping(self, image):
        data = image.numpy()
        threshold = np.percentile(data[data > 0], 10)
        mask = image > threshold
        
        mask_array = mask.numpy().astype(np.int32)
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

    def resample_and_pad_crop(self, image):
        resampled = ants.resample_image(
            image,
            self.target_resolution,
            use_voxels=False,
            interp_type=1
        )
        
        current_shape = np.array(resampled.shape)
        pad_or_crop = self.target_shape - current_shape
        
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
            for i, (curr, target) in enumerate(zip(current_shape, self.target_shape)):
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

    def normalize_intensity(self, image):
        data = image.numpy()
        data_min, data_max = data.min(), data.max()
        if data_max - data_min > 0:
            data = (data - data_min) / (data_max - data_min)
        return ants.from_numpy(
            data.astype(np.float32),
            spacing=image.spacing,
            origin=image.origin,
            direction=image.direction
        )

    def __call__(self, data, spacing=None, origin=None, direction=None):
        image = self._numpy_to_ants(data, spacing, origin, direction)
        
        if self.do_bias_correction:
            image = self.bias_field_correction(image)
        
        if self.do_skull_stripping:
            image = self.skull_stripping(image)
        
        image = self.resample_and_pad_crop(image)
        image = self.normalize_intensity(image)
        
        result = self._ants_to_numpy(image)
        
        return result, {
            'spacing': image.spacing,
            'origin': image.origin,
            'direction': image.direction
        }


def preprocess_3t_for_inference(data,
                                 spacing=None,
                                 origin=None,
                                 direction=None,
                                 target_shape=(512, 512, 320),
                                 target_resolution=(0.4, 0.4, 0.4),
                                 do_bias_correction=True,
                                 do_skull_stripping=True):
    pipeline = InferencePreprocessPipeline(
        target_shape=target_shape,
        target_resolution=target_resolution,
        do_bias_correction=do_bias_correction,
        do_skull_stripping=do_skull_stripping
    )
    return pipeline(data, spacing, origin, direction)


if __name__ == '__main__':
    print("=" * 60)
    print("推理预处理Pipeline示例")
    print("=" * 60)
    
    print("\n示例1: 基本使用")
    print("-" * 40)
    
    synthetic_3t = np.random.rand(256, 256, 160).astype(np.float32) * 100
    synthetic_3t[50:200, 50:200, 40:120] += 500
    
    print(f"输入数据形状: {synthetic_3t.shape}")
    print(f"输入数据范围: [{synthetic_3t.min():.2f}, {synthetic_3t.max():.2f}]")
    
    processed, metadata = preprocess_3t_for_inference(
        synthetic_3t,
        spacing=[0.5, 0.5, 0.5],
        origin=[0.0, 0.0, 0.0],
        target_shape=(512, 512, 320),
        target_resolution=(0.4, 0.4, 0.4)
    )
    
    print(f"输出数据形状: {processed.shape}")
    print(f"输出数据范围: [{processed.min():.2f}, {processed.max():.2f}]")
    print(f"输出spacing: {metadata['spacing']}")
    print(f"输出origin: {metadata['origin']}")
    
    print("\n示例2: 使用Pipeline类")
    print("-" * 40)
    
    pipeline = InferencePreprocessPipeline(
        target_shape=(256, 256, 128),
        target_resolution=(0.5, 0.5, 0.5),
        do_bias_correction=True,
        do_skull_stripping=True
    )
    
    test_data = np.random.rand(200, 200, 100).astype(np.float32) * 200
    processed2, metadata2 = pipeline(test_data)
    
    print(f"输入数据形状: {test_data.shape}")
    print(f"输出数据形状: {processed2.shape}")
    print(f"输出数据范围: [{processed2.min():.2f}, {processed2.max():.2f}]")
    
    print("\n示例3: 跳过某些步骤")
    print("-" * 40)
    
    processed3, metadata3 = preprocess_3t_for_inference(
        synthetic_3t,
        spacing=[0.5, 0.5, 0.5],
        do_bias_correction=False,
        do_skull_stripping=False,
        target_shape=(256, 256, 128)
    )
    
    print(f"跳过N4校正和颅骨去除后的输出形状: {processed3.shape}")
    
    print("\n示例4: 与深度学习模型集成")
    print("-" * 40)
    print("""
# 典型的推理流程
import torch

# 1. 加载原始3T数据 (假设从NIfTI文件读取)
import nibabel as nib
nii = nib.load('subject_001_3t.nii.gz')
data_3t = nii.get_fdata()
spacing = nii.header.get_zooms()[:3]
origin = nii.affine[:3, 3]
direction = nii.affine[:3, :3].T.flatten()

# 2. 预处理
processed, meta = preprocess_3t_for_inference(
    data_3t,
    spacing=spacing,
    origin=origin,
    direction=direction
)

# 3. 转换为模型输入
input_tensor = torch.from_numpy(processed).float()
input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]

# 4. 模型推理
with torch.no_grad():
    output_7t = model(input_tensor)

# 5. 后处理 (如果需要恢复原始空间)
output_7t_np = output_7t.squeeze().cpu().numpy()
""")
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)
