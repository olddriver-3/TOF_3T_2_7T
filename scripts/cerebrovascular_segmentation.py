"""
Cerebrovascular Segmentation Algorithms
Based on: "A comparative analysis framework of 3T and 7T TOF-MRA based on automated cerebrovascular segmentation"
Paper: Computerized Medical Imaging and Graphics 89 (2021) 101830

This module implements:
- Algorithm 1: Automatic cerebrovascular segmentation of TOF-MRA
- Algorithm 2: Non-vascular brain-tissue and noise regions segmentation
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, label
from skimage.morphology import skeletonize_3d
from collections import deque
import warnings

warnings.filterwarnings('ignore')

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter
    from cupyx.scipy.ndimage import median_filter as gpu_median_filter
    from cupyx.scipy.ndimage import minimum_filter as gpu_minimum_filter
    from cupyx.scipy.ndimage import maximum_filter as gpu_maximum_filter
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

def is_gpu_available():
    if not CUPY_AVAILABLE:
        return False
    try:
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False

USE_GPU = is_gpu_available()

def to_gpu(arr):
    if USE_GPU and isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr

def to_cpu(arr):
    if USE_GPU and not isinstance(arr, np.ndarray):
        return cp.asnumpy(arr)
    return arr

def free_gpu_memory():
    if USE_GPU:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


# ==============================================================================
# Default Parameters Configuration
# ==============================================================================

DEFAULT_ALGORITHM1_PARAMS = {
    'denoise': True,
    'denoise_window_size': 1,
    'denoise_max_iterations': 0,
    'sigmas': [1,3],
    'frangi_alpha': 0.5,
    'frangi_k': 500,
    'tau': 0.5,
    'threshold': 0.3,
    'min_skeleton_area': 10,
    'region_threshold_low': 0.4,
    'region_threshold_high': 10,
    'region_max_iterations': 1000000,
    'enhancement_method': 'improved',
    'verbose': True
}

DEFAULT_ALGORITHM2_PARAMS = {
    'brain_threshold_low': 0.00,
    'brain_threshold_high': 1,
    'verbose': True
}

DEFAULT_FRANGI_PARAMS = {
    'alpha': 0.5,
    'k': 500
}



# ==============================================================================
# Module 1: Preprocessing Functions (公式模块 - 可编辑)
# ==============================================================================

def nafsm_filter(image, window_size=3, max_iterations=1, use_gpu=None):
    """
    Noise Adaptive Fuzzy Switching Median Filter (NAFSM)
    去噪预处理模块 - 用于消除MRA成像过程中产生的噪声
    
    Reference: Toh and Isa, 2010
    
    Parameters:
    -----------
    image : ndarray
        Input MRA image
    window_size : int
        Size of the filtering window (default: 3)
    max_iterations : int
        Maximum number of iterations
    use_gpu : bool, optional
        Force GPU usage. If None, auto-detect.
        
    Returns:
    --------
    filtered_image : ndarray
        Denoised image
    """
    gpu_enabled = USE_GPU if use_gpu is None else use_gpu
    
    if gpu_enabled:
        from cupyx.scipy.ndimage import median_filter as gpu_median_filter
        from cupyx.scipy.ndimage import minimum_filter as gpu_minimum_filter
        from cupyx.scipy.ndimage import maximum_filter as gpu_maximum_filter
        
        filtered_gpu = to_gpu(image.copy())
        
        for iteration in range(max_iterations):
            median_vals = gpu_median_filter(filtered_gpu, size=window_size, mode='reflect')
            min_vals = gpu_minimum_filter(filtered_gpu, size=window_size, mode='reflect')
            max_vals = gpu_maximum_filter(filtered_gpu, size=window_size, mode='reflect')
            
            is_noise = (filtered_gpu == min_vals) | (filtered_gpu == max_vals)
            
            new_filtered = cp.where(is_noise, median_vals, filtered_gpu)
            
            if cp.array_equal(filtered_gpu, new_filtered):
                del median_vals, min_vals, max_vals, is_noise, new_filtered
                break
            
            del filtered_gpu, median_vals, min_vals, max_vals, is_noise
            filtered_gpu = new_filtered
        
        result = to_cpu(filtered_gpu)
        del filtered_gpu
        free_gpu_memory()
        return result
    else:
        from scipy.ndimage import median_filter, minimum_filter, maximum_filter
        
        filtered = image.copy()
        
        for iteration in range(max_iterations):
            median_vals = median_filter(filtered, size=window_size, mode='reflect')
            min_vals = minimum_filter(filtered, size=window_size, mode='reflect')
            max_vals = maximum_filter(filtered, size=window_size, mode='reflect')
            
            is_noise = (filtered == min_vals) | (filtered == max_vals)
            
            new_filtered = np.where(is_noise, median_vals, filtered)
            
            if np.array_equal(filtered, new_filtered):
                break
            
            filtered = new_filtered
        
        return filtered


# ==============================================================================
# Module 2: Hessian Matrix Functions (公式模块 - 可编辑)
# ==============================================================================

def compute_hessian_matrix(image, sigma, use_gpu=None):
    """
    计算Hessian矩阵 - 公式(1)
    
    H_ij(x,s) = s^2 * I(x) * ∂²/∂x_i∂x_j G(x,s)
    
    其中 G(x,s) = (1/(2πs²)^(D/2)) * exp(-x^T x / 2s²)
    
    Parameters:
    -----------
    image : ndarray
        Input image
    sigma : float
        Scale parameter for Gaussian derivative
    use_gpu : bool, optional
        Force GPU usage. If None, auto-detect.
        
    Returns:
    --------
    hessian : ndarray
        Hessian matrix at each voxel, shape: (H, W, D, 3, 3)
    """
    gpu_enabled = USE_GPU if use_gpu is None else use_gpu
    if gpu_enabled:
        image_gpu = to_gpu(image.astype(np.float32))
        smoothed = gpu_gaussian_filter(image_gpu, sigma=sigma)
        
        grad_z = cp.gradient(smoothed, axis=0)
        grad_y = cp.gradient(smoothed, axis=1)
        grad_x = cp.gradient(smoothed, axis=2)
        
        hessian_gpu = cp.zeros((*image.shape, 3, 3), dtype=np.float32)
        
        hessian_gpu[..., 0, 0] = sigma**2 * cp.gradient(grad_z, axis=0)
        hessian_gpu[..., 1, 1] = sigma**2 * cp.gradient(grad_y, axis=1)
        hessian_gpu[..., 2, 2] = sigma**2 * cp.gradient(grad_x, axis=2)
        
        hessian_gpu[..., 0, 1] = hessian_gpu[..., 1, 0] = sigma**2 * cp.gradient(grad_z, axis=1)
        hessian_gpu[..., 0, 2] = hessian_gpu[..., 2, 0] = sigma**2 * cp.gradient(grad_z, axis=2)
        hessian_gpu[..., 1, 2] = hessian_gpu[..., 2, 1] = sigma**2 * cp.gradient(grad_y, axis=2)
        
        result = to_cpu(hessian_gpu)
        del image_gpu, smoothed, grad_z, grad_y, grad_x, hessian_gpu
        free_gpu_memory()
        return result
    else:
        smoothed = gaussian_filter(image.astype(np.float32), sigma=sigma)
        
        grad_z = np.gradient(smoothed, axis=0)
        grad_y = np.gradient(smoothed, axis=1)
        grad_x = np.gradient(smoothed, axis=2)
        
        hessian = np.zeros((*image.shape, 3, 3), dtype=np.float32)
        
        hessian[..., 0, 0] = sigma**2 * np.gradient(grad_z, axis=0)
        hessian[..., 1, 1] = sigma**2 * np.gradient(grad_y, axis=1)
        hessian[..., 2, 2] = sigma**2 * np.gradient(grad_x, axis=2)
        
        hessian[..., 0, 1] = hessian[..., 1, 0] = sigma**2 * np.gradient(grad_z, axis=1)
        hessian[..., 0, 2] = hessian[..., 2, 0] = sigma**2 * np.gradient(grad_z, axis=2)
        hessian[..., 1, 2] = hessian[..., 2, 1] = sigma**2 * np.gradient(grad_y, axis=2)
        
        return hessian


def compute_eigenvalues(hessian, use_gpu=None):
    """
    计算Hessian矩阵的特征值
    
    Parameters:
    -----------
    hessian : ndarray
        Hessian matrix at each voxel, shape: (H, W, D, 3, 3)
    use_gpu : bool, optional
        Force GPU usage. If None, auto-detect.
        
    Returns:
    --------
    eigenvalues : ndarray
        Sorted eigenvalues by absolute value, shape: (H, W, D, 3)
        λ1, λ2, λ3 where |λ1| ≤ |λ2| ≤ |λ3|
        注意：保留原始符号，仅按绝对值排序
    """
    gpu_enabled = USE_GPU if use_gpu is None else use_gpu
    
    if gpu_enabled:
        hessian_gpu = to_gpu(hessian)
        eigenvalues_gpu = cp.linalg.eigvalsh(hessian_gpu)
        sorted_indices = cp.argsort(cp.abs(eigenvalues_gpu), axis=-1)
        eigenvalues_gpu = cp.take_along_axis(eigenvalues_gpu, sorted_indices, axis=-1)
        result = to_cpu(eigenvalues_gpu)
        del hessian_gpu, eigenvalues_gpu, sorted_indices
        free_gpu_memory()
        return result
    else:
        eigenvalues = np.linalg.eigvals(hessian)
        sorted_indices = np.argsort(np.abs(eigenvalues), axis=-1)
        eigenvalues = np.take_along_axis(eigenvalues, sorted_indices, axis=-1)

    return eigenvalues


# ==============================================================================
# Module 3: Vessel Enhancement Functions (公式模块 - 可编辑)
# ==============================================================================

def redefine_eigenvalue(eigenvals, structure_type='bright'):
    """
    重新定义特征值 - 公式(2)
    
    根据结构类型(亮结构或暗结构)重新定义特征值
    
    Parameters:
    -----------
    lambda_i : float
        Original eigenvalue
    structure_type : str
        'bright' for bright structure on dark background
        'dark' for dark structure on bright background
        
    Returns:
    --------
    redefined_lambda : float
        Redefined eigenvalue
    """
    if structure_type == 'bright':
        return -eigenvals
    else:
        return eigenvals


def compute_lambda_rho(lambda_3, tau=0.5, use_gpu=None):
    """
    计算修正的特征值 λ_ρ - 公式(5)
    
    λ_ρ = {
        λ_3,                    if λ_3 > τ * max_x λ_3
        τ * max_x λ_3,          if 0 < λ_3 ≤ τ * max_x λ_3
        0,                      otherwise
    }
    
    Parameters:
    -----------
    lambda_3 : ndarray
        Third eigenvalue at each voxel
    tau : float
        Cutoff threshold between 0 and 1
    use_gpu : bool, optional
        Force GPU usage. If None, auto-detect.
        
    Returns:
    --------
    lambda_rho : ndarray
        Corrected eigenvalue
    """
    gpu_enabled = USE_GPU if use_gpu is None else use_gpu
    
    if gpu_enabled:
        lambda_3_gpu = to_gpu(lambda_3)
        max_lambda_3 = cp.max(lambda_3_gpu)
        lambda_rho_gpu = cp.zeros_like(lambda_3_gpu)
        
        mask_high = lambda_3_gpu > tau * max_lambda_3
        mask_mid = (lambda_3_gpu > 0) & (lambda_3_gpu <= tau * max_lambda_3)
        
        lambda_rho_gpu[mask_high] = lambda_3_gpu[mask_high]
        lambda_rho_gpu[mask_mid] = tau * max_lambda_3
        
        result = to_cpu(lambda_rho_gpu)
        del lambda_3_gpu, lambda_rho_gpu, mask_high, mask_mid
        free_gpu_memory()
        return result
    else:
        max_lambda_3 = np.max(lambda_3)
        lambda_rho = np.zeros_like(lambda_3)
        
        mask_high = lambda_3 > tau * max_lambda_3
        mask_mid = (lambda_3 > 0) & (lambda_3 <= tau * max_lambda_3)
        
        lambda_rho[mask_high] = lambda_3[mask_high]
        lambda_rho[mask_mid] = tau * max_lambda_3
        
        return lambda_rho


def vessel_enhancement_frangi(eigenvalues, alpha=0.5, k=500, use_gpu=None):
    """
    Frangi血管增强函数 - 公式(4)
    
    V_F = (1 - exp(-R_A² / (2α²))) * (1 - exp(-S² / (2k²)))
    
    其中:
    S = √(λ1² + λ2² + λ3²)
    R_A = λ2 / λ3
    
    对于MRA中的亮血管结构 (管状):
    - 血管在MRA中是亮结构
    - Hessian矩阵的特征值: λ1 ≈ 0, λ2 < 0, λ3 < 0
    - 按绝对值排序后: |λ1| ≤ |λ2| ≤ |λ3|
    
    Parameters:
    -----------
    eigenvalues : ndarray
        Sorted eigenvalues by absolute value, shape: (H, W, D, 3)
    alpha, k : float
        Parameters controlling sensitivity
    use_gpu : bool, optional
        Force GPU usage. If None, auto-detect.
        
    Returns:
    --------
    enhanced : ndarray
        Enhanced vessel image
    """
    gpu_enabled = USE_GPU if use_gpu is None else use_gpu
    
    if gpu_enabled:
        eigenvalues_gpu = to_gpu(eigenvalues)
        
        lambda_1 = cp.abs(eigenvalues_gpu[..., 0])
        lambda_2 = cp.abs(eigenvalues_gpu[..., 1])
        lambda_3 = cp.abs(eigenvalues_gpu[..., 2])
        
        lambda_2_safe = cp.maximum(lambda_2, 1e-10)
        lambda_3_safe = cp.maximum(lambda_3, 1e-10)
        
        S = cp.sqrt(lambda_1**2 + lambda_2_safe**2 + lambda_3_safe**2)
        R_A = lambda_2_safe / lambda_3_safe
        
        term1 = 1 - cp.exp(-(R_A**2) / (2 * alpha**2))
        term3 = 1 - cp.exp(-(S**2) / (2 * k**2))
        
        enhanced_gpu = term1 * term3
        
        is_bright_structure = (eigenvalues_gpu[..., 1] < 0) & (eigenvalues_gpu[..., 2] < 0)
        enhanced_gpu[~is_bright_structure] = 0
        
        result = to_cpu(enhanced_gpu)
        del eigenvalues_gpu, lambda_1, lambda_2, lambda_3, lambda_2_safe, lambda_3_safe
        del S, R_A, term1, term3, enhanced_gpu, is_bright_structure
        free_gpu_memory()
        return result
    else:
        lambda_1 = np.abs(eigenvalues[..., 0])
        lambda_2 = np.abs(eigenvalues[..., 1])
        lambda_3 = np.abs(eigenvalues[..., 2])
        
        lambda_2_safe = np.maximum(lambda_2, 1e-10)
        lambda_3_safe = np.maximum(lambda_3, 1e-10)
        
        S = np.sqrt(lambda_1**2 + lambda_2_safe**2 + lambda_3_safe**2)
        R_A = lambda_2_safe / lambda_3_safe
        
        term1 = 1 - np.exp(-(R_A**2) / (2 * alpha**2))
        term3 = 1 - np.exp(-(S**2) / (2 * k**2))
        
        enhanced = term1 * term3
        
        is_bright_structure = (eigenvalues[..., 1] < 0) & (eigenvalues[..., 2] < 0)
        enhanced[~is_bright_structure] = 0
        
        return enhanced


def vessel_enhancement_improved(eigenvalues, tau=0.5, use_gpu=None):
    """
    改进的血管增强函数 VP - 公式(6)
    
    VP = {
        0,                              if λ2 ≤ 0 ∨ λρ ≤ 0
        1,                              if λ2 >= 0.5*λρ > 0
        27 * λ2² * (λp - λ2) / (λ2 + λp)^3,    otherwise
    }
    
    对于MRA中的亮血管结构 (管状):
    - 血管在MRA中是亮结构
    - Hessian矩阵的特征值: λ1 ≈ 0, λ2 < 0, λ3 < 0
    - 按绝对值排序后: |λ1| ≤ |λ2| ≤ |λ3|
    
    Parameters:
    -----------
    eigenvalues : ndarray
        Sorted eigenvalues by absolute value, shape: (H, W, D, 3)
    tau : float
        Cutoff threshold for lambda_rho calculation
    use_gpu : bool, optional
        Force GPU usage. If None, auto-detect.
        
    Returns:
    --------
    enhanced : ndarray
        Enhanced vessel image
    """
    gpu_enabled = USE_GPU if use_gpu is None else use_gpu
    
    if gpu_enabled:
        eigenvalues_gpu = to_gpu(eigenvalues)
        
        lambda_1 = eigenvalues_gpu[..., 0]
        lambda_2 = eigenvalues_gpu[..., 1]
        lambda_3 = eigenvalues_gpu[..., 2]

        max_lambda_3 = cp.max(lambda_3)
        lambda_rho = cp.zeros_like(lambda_3)
        mask_high = lambda_3 > tau * max_lambda_3
        mask_mid = (lambda_3 > 0) & (lambda_3 <= tau * max_lambda_3)
        lambda_rho[mask_high] = lambda_3[mask_high]
        lambda_rho[mask_mid] = tau * max_lambda_3
        
        enhanced_gpu = cp.zeros_like(lambda_1)
        
        mask_zero = (lambda_2 <= 0) | (lambda_rho <= 0)
        mask_one = (lambda_2 >= 0.5 * lambda_rho) & (lambda_rho > 0)
        mask_other = ~mask_zero & ~mask_one
        
        enhanced_gpu[mask_one] = 1.0
        
        if cp.any(mask_other):
            lambda_2_other = lambda_2[mask_other]
            lambda_rho_other = lambda_rho[mask_other]

            denominator = (lambda_2_other + lambda_rho_other) ** 3
            safe_denom = cp.where(denominator != 0, denominator, 1e-12)

            numerator = 27 * (lambda_2_other ** 2) * (lambda_rho_other - lambda_2_other)
            vp_other = numerator / safe_denom

            enhanced_gpu[mask_other] = cp.clip(vp_other, 0.0, 1.0)
        
        result = to_cpu(enhanced_gpu)
        del eigenvalues_gpu, lambda_1, lambda_2, lambda_3, lambda_rho
        del mask_high, mask_mid, mask_zero, mask_one, mask_other, enhanced_gpu
        free_gpu_memory()
        return result
    else:
        lambda_1 = eigenvalues[..., 0]
        lambda_2 = eigenvalues[..., 1]
        lambda_3 = eigenvalues[..., 2]

        lambda_rho = compute_lambda_rho(lambda_3, tau)

        enhanced = np.zeros_like(lambda_1)

        mask_zero = (lambda_2 <= 0) | (lambda_rho <= 0)

        mask_one = (lambda_2 >= 0.5 * lambda_rho) & (lambda_rho > 0)

        mask_other = ~mask_zero & ~mask_one

        enhanced[mask_one] = 1.0

        if np.any(mask_other):
            lambda_2_other = lambda_2[mask_other]
            lambda_rho_other = lambda_rho[mask_other]

            denominator = (lambda_2_other + lambda_rho_other) ** 3
            safe_denom = np.where(denominator != 0, denominator, 1e-12)

            numerator = 27 * (lambda_2_other ** 2) * (lambda_rho_other - lambda_2_other)
            vp_other = numerator / safe_denom

            enhanced[mask_other] = np.clip(vp_other, 0.0, 1.0)

    return enhanced


def multi_scale_vessel_enhancement(image, sigmas=[1.0, 2.0, 3.0], tau=0.5, method='improved',
                                   frangi_alpha=0.5, frangi_k=500, use_gpu=None):
    """
    多尺度血管增强
    
    Parameters:
    -----------
    image : ndarray
        Input image
    sigmas : list
        List of scale parameters
    tau : float
        Cutoff threshold for improved method
    method : str
        'improved' for VP function, 'frangi' for Frangi function
    frangi_alpha : float
        Frangi parameter alpha
    frangi_k : float
        Frangi parameter k
    use_gpu : bool, optional
        Force GPU usage. If None, auto-detect.
        
    Returns:
    --------
    enhanced : ndarray
        Multi-scale enhanced vessel image
    """
    gpu_enabled = USE_GPU if use_gpu is None else use_gpu
    enhanced_images = []
    
    if gpu_enabled:
        for sigma in sigmas:
            hessian = compute_hessian_matrix(image, sigma, use_gpu=True)
            eigenvalues = compute_eigenvalues(hessian, use_gpu=True)
            eigenvalues = redefine_eigenvalue(eigenvalues, structure_type='bright')
            
            if method == 'improved':
                enhanced = vessel_enhancement_improved(eigenvalues, tau, use_gpu=True)
            else:
                enhanced = vessel_enhancement_frangi(eigenvalues, alpha=frangi_alpha, k=frangi_k, use_gpu=True)
            
            enhanced_images.append(enhanced)
            free_gpu_memory()
        
        stacked = np.stack(enhanced_images, axis=0)
        multi_scale_enhanced = np.max(stacked, axis=0)
        
        return multi_scale_enhanced
    else:
        for sigma in sigmas:
            hessian = compute_hessian_matrix(image, sigma, use_gpu=False)
            eigenvalues = compute_eigenvalues(hessian, use_gpu=False)
            eigenvalues = redefine_eigenvalue(eigenvalues, structure_type='bright')
            
            if method == 'improved':
                enhanced = vessel_enhancement_improved(eigenvalues, tau, use_gpu=False)
            else:
                enhanced = vessel_enhancement_frangi(eigenvalues, alpha=frangi_alpha, k=frangi_k, use_gpu=False)
            
            enhanced_images.append(enhanced)
        
        multi_scale_enhanced = np.max(np.stack(enhanced_images, axis=0), axis=0)
        
        return multi_scale_enhanced


# ==============================================================================
# Module 4: Skeleton Extraction Functions (公式模块 - 可编辑)
# ==============================================================================

def compute_euler_number_3d(binary_image):
    """
    计算3D图像的欧拉数 - 公式(7)
    
    χ(S) = O(S) - H(S) + C(S)
    
    其中:
    O(S): 连通分量数
    H(S): 环孔数
    C(S): 洞数
    
    Parameters:
    -----------
    binary_image : ndarray
        Binary 3D image
        
    Returns:
    --------
    euler_number : int
        Euler number of the image
    """
    labeled_array, num_features = label(binary_image)
    return num_features


def get_26_neighbors(shape, i, j, k):
    """
    获取26邻域点 - 公式(8)
    
    P26(m) = {n(r,s,t) ∈ Z³ | max(|i-r|, |j-s|, |k-t|) = 1}
    
    Parameters:
    -----------
    shape : tuple
        Shape of the image
    i, j, k : int
        Center coordinates
        
    Returns:
    --------
    neighbors : list
        List of valid neighbor coordinates
    """
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni, nj, nk = i + di, j + dj, k + dk
                if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
                    neighbors.append((ni, nj, nk))
    return neighbors


def is_simple_point(binary_image, i, j, k):
    """
    判断是否为简单点
    
    简单点: 删除后不改变图像拓扑结构的边界点
    删除前后欧拉数不变，连通分量数不变
    
    Parameters:
    -----------
    binary_image : ndarray
        Binary 3D image
    i, j, k : int
        Point coordinates
        
    Returns:
    --------
    is_simple : bool
        True if the point is a simple point
    """
    if binary_image[i, j, k] == 0:
        return False
    
    neighbors = get_26_neighbors(binary_image.shape, i, j, k)
    neighbor_values = [binary_image[ni, nj, nk] for ni, nj, nk in neighbors]
    
    if all(v == 0 for v in neighbor_values):
        return False
    
    if all(v == 1 for v in neighbor_values):
        return False
    
    return True


def thinning_3d(binary_image, max_iterations=100):
    """
    3D细化算法 - 提取骨架
    
    保持拓扑不变性:
    1. 只删除简单点
    2. 删除前后欧拉数不变
    3. 连通分量、环孔和洞的数量不变
    
    Parameters:
    -----------
    binary_image : ndarray
        Binary 3D image
    max_iterations : int
        Maximum number of iterations
        
    Returns:
    --------
    skeleton : ndarray
        Skeleton of the binary image
    """
    skeleton = binary_image.copy()
    
    directions = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1)
    ]
    
    for iteration in range(max_iterations):
        changed = False
        
        for direction in directions:
            boundary_points = []
            
            for i in range(1, skeleton.shape[0] - 1):
                for j in range(1, skeleton.shape[1] - 1):
                    for k in range(1, skeleton.shape[2] - 1):
                        if skeleton[i, j, k] == 1:
                            ni = i + direction[0]
                            nj = j + direction[1]
                            nk = k + direction[2]
                            
                            if (0 <= ni < skeleton.shape[0] and 
                                0 <= nj < skeleton.shape[1] and 
                                0 <= nk < skeleton.shape[2]):
                                if skeleton[ni, nj, nk] == 0:
                                    boundary_points.append((i, j, k))
            
            for point in boundary_points:
                i, j, k = point
                if is_simple_point(skeleton, i, j, k):
                    skeleton[i, j, k] = 0
                    changed = True
        
        if not changed:
            break
    
    return skeleton


def extract_skeleton(binary_image):
    """
    提取骨架 - 使用skimage的skeletonize_3d函数
    
    Parameters:
    -----------
    binary_image : ndarray
        Binary 3D image
        
    Returns:
    --------
    skeleton : ndarray
        Skeleton of the binary image
    """
    return skeletonize_3d(binary_image.astype(np.uint8))


# ==============================================================================
# Module 5: Seed Point Detection Functions
# ==============================================================================

def detect_seed_points_from_region(binary, enhanced_image, min_area=20):
    """
    从二值化图像的连通域中自动检测种子点
    
    根据血管形态学特征，连通区域的长度与脑血管长度几乎相同
    设置连接区域面积大于min_area作为候选区域
    
    优化逻辑：
    1. 计算每个连通域内像素值的中位数
    2. 选取一个值为中位数的点作为种子点
    3. 返回种子点及其对应的中位数值
    
    Parameters:
    -----------
    binary : ndarray
        Binary image (thresholded enhanced vessel image)
    enhanced_image : ndarray
        Enhanced vessel image for computing median values
    min_area : int
        Minimum area threshold for connected regions
        
    Returns:
    --------
    seed_info : list of tuples
        List of (seed_point, median_value) tuples
        seed_point: (z, y, x) coordinates
        median_value: median intensity value of the connected region
    """
    labeled_binary, num_regions = label(binary)
    seed_info = []
    
    region_sizes = ndimage.sum(np.ones_like(binary), labeled_binary, range(1, num_regions + 1))
    
    for region_id in range(1, num_regions + 1):
        if region_sizes[region_id - 1] >= min_area:
            region_mask = (labeled_binary == region_id)
            region_values = enhanced_image[region_mask]
            median_value = np.median(region_values)
            
            median_diff = np.abs(region_values - median_value)
            median_idx = np.argmin(median_diff)
            
            coords = np.where(region_mask)
            seed_z = coords[0][median_idx]
            seed_y = coords[1][median_idx]
            seed_x = coords[2][median_idx]
            
            seed_info.append(((int(seed_z), int(seed_y), int(seed_x)), median_value))
    
    return seed_info


# ==============================================================================
# Module 6: Region Growing Functions
# ==============================================================================

def region_growing_3d(image, seed_info, threshold_low_multiplier=0.5, threshold_high_multiplier=5.0,
                      connectivity=26, max_iterations=100000):
    """
    3D区域生长算法
    
    基于图像灰度值的相似性，从种子点开始生长，合并相邻的相似像素
    
    优化逻辑：
    - 使用种子点对应连通域的中位数值作为基准
    - 生长阈值 = 中位数值 × 倍数
    - threshold_low = median_value × threshold_low_multiplier
    - threshold_high = median_value × threshold_high_multiplier
    
    Parameters:
    -----------
    image : ndarray
        Input image (enhanced vessel image)
    seed_info : list of tuples
        List of (seed_point, median_value) tuples from detect_seed_points_from_skeleton
    threshold_low_multiplier : float
        Multiplier for lower intensity threshold (default: 0.5)
    threshold_high_multiplier : float
        Multiplier for upper intensity threshold (default: 5.0)
    connectivity : int
        Connectivity (6, 18, or 26)
    max_iterations : int
        Maximum number of iterations
        
    Returns:
    --------
    segmentation : ndarray
        Binary segmentation result
    """
    segmentation = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=np.uint8)
    
    queue = deque()
    
    for seed_point, median_value in seed_info:
        if 0 <= seed_point[0] < image.shape[0] and \
           0 <= seed_point[1] < image.shape[1] and \
           0 <= seed_point[2] < image.shape[2]:
            threshold_low = median_value * threshold_low_multiplier
            threshold_high = median_value * threshold_high_multiplier
            queue.append((seed_point, threshold_low, threshold_high))
            visited[seed_point] = 1
    
    iterations = 0
    
    while queue and iterations < max_iterations:
        current, threshold_low, threshold_high = queue.popleft()
        z, y, x = current
        
        if threshold_low <= image[z, y, x] <= threshold_high:
            segmentation[z, y, x] = 1
            
            neighbors = get_26_neighbors(image.shape, z, y, x)
            
            for nz, ny, nx in neighbors:
                if visited[nz, ny, nx] == 0:
                    if threshold_low <= image[nz, ny, nx] <= threshold_high:
                        visited[nz, ny, nx] = 1
                        queue.append(((nz, ny, nx), threshold_low, threshold_high))
        
        iterations += 1
    
    return segmentation


# ==============================================================================
# Algorithm 1: Automatic Cerebrovascular Segmentation of TOF-MRA
# ==============================================================================

def algorithm1_cerebrovascular_segmentation(mra_image, params=None, use_gpu=None):
    """
    Algorithm 1: TOF-MRA脑血管自动分割流程
    
    流程:
    1. 输入原始TOF-MRA图像
    2. NAFSM去噪预处理
    3. 多尺度Hessian矩阵血管增强
    4. 阈值二值化
    5. 骨架提取
    6. 自动种子点检测
    7. 区域生长分割
    8. 输出分割结果
    
    Parameters:
    -----------
    mra_image : ndarray
        Input TOF-MRA image
    params : dict, optional
        参数字典，包含以下键:
        - 'denoise': bool, 是否应用去噪 (默认: True)
        - 'denoise_window_size': int, 去噪窗口大小 (默认: 5)
        - 'denoise_max_iterations': int, 去噪最大迭代次数 (默认: 1)
        - 'sigmas': list, 多尺度增强的尺度参数 (默认: [2,5,10])
        - 'tau': float, 改进增强函数的截止阈值 (默认: 0.5)
        - 'threshold': float, 二值化阈值 (默认: 0.5)
        - 'min_skeleton_area': int, 骨架区域最小面积 (默认: 20)
        - 'region_threshold_low': float, 区域生长下阈值倍数 (默认: 0.5，即中位数的0.5倍)
        - 'region_threshold_high': float, 区域生长上阈值倍数 (默认: 5.0，即中位数的5倍)
        - 'region_max_iterations': int, 区域生长最大迭代次数 (默认: 100000)
        - 'enhancement_method': str, 增强方法 ('improved' 或 'frangi') (默认: 'improved')
        - 'frangi_alpha': float, Frangi参数alpha (默认: 0.5)
        - 'frangi_c': float, Frangi参数c (默认: 500)
        - 'verbose': bool, 是否打印进度信息 (默认: True)
    use_gpu : bool, optional
        Force GPU usage. If None, auto-detect.
        
    Returns:
    --------
    result : dict
        Dictionary containing:
        - 'segmentation': Binary segmentation result
        - 'enhanced': Enhanced vessel image
        - 'skeleton': Extracted skeleton
        - 'seed_points': Detected seed point coordinates (list of tuples)
        - 'seed_info': List of (seed_point, median_value) tuples for each connected region
    """
    gpu_enabled = USE_GPU if use_gpu is None else use_gpu
    p = DEFAULT_ALGORITHM1_PARAMS.copy()
    if params is not None:
        p.update(params)
    
    if p['verbose']:
        print("=" * 60)
        print("Algorithm 1: Automatic Cerebrovascular Segmentation")
        print(f"GPU Acceleration: {'Enabled' if gpu_enabled else 'Disabled'}")
        print("=" * 60)
    
    image = mra_image.astype(np.float32)
    
    if np.max(image) > 1:
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image.astype(np.float32)
    
    if p['verbose']:
        print("Step 1: Denoising with NAFSM filter...")
    if p['denoise']:
        denoised = nafsm_filter(image, 
                               window_size=p['denoise_window_size'],
                               max_iterations=p['denoise_max_iterations'],
                               use_gpu=gpu_enabled)
    else:
        denoised = image
    
    if p['verbose']:
        print("Step 2: Multi-scale Hessian vessel enhancement...")
    enhanced = multi_scale_vessel_enhancement(
        denoised, 
        sigmas=p['sigmas'], 
        tau=p['tau'], 
        method=p['enhancement_method'],
        frangi_alpha=p.get('frangi_alpha', DEFAULT_FRANGI_PARAMS['alpha']),
        frangi_k=p.get('frangi_k', DEFAULT_FRANGI_PARAMS['k']),
        use_gpu=False
    )
    
    if p['verbose']:
        print("Step 3: Thresholding...")
    binary = (enhanced > p['threshold']).astype(np.uint8)
    
    if p['verbose']:
        print("Step 4: Skeleton extraction...")
    skeleton = extract_skeleton(binary)
    
    if p['verbose']:
        print("Step 5: Automatic seed point detection...")
    seed_info = detect_seed_points_from_region(binary, enhanced, p['min_skeleton_area'])
    
    if p['verbose']:
        print(f"  Detected {len(seed_info)} seed points")
    
    if p['verbose']:
        print("Step 6: Region growing segmentation...")
    if len(seed_info) > 0:
        segmentation = region_growing_3d(
            enhanced, 
            seed_info, 
            threshold_low_multiplier=p['region_threshold_low'],
            threshold_high_multiplier=p['region_threshold_high'],
            max_iterations=p['region_max_iterations']
        )
    else:
        segmentation = binary
    
    if p['verbose']:
        print("Segmentation complete!")
        print(f"  Total vessel voxels: {np.sum(segmentation)}")
    
    seed_points = [sp for sp, _ in seed_info]
    return {
        'segmentation': segmentation,
        'enhanced': enhanced,
        'skeleton': skeleton,
        'seed_points': seed_points,
        'seed_info': seed_info
    }


# ==============================================================================
# Algorithm 2: Non-vascular Brain-tissue and Noise Regions Segmentation
# ==============================================================================

def algorithm2_non_vascular_segmentation(mra_image, vessel_segmentation, params=None):
    """
    Algorithm 2: 非血管脑组织和噪声区域分割
    
    流程:
    1. 输入原始MRA图像和血管分割结果
    2. 对原始图像进行阈值处理得到脑区域
    3. 从脑区域中减去血管区域得到非血管脑组织
    4. 非脑区域作为噪声区域
    
    Parameters:
    -----------
    mra_image : ndarray
        Input MRA image
    vessel_segmentation : ndarray
        Binary vessel segmentation from Algorithm 1
    params : dict, optional
        参数字典，包含以下键:
        - 'brain_threshold_low': float, 脑组织下阈值 (默认: 0.05)
        - 'brain_threshold_high': float, 脑组织上阈值 (默认: 0.95)
        - 'verbose': bool, 是否打印进度信息 (默认: True)
        
    Returns:
    --------
    result : dict
        Dictionary containing:
        - 'vessel_region': Vessel region mask
        - 'non_vessel_brain_tissue': Non-vascular brain tissue mask
        - 'noise_region': Noise region mask (non-brain area)
    """
    p = DEFAULT_ALGORITHM2_PARAMS.copy()
    if params is not None:
        p.update(params)
    
    if p['verbose']:
        print("=" * 60)
        print("Algorithm 2: Non-vascular Brain-tissue and Noise Segmentation")
        print("=" * 60)
    
    image = mra_image.astype(np.float32)
    
    if np.max(image) > 1:
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    if p['verbose']:
        print("Step 1: Identifying brain region...")
    
    brain_mask = (image > p['brain_threshold_low']) & (image < p['brain_threshold_high'])
    brain_mask = brain_mask.astype(np.uint8)
    
    if p['verbose']:
        print("Step 2: Extracting vessel region...")
    vessel_region = vessel_segmentation.astype(np.uint8)
    
    if p['verbose']:
        print("Step 3: Computing non-vascular brain tissue...")
    non_vessel_brain_tissue = brain_mask.copy()
    non_vessel_brain_tissue[vessel_region == 1] = 0
    
    if p['verbose']:
        print("Step 4: Computing noise region (non-brain area)...")
    noise_region = (1 - brain_mask).astype(np.uint8)
    
    if p['verbose']:
        print("Segmentation complete!")
        print(f"  Vessel region voxels: {np.sum(vessel_region)}")
        print(f"  Non-vascular brain tissue voxels: {np.sum(non_vessel_brain_tissue)}")
        print(f"  Noise region voxels: {np.sum(noise_region)}")
    
    return {
        'vessel_region': vessel_region,
        'non_vessel_brain_tissue': non_vessel_brain_tissue,
        'noise_region': noise_region,
        'brain_mask': brain_mask
    }


# ==============================================================================
# Utility Functions for Image Quality Evaluation
# ==============================================================================

def compute_bbc(signal_vessel, signal_non_vessel_tissue):
    """
    计算Blood-to-Background Contrast (BBC) - 公式(9)
    
    BBC = (Mean(Signal_vessel) - Mean(Signal_non_vessel_tissue)) / Mean(Signal_non_vessel_tissue)
    
    Parameters:
    -----------
    signal_vessel : ndarray
        Signal values in vessel region
    signal_non_vessel_tissue : ndarray
        Signal values in non-vascular brain tissue region
        
    Returns:
    --------
    bbc : float
        Blood-to-background contrast
    """
    mean_vessel = np.mean(signal_vessel)
    mean_non_vessel = np.mean(signal_non_vessel_tissue)
    
    if mean_non_vessel == 0:
        return 0
    
    return (mean_vessel - mean_non_vessel) / mean_non_vessel


def compute_cnr(signal_vessel, signal_non_vessel_tissue, noise):
    """
    计算Contrast-to-Noise Ratio (CNR) - 公式(10)
    
    CNR = (Mean(Signal_vessel) - Mean(Signal_non_vessel_tissue)) / Std(Noise)
    
    Parameters:
    -----------
    signal_vessel : ndarray
        Signal values in vessel region
    signal_non_vessel_tissue : ndarray
        Signal values in non-vascular brain tissue region
    noise : ndarray
        Signal values in noise region
        
    Returns:
    --------
    cnr : float
        Contrast-to-noise ratio
    """
    mean_vessel = np.mean(signal_vessel)
    mean_non_vessel = np.mean(signal_non_vessel_tissue)
    std_noise = np.std(noise)
    
    if std_noise == 0:
        return 0
    
    return (mean_vessel - mean_non_vessel) / std_noise


def compute_snr(signal_vessel, noise):
    """
    计算Signal-to-Noise Ratio (SNR) - 公式(11)
    
    SNR = Mean(Signal_vessel) / Std(Noise)
    
    Parameters:
    -----------
    signal_vessel : ndarray
        Signal values in vessel region
    noise : ndarray
        Signal values in noise region
        
    Returns:
    --------
    snr : float
        Signal-to-noise ratio
    """
    mean_vessel = np.mean(signal_vessel)
    std_noise = np.std(noise)
    
    if std_noise == 0:
        return 0
    
    return mean_vessel / std_noise


def compute_vessel_diameter(segmentation, skeleton, voxel_size=(0.5, 0.5, 0.5)):
    """
    计算血管平均直径
    
    Da = 2 × voxel_size × (vt / Lt) / (2π)
    
    其中:
    vt = 血管总体积
    Lt = 血管总长度 = Is × voxel_size
    
    Parameters:
    -----------
    segmentation : ndarray
        Binary vessel segmentation
    skeleton : ndarray
        Vessel skeleton
    voxel_size : tuple
        Voxel size in mm
        
    Returns:
    --------
    diameter : float
        Average vessel diameter in mm
    """
    vt = np.sum(segmentation)
    is_length = np.sum(skeleton)
    
    voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2]
    voxel_length = np.mean(voxel_size)
    
    lt = is_length * voxel_length
    vt_mm3 = vt * voxel_volume
    
    if lt == 0:
        return 0
    
    diameter = 2 * (vt_mm3 / lt) / (2 * np.pi)
    
    return diameter


def compute_vessel_length(skeleton, voxel_size=(0.5, 0.5, 0.5)):
    """
    计算血管总长度
    
    Lt = Is × voxel_size
    
    Parameters:
    -----------
    skeleton : ndarray
        Vessel skeleton
    voxel_size : tuple
        Voxel size in mm
        
    Returns:
    --------
    length : float
        Total vessel length in mm
    """
    is_length = np.sum(skeleton)
    voxel_length = np.mean(voxel_size)
    
    return is_length * voxel_length


# ==============================================================================
# Main Pipeline
# ==============================================================================

def run_complete_pipeline(mra_image, 
                          voxel_size=(0.5, 0.5, 0.5),
                          verbose=True):
    """
    运行完整的脑血管分割和分析流程
    
    Parameters:
    -----------
    mra_image : ndarray
        Input TOF-MRA image
    voxel_size : tuple
        Voxel size in mm
    verbose : bool
        Print progress information
        
    Returns:
    --------
    results : dict
        Complete results including segmentation and metrics
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Running Complete Cerebrovascular Segmentation Pipeline")
        print("=" * 60 + "\n")
    
    algo1_results = algorithm1_cerebrovascular_segmentation(
        mra_image, verbose=verbose
    )
    
    algo2_results = algorithm2_non_vascular_segmentation(
        mra_image, algo1_results['segmentation'], verbose=verbose
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print("Computing Image Quality Metrics")
        print("=" * 60)
    
    vessel_region = algo1_results['segmentation']
    non_vessel_tissue = algo2_results['non_vessel_brain_tissue']
    noise_region = algo2_results['noise_region']
    skeleton = algo1_results['skeleton']
    
    image_normalized = mra_image.astype(np.float32)
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
    
    diameter = compute_vessel_diameter(vessel_region, skeleton, voxel_size)
    length = compute_vessel_length(skeleton, voxel_size)
    
    if verbose:
        print(f"\nImage Quality Metrics:")
        print(f"  BBC: {bbc:.4f}")
        print(f"  CNR: {cnr:.4f}")
        print(f"  SNR: {snr:.4f}")
        print(f"\nVessel Characteristics:")
        print(f"  Average Diameter: {diameter:.4f} mm")
        print(f"  Total Length: {length:.4f} mm")

    
    return {
        'segmentation': algo1_results,
        'regions': algo2_results,
        'metrics': {
            'bbc': bbc,
            'cnr': cnr,
            'snr': snr,
            'diameter': diameter,
            'length': length,
        }
    }


if __name__ == "__main__":
    print("Cerebrovascular Segmentation Algorithms Module")
    print("=" * 60)
    print("\nThis module implements:")
    print("  - Algorithm 1: Automatic cerebrovascular segmentation of TOF-MRA")
    print("  - Algorithm 2: Non-vascular brain-tissue and noise regions segmentation")
    print("\nUsage:")
    print("  from cerebrovascular_segmentation import algorithm1_cerebrovascular_segmentation")
    print("  from cerebrovascular_segmentation import algorithm2_non_vascular_segmentation")
    print("\nFor modular formula editing, modify the following functions:")
    print("  - compute_hessian_matrix(): Formula (1)")
    print("  - redefine_eigenvalue(): Formula (2)")
    print("  - vessel_enhancement_frangi(): Formula (4)")
    print("  - compute_lambda_rho(): Formula (5)")
    print("  - vessel_enhancement_improved(): Formula (6)")
    print("  - compute_euler_number_3d(): Formula (7)")
    print("  - get_26_neighbors(): Formula (8)")
    print("  - compute_bbc(): Formula (9)")
    print("  - compute_cnr(): Formula (10)")
    print("  - compute_snr(): Formula (11)")
