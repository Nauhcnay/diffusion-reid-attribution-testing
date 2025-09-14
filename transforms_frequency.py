"""
Frequency and contrast transforms for ReID diagnostics.

A) Frequency / contrast transforms:
  A1. Low-pass smoothing: Gaussian, Bilateral, Median
  A2. High-pass / edge emphasize: Laplacian, Sobel, Unsharp masking
  A3. Band-pass / notch via FFT masks
  A4. Photometric: Gamma correction, Global contrast stretch, CLAHE
"""

import itertools
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from scipy import ndimage
from skimage import exposure, filters
from skimage.restoration import denoise_tv_chambolle

from io_utils import uint8_to_float, float_to_uint8


def apply_gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply Gaussian blur filter.
    
    Args:
        image: Input RGB image (uint8)
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Filtered image and parameters dictionary
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    params = {
        'kernel_size': kernel_size,
        'sigma': sigma
    }
    
    return blurred, params


def apply_bilateral_filter(image: np.ndarray, diameter: int, sigma_color: float, sigma_space: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply bilateral filter for edge-preserving smoothing.
    
    Args:
        image: Input RGB image (uint8)
        diameter: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
        
    Returns:
        Filtered image and parameters dictionary
    """
    filtered = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    
    params = {
        'diameter': diameter,
        'sigma_color': sigma_color,
        'sigma_space': sigma_space
    }
    
    return filtered, params


def apply_median_filter(image: np.ndarray, kernel_size: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply median filter.
    
    Args:
        image: Input RGB image (uint8)
        kernel_size: Size of median filter kernel (must be odd)
        
    Returns:
        Filtered image and parameters dictionary
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    filtered = cv2.medianBlur(image, kernel_size)
    
    params = {
        'kernel_size': kernel_size
    }
    
    return filtered, params


def apply_laplacian_filter(image: np.ndarray, kernel_size: int, scale: float = 1.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply Laplacian edge detection filter.
    
    Args:
        image: Input RGB image (uint8)
        kernel_size: Size of Laplacian kernel
        scale: Scale factor for Laplacian response
        
    Returns:
        Edge-enhanced image and parameters dictionary
    """
    # Convert to float for processing
    float_image = uint8_to_float(image)
    
    # Apply Laplacian to each channel
    result = np.zeros_like(float_image)
    for i in range(3):
        laplacian = cv2.Laplacian(float_image[:, :, i], cv2.CV_32F, ksize=kernel_size)
        # Add scaled Laplacian back to original
        result[:, :, i] = float_image[:, :, i] + scale * laplacian
    
    # Clip and convert back to uint8
    result = np.clip(result, 0, 1)
    result_uint8 = float_to_uint8(result)
    
    params = {
        'kernel_size': kernel_size,
        'scale': scale
    }
    
    return result_uint8, params


def apply_sobel_filter(image: np.ndarray, direction: str, kernel_size: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply Sobel edge detection filter.
    
    Args:
        image: Input RGB image (uint8)
        direction: 'x', 'y', or 'magnitude'
        kernel_size: Size of Sobel kernel (3 or 5)
        
    Returns:
        Edge-detected image and parameters dictionary
    """
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_float = uint8_to_float(gray)
    
    if direction == 'x':
        sobel = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=kernel_size)
    elif direction == 'y':
        sobel = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=kernel_size)
    elif direction == 'magnitude':
        sobel_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=kernel_size)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    else:
        raise ValueError(f"Invalid direction: {direction}")
    
    # Normalize to [0, 1]
    sobel = np.abs(sobel)
    if sobel.max() > 0:
        sobel = sobel / sobel.max()
    
    # Convert to 3-channel by replicating
    sobel_rgb = np.stack([sobel, sobel, sobel], axis=2)
    result_uint8 = float_to_uint8(sobel_rgb)
    
    params = {
        'direction': direction,
        'kernel_size': kernel_size
    }
    
    return result_uint8, params


def apply_unsharp_mask(image: np.ndarray, amount: float, radius: float, threshold: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply unsharp masking for edge enhancement.
    
    Args:
        image: Input RGB image (uint8)
        amount: Strength of sharpening
        radius: Radius of blur for mask creation
        threshold: Threshold for applying sharpening
        
    Returns:
        Sharpened image and parameters dictionary
    """
    float_image = uint8_to_float(image)
    
    # Create Gaussian blur
    # Handle both old and new scikit-image versions
    try:
        # Try new API first (channel_axis parameter for newer versions)
        blurred = filters.gaussian(float_image, sigma=radius, channel_axis=-1)
    except TypeError:
        try:
            # Fall back to multichannel parameter for older versions
            blurred = filters.gaussian(float_image, sigma=radius, multichannel=True)
        except TypeError:
            # Final fallback - apply to each channel separately
            blurred = np.zeros_like(float_image)
            for i in range(float_image.shape[2]):
                blurred[:, :, i] = filters.gaussian(float_image[:, :, i], sigma=radius)
    
    # Create unsharp mask
    mask = float_image - blurred
    
    # Apply threshold
    if threshold > 0:
        mask = np.where(np.abs(mask) > threshold / 255.0, mask, 0)
    
    # Apply sharpening
    result = float_image + amount * mask
    result = np.clip(result, 0, 1)
    result_uint8 = float_to_uint8(result)
    
    params = {
        'amount': amount,
        'radius': radius,
        'threshold': threshold
    }
    
    return result_uint8, params


def apply_fft_lowpass(image: np.ndarray, cutoff_ratio: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply FFT-based low-pass filter.
    
    Args:
        image: Input RGB image (uint8)
        cutoff_ratio: Cutoff frequency as ratio of Nyquist (0 to 1)
        
    Returns:
        Filtered image and parameters dictionary
    """
    float_image = uint8_to_float(image)
    h, w = float_image.shape[:2]
    
    # Create circular low-pass mask
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    cutoff = cutoff_ratio * min(h, w) / 2
    mask = distance <= cutoff
    
    # Apply filter to each channel
    result = np.zeros_like(float_image)
    for i in range(3):
        # FFT
        fft = np.fft.fft2(float_image[:, :, i])
        fft_shifted = np.fft.fftshift(fft)
        
        # Apply mask
        filtered_fft = fft_shifted * mask
        
        # Inverse FFT
        ifft_shifted = np.fft.ifftshift(filtered_fft)
        ifft = np.fft.ifft2(ifft_shifted)
        result[:, :, i] = np.real(ifft)
    
    result = np.clip(result, 0, 1)
    result_uint8 = float_to_uint8(result)
    
    params = {
        'cutoff_ratio': cutoff_ratio
    }
    
    return result_uint8, params


def apply_fft_highpass(image: np.ndarray, cutoff_ratio: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply FFT-based high-pass filter.
    
    Args:
        image: Input RGB image (uint8)
        cutoff_ratio: Cutoff frequency as ratio of Nyquist (0 to 1)
        
    Returns:
        Filtered image and parameters dictionary
    """
    float_image = uint8_to_float(image)
    h, w = float_image.shape[:2]
    
    # Create circular high-pass mask
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    cutoff = cutoff_ratio * min(h, w) / 2
    mask = distance > cutoff
    
    # Apply filter to each channel
    result = np.zeros_like(float_image)
    for i in range(3):
        # FFT
        fft = np.fft.fft2(float_image[:, :, i])
        fft_shifted = np.fft.fftshift(fft)
        
        # Apply mask
        filtered_fft = fft_shifted * mask
        
        # Inverse FFT
        ifft_shifted = np.fft.ifftshift(filtered_fft)
        ifft = np.fft.ifft2(ifft_shifted)
        result[:, :, i] = np.real(ifft)
    
    result = np.clip(result, 0, 1)
    result_uint8 = float_to_uint8(result)
    
    params = {
        'cutoff_ratio': cutoff_ratio
    }
    
    return result_uint8, params


def apply_fft_bandpass(image: np.ndarray, low_ratio: float, high_ratio: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply FFT-based band-pass filter.
    
    Args:
        image: Input RGB image (uint8)
        low_ratio: Low cutoff frequency as ratio of Nyquist
        high_ratio: High cutoff frequency as ratio of Nyquist
        
    Returns:
        Filtered image and parameters dictionary
    """
    float_image = uint8_to_float(image)
    h, w = float_image.shape[:2]
    
    # Create annular band-pass mask
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    low_cutoff = low_ratio * min(h, w) / 2
    high_cutoff = high_ratio * min(h, w) / 2
    mask = (distance > low_cutoff) & (distance < high_cutoff)
    
    # Apply filter to each channel
    result = np.zeros_like(float_image)
    for i in range(3):
        # FFT
        fft = np.fft.fft2(float_image[:, :, i])
        fft_shifted = np.fft.fftshift(fft)
        
        # Apply mask
        filtered_fft = fft_shifted * mask
        
        # Inverse FFT
        ifft_shifted = np.fft.ifftshift(filtered_fft)
        ifft = np.fft.ifft2(ifft_shifted)
        result[:, :, i] = np.real(ifft)
    
    result = np.clip(result, 0, 1)
    result_uint8 = float_to_uint8(result)
    
    params = {
        'low_ratio': low_ratio,
        'high_ratio': high_ratio
    }
    
    return result_uint8, params


def apply_fft_notch(image: np.ndarray, center_ratio: float, notch_radius: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply FFT-based notch filter to remove periodic textures.
    
    Args:
        image: Input RGB image (uint8)
        center_ratio: Center of notch as ratio of image size
        notch_radius: Radius of notch filter
        
    Returns:
        Filtered image and parameters dictionary
    """
    float_image = uint8_to_float(image)
    h, w = float_image.shape[:2]
    
    # Create notch filter mask
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    
    # Notch at specified location and its symmetric point
    notch_y = center_y + center_ratio * h / 2
    notch_x = center_x + center_ratio * w / 2
    
    dist1 = np.sqrt((y - notch_y)**2 + (x - notch_x)**2)
    dist2 = np.sqrt((y - (2*center_y - notch_y))**2 + (x - (2*center_x - notch_x))**2)
    
    notch_size = notch_radius * min(h, w) / 2
    mask = (dist1 > notch_size) & (dist2 > notch_size)
    
    # Apply filter to each channel
    result = np.zeros_like(float_image)
    for i in range(3):
        # FFT
        fft = np.fft.fft2(float_image[:, :, i])
        fft_shifted = np.fft.fftshift(fft)
        
        # Apply mask
        filtered_fft = fft_shifted * mask
        
        # Inverse FFT
        ifft_shifted = np.fft.ifftshift(filtered_fft)
        ifft = np.fft.ifft2(ifft_shifted)
        result[:, :, i] = np.real(ifft)
    
    result = np.clip(result, 0, 1)
    result_uint8 = float_to_uint8(result)
    
    params = {
        'center_ratio': center_ratio,
        'notch_radius': notch_radius
    }
    
    return result_uint8, params


def apply_gamma_correction(image: np.ndarray, gamma: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply gamma correction.
    
    Args:
        image: Input RGB image (uint8)
        gamma: Gamma value (< 1 brightens, > 1 darkens)
        
    Returns:
        Gamma-corrected image and parameters dictionary
    """
    float_image = uint8_to_float(image)
    
    # Apply gamma correction
    corrected = np.power(float_image, gamma)
    result_uint8 = float_to_uint8(corrected)
    
    params = {
        'gamma': gamma
    }
    
    return result_uint8, params


def apply_contrast_stretch(image: np.ndarray, percentile_range: Tuple[float, float]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply global contrast stretching.
    
    Args:
        image: Input RGB image (uint8)
        percentile_range: (low_percentile, high_percentile) for stretching
        
    Returns:
        Contrast-stretched image and parameters dictionary
    """
    low_p, high_p = percentile_range
    
    # Apply to each channel separately
    result = np.zeros_like(image)
    for i in range(3):
        channel = image[:, :, i]
        p_low, p_high = np.percentile(channel, [low_p, high_p])
        
        # Stretch contrast
        stretched = exposure.rescale_intensity(
            channel, 
            in_range=(p_low, p_high), 
            out_range=(0, 255)
        ).astype(np.uint8)
        
        result[:, :, i] = stretched
    
    params = {
        'low_percentile': low_p,
        'high_percentile': high_p
    }
    
    return result, params


def apply_clahe(image: np.ndarray, clip_limit: float, tile_grid: Tuple[int, int]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        image: Input RGB image (uint8)
        clip_limit: Clipping limit for contrast limiting
        tile_grid: (grid_height, grid_width) for adaptive tiles
        
    Returns:
        CLAHE-processed image and parameters dictionary
    """
    # Convert to LAB color space for better results
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    params = {
        'clip_limit': clip_limit,
        'tile_grid_height': tile_grid[0],
        'tile_grid_width': tile_grid[1]
    }
    
    return result, params


# Transform registry for frequency/contrast transforms
FREQUENCY_TRANSFORMS = {
    'frequency_gaussian_blur': {
        'function': apply_gaussian_blur,
        'param_combinations': lambda params: [
            {'kernel_size': ks, 'sigma': s}
            for ks in params.get('kernel_sizes', [5, 9, 15])
            for s in params.get('sigmas', [1.0, 2.0, 3.0])
        ]
    },
    'frequency_bilateral': {
        'function': apply_bilateral_filter,
        'param_combinations': lambda params: [
            {'diameter': d, 'sigma_color': sc, 'sigma_space': ss}
            for d in params.get('diameters', [9, 15])
            for sc in params.get('sigma_colors', [20, 50])
            for ss in params.get('sigma_spaces', [20, 50])
        ]
    },
    'frequency_median': {
        'function': apply_median_filter,
        'param_combinations': lambda params: [
            {'kernel_size': ks}
            for ks in params.get('kernel_sizes', [3, 5, 7])
        ]
    },
    'frequency_laplacian': {
        'function': apply_laplacian_filter,
        'param_combinations': lambda params: [
            {'kernel_size': ks, 'scale': s}
            for ks in params.get('kernel_sizes', [1, 3])
            for s in params.get('scales', [1.0, 2.0])
        ]
    },
    'frequency_sobel': {
        'function': apply_sobel_filter,
        'param_combinations': lambda params: [
            {'direction': d, 'kernel_size': ks}
            for d in params.get('directions', ['x', 'y', 'magnitude'])
            for ks in params.get('kernel_sizes', [3, 5])
        ]
    },
    'frequency_unsharp': {
        'function': apply_unsharp_mask,
        'param_combinations': lambda params: [
            {'amount': a, 'radius': r, 'threshold': t}
            for a in params.get('amounts', [0.5, 1.0, 2.0])
            for r in params.get('radii', [1.0, 2.0])
            for t in params.get('thresholds', [0, 1])
        ]
    },
    'frequency_fft_lowpass': {
        'function': apply_fft_lowpass,
        'param_combinations': lambda params: [
            {'cutoff_ratio': cr}
            for cr in params.get('cutoff_ratios', [0.1, 0.2, 0.3])
        ]
    },
    'frequency_fft_highpass': {
        'function': apply_fft_highpass,
        'param_combinations': lambda params: [
            {'cutoff_ratio': cr}
            for cr in params.get('cutoff_ratios', [0.1, 0.2, 0.3])
        ]
    },
    'frequency_fft_bandpass': {
        'function': apply_fft_bandpass,
        'param_combinations': lambda params: [
            {'low_ratio': lr, 'high_ratio': hr}
            for lr in params.get('low_ratios', [0.05, 0.1])
            for hr in params.get('high_ratios', [0.3, 0.5])
            if lr < hr
        ]
    },
    'frequency_fft_notch': {
        'function': apply_fft_notch,
        'param_combinations': lambda params: [
            {'center_ratio': cr, 'notch_radius': nr}
            for cr in params.get('center_ratios', [0.25, 0.5])
            for nr in params.get('notch_radii', [0.02, 0.05])
        ]
    },
    'photometric_gamma': {
        'function': apply_gamma_correction,
        'param_combinations': lambda params: [
            {'gamma': g}
            for g in params.get('gammas', [0.5, 0.7, 1.4, 2.0])
        ]
    },
    'photometric_contrast_stretch': {
        'function': apply_contrast_stretch,
        'param_combinations': lambda params: [
            {'percentile_range': pr}
            for pr in params.get('percentiles', [(1, 99), (2, 98), (5, 95)])
        ]
    },
    'photometric_clahe': {
        'function': apply_clahe,
        'param_combinations': lambda params: [
            {'clip_limit': cl, 'tile_grid': tg}
            for cl in params.get('clip_limits', [2.0, 4.0])
            for tg in params.get('tile_grids', [(4, 4), (8, 8)])
        ]
    }
}


def get_frequency_transforms() -> Dict[str, Any]:
    """Get all frequency/contrast transforms."""
    return FREQUENCY_TRANSFORMS


if __name__ == "__main__":
    # Test transforms with synthetic data
    test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    # Test a few transforms
    result, params = apply_gaussian_blur(test_image, 5, 2.0)
    print(f"Gaussian blur: {params}, output shape: {result.shape}")
    
    result, params = apply_gamma_correction(test_image, 0.8)
    print(f"Gamma correction: {params}, output shape: {result.shape}")
    
    print(f"Available frequency transforms: {list(FREQUENCY_TRANSFORMS.keys())}")