"""
Surveillance-like degradation transforms for ReID diagnostics.

B) "Surveillance-like" degradations (domain-consistent):
  B1. JPEG compression sweep
  B2. Motion blur PSF
  B3. Sensor noise: Poisson-Gaussian noise
  B4. Low-light gain + clipping + noise
  B5. Downscale→Upscale
"""

from typing import Dict, List, Tuple, Any
import io

import cv2
import numpy as np
from PIL import Image

from io_utils import uint8_to_float, float_to_uint8


def apply_jpeg_compression(image: np.ndarray, quality: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply JPEG compression with specified quality.
    
    Args:
        image: Input RGB image (uint8)
        quality: JPEG quality (1-100, higher = better quality)
        
    Returns:
        JPEG-compressed image and parameters dictionary
    """
    # Convert RGB to PIL Image
    pil_image = Image.fromarray(image, mode='RGB')
    
    # Compress to JPEG in memory
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
    
    # Load back from compressed buffer
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    result = np.array(compressed_image, dtype=np.uint8)
    buffer.close()
    
    params = {
        'quality': quality
    }
    
    return result, params


def create_motion_blur_kernel(length: int, angle: float) -> np.ndarray:
    """
    Create motion blur kernel.
    
    Args:
        length: Length of motion blur
        angle: Angle in degrees (0=horizontal, 90=vertical)
        
    Returns:
        Motion blur kernel
    """
    # Create kernel
    kernel = np.zeros((length, length))
    
    # Calculate end points of line
    angle_rad = np.radians(angle)
    center = length // 2
    
    # Draw line in kernel
    for i in range(length):
        offset = i - center
        x = center + int(offset * np.cos(angle_rad))
        y = center + int(offset * np.sin(angle_rad))
        
        # Ensure coordinates are within bounds
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    
    # Normalize kernel
    if kernel.sum() > 0:
        kernel = kernel / kernel.sum()
    
    return kernel


def apply_motion_blur(image: np.ndarray, length: int, angle: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply motion blur using directional kernel.
    
    Args:
        image: Input RGB image (uint8)
        length: Length of motion blur
        angle: Angle in degrees
        
    Returns:
        Motion-blurred image and parameters dictionary
    """
    # Create motion blur kernel
    kernel = create_motion_blur_kernel(length, angle)
    
    # Apply convolution to each channel
    blurred = cv2.filter2D(image, -1, kernel)
    
    params = {
        'length': length,
        'angle': angle
    }
    
    return blurred, params


def apply_poisson_gaussian_noise(image: np.ndarray, sigma_read: float, gain: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply Poisson-Gaussian sensor noise model.
    
    Args:
        image: Input RGB image (uint8)
        sigma_read: Read noise standard deviation
        gain: Sensor gain factor
        
    Returns:
        Noisy image and parameters dictionary
    """
    # Convert to float for noise processing
    float_image = uint8_to_float(image)
    
    # Apply gain
    gained = float_image * gain
    
    # Add Poisson noise (shot noise)
    # Scale to reasonable range for Poisson
    scaled = gained * 255
    noisy_scaled = np.random.poisson(scaled).astype(np.float32)
    noisy = noisy_scaled / 255
    
    # Add Gaussian read noise
    read_noise = np.random.normal(0, sigma_read / 255, image.shape).astype(np.float32)
    noisy = noisy + read_noise
    
    # Clip and convert back
    noisy = np.clip(noisy, 0, 1)
    result_uint8 = float_to_uint8(noisy)
    
    params = {
        'sigma_read': sigma_read,
        'gain': gain
    }
    
    return result_uint8, params


def apply_low_light_degradation(image: np.ndarray, gain: float, noise_sigma: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply low-light degradation with gain and noise.
    
    Args:
        image: Input RGB image (uint8)
        gain: Brightness gain factor (< 1 for darkening)
        noise_sigma: Noise standard deviation
        
    Returns:
        Low-light degraded image and parameters dictionary
    """
    # Convert to float
    float_image = uint8_to_float(image)
    
    # Apply gain (darken)
    darkened = float_image * gain
    
    # Add noise (more visible in dark regions)
    noise = np.random.normal(0, noise_sigma, image.shape).astype(np.float32)
    noisy = darkened + noise
    
    # Clip to valid range
    noisy = np.clip(noisy, 0, 1)
    result_uint8 = float_to_uint8(noisy)
    
    params = {
        'gain': gain,
        'noise_sigma': noise_sigma
    }
    
    return result_uint8, params


def apply_downscale_upscale(image: np.ndarray, scale_factor: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply downscale followed by upscale (resolution degradation).
    
    Args:
        image: Input RGB image (uint8)
        scale_factor: Scale factor for downscaling (e.g., 0.5 = half resolution)
        
    Returns:
        Downscale-upscale processed image and parameters dictionary
    """
    h, w = image.shape[:2]
    
    # Calculate new dimensions
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    
    # Downscale using bicubic interpolation
    downscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Upscale back to original size
    upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_CUBIC)
    
    params = {
        'scale_factor': scale_factor,
        'intermediate_size': (new_w, new_h)
    }
    
    return upscaled, params


def apply_gaussian_blur_simple(image: np.ndarray, sigma: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply simple Gaussian blur (for combining with other degradations).
    
    Args:
        image: Input RGB image (uint8)
        sigma: Standard deviation for Gaussian blur
        
    Returns:
        Blurred image and parameters dictionary
    """
    # Calculate kernel size from sigma
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    params = {
        'sigma': sigma,
        'kernel_size': kernel_size
    }
    
    return blurred, params


def apply_defocus_blur(image: np.ndarray, radius: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply defocus blur using circular averaging kernel.
    
    Args:
        image: Input RGB image (uint8)
        radius: Radius of defocus blur
        
    Returns:
        Defocus-blurred image and parameters dictionary
    """
    # Create circular kernel
    kernel_size = int(2 * radius + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    center = kernel_size // 2
    y, x = np.ogrid[:kernel_size, :kernel_size]
    mask = (x - center)**2 + (y - center)**2 <= radius**2
    
    kernel = mask.astype(np.float32)
    kernel = kernel / kernel.sum()  # Normalize
    
    # Apply convolution
    blurred = cv2.filter2D(image, -1, kernel)
    
    params = {
        'radius': radius,
        'kernel_size': kernel_size
    }
    
    return blurred, params


def apply_atmospheric_scattering(image: np.ndarray, scattering_coefficient: float, 
                                atmosphere_color: Tuple[float, float, float] = (0.8, 0.8, 0.9)) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply atmospheric scattering effect (haze/fog simulation).
    
    Args:
        image: Input RGB image (uint8)
        scattering_coefficient: Strength of scattering effect (0-1)
        atmosphere_color: RGB color of atmospheric scattering
        
    Returns:
        Atmosphere-affected image and parameters dictionary
    """
    float_image = uint8_to_float(image)
    
    # Create atmosphere layer
    atmosphere = np.full_like(float_image, atmosphere_color, dtype=np.float32)
    
    # Blend with atmosphere based on scattering coefficient
    result = (1 - scattering_coefficient) * float_image + scattering_coefficient * atmosphere
    result = np.clip(result, 0, 1)
    
    result_uint8 = float_to_uint8(result)
    
    params = {
        'scattering_coefficient': scattering_coefficient,
        'atmosphere_color': atmosphere_color
    }
    
    return result_uint8, params


def apply_vignetting(image: np.ndarray, strength: float, falloff: float = 2.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply vignetting effect (darkening towards edges).
    
    Args:
        image: Input RGB image (uint8)
        strength: Strength of vignetting (0-1)
        falloff: Falloff rate (higher = more abrupt)
        
    Returns:
        Vignetted image and parameters dictionary
    """
    float_image = uint8_to_float(image)
    h, w = image.shape[:2]
    
    # Create vignette mask
    center_x, center_y = w / 2, h / 2
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Normalize distance and apply falloff
    normalized_distance = distance / max_distance
    vignette = 1 - strength * np.power(normalized_distance, falloff)
    vignette = np.clip(vignette, 0, 1)
    
    # Apply vignette to each channel
    result = float_image * vignette[:, :, np.newaxis]
    result_uint8 = float_to_uint8(result)
    
    params = {
        'strength': strength,
        'falloff': falloff
    }
    
    return result_uint8, params


# Transform registry for degradation transforms
DEGRADATION_TRANSFORMS = {
    'degradation_jpeg': {
        'function': apply_jpeg_compression,
        'param_combinations': lambda params: [
            {'quality': q}
            for q in params.get('qualities', [90, 70, 50, 30])
        ]
    },
    'degradation_motion_blur': {
        'function': apply_motion_blur,
        'param_combinations': lambda params: [
            {'length': l, 'angle': a}
            for l in params.get('lengths', [5, 9, 15])
            for a in params.get('angles', [0, 30, 60, 90])
        ]
    },
    'degradation_noise_poisson_gaussian': {
        'function': apply_poisson_gaussian_noise,
        'param_combinations': lambda params: [
            {'sigma_read': sr, 'gain': g}
            for sr in params.get('sigma_reads', [2, 5, 10])
            for g in params.get('gains', [0.8, 1.0, 1.2])
        ]
    },
    'degradation_low_light': {
        'function': apply_low_light_degradation,
        'param_combinations': lambda params: [
            {'gain': g, 'noise_sigma': ns}
            for g in params.get('gains', [0.5, 0.7])
            for ns in params.get('noise_sigmas', [0.01, 0.02])
        ]
    },
    'degradation_downscale_upscale': {
        'function': apply_downscale_upscale,
        'param_combinations': lambda params: [
            {'scale_factor': sf}
            for sf in params.get('factors', [0.75, 0.5])
        ]
    },
    'degradation_gaussian_blur': {
        'function': apply_gaussian_blur_simple,
        'param_combinations': lambda params: [
            {'sigma': s}
            for s in params.get('sigmas', [0.5, 1.0, 2.0, 3.0])
        ]
    },
    'degradation_defocus_blur': {
        'function': apply_defocus_blur,
        'param_combinations': lambda params: [
            {'radius': r}
            for r in params.get('radii', [1.0, 2.0, 3.0, 5.0])
        ]
    },
    'degradation_atmospheric_scattering': {
        'function': apply_atmospheric_scattering,
        'param_combinations': lambda params: [
            {'scattering_coefficient': sc, 'atmosphere_color': ac}
            for sc in params.get('scattering_coefficients', [0.1, 0.2, 0.3])
            for ac in params.get('atmosphere_colors', [(0.8, 0.8, 0.9), (0.9, 0.85, 0.8)])
        ]
    },
    'degradation_vignetting': {
        'function': apply_vignetting,
        'param_combinations': lambda params: [
            {'strength': s, 'falloff': f}
            for s in params.get('strengths', [0.2, 0.4, 0.6])
            for f in params.get('falloffs', [1.5, 2.0, 3.0])
        ]
    }
}


def get_degradation_transforms() -> Dict[str, Any]:
    """Get all degradation transforms."""
    return DEGRADATION_TRANSFORMS


if __name__ == "__main__":
    # Test transforms with synthetic data
    test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    # Set random seed for deterministic testing
    np.random.seed(42)
    
    # Test a few transforms
    result, params = apply_jpeg_compression(test_image, 50)
    print(f"JPEG compression: {params}, output shape: {result.shape}")
    
    result, params = apply_motion_blur(test_image, 9, 45)
    print(f"Motion blur: {params}, output shape: {result.shape}")
    
    result, params = apply_poisson_gaussian_noise(test_image, 5.0, 1.0)
    print(f"Poisson-Gaussian noise: {params}, output shape: {result.shape}")
    
    print(f"Available degradation transforms: {list(DEGRADATION_TRANSFORMS.keys())}")