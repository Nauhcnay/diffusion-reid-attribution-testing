"""
Color and texture transforms for ReID diagnostics.

C) Color / texture (lightweight, domain-safe):
  C1. Desaturate (grayscale)
  C2. HSV jitter
  C3. Texture suppression: Non-Local Means, Bilateral strong, TV-denoise
"""

from typing import Dict, List, Tuple, Any
import warnings

import cv2
import numpy as np
from skimage import color
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means

from io_utils import uint8_to_float, float_to_uint8


def apply_desaturate(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert image to grayscale (desaturate).
    
    Args:
        image: Input RGB image (uint8)
        
    Returns:
        Grayscale image converted back to RGB and parameters dictionary
    """
    # Convert to grayscale using standard luminance weights
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Convert back to 3-channel RGB (all channels identical)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    params = {
        'method': 'luminance_weighted'
    }
    
    return gray_rgb, params


def apply_hsv_jitter(image: np.ndarray, hue_delta: float, saturation_delta: float, value_delta: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply HSV color jittering.
    
    Args:
        image: Input RGB image (uint8)
        hue_delta: Change in hue (degrees, ±180)
        saturation_delta: Change in saturation (±1.0)
        value_delta: Change in value/brightness (±1.0)
        
    Returns:
        HSV-jittered image and parameters dictionary
    """
    # Convert to HSV (OpenCV uses H: 0-179, S: 0-255, V: 0-255)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Apply hue shift (wrap around)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta * 179 / 180) % 180
    
    # Apply saturation change (clamp to valid range)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + saturation_delta), 0, 255)
    
    # Apply value change (clamp to valid range)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + value_delta), 0, 255)
    
    # Convert back to RGB
    hsv_uint8 = hsv.astype(np.uint8)
    result = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2RGB)
    
    params = {
        'hue_delta': hue_delta,
        'saturation_delta': saturation_delta,
        'value_delta': value_delta
    }
    
    return result, params


def apply_color_balance_shift(image: np.ndarray, red_gain: float, green_gain: float, blue_gain: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply color balance shift by adjusting channel gains.
    
    Args:
        image: Input RGB image (uint8)
        red_gain: Multiplicative gain for red channel
        green_gain: Multiplicative gain for green channel
        blue_gain: Multiplicative gain for blue channel
        
    Returns:
        Color-balanced image and parameters dictionary
    """
    float_image = uint8_to_float(image)
    
    # Apply channel gains
    result = float_image.copy()
    result[:, :, 0] *= red_gain    # Red
    result[:, :, 1] *= green_gain  # Green
    result[:, :, 2] *= blue_gain   # Blue
    
    # Clip to valid range
    result = np.clip(result, 0, 1)
    result_uint8 = float_to_uint8(result)
    
    params = {
        'red_gain': red_gain,
        'green_gain': green_gain,
        'blue_gain': blue_gain
    }
    
    return result_uint8, params


def apply_nonlocal_means_denoising(image: np.ndarray, h: float, template_window_size: int = 7, search_window_size: int = 21) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply Non-Local Means denoising for texture suppression.
    
    Args:
        image: Input RGB image (uint8)
        h: Filter strength. Higher h removes more noise but removes detail too
        template_window_size: Size of template patch (should be odd)
        search_window_size: Size of search window (should be odd)
        
    Returns:
        Denoised image and parameters dictionary
    """
    # OpenCV Non-Local Means works on uint8
    denoised = cv2.fastNlMeansDenoisingColored(
        image, 
        None, 
        h, 
        h,  # hColor (same as h for color images)
        template_window_size, 
        search_window_size
    )
    
    params = {
        'h': h,
        'template_window_size': template_window_size,
        'search_window_size': search_window_size
    }
    
    return denoised, params


def apply_nonlocal_means_scikit(image: np.ndarray, h: float, patch_size: int = 5, patch_distance: int = 6) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply scikit-image Non-Local Means denoising.
    
    Args:
        image: Input RGB image (uint8)
        h: Cut-off distance (smaller means more denoising)
        patch_size: Size of patches used for denoising
        patch_distance: Maximal distance in pixels for patches
        
    Returns:
        Denoised image and parameters dictionary
    """
    float_image = uint8_to_float(image)
    
    # Apply NL-Means (scikit-image version)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress scikit-image warnings
        try:
            # Try new API first (channel_axis parameter for newer versions)
            denoised = denoise_nl_means(
                float_image,
                h=h / 255.0,  # Convert to float range
                fast_mode=True,
                patch_size=patch_size,
                patch_distance=patch_distance,
                channel_axis=-1
            )
        except TypeError:
            try:
                # Fall back to multichannel parameter for older versions
                denoised = denoise_nl_means(
                    float_image,
                    h=h / 255.0,  # Convert to float range
                    fast_mode=True,
                    patch_size=patch_size,
                    patch_distance=patch_distance,
                    multichannel=True
                )
            except TypeError:
                # Final fallback - apply to each channel separately
                denoised = np.zeros_like(float_image)
                for i in range(float_image.shape[2]):
                    denoised[:, :, i] = denoise_nl_means(
                        float_image[:, :, i],
                        h=h / 255.0,
                        fast_mode=True,
                        patch_size=patch_size,
                        patch_distance=patch_distance
                    )
    
    result_uint8 = float_to_uint8(denoised)
    
    params = {
        'h': h,
        'patch_size': patch_size,
        'patch_distance': patch_distance,
        'implementation': 'scikit-image'
    }
    
    return result_uint8, params


def apply_bilateral_strong(image: np.ndarray, diameter: int, sigma_color: float, sigma_space: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply strong bilateral filter for texture suppression.
    
    Args:
        image: Input RGB image (uint8)
        diameter: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space (higher = more averaging)
        sigma_space: Filter sigma in coordinate space (higher = farther pixels influence)
        
    Returns:
        Bilateral filtered image and parameters dictionary
    """
    filtered = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    
    params = {
        'diameter': diameter,
        'sigma_color': sigma_color,
        'sigma_space': sigma_space,
        'purpose': 'texture_suppression'
    }
    
    return filtered, params


def apply_tv_denoising(image: np.ndarray, weight: float, max_iter: int = 200) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply Total Variation denoising for texture suppression.

    Args:
        image: Input RGB image (uint8)
        weight: Denoising weight (higher = more denoising)
        max_iter: Maximum number of iterations (deprecated in newer scikit-image)

    Returns:
        TV-denoised image and parameters dictionary
    """
    float_image = uint8_to_float(image)

    # Apply TV denoising to each channel
    result = np.zeros_like(float_image)
    for i in range(3):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress convergence warnings
            try:
                # Try with max_iter parameter (older scikit-image)
                result[:, :, i] = denoise_tv_chambolle(
                    float_image[:, :, i],
                    weight=weight,
                    max_iter=max_iter
                )
            except TypeError:
                # Fall back to newer API without max_iter
                result[:, :, i] = denoise_tv_chambolle(
                    float_image[:, :, i],
                    weight=weight
                )

    result_uint8 = float_to_uint8(result)

    params = {
        'weight': weight,
        'max_iter': max_iter
    }

    return result_uint8, params


def apply_edge_preserving_filter(image: np.ndarray, flags: int = 1, sigma_s: float = 50, sigma_r: float = 0.4) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply OpenCV edge-preserving filter.
    
    Args:
        image: Input RGB image (uint8)
        flags: Edge preserving filters: 1=RECURS_FILTER, 2=NORMCONV_FILTER
        sigma_s: Size of neighborhood area
        sigma_r: How dissimilar colors average
        
    Returns:
        Edge-preserving filtered image and parameters dictionary
    """
    filtered = cv2.edgePreservingFilter(image, flags=flags, sigma_s=sigma_s, sigma_r=sigma_r)
    
    params = {
        'flags': flags,
        'sigma_s': sigma_s,
        'sigma_r': sigma_r
    }
    
    return filtered, params


def apply_detail_enhancement(image: np.ndarray, sigma_s: float = 10, sigma_r: float = 0.15) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply detail enhancement filter.
    
    Args:
        image: Input RGB image (uint8)
        sigma_s: Size of neighborhood area
        sigma_r: How dissimilar colors average
        
    Returns:
        Detail-enhanced image and parameters dictionary
    """
    enhanced = cv2.detailEnhance(image, sigma_s=sigma_s, sigma_r=sigma_r)
    
    params = {
        'sigma_s': sigma_s,
        'sigma_r': sigma_r
    }
    
    return enhanced, params


def apply_stylization(image: np.ndarray, sigma_s: float = 150, sigma_r: float = 0.25) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply stylization filter to reduce texture detail.
    
    Args:
        image: Input RGB image (uint8)
        sigma_s: Size of neighborhood area
        sigma_r: How dissimilar colors average
        
    Returns:
        Stylized image and parameters dictionary
    """
    stylized = cv2.stylization(image, sigma_s=sigma_s, sigma_r=sigma_r)
    
    params = {
        'sigma_s': sigma_s,
        'sigma_r': sigma_r
    }
    
    return stylized, params


def apply_color_quantization(image: np.ndarray, k: int = 8) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply color quantization using K-means clustering.
    
    Args:
        image: Input RGB image (uint8)
        k: Number of color clusters
        
    Returns:
        Quantized image and parameters dictionary
    """
    # Reshape image to be a list of pixels
    data = image.reshape((-1, 3))
    data = np.float32(data)
    
    # Define criteria and apply K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8 and reshape to original image shape
    centers = np.uint8(centers)
    quantized_data = centers[labels.flatten()]
    quantized = quantized_data.reshape(image.shape)
    
    params = {
        'k_clusters': k,
        'method': 'kmeans'
    }
    
    return quantized, params


# Transform registry for color/texture transforms
COLOR_TEXTURE_TRANSFORMS = {
    'color_desaturate': {
        'function': apply_desaturate,
        'param_combinations': lambda params: [{}]  # No parameters needed
    },
    'color_hsv_jitter': {
        'function': apply_hsv_jitter,
        'param_combinations': lambda params: [
            {'hue_delta': hd, 'saturation_delta': sd, 'value_delta': vd}
            for hd in params.get('hue_deltas', [-5, 0, 5])
            for sd in params.get('saturation_deltas', [-0.1, -0.05, 0.05, 0.1])
            for vd in params.get('value_deltas', [-0.1, -0.05, 0.05, 0.1])
            if not (hd == 0 and sd == 0 and vd == 0)  # Skip identity transform
        ]
    },
    'color_balance_shift': {
        'function': apply_color_balance_shift,
        'param_combinations': lambda params: [
            {'red_gain': rg, 'green_gain': gg, 'blue_gain': bg}
            for rg in params.get('red_gains', [0.8, 1.0, 1.2])
            for gg in params.get('green_gains', [0.8, 1.0, 1.2])
            for bg in params.get('blue_gains', [0.8, 1.0, 1.2])
            if not (rg == 1.0 and gg == 1.0 and bg == 1.0)  # Skip identity
        ]
    },
    'texture_nonlocal_means': {
        'function': apply_nonlocal_means_denoising,
        'param_combinations': lambda params: [
            {'h': h, 'template_window_size': tws, 'search_window_size': sws}
            for h in params.get('h_values', [5, 10])
            for tws in params.get('template_sizes', [7])
            for sws in params.get('search_sizes', [21])
        ]
    },
    'texture_nonlocal_means_scikit': {
        'function': apply_nonlocal_means_scikit,
        'param_combinations': lambda params: [
            {'h': h, 'patch_size': ps, 'patch_distance': pd}
            for h in params.get('h_values', [5, 10, 20])
            for ps in params.get('patch_sizes', [5, 7])
            for pd in params.get('patch_distances', [6, 11])
        ]
    },
    'texture_bilateral_strong': {
        'function': apply_bilateral_strong,
        'param_combinations': lambda params: [
            {'diameter': d, 'sigma_color': sc, 'sigma_space': ss}
            for d in params.get('diameters', [9, 15])
            for sc in params.get('sigma_colors', [50, 75])
            for ss in params.get('sigma_spaces', [9, 15])
        ]
    },
    'texture_tv_denoise': {
        'function': apply_tv_denoising,
        'param_combinations': lambda params: [
            {'weight': w, 'max_iter': mi}
            for w in params.get('weights', [0.05, 0.1])
            for mi in params.get('max_iters', [200])
        ]
    },
    'texture_edge_preserving': {
        'function': apply_edge_preserving_filter,
        'param_combinations': lambda params: [
            {'flags': f, 'sigma_s': ss, 'sigma_r': sr}
            for f in params.get('flags', [1, 2])
            for ss in params.get('sigma_s_values', [30, 50, 80])
            for sr in params.get('sigma_r_values', [0.2, 0.4])
        ]
    },
    'texture_detail_enhance': {
        'function': apply_detail_enhancement,
        'param_combinations': lambda params: [
            {'sigma_s': ss, 'sigma_r': sr}
            for ss in params.get('sigma_s_values', [10, 20])
            for sr in params.get('sigma_r_values', [0.1, 0.15, 0.25])
        ]
    },
    'texture_stylization': {
        'function': apply_stylization,
        'param_combinations': lambda params: [
            {'sigma_s': ss, 'sigma_r': sr}
            for ss in params.get('sigma_s_values', [100, 150, 200])
            for sr in params.get('sigma_r_values', [0.2, 0.25, 0.3])
        ]
    },
    'color_quantization': {
        'function': apply_color_quantization,
        'param_combinations': lambda params: [
            {'k': k}
            for k in params.get('k_values', [4, 8, 16, 32])
        ]
    }
}


def get_color_texture_transforms() -> Dict[str, Any]:
    """Get all color/texture transforms."""
    return COLOR_TEXTURE_TRANSFORMS


if __name__ == "__main__":
    # Test transforms with synthetic data
    test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    # Test a few transforms
    result, params = apply_desaturate(test_image)
    print(f"Desaturate: {params}, output shape: {result.shape}")
    
    result, params = apply_hsv_jitter(test_image, 10, 0.1, -0.05)
    print(f"HSV jitter: {params}, output shape: {result.shape}")
    
    result, params = apply_bilateral_strong(test_image, 9, 50, 50)
    print(f"Bilateral strong: {params}, output shape: {result.shape}")
    
    print(f"Available color/texture transforms: {list(COLOR_TEXTURE_TRANSFORMS.keys())}")