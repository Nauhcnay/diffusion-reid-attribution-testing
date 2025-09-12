"""
Background and context transforms for ReID diagnostics.

D) Background / context:
  D1. Foreground keep, background to black / mean color / heavy Gaussian blur
  D2. "Background only" control (foreground zeroed; expect terrible retrieval)
  
Note: All background operations require a foreground mask and gracefully skip if unavailable.
"""

from typing import Dict, List, Tuple, Any, Optional
import random

import cv2
import numpy as np

from io_utils import get_foreground_mask, load_background_images, resize_background_to_fit, uint8_to_float, float_to_uint8


def apply_background_to_black(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Replace background with black, keep foreground.
    
    Args:
        image: Input RGB image (uint8)
        mask: Foreground mask (bool array), if None will try to generate
        
    Returns:
        Image with black background and parameters dictionary, or None if no mask available
    """
    if mask is None:
        mask = get_foreground_mask(image)
    
    if mask is None:
        return None, {'error': 'no_mask_available'}
    
    # Create output image
    result = np.zeros_like(image)
    
    # Copy foreground pixels
    result[mask] = image[mask]
    
    params = {
        'background_color': 'black',
        'foreground_pixels': int(np.sum(mask))
    }
    
    return result, params


def apply_background_to_white(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Replace background with white, keep foreground.
    
    Args:
        image: Input RGB image (uint8)
        mask: Foreground mask (bool array), if None will try to generate
        
    Returns:
        Image with white background and parameters dictionary, or None if no mask available
    """
    if mask is None:
        mask = get_foreground_mask(image)
    
    if mask is None:
        return None, {'error': 'no_mask_available'}
    
    # Create white background
    result = np.full_like(image, 255)
    
    # Copy foreground pixels
    result[mask] = image[mask]
    
    params = {
        'background_color': 'white',
        'foreground_pixels': int(np.sum(mask))
    }
    
    return result, params


def apply_background_to_mean_color(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Replace background with mean color of original background, keep foreground.
    
    Args:
        image: Input RGB image (uint8)
        mask: Foreground mask (bool array), if None will try to generate
        
    Returns:
        Image with mean color background and parameters dictionary, or None if no mask available
    """
    if mask is None:
        mask = get_foreground_mask(image)
    
    if mask is None:
        return None, {'error': 'no_mask_available'}
    
    # Calculate mean color of background pixels
    background_mask = ~mask
    if np.sum(background_mask) == 0:
        # No background pixels, use image mean
        mean_color = np.mean(image.reshape(-1, 3), axis=0)
    else:
        mean_color = np.mean(image[background_mask], axis=0)
    
    # Create background with mean color
    result = np.full_like(image, mean_color.astype(np.uint8))
    
    # Copy foreground pixels
    result[mask] = image[mask]
    
    params = {
        'background_color': 'mean',
        'mean_rgb': mean_color.tolist(),
        'foreground_pixels': int(np.sum(mask))
    }
    
    return result, params


def apply_background_to_blur(image: np.ndarray, blur_sigma: float, mask: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Replace background with heavily blurred version, keep foreground.
    
    Args:
        image: Input RGB image (uint8)
        blur_sigma: Standard deviation for Gaussian blur
        mask: Foreground mask (bool array), if None will try to generate
        
    Returns:
        Image with blurred background and parameters dictionary, or None if no mask available
    """
    if mask is None:
        mask = get_foreground_mask(image)
    
    if mask is None:
        return None, {'error': 'no_mask_available'}
    
    # Create heavily blurred version of entire image
    kernel_size = int(2 * np.ceil(3 * blur_sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), blur_sigma)
    
    # Start with blurred image
    result = blurred.copy()
    
    # Replace foreground with original
    result[mask] = image[mask]
    
    params = {
        'background_type': 'blurred',
        'blur_sigma': blur_sigma,
        'kernel_size': kernel_size,
        'foreground_pixels': int(np.sum(mask))
    }
    
    return result, params


def apply_background_replacement(image: np.ndarray, background_images: List[np.ndarray], 
                               mask: Optional[np.ndarray] = None, seed: Optional[int] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Replace background with random background image, keep foreground.
    
    Args:
        image: Input RGB image (uint8)
        background_images: List of background images to choose from
        mask: Foreground mask (bool array), if None will try to generate
        seed: Random seed for reproducible background selection
        
    Returns:
        Image with replaced background and parameters dictionary, or None if no mask/backgrounds available
    """
    if not background_images:
        return None, {'error': 'no_backgrounds_available'}
    
    if mask is None:
        mask = get_foreground_mask(image)
    
    if mask is None:
        return None, {'error': 'no_mask_available'}
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Select random background
    background_idx = random.randint(0, len(background_images) - 1)
    background = background_images[background_idx]
    
    # Resize background to fit image dimensions
    target_shape = image.shape[:2]  # (height, width)
    resized_background = resize_background_to_fit(background, target_shape)
    
    # Start with resized background
    result = resized_background.copy()
    
    # Replace foreground with original
    result[mask] = image[mask]
    
    params = {
        'background_type': 'replaced',
        'background_index': background_idx,
        'background_original_shape': background.shape,
        'foreground_pixels': int(np.sum(mask))
    }
    
    return result, params


def apply_background_only(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Keep background only, zero out foreground (control experiment).
    This should result in poor ReID performance if the model depends on foreground.
    
    Args:
        image: Input RGB image (uint8)
        mask: Foreground mask (bool array), if None will try to generate
        
    Returns:
        Background-only image and parameters dictionary, or None if no mask available
    """
    if mask is None:
        mask = get_foreground_mask(image)
    
    if mask is None:
        return None, {'error': 'no_mask_available'}
    
    # Start with original image
    result = image.copy()
    
    # Zero out foreground pixels
    result[mask] = 0
    
    params = {
        'operation': 'background_only',
        'foreground_pixels_zeroed': int(np.sum(mask)),
        'background_pixels': int(np.sum(~mask))
    }
    
    return result, params


def apply_foreground_only(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Keep foreground only, zero out background.
    
    Args:
        image: Input RGB image (uint8)
        mask: Foreground mask (bool array), if None will try to generate
        
    Returns:
        Foreground-only image and parameters dictionary, or None if no mask available
    """
    if mask is None:
        mask = get_foreground_mask(image)
    
    if mask is None:
        return None, {'error': 'no_mask_available'}
    
    # Create black image
    result = np.zeros_like(image)
    
    # Copy only foreground pixels
    result[mask] = image[mask]
    
    params = {
        'operation': 'foreground_only',
        'foreground_pixels': int(np.sum(mask)),
        'background_pixels_zeroed': int(np.sum(~mask))
    }
    
    return result, params


def apply_background_inpainting(image: np.ndarray, mask: Optional[np.ndarray] = None, method: str = 'telea') -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Inpaint foreground area with background-like content, then keep original foreground.
    This creates a version where background context is extended under the person.
    
    Args:
        image: Input RGB image (uint8)
        mask: Foreground mask (bool array), if None will try to generate
        method: Inpainting method ('telea' or 'ns')
        
    Returns:
        Background-inpainted image and parameters dictionary, or None if no mask available
    """
    if mask is None:
        mask = get_foreground_mask(image)
    
    if mask is None:
        return None, {'error': 'no_mask_available'}
    
    # Convert mask to uint8 for OpenCV
    inpaint_mask = mask.astype(np.uint8) * 255
    
    # Choose inpainting method
    if method == 'telea':
        inpaint_flag = cv2.INPAINT_TELEA
    else:  # 'ns'
        inpaint_flag = cv2.INPAINT_NS
    
    # Inpaint the foreground region
    inpainted = cv2.inpaint(image, inpaint_mask, 3, inpaint_flag)
    
    # Keep original foreground
    result = inpainted.copy()
    result[mask] = image[mask]
    
    params = {
        'operation': 'background_inpainting',
        'method': method,
        'inpainted_pixels': int(np.sum(mask)),
        'foreground_pixels': int(np.sum(mask))
    }
    
    return result, params


def apply_background_gradual_transition(image: np.ndarray, transition_color: Tuple[int, int, int] = (128, 128, 128),
                                      transition_width: int = 20, mask: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Apply gradual transition from foreground to background color.
    
    Args:
        image: Input RGB image (uint8)
        transition_color: RGB color for background
        transition_width: Width of transition zone in pixels
        mask: Foreground mask (bool array), if None will try to generate
        
    Returns:
        Image with gradual background transition and parameters dictionary, or None if no mask available
    """
    if mask is None:
        mask = get_foreground_mask(image)
    
    if mask is None:
        return None, {'error': 'no_mask_available'}
    
    # Create distance transform from foreground mask
    dist_transform = cv2.distanceTransform(
        (~mask).astype(np.uint8), 
        cv2.DIST_L2, 
        cv2.DIST_MASK_PRECISE
    )
    
    # Create transition weights
    transition_weights = np.clip(dist_transform / transition_width, 0, 1)
    
    # Create background with transition color
    background = np.full_like(image, transition_color)
    
    # Blend based on distance
    result = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        result[:, :, i] = (
            (1 - transition_weights) * image[:, :, i] + 
            transition_weights * background[:, :, i]
        )
    
    # Convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    params = {
        'operation': 'gradual_transition',
        'transition_color': transition_color,
        'transition_width': transition_width,
        'foreground_pixels': int(np.sum(mask))
    }
    
    return result, params


# Transform registry for background transforms
BACKGROUND_TRANSFORMS = {
    'background_to_black': {
        'function': apply_background_to_black,
        'param_combinations': lambda params: [{}],
        'requires_mask': True
    },
    'background_to_white': {
        'function': apply_background_to_white,
        'param_combinations': lambda params: [{}],
        'requires_mask': True
    },
    'background_to_mean': {
        'function': apply_background_to_mean_color,
        'param_combinations': lambda params: [{}],
        'requires_mask': True
    },
    'background_to_blur': {
        'function': apply_background_to_blur,
        'param_combinations': lambda params: [
            {'blur_sigma': bs}
            for bs in params.get('blur_sigmas', [10, 20, 30])
        ],
        'requires_mask': True
    },
    'background_replace': {
        'function': apply_background_replacement,
        'param_combinations': lambda params: [{}],
        'requires_mask': True,
        'requires_backgrounds': True
    },
    'background_only': {
        'function': apply_background_only,
        'param_combinations': lambda params: [{}],
        'requires_mask': True
    },
    'foreground_only': {
        'function': apply_foreground_only,
        'param_combinations': lambda params: [{}],
        'requires_mask': True
    },
    'background_inpaint': {
        'function': apply_background_inpainting,
        'param_combinations': lambda params: [
            {'method': method}
            for method in params.get('methods', ['telea', 'ns'])
        ],
        'requires_mask': True
    },
    'background_gradual_transition': {
        'function': apply_background_gradual_transition,
        'param_combinations': lambda params: [
            {'transition_color': tc, 'transition_width': tw}
            for tc in params.get('transition_colors', [(128, 128, 128), (64, 64, 64), (192, 192, 192)])
            for tw in params.get('transition_widths', [10, 20, 30])
        ],
        'requires_mask': True
    }
}


def get_background_transforms() -> Dict[str, Any]:
    """Get all background transforms."""
    return BACKGROUND_TRANSFORMS


def apply_background_transform(transform_name: str, image: np.ndarray, params: Dict[str, Any], 
                             mask: Optional[np.ndarray] = None, 
                             background_images: Optional[List[np.ndarray]] = None,
                             seed: Optional[int] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Apply a background transform by name with given parameters.
    
    Args:
        transform_name: Name of transform to apply
        image: Input RGB image (uint8)
        params: Parameters for the transform
        mask: Optional foreground mask
        background_images: Optional list of background images
        seed: Optional random seed
        
    Returns:
        Transformed image and actual parameters used, or None if transform failed
    """
    if transform_name not in BACKGROUND_TRANSFORMS:
        return None, {'error': f'unknown_transform: {transform_name}'}
    
    transform_info = BACKGROUND_TRANSFORMS[transform_name]
    
    # Check requirements
    if transform_info.get('requires_backgrounds', False) and not background_images:
        return None, {'error': 'backgrounds_required_but_not_provided'}
    
    # Get transform function
    transform_func = transform_info['function']
    
    # Apply transform with special handling for background replacement
    try:
        if transform_name == 'background_replace':
            return transform_func(image, background_images, mask, seed)
        else:
            return transform_func(image, mask=mask, **params)
    except Exception as e:
        return None, {'error': f'transform_failed: {str(e)}'}


if __name__ == "__main__":
    # Test transforms with synthetic data
    test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    # Create simple synthetic mask (circle in center)
    y, x = np.ogrid[:64, :64]
    center_y, center_x = 32, 32
    radius = 20
    test_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    # Test a few transforms
    result, params = apply_background_to_black(test_image, test_mask)
    print(f"Background to black: {params}, output shape: {result.shape if result is not None else None}")
    
    result, params = apply_background_to_blur(test_image, 15.0, test_mask)
    print(f"Background blur: {params}, output shape: {result.shape if result is not None else None}")
    
    result, params = apply_background_only(test_image, test_mask)
    print(f"Background only: {params}, output shape: {result.shape if result is not None else None}")
    
    print(f"Available background transforms: {list(BACKGROUND_TRANSFORMS.keys())}")