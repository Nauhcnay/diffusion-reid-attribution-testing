"""
Morphological transforms for ReID diagnostics.

F) Morphology on foreground mask, then composite back to image (image-domain result):
  F1. Opening/Closing (kernel shapes: ellipse/square; sizes)
  F2. Light Erode/Dilate (iterations)
  F3. Fill holes before compositing (connected-component hole filling)
  
Implementation rule: these operate on a binary mask, then re-composite original 
pixels under the morphologically adjusted mask onto a neutral background.
"""

from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
from scipy import ndimage

from io_utils import get_foreground_mask


def create_morphological_kernel(shape: str, size: int) -> np.ndarray:
    """
    Create morphological kernel of specified shape and size.
    
    Args:
        shape: 'ellipse', 'rectangle', or 'cross'
        size: Kernel size (odd number)
        
    Returns:
        Morphological kernel
    """
    if size % 2 == 0:
        size += 1  # Ensure odd size
    
    if shape == 'ellipse':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif shape == 'rectangle':
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif shape == 'cross':
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    else:
        # Default to ellipse
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def apply_morphological_opening(image: np.ndarray, kernel_shape: str, kernel_size: int, 
                              mask: Optional[np.ndarray] = None, background_color: Tuple[int, int, int] = (128, 128, 128)) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Apply morphological opening to foreground mask, then composite back to image.
    
    Args:
        image: Input RGB image (uint8)
        kernel_shape: Shape of morphological kernel ('ellipse', 'rectangle', 'cross')
        kernel_size: Size of kernel (will be made odd)
        mask: Foreground mask (bool array), if None will try to generate
        background_color: RGB color for background
        
    Returns:
        Morphologically processed image and parameters dictionary, or None if no mask available
    """
    if mask is None:
        mask = get_foreground_mask(image)
    
    if mask is None:
        return None, {'error': 'no_mask_available'}
    
    # Create morphological kernel
    kernel = create_morphological_kernel(kernel_shape, kernel_size)
    
    # Convert mask to uint8 for OpenCV
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # Apply morphological opening
    opened_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    
    # Convert back to bool
    processed_mask = opened_mask > 128
    
    # Create result image with background color
    result = np.full_like(image, background_color)
    
    # Composite original pixels under processed mask
    result[processed_mask] = image[processed_mask]
    
    params = {
        'operation': 'opening',
        'kernel_shape': kernel_shape,
        'kernel_size': kernel_size,
        'background_color': background_color,
        'original_foreground_pixels': int(np.sum(mask)),
        'processed_foreground_pixels': int(np.sum(processed_mask))
    }
    
    return result, params


def apply_morphological_closing(image: np.ndarray, kernel_shape: str, kernel_size: int, 
                               mask: Optional[np.ndarray] = None, background_color: Tuple[int, int, int] = (128, 128, 128)) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Apply morphological closing to foreground mask, then composite back to image.
    
    Args:
        image: Input RGB image (uint8)
        kernel_shape: Shape of morphological kernel
        kernel_size: Size of kernel (will be made odd)
        mask: Foreground mask (bool array), if None will try to generate
        background_color: RGB color for background
        
    Returns:
        Morphologically processed image and parameters dictionary, or None if no mask available
    """
    if mask is None:
        mask = get_foreground_mask(image)
    
    if mask is None:
        return None, {'error': 'no_mask_available'}
    
    # Create morphological kernel
    kernel = create_morphological_kernel(kernel_shape, kernel_size)
    
    # Convert mask to uint8 for OpenCV
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # Apply morphological closing
    closed_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to bool
    processed_mask = closed_mask > 128
    
    # Create result image with background color
    result = np.full_like(image, background_color)
    
    # Composite original pixels under processed mask
    result[processed_mask] = image[processed_mask]
    
    params = {
        'operation': 'closing',
        'kernel_shape': kernel_shape,
        'kernel_size': kernel_size,
        'background_color': background_color,
        'original_foreground_pixels': int(np.sum(mask)),
        'processed_foreground_pixels': int(np.sum(processed_mask))
    }
    
    return result, params


def apply_morphological_erosion(image: np.ndarray, kernel_shape: str, kernel_size: int, iterations: int,
                               mask: Optional[np.ndarray] = None, background_color: Tuple[int, int, int] = (128, 128, 128)) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Apply morphological erosion to foreground mask, then composite back to image.
    
    Args:
        image: Input RGB image (uint8)
        kernel_shape: Shape of morphological kernel
        kernel_size: Size of kernel (will be made odd)
        iterations: Number of erosion iterations
        mask: Foreground mask (bool array), if None will try to generate
        background_color: RGB color for background
        
    Returns:
        Morphologically processed image and parameters dictionary, or None if no mask available
    """
    if mask is None:
        mask = get_foreground_mask(image)
    
    if mask is None:
        return None, {'error': 'no_mask_available'}
    
    # Create morphological kernel
    kernel = create_morphological_kernel(kernel_shape, kernel_size)
    
    # Convert mask to uint8 for OpenCV
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # Apply morphological erosion
    eroded_mask = cv2.erode(mask_uint8, kernel, iterations=iterations)
    
    # Convert back to bool
    processed_mask = eroded_mask > 128
    
    # Create result image with background color
    result = np.full_like(image, background_color)
    
    # Composite original pixels under processed mask
    result[processed_mask] = image[processed_mask]
    
    params = {
        'operation': 'erosion',
        'kernel_shape': kernel_shape,
        'kernel_size': kernel_size,
        'iterations': iterations,
        'background_color': background_color,
        'original_foreground_pixels': int(np.sum(mask)),
        'processed_foreground_pixels': int(np.sum(processed_mask))
    }
    
    return result, params


def apply_morphological_dilation(image: np.ndarray, kernel_shape: str, kernel_size: int, iterations: int,
                                mask: Optional[np.ndarray] = None, background_color: Tuple[int, int, int] = (128, 128, 128)) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Apply morphological dilation to foreground mask, then composite back to image.
    
    Args:
        image: Input RGB image (uint8)
        kernel_shape: Shape of morphological kernel
        kernel_size: Size of kernel (will be made odd)
        iterations: Number of dilation iterations
        mask: Foreground mask (bool array), if None will try to generate
        background_color: RGB color for background
        
    Returns:
        Morphologically processed image and parameters dictionary, or None if no mask available
    """
    if mask is None:
        mask = get_foreground_mask(image)
    
    if mask is None:
        return None, {'error': 'no_mask_available'}
    
    # Create morphological kernel
    kernel = create_morphological_kernel(kernel_shape, kernel_size)
    
    # Convert mask to uint8 for OpenCV
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # Apply morphological dilation
    dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=iterations)
    
    # Convert back to bool
    processed_mask = dilated_mask > 128
    
    # Create result image with background color
    result = np.full_like(image, background_color)
    
    # Composite original pixels under processed mask
    # Note: For dilation, we may be extending beyond original foreground
    # We can only use pixels that exist in the original image
    valid_pixels = processed_mask
    result[valid_pixels] = image[valid_pixels]
    
    params = {
        'operation': 'dilation',
        'kernel_shape': kernel_shape,
        'kernel_size': kernel_size,
        'iterations': iterations,
        'background_color': background_color,
        'original_foreground_pixels': int(np.sum(mask)),
        'processed_foreground_pixels': int(np.sum(processed_mask))
    }
    
    return result, params


def fill_holes_in_mask(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in binary mask using connected component analysis.
    
    Args:
        mask: Binary mask as bool array
        
    Returns:
        Mask with holes filled
    """
    # Convert to uint8
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # Find contours
    contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create filled mask
    filled_mask = np.zeros_like(mask_uint8)
    
    # Fill external contours (hierarchy[i][3] == -1 means external)
    if hierarchy is not None:
        for i in range(len(contours)):
            if hierarchy[0][i][3] == -1:  # External contour
                cv2.drawContours(filled_mask, contours, i, 255, thickness=cv2.FILLED)
    
    return filled_mask > 128


def apply_hole_filling(image: np.ndarray, mask: Optional[np.ndarray] = None, 
                      background_color: Tuple[int, int, int] = (128, 128, 128)) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Fill holes in foreground mask, then composite back to image.
    
    Args:
        image: Input RGB image (uint8)
        mask: Foreground mask (bool array), if None will try to generate
        background_color: RGB color for background
        
    Returns:
        Image with hole-filled mask and parameters dictionary, or None if no mask available
    """
    if mask is None:
        mask = get_foreground_mask(image)
    
    if mask is None:
        return None, {'error': 'no_mask_available'}
    
    # Fill holes in mask
    filled_mask = fill_holes_in_mask(mask)
    
    # Create result image with background color
    result = np.full_like(image, background_color)
    
    # Composite original pixels under filled mask
    result[filled_mask] = image[filled_mask]
    
    params = {
        'operation': 'hole_filling',
        'background_color': background_color,
        'original_foreground_pixels': int(np.sum(mask)),
        'filled_foreground_pixels': int(np.sum(filled_mask)),
        'holes_filled_pixels': int(np.sum(filled_mask) - np.sum(mask))
    }
    
    return result, params


def apply_morphological_gradient(image: np.ndarray, kernel_shape: str, kernel_size: int,
                                mask: Optional[np.ndarray] = None, background_color: Tuple[int, int, int] = (128, 128, 128)) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Apply morphological gradient (dilation - erosion) to foreground mask, then composite back to image.
    This highlights the boundary of the foreground.
    
    Args:
        image: Input RGB image (uint8)
        kernel_shape: Shape of morphological kernel
        kernel_size: Size of kernel (will be made odd)
        mask: Foreground mask (bool array), if None will try to generate
        background_color: RGB color for background
        
    Returns:
        Morphological gradient image and parameters dictionary, or None if no mask available
    """
    if mask is None:
        mask = get_foreground_mask(image)
    
    if mask is None:
        return None, {'error': 'no_mask_available'}
    
    # Create morphological kernel
    kernel = create_morphological_kernel(kernel_shape, kernel_size)
    
    # Convert mask to uint8 for OpenCV
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # Apply morphological gradient
    gradient_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_GRADIENT, kernel)
    
    # Convert back to bool
    processed_mask = gradient_mask > 128
    
    # Create result image with background color
    result = np.full_like(image, background_color)
    
    # Composite original pixels under gradient mask (boundary pixels only)
    result[processed_mask] = image[processed_mask]
    
    params = {
        'operation': 'gradient',
        'kernel_shape': kernel_shape,
        'kernel_size': kernel_size,
        'background_color': background_color,
        'original_foreground_pixels': int(np.sum(mask)),
        'gradient_pixels': int(np.sum(processed_mask))
    }
    
    return result, params


def apply_tophat_transform(image: np.ndarray, kernel_shape: str, kernel_size: int,
                          mask: Optional[np.ndarray] = None, background_color: Tuple[int, int, int] = (128, 128, 128)) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Apply morphological top-hat transform (original - opening) to foreground mask.
    This highlights small bright details that were removed by opening.
    
    Args:
        image: Input RGB image (uint8)
        kernel_shape: Shape of morphological kernel
        kernel_size: Size of kernel (will be made odd)
        mask: Foreground mask (bool array), if None will try to generate
        background_color: RGB color for background
        
    Returns:
        Top-hat transformed image and parameters dictionary, or None if no mask available
    """
    if mask is None:
        mask = get_foreground_mask(image)
    
    if mask is None:
        return None, {'error': 'no_mask_available'}
    
    # Create morphological kernel
    kernel = create_morphological_kernel(kernel_shape, kernel_size)
    
    # Convert mask to uint8 for OpenCV
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # Apply morphological top-hat
    tophat_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_TOPHAT, kernel)
    
    # Convert back to bool
    processed_mask = tophat_mask > 128
    
    # Create result image with background color
    result = np.full_like(image, background_color)
    
    # Composite original pixels under top-hat mask
    result[processed_mask] = image[processed_mask]
    
    params = {
        'operation': 'tophat',
        'kernel_shape': kernel_shape,
        'kernel_size': kernel_size,
        'background_color': background_color,
        'original_foreground_pixels': int(np.sum(mask)),
        'tophat_pixels': int(np.sum(processed_mask))
    }
    
    return result, params


def apply_blackhat_transform(image: np.ndarray, kernel_shape: str, kernel_size: int,
                            mask: Optional[np.ndarray] = None, background_color: Tuple[int, int, int] = (128, 128, 128)) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Apply morphological black-hat transform (closing - original) to foreground mask.
    This highlights small dark details that were filled by closing.
    
    Args:
        image: Input RGB image (uint8)
        kernel_shape: Shape of morphological kernel
        kernel_size: Size of kernel (will be made odd)
        mask: Foreground mask (bool array), if None will try to generate
        background_color: RGB color for background
        
    Returns:
        Black-hat transformed image and parameters dictionary, or None if no mask available
    """
    if mask is None:
        mask = get_foreground_mask(image)
    
    if mask is None:
        return None, {'error': 'no_mask_available'}
    
    # Create morphological kernel
    kernel = create_morphological_kernel(kernel_shape, kernel_size)
    
    # Convert mask to uint8 for OpenCV
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # Apply morphological black-hat
    blackhat_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_BLACKHAT, kernel)
    
    # Convert back to bool
    processed_mask = blackhat_mask > 128
    
    # Create result image with background color
    result = np.full_like(image, background_color)
    
    # Composite original pixels under black-hat mask
    result[processed_mask] = image[processed_mask]
    
    params = {
        'operation': 'blackhat',
        'kernel_shape': kernel_shape,
        'kernel_size': kernel_size,
        'background_color': background_color,
        'original_foreground_pixels': int(np.sum(mask)),
        'blackhat_pixels': int(np.sum(processed_mask))
    }
    
    return result, params


# Transform registry for morphological transforms
MORPHOLOGY_TRANSFORMS = {
    'morphology_opening': {
        'function': apply_morphological_opening,
        'param_combinations': lambda params: [
            {'kernel_shape': ks, 'kernel_size': ksize, 'background_color': bc}
            for ks in params.get('kernel_shapes', ['ellipse', 'rectangle'])
            for ksize in params.get('kernel_sizes', [3, 5])
            for bc in params.get('background_colors', [(128, 128, 128)])
        ],
        'requires_mask': True
    },
    'morphology_closing': {
        'function': apply_morphological_closing,
        'param_combinations': lambda params: [
            {'kernel_shape': ks, 'kernel_size': ksize, 'background_color': bc}
            for ks in params.get('kernel_shapes', ['ellipse', 'rectangle'])
            for ksize in params.get('kernel_sizes', [3, 5])
            for bc in params.get('background_colors', [(128, 128, 128)])
        ],
        'requires_mask': True
    },
    'morphology_erode': {
        'function': apply_morphological_erosion,
        'param_combinations': lambda params: [
            {'kernel_shape': ks, 'kernel_size': ksize, 'iterations': it, 'background_color': bc}
            for ks in params.get('kernel_shapes', ['ellipse', 'rectangle'])
            for ksize in params.get('kernel_sizes', [3, 5])
            for it in params.get('iterations', [1, 2])
            for bc in params.get('background_colors', [(128, 128, 128)])
        ],
        'requires_mask': True
    },
    'morphology_dilate': {
        'function': apply_morphological_dilation,
        'param_combinations': lambda params: [
            {'kernel_shape': ks, 'kernel_size': ksize, 'iterations': it, 'background_color': bc}
            for ks in params.get('kernel_shapes', ['ellipse', 'rectangle'])
            for ksize in params.get('kernel_sizes', [3, 5])
            for it in params.get('iterations', [1, 2])
            for bc in params.get('background_colors', [(128, 128, 128)])
        ],
        'requires_mask': True
    },
    'morphology_fill_holes': {
        'function': apply_hole_filling,
        'param_combinations': lambda params: [
            {'background_color': bc}
            for bc in params.get('background_colors', [(128, 128, 128)])
        ],
        'requires_mask': True
    },
    'morphology_gradient': {
        'function': apply_morphological_gradient,
        'param_combinations': lambda params: [
            {'kernel_shape': ks, 'kernel_size': ksize, 'background_color': bc}
            for ks in params.get('kernel_shapes', ['ellipse', 'rectangle'])
            for ksize in params.get('kernel_sizes', [3, 5])
            for bc in params.get('background_colors', [(128, 128, 128)])
        ],
        'requires_mask': True
    },
    'morphology_tophat': {
        'function': apply_tophat_transform,
        'param_combinations': lambda params: [
            {'kernel_shape': ks, 'kernel_size': ksize, 'background_color': bc}
            for ks in params.get('kernel_shapes', ['ellipse', 'rectangle'])
            for ksize in params.get('kernel_sizes', [5, 7, 9])
            for bc in params.get('background_colors', [(128, 128, 128)])
        ],
        'requires_mask': True
    },
    'morphology_blackhat': {
        'function': apply_blackhat_transform,
        'param_combinations': lambda params: [
            {'kernel_shape': ks, 'kernel_size': ksize, 'background_color': bc}
            for ks in params.get('kernel_shapes', ['ellipse', 'rectangle'])
            for ksize in params.get('kernel_sizes', [5, 7, 9])
            for bc in params.get('background_colors', [(128, 128, 128)])
        ],
        'requires_mask': True
    }
}


def get_morphology_transforms() -> Dict[str, Any]:
    """Get all morphological transforms."""
    return MORPHOLOGY_TRANSFORMS


if __name__ == "__main__":
    # Test transforms with synthetic data
    test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    # Create simple synthetic mask (circle in center with small hole)
    y, x = np.ogrid[:64, :64]
    center_y, center_x = 32, 32
    
    # Main circle
    main_radius = 20
    main_mask = (x - center_x)**2 + (y - center_y)**2 <= main_radius**2
    
    # Small hole
    hole_radius = 5
    hole_mask = (x - center_x)**2 + (y - center_y)**2 <= hole_radius**2
    
    # Combine to create mask with hole
    test_mask = main_mask & ~hole_mask
    
    # Test a few transforms
    result, params = apply_morphological_opening(test_image, 'ellipse', 3, test_mask)
    print(f"Morphological opening: {params}, output shape: {result.shape if result is not None else None}")
    
    result, params = apply_hole_filling(test_image, test_mask)
    print(f"Hole filling: {params}, output shape: {result.shape if result is not None else None}")
    
    result, params = apply_morphological_gradient(test_image, 'ellipse', 3, test_mask)
    print(f"Morphological gradient: {params}, output shape: {result.shape if result is not None else None}")
    
    print(f"Available morphological transforms: {list(MORPHOLOGY_TRANSFORMS.keys())}")