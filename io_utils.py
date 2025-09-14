"""
I/O utilities for image loading, saving, masking, and hashing operations.
"""

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

import cv2
import numpy as np
from PIL import Image


def load_image(image_path: str) -> np.ndarray:
    """
    Load an RGB image from file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        RGB image as uint8 numpy array with shape (H, W, 3)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded or is not RGB
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Use PIL for better format support
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array
            image = np.array(img, dtype=np.uint8)
            
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected RGB image, got shape: {image.shape}")
                
            return image
            
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


def load_mask(mask_path: str) -> np.ndarray:
    """
    Load a binary mask from file.
    
    Args:
        mask_path: Path to mask file (should be binary/grayscale)
        
    Returns:
        Binary mask as bool numpy array with shape (H, W)
        
    Raises:
        FileNotFoundError: If mask file doesn't exist
        ValueError: If mask cannot be loaded
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    try:
        # Load as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("Could not load mask with OpenCV")
        
        # Convert to binary (threshold at 128)
        mask = mask > 128
        
        return mask.astype(bool)
        
    except Exception as e:
        raise ValueError(f"Failed to load mask {mask_path}: {e}")


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save an image to file as PNG.
    
    Args:
        image: RGB image as uint8 numpy array
        output_path: Output file path
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to PIL and save
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Convert from [0,1] to [0,255]
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    pil_image = Image.fromarray(image, mode='RGB')
    pil_image.save(output_path, format='PNG', optimize=True)


def save_mask(mask: np.ndarray, output_path: str) -> None:
    """
    Save a binary mask to file as PNG.
    
    Args:
        mask: Binary mask as bool numpy array
        output_path: Output file path
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Save with PIL
    pil_mask = Image.fromarray(mask_uint8, mode='L')
    pil_mask.save(output_path, format='PNG', optimize=True)


def compute_image_hash(image: np.ndarray) -> str:
    """
    Compute SHA-256 hash of image data.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Hexadecimal hash string
    """
    # Ensure consistent data type for hashing
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    return hashlib.sha256(image.tobytes()).hexdigest()[:16]  # First 16 chars


def get_jpeg_size(image: np.ndarray, quality: int = 90) -> int:
    """
    Get JPEG compressed size in bytes (without saving to disk).
    
    Args:
        image: RGB image as uint8 numpy array
        quality: JPEG quality (1-100)
        
    Returns:
        Size in bytes
    """
    # Convert to BGR for OpenCV
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Encode as JPEG in memory
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode('.jpg', bgr_image, encode_param)
    
    return len(encoded.tobytes())


def create_grabcut_mask(image: np.ndarray, iterations: int = 5) -> Optional[np.ndarray]:
    """
    Create foreground mask using GrabCut algorithm.
    Uses automatic rectangle initialization (10% border).
    
    Args:
        image: RGB image as uint8 numpy array
        iterations: Number of GrabCut iterations
        
    Returns:
        Binary mask as bool array, or None if GrabCut fails
    """
    try:
        h, w = image.shape[:2]
        
        # Create initial rectangle (10% border)
        border = min(h, w) // 10
        rect = (border, border, w - 2*border, h - 2*border)
        
        # Initialize masks
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Convert to BGR for OpenCV
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Run GrabCut
        cv2.grabCut(bgr_image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        
        # Extract foreground (values 1 and 3)
        foreground_mask = np.where((mask == 1) | (mask == 3), True, False)
        
        return foreground_mask
        
    except Exception:
        return None


def create_threshold_mask(image: np.ndarray, method: str = 'otsu') -> Optional[np.ndarray]:
    """
    Create foreground mask using simple thresholding heuristics.
    
    Args:
        image: RGB image as uint8 numpy array
        method: Thresholding method ('otsu', 'mean')
        
    Returns:
        Binary mask as bool array, or None if thresholding fails
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if method == 'otsu':
            # Otsu's thresholding
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'mean':
            # Threshold at mean intensity
            threshold = np.mean(gray)
            _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        else:
            return None
        
        return mask > 128
        
    except Exception:
        return None


def get_foreground_mask(image: np.ndarray, provided_mask: Optional[np.ndarray] = None, 
                       use_advanced: bool = True, verbose: bool = False) -> Optional[np.ndarray]:
    """
    Get foreground mask using provided mask or automatic methods.
    
    Args:
        image: RGB image as uint8 numpy array
        provided_mask: Optional provided binary mask
        use_advanced: Whether to try advanced segmentation (SAM2/SAM)
        verbose: Enable verbose logging
        
    Returns:
        Binary mask as bool array, or None if no valid mask can be obtained
    """
    if provided_mask is not None:
        # Validate provided mask dimensions
        if provided_mask.shape[:2] != image.shape[:2]:
            if verbose:
                warnings.warn("Provided mask dimensions don't match image, trying to resize")
            try:
                provided_mask = cv2.resize(
                    provided_mask.astype(np.uint8), 
                    (image.shape[1], image.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                ) > 0
            except Exception:
                provided_mask = None
        
        if provided_mask is not None:
            return provided_mask
    
    # Try advanced segmentation first if enabled
    if use_advanced:
        try:
            from segmentation import get_advanced_foreground_mask
            mask = get_advanced_foreground_mask(image, provided_mask, use_advanced=True, verbose=verbose)
            if mask is not None:
                return mask
        except ImportError:
            if verbose:
                print("Advanced segmentation not available, using fallback methods")
        except Exception as e:
            if verbose:
                print(f"Advanced segmentation failed: {e}")
    
    # Try GrabCut
    mask = create_grabcut_mask(image)
    if mask is not None:
        return mask
    
    # Fall back to Otsu thresholding
    mask = create_threshold_mask(image, 'otsu')
    if mask is not None:
        return mask
    
    # Final fallback to mean thresholding
    return create_threshold_mask(image, 'mean')


def load_background_images(bg_bank_path: str) -> List[np.ndarray]:
    """
    Load all background images from a directory.
    
    Args:
        bg_bank_path: Path to directory containing background images
        
    Returns:
        List of RGB background images as uint8 numpy arrays
    """
    if not os.path.exists(bg_bank_path):
        return []
    
    backgrounds = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    for file_path in Path(bg_bank_path).glob('*'):
        if file_path.suffix.lower() in supported_extensions:
            try:
                bg_image = load_image(str(file_path))
                backgrounds.append(bg_image)
            except Exception:
                continue  # Skip problematic images
    
    return backgrounds


def resize_background_to_fit(background: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resize and crop background image to fit target shape.
    
    Args:
        background: Background image as uint8 numpy array
        target_shape: Target (height, width)
        
    Returns:
        Resized background image
    """
    target_h, target_w = target_shape
    bg_h, bg_w = background.shape[:2]
    
    # Calculate scaling to cover the target area
    scale_w = target_w / bg_w
    scale_h = target_h / bg_h
    scale = max(scale_w, scale_h)  # Scale to cover
    
    # Resize
    new_w = int(bg_w * scale)
    new_h = int(bg_h * scale)
    resized = cv2.resize(background, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Center crop to target size
    start_y = (new_h - target_h) // 2
    start_x = (new_w - target_w) // 2
    cropped = resized[start_y:start_y + target_h, start_x:start_x + target_w]
    
    return cropped


def validate_paths(input_path: str, mask_path: Optional[str] = None, bg_bank_path: Optional[str] = None) -> None:
    """
    Validate that required input paths exist.
    
    Args:
        input_path: Path to input image
        mask_path: Optional path to mask file
        bg_bank_path: Optional path to background bank directory
        
    Raises:
        FileNotFoundError: If required paths don't exist
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
    
    if mask_path and not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    if bg_bank_path and not os.path.exists(bg_bank_path):
        raise FileNotFoundError(f"Background bank directory not found: {bg_bank_path}")


def float_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert float image in [0,1] to uint8 in [0,255].
    
    Args:
        image: Float image array
        
    Returns:
        uint8 image array
    """
    if image.dtype == np.uint8:
        return image
    
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def uint8_to_float(image: np.ndarray) -> np.ndarray:
    """
    Convert uint8 image in [0,255] to float in [0,1].
    
    Args:
        image: uint8 image array
        
    Returns:
        float32 image array
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        return image.astype(np.float32)
    
    return image.astype(np.float32) / 255.0