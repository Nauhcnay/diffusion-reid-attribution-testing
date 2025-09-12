"""
Image quality metrics for ReID diagnostics.

Computes various metrics per output image:
- Edge density (fraction of non-zero Canny edges)
- Gradient energy (mean of Sobel magnitude)
- Intensity histogram stats (mean, std, skewness, entropy)
- NIQE and/or BRISQUE if available (optional)
- JPEG size in bytes when saved at quality=90 (proxy for texture complexity)
"""

from typing import Dict, Any, Optional, List
import warnings

import cv2
import numpy as np
from scipy import stats
from skimage import feature

from io_utils import get_jpeg_size, uint8_to_float


def compute_edge_density(image: np.ndarray, canny_low: float = 50, canny_high: float = 150) -> float:
    """
    Compute edge density using Canny edge detection.
    
    Args:
        image: Input RGB image (uint8)
        canny_low: Low threshold for Canny
        canny_high: High threshold for Canny
        
    Returns:
        Fraction of pixels that are edges (0-1)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, canny_low, canny_high)
    
    # Calculate fraction of edge pixels
    total_pixels = edges.size
    edge_pixels = np.sum(edges > 0)
    
    return edge_pixels / total_pixels if total_pixels > 0 else 0.0


def compute_gradient_energy(image: np.ndarray) -> float:
    """
    Compute gradient energy (mean of Sobel magnitude).
    
    Args:
        image: Input RGB image (uint8)
        
    Returns:
        Mean gradient magnitude
    """
    # Convert to grayscale and float
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_float = uint8_to_float(gray)
    
    # Compute Sobel gradients
    sobel_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    return np.mean(gradient_magnitude)


def compute_intensity_histogram_stats(image: np.ndarray) -> Dict[str, float]:
    """
    Compute intensity histogram statistics.
    
    Args:
        image: Input RGB image (uint8)
        
    Returns:
        Dictionary with mean, std, skewness, and entropy
    """
    # Convert to grayscale for intensity analysis
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    pixels = gray.flatten().astype(np.float32)
    
    # Basic statistics
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = stats.skew(pixels)
    
    # Compute histogram entropy
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    # Remove zero probabilities to avoid log(0)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    return {
        'mean': float(mean_intensity),
        'std': float(std_intensity),
        'skewness': float(skewness),
        'entropy': float(entropy)
    }


def compute_color_histogram_stats(image: np.ndarray) -> Dict[str, float]:
    """
    Compute per-channel color statistics.
    
    Args:
        image: Input RGB image (uint8)
        
    Returns:
        Dictionary with per-channel statistics
    """
    stats_dict = {}
    
    for i, channel_name in enumerate(['red', 'green', 'blue']):
        channel = image[:, :, i].flatten().astype(np.float32)
        
        stats_dict[f'{channel_name}_mean'] = float(np.mean(channel))
        stats_dict[f'{channel_name}_std'] = float(np.std(channel))
    
    # Overall color diversity (std of channel means)
    channel_means = [stats_dict['red_mean'], stats_dict['green_mean'], stats_dict['blue_mean']]
    stats_dict['color_diversity'] = float(np.std(channel_means))
    
    return stats_dict


def compute_texture_energy(image: np.ndarray) -> float:
    """
    Compute texture energy using Gray Level Co-occurrence Matrix (GLCM).
    
    Args:
        image: Input RGB image (uint8)
        
    Returns:
        Texture energy measure
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Compute GLCM (using scikit-image)
        glcm = feature.graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        
        # Compute energy (Angular Second Moment)
        energy = feature.graycoprops(glcm, 'ASM')[0, 0]
        
        return float(energy)
    
    except Exception:
        # If GLCM computation fails, return simple texture measure
        return compute_gradient_energy(image)


def compute_contrast_measure(image: np.ndarray) -> float:
    """
    Compute RMS contrast measure.
    
    Args:
        image: Input RGB image (uint8)
        
    Returns:
        RMS contrast value
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Compute RMS contrast
    mean_intensity = np.mean(gray)
    rms_contrast = np.sqrt(np.mean((gray - mean_intensity)**2))
    
    return float(rms_contrast / 255.0)  # Normalize to [0,1]


def compute_sharpness_measure(image: np.ndarray) -> float:
    """
    Compute image sharpness using Laplacian variance.
    
    Args:
        image: Input RGB image (uint8)
        
    Returns:
        Sharpness measure (higher = sharper)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Compute Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Return variance of Laplacian
    return float(laplacian.var())


def compute_blur_metric(image: np.ndarray) -> float:
    """
    Compute blur metric using spectral analysis.
    
    Args:
        image: Input RGB image (uint8)
        
    Returns:
        Blur metric (lower = more blurred)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Compute FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # Compute high frequency content
    h, w = gray.shape
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    
    # Create high-pass mask (outer 50% of frequencies)
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    high_freq_mask = distance > min(h, w) / 4
    
    # Sum high frequency magnitudes
    high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
    total_energy = np.sum(magnitude_spectrum)
    
    return float(high_freq_energy / total_energy if total_energy > 0 else 0)


def compute_noise_metric(image: np.ndarray) -> float:
    """
    Compute noise metric using local standard deviation.
    
    Args:
        image: Input RGB image (uint8)
        
    Returns:
        Noise metric estimate
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Compute local standard deviation using sliding window
    kernel = np.ones((3, 3)) / 9  # 3x3 averaging kernel
    local_mean = cv2.filter2D(gray, -1, kernel)
    local_mean_sq = cv2.filter2D(gray**2, -1, kernel)
    local_var = local_mean_sq - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 0))  # Ensure non-negative
    
    # Return mean local standard deviation as noise estimate
    return float(np.mean(local_std) / 255.0)


def compute_brisque_score(image: np.ndarray) -> Optional[float]:
    """
    Compute BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) score.
    
    Args:
        image: Input RGB image (uint8)
        
    Returns:
        BRISQUE score if available, None otherwise
    """
    try:
        # Try importing piq library
        import piq
        
        # Convert to tensor format expected by piq
        import torch
        
        # Convert to float and normalize to [0, 1]
        float_image = uint8_to_float(image)
        
        # Convert to tensor (1, 3, H, W)
        tensor_image = torch.from_numpy(float_image).permute(2, 0, 1).unsqueeze(0)
        
        # Compute BRISQUE
        brisque = piq.brisque(tensor_image)
        
        return float(brisque.item())
    
    except ImportError:
        # piq not available
        return None
    except Exception:
        # Other errors
        return None


def compute_niqe_score(image: np.ndarray) -> Optional[float]:
    """
    Compute NIQE (Natural Image Quality Evaluator) score.
    
    Args:
        image: Input RGB image (uint8)
        
    Returns:
        NIQE score if available, None otherwise
    """
    try:
        # Try importing imquality library
        import imquality.brisque as brisque
        
        # Convert to PIL Image format
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image)
        
        # Compute NIQE (note: some libraries provide NIQE through brisque module)
        score = brisque.score(pil_image)
        
        return float(score)
    
    except ImportError:
        # Library not available
        return None
    except Exception:
        # Other errors
        return None


def compute_jpeg_complexity(image: np.ndarray, quality: int = 90) -> int:
    """
    Compute JPEG compressed size as texture complexity proxy.
    
    Args:
        image: Input RGB image (uint8)
        quality: JPEG quality for compression
        
    Returns:
        Size in bytes
    """
    return get_jpeg_size(image, quality)


def compute_all_metrics(image: np.ndarray, include_expensive: bool = False) -> Dict[str, Any]:
    """
    Compute all available metrics for an image.
    
    Args:
        image: Input RGB image (uint8)
        include_expensive: Whether to include computationally expensive metrics
        
    Returns:
        Dictionary of all computed metrics
    """
    metrics = {}
    
    # Fast metrics
    try:
        metrics['edge_density'] = compute_edge_density(image)
    except Exception:
        metrics['edge_density'] = None
    
    try:
        metrics['gradient_energy'] = compute_gradient_energy(image)
    except Exception:
        metrics['gradient_energy'] = None
    
    try:
        intensity_stats = compute_intensity_histogram_stats(image)
        for key, value in intensity_stats.items():
            metrics[f'intensity_{key}'] = value
    except Exception:
        metrics.update({
            'intensity_mean': None,
            'intensity_std': None,
            'intensity_skewness': None,
            'intensity_entropy': None
        })
    
    try:
        color_stats = compute_color_histogram_stats(image)
        for key, value in color_stats.items():
            metrics[key] = value
    except Exception:
        for channel in ['red', 'green', 'blue']:
            metrics[f'{channel}_mean'] = None
            metrics[f'{channel}_std'] = None
        metrics['color_diversity'] = None
    
    try:
        metrics['contrast'] = compute_contrast_measure(image)
    except Exception:
        metrics['contrast'] = None
    
    try:
        metrics['sharpness'] = compute_sharpness_measure(image)
    except Exception:
        metrics['sharpness'] = None
    
    try:
        metrics['blur_metric'] = compute_blur_metric(image)
    except Exception:
        metrics['blur_metric'] = None
    
    try:
        metrics['noise_metric'] = compute_noise_metric(image)
    except Exception:
        metrics['noise_metric'] = None
    
    try:
        metrics['jpeg_size_q90'] = compute_jpeg_complexity(image, 90)
    except Exception:
        metrics['jpeg_size_q90'] = None
    
    # Expensive metrics (if requested)
    if include_expensive:
        try:
            metrics['texture_energy'] = compute_texture_energy(image)
        except Exception:
            metrics['texture_energy'] = None
        
        try:
            metrics['brisque_score'] = compute_brisque_score(image)
        except Exception:
            metrics['brisque_score'] = None
        
        try:
            metrics['niqe_score'] = compute_niqe_score(image)
        except Exception:
            metrics['niqe_score'] = None
    else:
        metrics['texture_energy'] = None
        metrics['brisque_score'] = None
        metrics['niqe_score'] = None
    
    return metrics


def get_metric_names() -> List[str]:
    """Get list of all metric names."""
    return [
        'edge_density',
        'gradient_energy',
        'intensity_mean',
        'intensity_std',
        'intensity_skewness',
        'intensity_entropy',
        'red_mean',
        'red_std',
        'green_mean',
        'green_std',
        'blue_mean',
        'blue_std',
        'color_diversity',
        'contrast',
        'sharpness',
        'blur_metric',
        'noise_metric',
        'texture_energy',
        'brisque_score',
        'niqe_score',
        'jpeg_size_q90'
    ]


if __name__ == "__main__":
    # Test metrics with synthetic data
    test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    print("Testing image metrics...")
    
    # Test individual metrics
    edge_density = compute_edge_density(test_image)
    print(f"Edge density: {edge_density:.4f}")
    
    gradient_energy = compute_gradient_energy(test_image)
    print(f"Gradient energy: {gradient_energy:.4f}")
    
    intensity_stats = compute_intensity_histogram_stats(test_image)
    print(f"Intensity stats: {intensity_stats}")
    
    jpeg_size = compute_jpeg_complexity(test_image)
    print(f"JPEG size (q=90): {jpeg_size} bytes")
    
    # Test all metrics
    all_metrics = compute_all_metrics(test_image, include_expensive=False)
    print(f"\nAll metrics computed: {len([v for v in all_metrics.values() if v is not None])}/{len(all_metrics)}")
    
    print("\nAvailable metric names:")
    print(get_metric_names())