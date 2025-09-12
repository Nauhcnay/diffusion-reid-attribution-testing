"""
Visualization utilities for ReID diagnostics.
Creates contact sheet mosaics showing original + all variants with captions.
"""

import os
import math
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont


def resize_image_to_thumbnail(image: np.ndarray, target_size: int, maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image to thumbnail size.
    
    Args:
        image: Input RGB image (uint8)
        target_size: Target thumbnail size (max dimension)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized thumbnail image
    """
    h, w = image.shape[:2]
    
    if maintain_aspect:
        # Calculate scale to fit within target_size x target_size
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create square thumbnail with padding
        thumbnail = np.full((target_size, target_size, 3), 255, dtype=np.uint8)  # White background
        
        # Center the resized image
        start_y = (target_size - new_h) // 2
        start_x = (target_size - new_w) // 2
        thumbnail[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        
    else:
        # Simply resize to target_size x target_size (may distort aspect ratio)
        thumbnail = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    return thumbnail


def create_caption(transform_name: str, param_key: str, font_size: int = 10) -> np.ndarray:
    """
    Create caption image for a transform.
    
    Args:
        transform_name: Name of the transform
        param_key: Parameter key string
        font_size: Font size for caption
        
    Returns:
        Caption image as RGB numpy array
    """
    # Create caption text
    if param_key:
        caption_text = f"{transform_name}\n{param_key}"
    else:
        caption_text = transform_name
    
    try:
        # Try to use PIL for better text rendering
        # Estimate caption size
        lines = caption_text.split('\n')
        max_line_length = max(len(line) for line in lines)
        
        # Rough estimate of image size needed
        char_width = font_size * 0.6  # Approximate character width
        line_height = font_size * 1.2  # Line height
        
        caption_width = int(max_line_length * char_width) + 10
        caption_height = int(len(lines) * line_height) + 10
        
        # Create PIL image for text
        pil_img = Image.new('RGB', (caption_width, caption_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(pil_img)
        
        # Try to load a font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # Draw text
        y_offset = 5
        for line in lines:
            draw.text((5, y_offset), line, fill=(0, 0, 0), font=font)
            y_offset += int(line_height)
        
        # Convert to numpy array
        caption_array = np.array(pil_img)
        
    except Exception:
        # Fallback: create simple text using OpenCV
        lines = caption_text.split('\n')
        
        # Estimate size needed
        line_height = 20
        caption_height = len(lines) * line_height + 10
        caption_width = 200  # Fixed width
        
        # Create white background
        caption_array = np.full((caption_height, caption_width, 3), 255, dtype=np.uint8)
        
        # Draw text using OpenCV
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        color = (0, 0, 0)  # Black text
        
        for i, line in enumerate(lines):
            y_pos = (i + 1) * line_height
            cv2.putText(caption_array, line, (5, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return caption_array


def create_contact_sheet(images_info: List[Dict[str, Any]], 
                        output_path: str,
                        thumbnail_size: int = 128,
                        grid_cols: int = 8,
                        max_images_per_sheet: int = 64,
                        font_size: int = 10,
                        caption_height: int = 20,
                        padding: int = 5,
                        background_color: Tuple[int, int, int] = (240, 240, 240)) -> List[str]:
    """
    Create contact sheet(s) from a list of images.
    
    Args:
        images_info: List of dictionaries with image info (path, transform_name, param_key, etc.)
        output_path: Base output path (will append _01.png, _02.png, etc. for multiple sheets)
        thumbnail_size: Size of each thumbnail
        grid_cols: Number of columns in grid
        max_images_per_sheet: Maximum images per sheet
        font_size: Font size for captions
        caption_height: Height reserved for captions
        padding: Padding between thumbnails
        background_color: Background color of contact sheet
        
    Returns:
        List of created contact sheet file paths
    """
    if not images_info:
        return []
    
    # Calculate grid dimensions
    total_images = len(images_info)
    images_per_sheet = min(max_images_per_sheet, total_images)
    num_sheets = math.ceil(total_images / max_images_per_sheet)
    
    created_files = []
    
    for sheet_idx in range(num_sheets):
        start_idx = sheet_idx * max_images_per_sheet
        end_idx = min(start_idx + max_images_per_sheet, total_images)
        sheet_images = images_info[start_idx:end_idx]
        
        # Calculate grid size for this sheet
        images_in_sheet = len(sheet_images)
        grid_rows = math.ceil(images_in_sheet / grid_cols)
        
        # Calculate contact sheet dimensions
        cell_width = thumbnail_size + padding * 2
        cell_height = thumbnail_size + caption_height + padding * 2
        
        sheet_width = grid_cols * cell_width
        sheet_height = grid_rows * cell_height
        
        # Create contact sheet canvas
        contact_sheet = np.full((sheet_height, sheet_width, 3), background_color, dtype=np.uint8)
        
        # Place thumbnails and captions
        for i, img_info in enumerate(sheet_images):
            row = i // grid_cols
            col = i % grid_cols
            
            # Calculate position
            x_start = col * cell_width + padding
            y_start = row * cell_height + padding
            
            try:
                # Load and resize image
                if 'image_array' in img_info:
                    # Image provided as array
                    image = img_info['image_array']
                else:
                    # Load from path
                    image_path = img_info['image_path']
                    if not os.path.exists(image_path):
                        continue
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Create thumbnail
                thumbnail = resize_image_to_thumbnail(image, thumbnail_size)
                
                # Place thumbnail
                thumb_h, thumb_w = thumbnail.shape[:2]
                x_thumb = x_start + (thumbnail_size - thumb_w) // 2
                y_thumb = y_start + (thumbnail_size - thumb_h) // 2
                
                contact_sheet[y_thumb:y_thumb + thumb_h, x_thumb:x_thumb + thumb_w] = thumbnail
                
                # Create and place caption
                transform_name = img_info.get('transform_name', 'unknown')
                param_key = img_info.get('param_key', '')
                
                caption = create_caption(transform_name, param_key, font_size)
                
                # Resize caption to fit
                caption_w_available = thumbnail_size
                if caption.shape[1] > caption_w_available:
                    caption_scale = caption_w_available / caption.shape[1]
                    new_caption_h = int(caption.shape[0] * caption_scale)
                    caption = cv2.resize(caption, (caption_w_available, new_caption_h), interpolation=cv2.INTER_AREA)
                
                # Place caption below thumbnail
                y_caption = y_start + thumbnail_size + 2
                caption_h, caption_w = caption.shape[:2]
                
                if y_caption + caption_h <= contact_sheet.shape[0] and caption_w <= contact_sheet.shape[1] - x_start:
                    x_caption = x_start + (thumbnail_size - caption_w) // 2
                    contact_sheet[y_caption:y_caption + caption_h, x_caption:x_caption + caption_w] = caption
                
            except Exception as e:
                # Skip problematic images
                print(f"Warning: Could not process image {img_info.get('image_path', 'unknown')}: {e}")
                continue
        
        # Save contact sheet
        if num_sheets == 1:
            sheet_path = output_path
        else:
            # Multi-sheet naming
            base_path = Path(output_path)
            sheet_path = str(base_path.with_suffix('')) + f"_{sheet_idx + 1:02d}" + base_path.suffix
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(sheet_path), exist_ok=True)
        
        # Save using PIL for better quality
        pil_image = Image.fromarray(contact_sheet)
        pil_image.save(sheet_path, format='PNG', optimize=True)
        
        created_files.append(sheet_path)
    
    return created_files


def create_comparison_grid(original_image: np.ndarray, 
                          transformed_images: List[Tuple[np.ndarray, str, str]],
                          output_path: str,
                          thumbnail_size: int = 128,
                          max_cols: int = 6) -> str:
    """
    Create a comparison grid with original image and transforms.
    
    Args:
        original_image: Original RGB image (uint8)
        transformed_images: List of (image, transform_name, param_key) tuples
        output_path: Output file path
        thumbnail_size: Size of each thumbnail
        max_cols: Maximum columns in grid
        
    Returns:
        Path to created comparison grid image
    """
    # Prepare all images (original + transforms)
    all_images = [
        {
            'image_array': original_image,
            'transform_name': 'Original',
            'param_key': ''
        }
    ]
    
    for img, transform_name, param_key in transformed_images:
        all_images.append({
            'image_array': img,
            'transform_name': transform_name,
            'param_key': param_key
        })
    
    # Create contact sheet
    created_files = create_contact_sheet(
        all_images,
        output_path,
        thumbnail_size=thumbnail_size,
        grid_cols=max_cols,
        max_images_per_sheet=len(all_images),
        font_size=8,
        caption_height=30,
        padding=5
    )
    
    return created_files[0] if created_files else output_path


def create_metrics_visualization(metrics_data: List[Dict[str, Any]], 
                               output_path: str,
                               metric_names: List[str],
                               figsize: Tuple[int, int] = (12, 8)) -> str:
    """
    Create visualization of metrics across transforms.
    
    Args:
        metrics_data: List of dictionaries with transform info and metrics
        output_path: Output file path for plot
        metric_names: Names of metrics to visualize
        figsize: Figure size for matplotlib
        
    Returns:
        Path to created visualization
    """
    if not metrics_data or not metric_names:
        return output_path
    
    try:
        # Filter valid metric names (those that have non-None values)
        valid_metrics = []
        for metric in metric_names:
            if any(item.get('metrics', {}).get(metric) is not None for item in metrics_data):
                valid_metrics.append(metric)
        
        if not valid_metrics:
            return output_path
        
        # Create subplots
        n_metrics = len(valid_metrics)
        n_cols = min(3, n_metrics)
        n_rows = math.ceil(n_metrics / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for i, metric_name in enumerate(valid_metrics):
            ax = axes[i]
            
            # Extract data for this metric
            transforms = []
            values = []
            
            for item in metrics_data:
                metric_value = item.get('metrics', {}).get(metric_name)
                if metric_value is not None:
                    transforms.append(item.get('transform_name', 'unknown'))
                    values.append(metric_value)
            
            if values:
                # Create bar plot
                ax.bar(range(len(values)), values)
                ax.set_title(metric_name.replace('_', ' ').title())
                ax.set_xticks(range(len(transforms)))
                ax.set_xticklabels(transforms, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save plot
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        print(f"Warning: Could not create metrics visualization: {e}")
    
    return output_path


def create_summary_report(diagnostics_results: Dict[str, Any], 
                         output_path: str) -> str:
    """
    Create a summary report with key statistics.
    
    Args:
        diagnostics_results: Dictionary with all diagnostics results
        output_path: Output file path for report
        
    Returns:
        Path to created report
    """
    try:
        with open(output_path, 'w') as f:
            f.write("ReID Diagnostics Summary Report\n")
            f.write("=" * 40 + "\n\n")
            
            # Basic information
            f.write(f"Input image: {diagnostics_results.get('input_path', 'unknown')}\n")
            f.write(f"Output directory: {diagnostics_results.get('output_dir', 'unknown')}\n")
            f.write(f"Processing timestamp: {diagnostics_results.get('timestamp', 'unknown')}\n\n")
            
            # Transform statistics
            transform_results = diagnostics_results.get('transform_results', [])
            f.write(f"Total transforms applied: {len(transform_results)}\n")
            
            # Count by family
            family_counts = {}
            for result in transform_results:
                family = result.get('family', 'unknown')
                family_counts[family] = family_counts.get(family, 0) + 1
            
            f.write("\nTransforms by family:\n")
            for family, count in sorted(family_counts.items()):
                f.write(f"  {family}: {count}\n")
            
            # Failed transforms
            failed_transforms = [r for r in transform_results if r.get('error')]
            if failed_transforms:
                f.write(f"\nFailed transforms: {len(failed_transforms)}\n")
                for result in failed_transforms[:5]:  # Show first 5
                    f.write(f"  {result.get('transform_name', 'unknown')}: {result.get('error', 'unknown error')}\n")
            
            # Metric summary (if available)
            successful_results = [r for r in transform_results if not r.get('error') and r.get('metrics')]
            if successful_results:
                f.write(f"\nMetrics computed for {len(successful_results)} successful transforms\n")
                
                # Find common metrics
                all_metrics = set()
                for result in successful_results:
                    all_metrics.update(result.get('metrics', {}).keys())
                
                # Show statistics for key metrics
                key_metrics = ['edge_density', 'gradient_energy', 'contrast', 'sharpness', 'jpeg_size_q90']
                for metric in key_metrics:
                    if metric in all_metrics:
                        values = [r['metrics'][metric] for r in successful_results 
                                if r.get('metrics', {}).get(metric) is not None]
                        if values:
                            f.write(f"  {metric}: min={min(values):.4f}, max={max(values):.4f}, mean={np.mean(values):.4f}\n")
            
            f.write(f"\nReport generated successfully.\n")
    
    except Exception as e:
        with open(output_path, 'w') as f:
            f.write(f"Error generating report: {e}\n")
    
    return output_path


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization functions...")
    
    # Create test images
    test_images = []
    for i in range(5):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        test_images.append({
            'image_array': img,
            'transform_name': f'Transform_{i}',
            'param_key': f'param={i}'
        })
    
    # Test contact sheet creation
    output_path = "test_contact_sheet.png"
    created_files = create_contact_sheet(test_images, output_path, thumbnail_size=64, grid_cols=3)
    print(f"Created contact sheets: {created_files}")
    
    # Test metrics visualization
    test_metrics = [
        {'transform_name': 'original', 'metrics': {'edge_density': 0.1, 'contrast': 0.5}},
        {'transform_name': 'blur', 'metrics': {'edge_density': 0.05, 'contrast': 0.3}},
        {'transform_name': 'sharpen', 'metrics': {'edge_density': 0.2, 'contrast': 0.8}}
    ]
    
    metrics_path = "test_metrics.png"
    create_metrics_visualization(test_metrics, metrics_path, ['edge_density', 'contrast'])
    print(f"Created metrics visualization: {metrics_path}")
    
    print("Visualization tests completed.")