#!/usr/bin/env python3
"""
Test script for the complete ReID diagnostics system.
Creates synthetic test data and runs the full pipeline.
"""

import os
import tempfile
import numpy as np
from pathlib import Path

from io_utils import save_image
from config import save_example_config


def create_synthetic_person_image(width: int = 128, height: int = 256, seed: int = 42) -> np.ndarray:
    """
    Create a synthetic person-like image for testing.
    
    Args:
        width: Image width
        height: Image height  
        seed: Random seed
        
    Returns:
        Synthetic RGB image
    """
    np.random.seed(seed)
    
    # Create base image with background
    image = np.full((height, width, 3), 120, dtype=np.uint8)  # Gray background
    
    # Add some background texture
    noise = np.random.randint(-20, 21, (height, width, 3))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # Create simple person silhouette (ellipse)
    y_center, x_center = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    
    # Body ellipse
    body_mask = ((x - x_center) / (width * 0.25))**2 + ((y - y_center) / (height * 0.4))**2 <= 1
    
    # Head circle
    head_y = int(height * 0.2)
    head_mask = ((x - x_center) / (width * 0.15))**2 + ((y - head_y) / (height * 0.1))**2 <= 1
    
    # Combine person mask
    person_mask = body_mask | head_mask
    
    # Fill person area with different colors/textures
    person_color = [80, 60, 120]  # Darker clothing color
    head_color = [180, 140, 120]  # Skin-like color
    
    # Apply colors
    image[body_mask] = person_color
    image[head_mask] = head_color
    
    # Add some texture to clothing
    clothing_texture = np.random.randint(-10, 11, (height, width, 3))
    clothing_area = body_mask & ~head_mask
    image[clothing_area] = np.clip(image[clothing_area] + clothing_texture[clothing_area], 0, 255)
    
    return image


def create_test_mask(width: int = 128, height: int = 256) -> np.ndarray:
    """Create a simple binary mask for the synthetic person."""
    y_center, x_center = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    
    # Body ellipse
    body_mask = ((x - x_center) / (width * 0.25))**2 + ((y - y_center) / (height * 0.4))**2 <= 1
    
    # Head circle
    head_y = int(height * 0.2)
    head_mask = ((x - x_center) / (width * 0.15))**2 + ((y - head_y) / (height * 0.1))**2 <= 1
    
    return body_mask | head_mask


def create_test_background_images(num_backgrounds: int = 3, width: int = 128, height: int = 256) -> list:
    """Create some synthetic background images."""
    backgrounds = []
    
    for i in range(num_backgrounds):
        np.random.seed(100 + i)  # Different seed for each background
        
        if i == 0:
            # Gradient background
            bg = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                intensity = int(50 + 150 * y / height)
                bg[y, :] = [intensity, intensity - 20, intensity + 10]
        elif i == 1:
            # Textured background
            bg = np.random.randint(40, 100, (height, width, 3)).astype(np.uint8)
        else:
            # Solid color background
            color = [200, 180, 160]  # Light brown
            bg = np.full((height, width, 3), color, dtype=np.uint8)
            # Add slight texture
            noise = np.random.randint(-10, 11, (height, width, 3))
            bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
        
        backgrounds.append(bg)
    
    return backgrounds


def run_system_test(verbose: bool = True):
    """Run complete system test."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        if verbose:
            print("Creating test data...")
        
        # Create synthetic test image
        test_image = create_synthetic_person_image()
        test_image_path = tmpdir / "test_person.png"
        save_image(test_image, str(test_image_path))
        
        # Create test mask
        test_mask = create_test_mask()
        test_mask_path = tmpdir / "test_mask.png"
        mask_uint8 = (test_mask * 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(mask_uint8, mode='L').save(test_mask_path)
        
        # Create background images
        bg_dir = tmpdir / "backgrounds"
        bg_dir.mkdir()
        backgrounds = create_test_background_images()
        for i, bg in enumerate(backgrounds):
            bg_path = bg_dir / f"background_{i}.png"
            save_image(bg, str(bg_path))
        
        # Create test configuration (limited transforms for quick testing)
        config_path = tmpdir / "test_config.yaml"
        save_example_config(str(config_path))
        
        # Output directory
        output_dir = tmpdir / "diagnostics_output"
        
        if verbose:
            print(f"Test image: {test_image_path}")
            print(f"Test mask: {test_mask_path}")
            print(f"Backgrounds: {bg_dir}")
            print(f"Config: {config_path}")
            print(f"Output: {output_dir}")
            print("\nRunning diagnostics...")
        
        # Run the CLI tool
        import subprocess
        import sys
        
        cmd = [
            sys.executable, "reid_diagnostics.py",
            "--input", str(test_image_path),
            "--outdir", str(output_dir),
            "--mask", str(test_mask_path),
            "--bg_bank", str(bg_dir),
            "--config", str(config_path),
            "--seed", "42"
        ]
        
        if verbose:
            cmd.append("--verbose")
        
        try:
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  cwd=Path(__file__).parent,
                                  timeout=300)  # 5 minute timeout
            
            if verbose:
                print("STDOUT:")
                print(result.stdout)
                
                if result.stderr:
                    print("\nSTDERR:")
                    print(result.stderr)
            
            if result.returncode != 0:
                print(f"Command failed with return code {result.returncode}")
                return False
            
            # Check outputs
            if not output_dir.exists():
                print("Output directory not created")
                return False
            
            # Check for expected files
            expected_files = [
                "diagnostics_index.csv",
                "summary_report.txt",
                "images/original.png"
            ]
            
            missing_files = []
            for file_path in expected_files:
                if not (output_dir / file_path).exists():
                    missing_files.append(file_path)
            
            if missing_files:
                print(f"Missing expected files: {missing_files}")
                return False
            
            # List all created files
            all_files = list(output_dir.rglob("*"))
            image_files = [f for f in all_files if f.suffix.lower() in ['.png', '.jpg']]
            
            if verbose:
                print(f"\nTest completed successfully!")
                print(f"Created {len(all_files)} total files")
                print(f"Created {len(image_files)} image files")
                print(f"Output directory: {output_dir}")
                
                # Show some key files
                csv_file = output_dir / "diagnostics_index.csv"
                if csv_file.exists():
                    import pandas as pd
                    try:
                        df = pd.read_csv(csv_file)
                        print(f"CSV index contains {len(df)} entries")
                        successful = len(df[df['error'].isna()])
                        failed = len(df[df['error'].notna()])
                        print(f"Transforms: {successful} successful, {failed} failed")
                    except Exception as e:
                        print(f"Could not read CSV: {e}")
            
            return True
            
        except subprocess.TimeoutExpired:
            print("Test timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"Error running test: {e}")
            return False


if __name__ == "__main__":
    import sys
    
    print("Starting ReID Diagnostics System Test")
    print("=" * 50)
    
    success = run_system_test(verbose=True)
    
    if success:
        print("\n✓ System test PASSED")
        sys.exit(0)
    else:
        print("\n✗ System test FAILED") 
        sys.exit(1)