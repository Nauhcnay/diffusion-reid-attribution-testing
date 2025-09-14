#!/usr/bin/env python3
"""
Setup script for advanced segmentation capabilities.
Downloads and installs SAM2/SAM models and dependencies.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import urllib.request
import tempfile


def check_torch():
    """Check if PyTorch is installed."""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} is installed")
        if torch.cuda.is_available():
            print(f"✓ CUDA is available: {torch.version.cuda}")
        else:
            print("⚠ CUDA not available, will use CPU")
        return True
    except ImportError:
        print("✗ PyTorch is not installed")
        return False


def install_torch():
    """Install PyTorch."""
    print("Installing PyTorch...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
        print("✓ PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install PyTorch: {e}")
        return False


def install_sam2():
    """Install SAM2 from source."""
    print("Installing SAM2...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/facebookresearch/segment-anything-2.git"
        ])
        print("✓ SAM2 installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install SAM2: {e}")
        return False


def install_sam():
    """Install original SAM from source."""
    print("Installing original SAM...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/facebookresearch/segment-anything.git"
        ])
        print("✓ SAM installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install SAM: {e}")
        return False


def download_model(url: str, filename: str, model_dir: Path) -> bool:
    """Download a model checkpoint."""
    model_path = model_dir / filename
    
    if model_path.exists():
        print(f"✓ {filename} already exists")
        return True
    
    print(f"Downloading {filename}...")
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with urllib.request.urlopen(url) as response, open(model_path, 'wb') as f:
            data = response.read()
            f.write(data)
        
        print(f"✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False


def setup_sam2_models(model_dir: Path = Path("./checkpoints")):
    """Download SAM2 model checkpoints."""
    print("Setting up SAM2 models...")
    
    models = [
        ("sam2_hiera_tiny.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"),
        ("sam2_hiera_small.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"),
        ("sam2_hiera_base_plus.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"),
        ("sam2_hiera_large.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")
    ]
    
    success_count = 0
    for filename, url in models:
        if download_model(url, filename, model_dir):
            success_count += 1
    
    print(f"Downloaded {success_count}/{len(models)} SAM2 models")
    return success_count > 0


def setup_sam_models(model_dir: Path = Path("./checkpoints")):
    """Download original SAM model checkpoints."""
    print("Setting up SAM models...")
    
    models = [
        ("sam_vit_b_01ec64.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
        ("sam_vit_l_0b3195.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"),
        ("sam_vit_h_4b8939.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
    ]
    
    success_count = 0
    for filename, url in models:
        if download_model(url, filename, model_dir):
            success_count += 1
    
    print(f"Downloaded {success_count}/{len(models)} SAM models")
    return success_count > 0


def test_installation():
    """Test if advanced segmentation is working."""
    print("Testing advanced segmentation...")
    
    try:
        from segmentation import AdvancedSegmentationEngine
        import numpy as np
        
        # Create test image
        test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # Test engine
        engine = AdvancedSegmentationEngine(verbose=True)
        info = engine.get_model_info()
        
        print("Model info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Try segmentation
        mask = engine.segment_person(test_image)
        
        if mask is not None:
            print("✓ Advanced segmentation is working!")
            return True
        else:
            print("⚠ Segmentation returned no mask (may be normal for test data)")
            return True
            
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Setup advanced segmentation for ReID diagnostics")
    parser.add_argument("--models-only", action="store_true", help="Only download models, skip package installation")
    parser.add_argument("--sam2-only", action="store_true", help="Only setup SAM2")
    parser.add_argument("--sam-only", action="store_true", help="Only setup original SAM")
    parser.add_argument("--model-dir", default="./checkpoints", help="Directory to store model checkpoints")
    parser.add_argument("--skip-test", action="store_true", help="Skip testing installation")
    
    args = parser.parse_args()
    
    print("ReID Diagnostics - Advanced Segmentation Setup")
    print("=" * 50)
    
    model_dir = Path(args.model_dir)
    
    success = True
    
    if not args.models_only:
        # Install packages
        if not check_torch():
            if not install_torch():
                success = False
        
        if success and not args.sam_only:
            if not install_sam2():
                print("⚠ SAM2 installation failed, continuing with SAM...")
        
        if success and not args.sam2_only:
            if not install_sam():
                print("⚠ SAM installation failed")
    
    # Download models
    if success:
        if not args.sam_only:
            setup_sam2_models(model_dir)
        
        if not args.sam2_only:
            setup_sam_models(model_dir)
    
    # Test installation
    if success and not args.skip_test:
        test_installation()
    
    print("\nSetup completed!")
    print(f"Models are stored in: {model_dir.absolute()}")
    print("\nTo use advanced segmentation, run:")
    print("python reid_diagnostics.py --input image.jpg --outdir results/ --segmentation sam2")


if __name__ == "__main__":
    main()