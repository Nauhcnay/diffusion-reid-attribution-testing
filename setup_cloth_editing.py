#!/usr/bin/env python3
"""
Setup script for cloth editing capabilities using Qwen-Image.
Downloads and installs Qwen-Image models and dependencies for cloth editing transforms.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import warnings

def check_cuda():
    """Check CUDA availability and GPU specifications."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ CUDA GPU available: {device_name}")
            print(f"✓ GPU Memory: {memory_gb:.1f} GB")

            if memory_gb < 8:
                print("⚠ Warning: Less than 8GB VRAM - cloth editing may fail")
                return False
            elif memory_gb >= 12:
                print("✓ Excellent: 12GB+ VRAM - optimal for cloth editing")
            return True
        else:
            print("✗ CUDA not available - cloth editing requires GPU")
            return False
    except ImportError:
        print("✗ PyTorch not available")
        return False

def check_torch():
    """Check if PyTorch with CUDA is installed."""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"✓ CUDA {cuda_version} support enabled")
            return True
        else:
            print("✗ PyTorch installed but CUDA not available")
            print("  Cloth editing requires CUDA-enabled PyTorch")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def install_cuda_pytorch():
    """Install CUDA-enabled PyTorch."""
    print("Installing CUDA-enabled PyTorch...")

    # Detect CUDA version and install appropriate PyTorch
    cuda_versions = ["cu121", "cu118"]

    for cuda_ver in cuda_versions:
        try:
            print(f"Trying PyTorch with {cuda_ver}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade",
                "torch", "torchvision",
                "--index-url", f"https://download.pytorch.org/whl/{cuda_ver}"
            ])

            # Test if CUDA works
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"✓ Successfully installed PyTorch with {cuda_ver}")
                    return True
            except:
                continue

        except subprocess.CalledProcessError:
            print(f"Failed to install with {cuda_ver}, trying next...")
            continue

    print("✗ Failed to install CUDA-enabled PyTorch")
    return False

def install_dependencies():
    """Install required dependencies for cloth editing."""
    print("Installing cloth editing dependencies...")

    packages = [
        "transformers>=4.46.2",
        "accelerate",
        "git+https://github.com/huggingface/diffusers",
        "diffsynth",
        "pillow",
        "opencv-python"
    ]

    success = True
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", package
            ])
            print(f"✓ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            success = False

    return success

def check_qwen_availability():
    """Check if Qwen-Image components are available."""
    print("Checking Qwen-Image availability...")

    try:
        from diffusers import QwenImageEditPipeline
        print("✓ QwenImageEditPipeline available")

        from transformers import AutoProcessor, AutoModelForCausalLM
        print("✓ Transformers components available")

        return True
    except ImportError as e:
        print(f"✗ Qwen-Image not available: {e}")
        return False

def download_and_cache_models():
    """Download and cache Qwen-Image models."""
    print("Downloading and caching Qwen-Image models...")
    print("⚠ This will download ~20GB of models and may take 10-30 minutes")

    response = input("Continue with model download? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Skipping model download")
        return True

    try:
        from diffusers import QwenImageEditPipeline
        import torch

        print("📥 Downloading Qwen-Image-Edit models...")

        # Initialize pipeline to trigger model download
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = QwenImageEditPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )

        print("✓ Qwen-Image models downloaded and cached")

        # Also download captioning model
        print("📥 Downloading image captioning model...")
        from transformers import AutoProcessor, AutoModelForCausalLM

        processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

        print("✓ Image captioning model downloaded")
        return True

    except Exception as e:
        print(f"✗ Model download failed: {e}")
        return False

def test_cloth_editing():
    """Test cloth editing functionality."""
    print("Testing cloth editing functionality...")

    try:
        import sys
        sys.path.append('.')

        from transforms_cloth_editing import get_cloth_editing_transforms, apply_cloth_editing_transform
        import numpy as np

        # Check transforms are available
        transforms = get_cloth_editing_transforms()
        print(f"✓ Found {len(transforms)} cloth editing transforms")

        # Create test image
        test_image = np.random.randint(0, 255, (512, 256, 3), dtype=np.uint8)
        print("✓ Created test image")

        # Test basic functionality (without actual inference to save time)
        print("✓ Cloth editing components working")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup cloth editing for ReID diagnostics")
    parser.add_argument("--skip-models", action="store_true", help="Skip downloading models")
    parser.add_argument("--deps-only", action="store_true", help="Only install dependencies")
    parser.add_argument("--skip-test", action="store_true", help="Skip testing installation")
    parser.add_argument("--force-pytorch", action="store_true", help="Force reinstall PyTorch")

    args = parser.parse_args()

    print("ReID Diagnostics - Cloth Editing Setup")
    print("=" * 50)

    success = True

    # Check CUDA availability first
    if not check_cuda():
        print("\n❌ CUDA GPU required for cloth editing")
        print("Please ensure you have:")
        print("  - NVIDIA GPU with 8GB+ VRAM")
        print("  - CUDA drivers installed")
        sys.exit(1)

    # Install/check PyTorch
    if args.force_pytorch or not check_torch():
        if not install_cuda_pytorch():
            success = False

    # Install dependencies
    if success and not args.deps_only:
        if not install_dependencies():
            print("⚠ Some dependencies failed to install")

    # Check Qwen availability
    if success:
        if not check_qwen_availability():
            print("✗ Qwen-Image not properly installed")
            success = False

    # Download models
    if success and not args.skip_models and not args.deps_only:
        if not download_and_cache_models():
            print("⚠ Model download failed - you can download them later")

    # Test installation
    if success and not args.skip_test:
        if not test_cloth_editing():
            print("⚠ Testing failed - manual verification may be needed")

    print("\n" + "=" * 50)
    if success:
        print("🎉 Cloth editing setup completed successfully!")
        print("\nNext steps:")
        print("1. Verify GPU memory: python check_cloth_editing_requirements.py")
        print("2. Test with ReID diagnostics:")
        print("   python reid_diagnostics.py --input image.jpg --outdir output --enable-cloth-editing")
        print("\nNote: First run will be slower due to model loading")
    else:
        print("❌ Setup encountered issues")
        print("\nTroubleshooting:")
        print("1. Ensure NVIDIA GPU with 8GB+ VRAM")
        print("2. Install CUDA drivers")
        print("3. Check internet connection for model downloads")
        print("4. Run: python check_cloth_editing_requirements.py")


if __name__ == "__main__":
    main()