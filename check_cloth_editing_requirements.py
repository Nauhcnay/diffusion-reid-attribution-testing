#!/usr/bin/env python3
"""
Check requirements for cloth editing functionality in ReID diagnostics.
"""

def check_cloth_editing_requirements():
    """Check if cloth editing requirements are met."""
    print("🔍 Checking Cloth Editing Requirements")
    print("=" * 45)

    requirements_met = True

    # Check Python packages
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch: Not installed")
        requirements_met = False

    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers: Not installed")
        requirements_met = False

    try:
        from diffusers import QwenImageEditPipeline
        print("✅ Diffusers with Qwen-Image: Available")
    except ImportError:
        print("❌ Diffusers with Qwen-Image: Not available")
        print("   Install with: pip install git+https://github.com/huggingface/diffusers")
        requirements_met = False

    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ CUDA: Available ({device_name}, {memory_gb:.1f} GB)")

            if memory_gb < 8:
                print("⚠️  Warning: Less than 8GB VRAM - may cause out-of-memory errors")
            elif memory_gb >= 12:
                print("🚀 Excellent: 12GB+ VRAM - optimal for cloth editing")
        else:
            print("❌ CUDA: Not available")
            print("   Cloth editing requires NVIDIA GPU with CUDA support")
            requirements_met = False
    except:
        print("❌ CUDA: Cannot check GPU status")
        requirements_met = False

    # Check internet connectivity (needed for model downloads)
    try:
        import urllib.request
        urllib.request.urlopen('https://huggingface.co', timeout=3)
        print("✅ Internet: Available (for model downloads)")
    except:
        print("⚠️  Internet: May not be available")
        print("   First-time use requires internet to download models")

    print("\n" + "=" * 45)
    if requirements_met:
        print("🎉 All requirements met! Cloth editing should work.")
        print("\nTo enable cloth editing:")
        print("  python reid_diagnostics.py --input image.jpg --outdir output --enable-cloth-editing")
    else:
        print("❌ Some requirements not met. Cloth editing will fail.")
        print("\nInstallation steps:")
        print("  1. Install CUDA-compatible PyTorch")
        print("  2. pip install transformers>=4.51.3")
        print("  3. pip install git+https://github.com/huggingface/diffusers")
        print("  4. Ensure NVIDIA GPU with 8GB+ VRAM")

    return requirements_met

if __name__ == "__main__":
    check_cloth_editing_requirements()