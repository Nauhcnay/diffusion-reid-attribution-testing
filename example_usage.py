#!/usr/bin/env python3
"""
Example usage of the ReID diagnostics tool.
This script shows how to use the tool both as a library and via CLI.
"""

def example_cli_usage():
    """Show example CLI commands."""
    
    print("CLI Usage Examples:")
    print("=" * 50)
    
    print("\n1. Basic usage with image only:")
    print("python reid_diagnostics.py --input person.jpg --outdir results/")
    
    print("\n2. With foreground mask:")
    print("python reid_diagnostics.py --input person.jpg --outdir results/ --mask mask.png")
    
    print("\n3. With background replacement:")
    print("python reid_diagnostics.py --input person.jpg --outdir results/ --bg_bank backgrounds/")
    
    print("\n4. With custom configuration:")
    print("python reid_diagnostics.py --input person.jpg --outdir results/ --config custom.yaml")
    
    print("\n5. Limit transforms and use verbose output:")
    print("python reid_diagnostics.py --input person.jpg --outdir results/ --max_examples_per_family 3 --verbose")
    
    print("\n6. Complete example:")
    print("python reid_diagnostics.py \\")
    print("    --input person.jpg \\")
    print("    --outdir results/ \\")
    print("    --mask person_mask.png \\")
    print("    --bg_bank background_images/ \\")
    print("    --config config.yaml \\")
    print("    --seed 42 \\")
    print("    --max_examples_per_family 5 \\")
    print("    --verbose")


def example_library_usage():
    """Show how to use the tool as a library."""
    
    print("\n\nLibrary Usage Example:")
    print("=" * 50)
    
    code_example = '''
import numpy as np
from pathlib import Path

from config import load_config
from io_utils import load_image, load_mask
from runner import DiagnosticsRunner

# Load input data
input_image = load_image("person.jpg")
mask = load_mask("person_mask.png")  # Optional

# Load configuration
config = load_config("config.yaml")  # Optional, uses defaults if None

# Create and run diagnostics
runner = DiagnosticsRunner(config=config, seed=42, verbose=True)
results = runner.run_diagnostics(
    input_image=input_image,
    input_path="person.jpg", 
    outdir=Path("results"),
    mask=mask,
    bg_bank="background_images"  # Optional
)

# Access results
transform_results = results['transform_results']
successful = [r for r in transform_results if not r.get('error')]
print(f"Successfully processed {len(successful)} transforms")

# CSV data is available at
csv_path = results['csv_path']
contact_sheets = results['contact_sheet_paths']
'''
    
    print(code_example)


def create_minimal_config_example():
    """Show how to create a minimal configuration."""
    
    print("\n\nMinimal Configuration Example:")
    print("=" * 50)
    
    config_yaml = '''
# Limit to a few quick transforms for testing
max_examples_per_family: 2

transforms:
  # Enable only fast transforms
  frequency_gaussian_blur:
    enabled: true
    params:
      kernel_sizes: [5, 9]
      sigmas: [1.0, 2.0]
  
  degradation_jpeg:
    enabled: true
    params:
      qualities: [70, 30]
  
  color_desaturate:
    enabled: true
  
  # Disable slow or mask-dependent transforms
  frequency_fft_lowpass:
    enabled: false
  
  texture_nonlocal_means:
    enabled: false
  
  background_to_black:
    enabled: false
  
  morphology_opening:
    enabled: false

output:
  save_images: true
  save_csv: true

contact_sheet:
  enabled: true
  thumbnail_size: 64  # Smaller for faster processing
  grid_cols: 6
'''
    
    print(config_yaml)
    
    print("\nSave this as 'minimal_config.yaml' and use with:")
    print("python reid_diagnostics.py --input image.jpg --outdir results/ --config minimal_config.yaml")


def show_expected_outputs():
    """Show what outputs to expect."""
    
    print("\n\nExpected Output Structure:")
    print("=" * 50)
    
    output_structure = '''
results/
├── images/
│   ├── original.png
│   ├── frequency/
│   │   ├── frequency_gaussian_blur__kernel_size=5_sigma=1.0.png
│   │   ├── frequency_gaussian_blur__kernel_size=5_sigma=2.0.png
│   │   └── ...
│   ├── degradation/
│   │   ├── degradation_jpeg__quality=70.png
│   │   └── ...
│   └── color/
│       └── ...
├── diagnostics_index.csv
├── summary_report.txt
├── contact_sheet_01.png
└── metrics_visualization.png
'''
    
    print(output_structure)
    
    print("\nKey files:")
    print("- diagnostics_index.csv: Complete data table with metrics")
    print("- contact_sheet_*.png: Visual overview of all transforms")
    print("- summary_report.txt: Human-readable summary")
    print("- images/*: All transformed images organized by category")


def troubleshooting_tips():
    """Provide troubleshooting information."""
    
    print("\n\nTroubleshooting Tips:")
    print("=" * 50)
    
    print("\n1. Missing dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n2. Out of memory with large images:")
    print("   - Resize input image first")
    print("   - Use --max_examples_per_family to limit transforms")
    print("   - Disable expensive transforms in config")
    
    print("\n3. No foreground mask available:")
    print("   - Tool will try automatic segmentation (GrabCut)")
    print("   - Background operations will be skipped if mask fails")
    print("   - Provide manual mask with --mask for best results")
    
    print("\n4. Slow processing:")
    print("   - Use minimal config (shown above)")
    print("   - Disable morphological and texture transforms")
    print("   - Reduce contact sheet thumbnail size")
    
    print("\n5. Transform failures:")
    print("   - Check summary_report.txt for error details")
    print("   - Use --verbose to see detailed progress")
    print("   - Failed transforms are logged but don't stop processing")
    
    print("\n6. Testing the installation:")
    print("   python test_system.py")


if __name__ == "__main__":
    print("ReID Diagnostics Tool - Usage Examples")
    print("=" * 50)
    
    example_cli_usage()
    example_library_usage()
    create_minimal_config_example() 
    show_expected_outputs()
    troubleshooting_tips()
    
    print("\n\nFor more information, see README.md")