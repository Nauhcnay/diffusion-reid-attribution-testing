#!/usr/bin/env python3
"""
Person ReID Image Diagnostics CLI Tool

A production-quality CLI tool that runs a battery of image-domain diagnostics
for person ReID attribution using classical, deterministic image transforms.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from config import load_config
from runner import DiagnosticsRunner
from io_utils import load_image, validate_paths


def main():
    parser = argparse.ArgumentParser(
        description="Run image-domain diagnostics for person ReID attribution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input RGB image"
    )
    
    parser.add_argument(
        "--outdir", "-o",
        required=True,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--mask", "-m",
        help="Path to foreground mask (binary PNG)"
    )
    
    parser.add_argument(
        "--bg_bank",
        help="Directory of background images for replacement"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="YAML configuration file to override defaults"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=123,
        help="Random seed for deterministic results"
    )
    
    parser.add_argument(
        "--max_examples_per_family", "-n",
        type=int,
        help="Maximum number of examples per transform family"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--segmentation", 
        choices=["auto", "sam2", "sam", "grabcut", "basic"],
        default="auto",
        help="Segmentation method for foreground mask (auto: try SAM2->SAM->GrabCut)"
    )
    
    parser.add_argument(
        "--no_advanced_segmentation",
        action="store_true",
        help="Disable advanced segmentation (SAM2/SAM), use only basic methods"
    )

    parser.add_argument(
        "--enable-cloth-editing",
        action="store_true",
        help="Enable cloth editing transforms (requires GPU and additional dependencies)"
    )
    
    args = parser.parse_args()
    
    # Validate input paths
    try:
        validate_paths(args.input, args.mask, args.bg_bank)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Load and validate configuration
    try:
        config = load_config(args.config, args.max_examples_per_family)

        # Enable cloth editing if requested
        if args.enable_cloth_editing:
            # Check basic requirements before enabling
            try:
                import torch
                if torch.cuda.is_available():
                    if args.verbose:
                        print("🎨 Enabling cloth editing transforms")
                        print(f"   🔧 GPU: {torch.cuda.get_device_name(0)}")
                        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        print(f"   💾 VRAM: {memory_gb:.1f} GB")
                    config.transforms['cloth_edit_casual'].enabled = True
                    config.transforms['cloth_edit_multiple'].enabled = True
                else:
                    print("⚠️  Warning: CUDA not available - cloth editing requires GPU")
                    print("   💡 Transforms will be enabled but will fail during execution")
                    print("   💡 Run 'python check_cloth_editing_requirements.py' for details")
                    config.transforms['cloth_edit_casual'].enabled = True
                    config.transforms['cloth_edit_multiple'].enabled = True
            except ImportError:
                print("⚠️  Warning: PyTorch not available - cloth editing requires PyTorch with CUDA")
                print("   💡 Run 'python check_cloth_editing_requirements.py' for details")
                config.transforms['cloth_edit_casual'].enabled = True
                config.transforms['cloth_edit_multiple'].enabled = True

    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load input image
    try:
        input_image = load_image(args.input)
        if args.verbose:
            h, w = input_image.shape[:2]
            print(f"Loaded input image: {w}x{h} from {args.input}")
    except Exception as e:
        print(f"Error loading input image: {e}", file=sys.stderr)
        return 1
    
    # Load mask if provided
    mask = None
    if args.mask:
        try:
            from io_utils import load_mask
            mask = load_mask(args.mask)
            if args.verbose:
                print(f"Loaded foreground mask from {args.mask}")
        except Exception as e:
            print(f"Error loading mask: {e}", file=sys.stderr)
            return 1
    
    # Initialize and run diagnostics
    try:
        runner = DiagnosticsRunner(
            config=config,
            seed=args.seed,
            verbose=args.verbose
        )
        
        # Pass segmentation preferences to runner
        segmentation_config = {
            'method': args.segmentation,
            'use_advanced': not args.no_advanced_segmentation and args.segmentation != "basic"
        }
        
        results = runner.run_diagnostics(
            input_image=input_image,
            input_path=args.input,
            outdir=outdir,
            mask=mask,
            bg_bank=args.bg_bank,
            segmentation_config=segmentation_config
        )
        
        print(f"Diagnostics completed successfully. Results saved to: {outdir}")
        
        # Print summary if verbose
        if args.verbose:
            transform_results = results.get('transform_results', [])
            successful = len([r for r in transform_results if not r.get('error')])
            failed = len([r for r in transform_results if r.get('error')])
            print(f"Summary: {successful} successful transforms, {failed} failed")
            
            if results.get('contact_sheet_paths'):
                print(f"Contact sheets: {', '.join(results['contact_sheet_paths'])}")
            
            if results.get('csv_path'):
                print(f"CSV index: {results['csv_path']}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error running diagnostics: {e}", file=sys.stderr)
        if args.verbose:
            print("\n📋 Full error trace:", file=sys.stderr)
            import traceback
            traceback.print_exc()
        else:
            print("💡 Use --verbose for detailed error information", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())