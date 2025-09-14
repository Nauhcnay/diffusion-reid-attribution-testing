"""
Main runner for orchestrating transform grids and diagnostics.
Coordinates all transforms, metrics computation, and output generation.
"""

import os
import json
import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DiagnosticsConfig, get_family_from_transform_name
from io_utils import (
    load_image, save_image, get_foreground_mask, 
    load_background_images, compute_image_hash
)
from metrics import compute_all_metrics, get_metric_names
from visualization import create_contact_sheet, create_metrics_visualization, create_summary_report

# Import all transform registries
from transforms_frequency import get_frequency_transforms
from transforms_degradation import get_degradation_transforms
from transforms_color_texture import get_color_texture_transforms
from transforms_background import get_background_transforms
from transforms_morphology import get_morphology_transforms
from transforms_cloth_editing import get_cloth_editing_transforms


@dataclass
class TransformResult:
    """Result of applying a single transform."""
    transform_name: str
    family: str
    param_key: str
    params: Dict[str, Any]
    output_path: Optional[str] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None


class DiagnosticsRunner:
    """Main runner for ReID diagnostics."""
    
    def __init__(self, config: DiagnosticsConfig, seed: int = 123, verbose: bool = False):
        self.config = config
        self.seed = seed
        self.verbose = verbose
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize transform registries
        self.transform_registries = {
            **get_frequency_transforms(),
            **get_degradation_transforms(),
            **get_color_texture_transforms(),
            **get_background_transforms(),
            **get_morphology_transforms(),
            **get_cloth_editing_transforms()
        }
        
        if self.verbose:
            print(f"Initialized runner with {len(self.transform_registries)} available transforms")
    
    def run_diagnostics(self, 
                       input_image: np.ndarray,
                       input_path: str,
                       outdir: Path,
                       mask: Optional[np.ndarray] = None,
                       bg_bank: Optional[str] = None,
                       segmentation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run complete diagnostics pipeline.
        
        Args:
            input_image: Input RGB image (uint8)
            input_path: Path to original input image
            outdir: Output directory
            mask: Optional foreground mask
            bg_bank: Optional background bank directory
            
        Returns:
            Dictionary with complete results
        """
        start_time = datetime.now()

        if self.verbose:
            print("=" * 80)
            print("🚀 STARTING REID DIAGNOSTICS")
            print("=" * 80)
            print(f"📁 Input image: {input_path}")
            print(f"📂 Output directory: {outdir}")
            print(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🌱 Random seed: {self.seed}")
            print()
        
        # Create output directory structure
        if self.verbose:
            print("📁 Creating output directory structure...")
        self._create_output_structure(outdir)
        if self.verbose:
            print("✅ Output directories created")

        # Load background images if provided
        background_images = []
        if bg_bank:
            if self.verbose:
                print(f"🖼️  Loading background images from: {bg_bank}")
            background_images = load_background_images(bg_bank)
            if self.verbose:
                print(f"✅ Loaded {len(background_images)} background images")
        elif self.verbose:
            print("ℹ️  No background bank provided - background replacement will be skipped")
        
        # Get or generate foreground mask
        if mask is None:
            if self.verbose:
                print("🎭 FOREGROUND SEGMENTATION")
                print("-" * 40)

            # Use segmentation configuration
            seg_config = segmentation_config or {'method': 'auto', 'use_advanced': True}
            method_name = seg_config.get('method', 'auto')

            if self.verbose:
                print(f"🔍 Segmentation method: {method_name}")
                print(f"🤖 Advanced segmentation: {'enabled' if seg_config.get('use_advanced', True) else 'disabled'}")

            if seg_config.get('method') == 'auto':
                if self.verbose:
                    print("🔄 Running automatic segmentation (will try SAM2 → SAM → GrabCut)...")
                mask = get_foreground_mask(
                    input_image,
                    use_advanced=seg_config.get('use_advanced', True),
                    verbose=self.verbose
                )
            else:
                # Use specific segmentation method
                if self.verbose:
                    print(f"🎯 Running specific segmentation method: {method_name}")

                try:
                    from segmentation import create_advanced_mask
                    method = seg_config.get('method')
                    if method in ["sam2", "sam"]:
                        if self.verbose:
                            print(f"🚀 Initializing {method.upper()} model...")
                        mask = create_advanced_mask(
                            input_image,
                            model_type=method,
                            verbose=self.verbose
                        )
                    elif method == "grabcut":
                        if self.verbose:
                            print("🎨 Using GrabCut segmentation...")
                        mask = get_foreground_mask(
                            input_image,
                            use_advanced=False,
                            verbose=self.verbose
                        )
                except ImportError:
                    if self.verbose:
                        print("⚠️  Advanced segmentation not available, falling back to basic methods")
                    mask = get_foreground_mask(
                        input_image,
                        use_advanced=False,
                        verbose=self.verbose
                    )
                except Exception as e:
                    if self.verbose:
                        print(f"❌ Advanced segmentation failed: {e}")
                        print("🔄 Falling back to basic methods...")
                        import traceback
                        print(f"📋 Segmentation error trace: {traceback.format_exc()}")
                    mask = get_foreground_mask(
                        input_image,
                        use_advanced=False,
                        verbose=self.verbose
                    )

            # Report segmentation results
            if mask is not None:
                fg_pixels = np.sum(mask)
                total_pixels = mask.size
                fg_ratio = fg_pixels / total_pixels
                if self.verbose:
                    print(f"✅ Segmentation successful!")
                    print(f"   🎯 Foreground pixels: {fg_pixels:,} ({fg_ratio:.1%})")
                    print(f"   📐 Mask dimensions: {mask.shape}")

                # Save debug visualization
                self._save_segmentation_debug(input_image, mask, outdir)
            else:
                if self.verbose:
                    print("❌ Segmentation failed - no foreground mask available")
                    print("⚠️  Background and morphological transforms will be skipped")
        else:
            if self.verbose:
                fg_pixels = np.sum(mask)
                total_pixels = mask.size
                fg_ratio = fg_pixels / total_pixels
                print("🎭 FOREGROUND SEGMENTATION")
                print("-" * 40)
                print("✅ Using provided mask")
                print(f"   🎯 Foreground pixels: {fg_pixels:,} ({fg_ratio:.1%})")
                print(f"   📐 Mask dimensions: {mask.shape}")

        if self.verbose:
            print()
        
        # Save original image
        if self.verbose:
            print("💾 Saving original image...")
        original_path = outdir / "images" / "original.png"
        save_image(input_image, str(original_path))
        if self.verbose:
            file_size = original_path.stat().st_size if original_path.exists() else 0
            print(f"✅ Original image saved ({file_size} bytes)")
            print()

        # Run all transforms
        if self.verbose:
            enabled_count = len(self.config.get_enabled_transforms())
            print("🔧 TRANSFORM PROCESSING")
            print("-" * 40)
            print(f"📋 Enabled transform families: {enabled_count}")
            print()

        transform_results = self._run_all_transforms(
            input_image, outdir, mask, background_images
        )

        if self.verbose:
            successful_transforms = [r for r in transform_results if r.error is None]
            failed_transforms = [r for r in transform_results if r.error is not None]
            total_processing_time = sum(r.processing_time for r in transform_results if r.processing_time)

            print()
            print("📊 TRANSFORM SUMMARY")
            print("-" * 40)
            print(f"✅ Successful: {len(successful_transforms)}")
            print(f"❌ Failed: {len(failed_transforms)}")
            print(f"⏱️  Total transform time: {total_processing_time:.2f}s")
            print()
        
        # Create CSV index
        if self.verbose:
            print("📄 CREATING DATA INDEX")
            print("-" * 40)
        csv_path = outdir / self.config.output['csv_filename']
        self._create_csv_index(transform_results, input_path, str(csv_path))
        if self.verbose:
            csv_size = csv_path.stat().st_size if csv_path.exists() else 0
            print(f"✅ CSV index created: {csv_path.name} ({csv_size} bytes)")

        # Create contact sheets
        contact_sheet_paths = []
        if self.config.contact_sheet['enabled']:
            if self.verbose:
                print("🖼️  Creating contact sheets...")
            contact_sheet_paths = self._create_contact_sheets(
                input_image, transform_results, outdir
            )
            if self.verbose and contact_sheet_paths:
                total_sheets = len(contact_sheet_paths)
                print(f"✅ Created {total_sheets} contact sheet{'s' if total_sheets > 1 else ''}")
                for sheet_path in contact_sheet_paths:
                    sheet_size = Path(sheet_path).stat().st_size if Path(sheet_path).exists() else 0
                    print(f"   📸 {Path(sheet_path).name} ({sheet_size} bytes)")
        elif self.verbose:
            print("ℹ️  Contact sheets disabled in configuration")

        # Create metrics visualization
        metrics_viz_path = None
        successful_results = [r for r in transform_results if r.error is None and r.metrics]
        if successful_results:
            if self.verbose:
                print("📊 Creating metrics visualization...")
            metrics_viz_path = outdir / "metrics_visualization.png"
            self._create_metrics_visualization(successful_results, str(metrics_viz_path))
            if self.verbose:
                viz_size = metrics_viz_path.stat().st_size if metrics_viz_path.exists() else 0
                print(f"✅ Metrics visualization created ({viz_size} bytes)")
        elif self.verbose:
            print("ℹ️  No successful results with metrics - skipping visualization")
        
        # Create summary report
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        if self.verbose:
            print("📄 Creating summary report...")
        summary_path = outdir / "summary_report.txt"
        results_dict = {
            'input_path': input_path,
            'output_dir': str(outdir),
            'timestamp': start_time.isoformat(),
            'processing_time_seconds': processing_time,
            'transform_results': [self._result_to_dict(r) for r in transform_results],
            'contact_sheet_paths': contact_sheet_paths,
            'metrics_visualization_path': str(metrics_viz_path) if metrics_viz_path else None,
            'csv_path': str(csv_path)
        }

        create_summary_report(results_dict, str(summary_path))
        if self.verbose:
            report_size = summary_path.stat().st_size if summary_path.exists() else 0
            print(f"✅ Summary report created ({report_size} bytes)")

        if self.verbose:
            print()
            print("🎉 DIAGNOSTICS COMPLETED")
            print("=" * 80)
            print(f"⏰ Total processing time: {processing_time:.2f} seconds")
            print(f"📂 Results saved to: {outdir}")

            # Count output files
            total_files = len(list(outdir.rglob("*"))) if outdir.exists() else 0
            image_files = len(list((outdir / "images").rglob("*.png"))) if (outdir / "images").exists() else 0

            print(f"📊 Generated {total_files} total files")
            print(f"🖼️  Created {image_files} transformed images")

            if transform_results:
                success_rate = len([r for r in transform_results if r.error is None]) / len(transform_results)
                print(f"📈 Transform success rate: {success_rate:.1%}")
            print("=" * 80)

        return results_dict
    
    def _create_output_structure(self, outdir: Path) -> None:
        """Create output directory structure."""
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "images").mkdir(exist_ok=True)

        # Create debug directory for segmentation visualization
        (outdir / "images" / "debug").mkdir(exist_ok=True)

        # Create family subdirectories
        families = set()
        for transform_name in self.config.get_enabled_transforms():
            family = get_family_from_transform_name(transform_name)
            families.add(family)

        for family in families:
            (outdir / "images" / family).mkdir(exist_ok=True)

    def _save_segmentation_debug(self, input_image: np.ndarray, mask: np.ndarray, outdir: Path) -> None:
        """Save segmentation debug visualizations."""
        debug_dir = outdir / "images" / "debug"

        try:
            # 1. Save raw mask as grayscale
            mask_uint8 = (mask.astype(np.uint8) * 255)
            # Convert to 3-channel grayscale for save_image compatibility
            if len(mask_uint8.shape) == 2:
                mask_uint8 = np.stack([mask_uint8, mask_uint8, mask_uint8], axis=2)
            save_image(mask_uint8, str(debug_dir / "segmentation_mask_raw.png"))

            # 2. Save mask overlay on original image
            overlay = input_image.copy()
            # Make foreground slightly transparent, background more transparent
            alpha = 0.7
            overlay[~mask] = (overlay[~mask] * 0.3).astype(np.uint8)  # Darken background
            overlay[mask] = (overlay[mask] * alpha + np.array([0, 255, 0]) * (1-alpha)).astype(np.uint8)  # Green tint for foreground
            save_image(overlay, str(debug_dir / "segmentation_overlay.png"))

            # 3. Save foreground only (same as background transform does)
            foreground_only = np.zeros_like(input_image)
            foreground_only[mask] = input_image[mask]
            save_image(foreground_only, str(debug_dir / "foreground_extracted.png"))

            # 4. Save background only
            background_only = input_image.copy()
            background_only[mask] = 0
            save_image(background_only, str(debug_dir / "background_extracted.png"))

            # 5. Save mask contours visualization
            contour_vis = input_image.copy()
            import cv2
            # Convert mask to uint8 for contour detection
            mask_uint8_contour = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_vis, contours, -1, (0, 255, 255), 2)  # Yellow contours
            save_image(contour_vis, str(debug_dir / "segmentation_contours.png"))

            if self.verbose:
                print(f"   🐛 Debug visualizations saved to {debug_dir}")
                print(f"      • segmentation_mask_raw.png - Raw boolean mask")
                print(f"      • segmentation_overlay.png - Mask overlaid on original")
                print(f"      • foreground_extracted.png - Foreground only")
                print(f"      • background_extracted.png - Background only")
                print(f"      • segmentation_contours.png - Mask boundaries")

        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ Failed to save debug visualizations: {e}")

    def _run_all_transforms(self, 
                           input_image: np.ndarray,
                           outdir: Path,
                           mask: Optional[np.ndarray],
                           background_images: List[np.ndarray]) -> List[TransformResult]:
        """Run all enabled transforms."""
        results = []
        enabled_transforms = self.config.get_enabled_transforms()
        
        # Create progress bar if verbose
        if self.verbose:
            transform_iterator = tqdm(enabled_transforms.items(), desc="Running transforms")
        else:
            transform_iterator = enabled_transforms.items()
        
        for transform_name, transform_config in transform_iterator:
            if self.verbose:
                print(f"\n🔄 Processing transform family: {transform_name}")

            if transform_name not in self.transform_registries:
                error_msg = f"❌ Transform {transform_name} not found in registry"
                if self.verbose:
                    print(f"   {error_msg}")
                results.append(TransformResult(
                    transform_name=transform_name,
                    family=get_family_from_transform_name(transform_name),
                    param_key="",
                    params={},
                    error="transform_not_found"
                ))
                continue

            # Get transform info
            transform_info = self.transform_registries[transform_name]
            family = get_family_from_transform_name(transform_name)

            if self.verbose:
                print(f"   📋 Family: {family}")
                print(f"   🔧 Enabled: {transform_config.enabled}")

            # Generate parameter combinations
            try:
                param_combinations = transform_info['param_combinations'](transform_config.params)
                if self.verbose:
                    print(f"   📊 Generated {len(param_combinations)} parameter combinations")
            except Exception as e:
                error_msg = f"param_generation_failed: {str(e)}"
                if self.verbose:
                    print(f"   ❌ Parameter generation failed: {e}")
                results.append(TransformResult(
                    transform_name=transform_name,
                    family=family,
                    param_key="",
                    params={},
                    error=error_msg
                ))
                continue

            # Limit number of examples if specified
            max_examples = transform_config.max_examples
            if max_examples and len(param_combinations) > max_examples:
                param_combinations = param_combinations[:max_examples]
                if self.verbose:
                    print(f"   ✂️  Limited to {max_examples} examples (from {len(param_combinations)} total)")

            # Check requirements
            requires_mask = transform_info.get('requires_mask', False)
            requires_backgrounds = transform_info.get('requires_backgrounds', False)

            if self.verbose:
                status_items = []
                if requires_mask:
                    mask_status = "✓" if mask is not None else "❌"
                    status_items.append(f"mask {mask_status}")
                if requires_backgrounds:
                    bg_status = "✓" if background_images else "❌"
                    status_items.append(f"backgrounds {bg_status}")

                if status_items:
                    print(f"   📋 Requirements: {', '.join(status_items)}")

            # Apply transform for each parameter combination
            successful_variants = 0
            failed_variants = 0

            for i, param_dict in enumerate(param_combinations):
                if self.verbose:
                    param_str = ", ".join([f"{k}={v}" for k, v in param_dict.items()])
                    print(f"   🎯 Variant {i+1}/{len(param_combinations)}: {param_str[:60]}{'...' if len(param_str) > 60 else ''}")

                result = self._apply_single_transform(
                    input_image, transform_name, transform_info,
                    param_dict, outdir, mask, background_images
                )

                if result.error is None:
                    successful_variants += 1
                    if self.verbose:
                        print(f"      ✅ Success: {result.output_path}")
                else:
                    failed_variants += 1
                    if self.verbose:
                        print(f"      ❌ Failed: {result.error}")

                results.append(result)

            if self.verbose:
                total_variants = successful_variants + failed_variants
                success_rate = (successful_variants / total_variants * 100) if total_variants > 0 else 0
                print(f"   📈 Summary: {successful_variants}/{total_variants} successful ({success_rate:.1f}%)")
        
        return results
    
    def _apply_single_transform(self, 
                               input_image: np.ndarray,
                               transform_name: str,
                               transform_info: Dict[str, Any],
                               param_dict: Dict[str, Any],
                               outdir: Path,
                               mask: Optional[np.ndarray],
                               background_images: List[np.ndarray]) -> TransformResult:
        """Apply a single transform with given parameters."""
        import time
        
        start_time = time.time()
        family = get_family_from_transform_name(transform_name)
        
        # Create parameter key for filename
        param_key = "_".join([f"{k}={v}" for k, v in sorted(param_dict.items())])
        param_key = param_key.replace(" ", "").replace("(", "").replace(")", "").replace(",", "-")
        if len(param_key) > 100:  # Truncate very long keys
            param_key = param_key[:97] + "..."
        
        # Create output filename
        output_filename = f"{transform_name}__{param_key}.png"
        output_path = outdir / "images" / family / output_filename
        
        try:
            # Get transform function
            transform_func = transform_info['function']
            
            # Check if transform requires mask
            requires_mask = transform_info.get('requires_mask', False)
            requires_backgrounds = transform_info.get('requires_backgrounds', False)
            
            # Skip if requirements not met
            if requires_mask and mask is None:
                error_msg = "mask_required_but_not_available"
                if self.verbose:
                    print(f"        ⚠️  Skipping: requires mask but none available")
                return TransformResult(
                    transform_name=transform_name,
                    family=family,
                    param_key=param_key,
                    params=param_dict,
                    error=error_msg
                )

            if requires_backgrounds and not background_images:
                error_msg = "backgrounds_required_but_not_available"
                if self.verbose:
                    print(f"        ⚠️  Skipping: requires backgrounds but none available")
                return TransformResult(
                    transform_name=transform_name,
                    family=family,
                    param_key=param_key,
                    params=param_dict,
                    error=error_msg
                )
            
            # Apply transform
            if self.verbose:
                print(f"        🔧 Applying transform...")

            try:
                if transform_name.startswith('cloth_edit'):
                    # Special handling for cloth editing transforms (return multiple results)
                    if self.verbose:
                        print(f"        👗 Cloth editing transform with AI models")
                    transform_results = transform_func(input_image, **param_dict)

                    # Handle multiple results from cloth editing
                    if transform_results and len(transform_results) > 0:
                        # Use the first result for now, could be extended to handle all variants
                        result_image, actual_params = transform_results[0]
                    else:
                        result_image = None
                        actual_params = {'error': 'no_cloth_variants_generated'}

                elif transform_name.startswith('background_') and 'replace' in transform_name:
                    # Special handling for background replacement
                    if self.verbose:
                        print(f"        🎨 Background replacement with {len(background_images)} options")
                    result_image, actual_params = transform_func(
                        input_image, background_images, mask, self.seed
                    )
                elif requires_mask:
                    # Transform that requires mask
                    if self.verbose:
                        mask_pixels = np.sum(mask) if mask is not None else 0
                        print(f"        🎭 Mask-based transform ({mask_pixels} foreground pixels)")
                    result_image, actual_params = transform_func(
                        input_image, mask=mask, **param_dict
                    )
                else:
                    # Regular transform
                    if self.verbose:
                        print(f"        🔄 Direct image transform")
                    result_image, actual_params = transform_func(
                        input_image, **param_dict
                    )

                if self.verbose:
                    print(f"        ✅ Transform function completed")

            except Exception as e:
                error_msg = f"transform_execution_failed: {str(e)}"
                if self.verbose:
                    print(f"        ❌ Transform execution failed: {e}")
                    import traceback
                    print(f"        📋 Stack trace: {traceback.format_exc()}")

                processing_time = time.time() - start_time
                return TransformResult(
                    transform_name=transform_name,
                    family=family,
                    param_key=param_key,
                    params=param_dict,
                    error=error_msg,
                    processing_time=processing_time
                )
            
            # Check if transform failed
            if result_image is None:
                error_msg = actual_params.get('error', 'unknown_transform_error')
                if self.verbose:
                    print(f"        ❌ Transform returned None: {error_msg}")
                return TransformResult(
                    transform_name=transform_name,
                    family=family,
                    param_key=param_key,
                    params=param_dict,
                    error=error_msg
                )

            # Save result image
            try:
                if self.verbose:
                    h, w = result_image.shape[:2]
                    print(f"        💾 Saving {w}x{h} image to {output_path.name}")
                save_image(result_image, str(output_path))
                if self.verbose:
                    file_size = output_path.stat().st_size if output_path.exists() else 0
                    print(f"        ✅ Image saved ({file_size} bytes)")
            except Exception as e:
                error_msg = f"image_save_failed: {str(e)}"
                if self.verbose:
                    print(f"        ❌ Failed to save image: {e}")
                processing_time = time.time() - start_time
                return TransformResult(
                    transform_name=transform_name,
                    family=family,
                    param_key=param_key,
                    params=actual_params,
                    error=error_msg,
                    processing_time=processing_time
                )

            # Compute metrics
            metrics = None
            try:
                if self.verbose:
                    print(f"        📊 Computing image quality metrics...")
                metrics = compute_all_metrics(result_image, include_expensive=False)
                # Add image hash
                metrics['image_hash'] = compute_image_hash(result_image)
                if self.verbose:
                    non_null_metrics = len([v for v in metrics.values() if v is not None])
                    print(f"        ✅ Computed {non_null_metrics}/{len(metrics)} metrics")
            except Exception as e:
                if self.verbose:
                    print(f"        ⚠️  Failed to compute metrics: {e}")
                    import traceback
                    print(f"        📋 Metrics error trace: {traceback.format_exc()}")

            processing_time = time.time() - start_time

            if self.verbose:
                print(f"        ⏱️  Processing time: {processing_time:.3f}s")

            return TransformResult(
                transform_name=transform_name,
                family=family,
                param_key=param_key,
                params=actual_params,
                output_path=str(output_path),
                metrics=metrics,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return TransformResult(
                transform_name=transform_name,
                family=family,
                param_key=param_key,
                params=param_dict,
                error=f"transform_failed: {str(e)}",
                processing_time=processing_time
            )
    
    def _create_csv_index(self, 
                         transform_results: List[TransformResult],
                         input_path: str,
                         csv_path: str) -> None:
        """Create CSV index of all results."""
        rows = []
        
        # Add original image entry
        try:
            original_image = load_image(input_path)
            original_metrics = compute_all_metrics(original_image, include_expensive=False)
            original_hash = compute_image_hash(original_image)
            original_metrics['image_hash'] = original_hash
        except:
            original_metrics = {}
        
        original_row = {
            'family': 'original',
            'transform': 'original',
            'param_key': '',
            'params_json': '{}',
            'input_path': input_path,
            'output_path': input_path,
            'width': original_image.shape[1] if 'original_image' in locals() else None,
            'height': original_image.shape[0] if 'original_image' in locals() else None,
            'error': None,
            'processing_time': None
        }
        
        # Add metrics with 'metrics_' prefix
        for metric_name in get_metric_names():
            original_row[f'metrics_{metric_name}'] = original_metrics.get(metric_name)
        
        rows.append(original_row)
        
        # Add transform results
        for result in transform_results:
            row = {
                'family': result.family,
                'transform': result.transform_name,
                'param_key': result.param_key,
                'params_json': json.dumps(result.params),
                'input_path': input_path,
                'output_path': result.output_path,
                'width': None,
                'height': None,
                'error': result.error,
                'processing_time': result.processing_time
            }
            
            # Add image dimensions if available
            if result.output_path and os.path.exists(result.output_path):
                try:
                    result_image = load_image(result.output_path)
                    row['width'] = result_image.shape[1]
                    row['height'] = result_image.shape[0]
                except:
                    pass
            
            # Add metrics
            metrics = result.metrics or {}
            for metric_name in get_metric_names():
                row[f'metrics_{metric_name}'] = metrics.get(metric_name)
            
            rows.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        if self.verbose:
            print(f"Created CSV index with {len(rows)} entries: {csv_path}")
    
    def _create_contact_sheets(self, 
                              input_image: np.ndarray,
                              transform_results: List[TransformResult],
                              outdir: Path) -> List[str]:
        """Create contact sheets showing all results."""
        # Prepare images for contact sheet
        images_info = []
        
        # Add original image
        images_info.append({
            'image_array': input_image,
            'transform_name': 'Original',
            'param_key': ''
        })
        
        # Add successful transforms
        successful_results = [r for r in transform_results if r.error is None and r.output_path]
        for result in successful_results:
            if os.path.exists(result.output_path):
                images_info.append({
                    'image_path': result.output_path,
                    'transform_name': result.transform_name,
                    'param_key': result.param_key
                })
        
        # Create contact sheets
        contact_sheet_path = outdir / self.config.contact_sheet['filename_pattern'].format(1)
        
        contact_sheet_paths = create_contact_sheet(
            images_info,
            str(contact_sheet_path),
            thumbnail_size=self.config.contact_sheet['thumbnail_size'],
            grid_cols=self.config.contact_sheet['grid_cols'],
            max_images_per_sheet=self.config.contact_sheet['max_images_per_sheet'],
            font_size=self.config.contact_sheet['font_size'],
            caption_height=self.config.contact_sheet['caption_height'],
            padding=self.config.contact_sheet['padding'],
            background_color=self.config.contact_sheet['background_color']
        )
        
        return contact_sheet_paths
    
    def _create_metrics_visualization(self, 
                                    successful_results: List[TransformResult],
                                    output_path: str) -> str:
        """Create metrics visualization."""
        metrics_data = []
        
        for result in successful_results:
            if result.metrics:
                metrics_data.append({
                    'transform_name': result.transform_name,
                    'param_key': result.param_key,
                    'metrics': result.metrics
                })
        
        # Select key metrics to visualize
        key_metrics = ['edge_density', 'gradient_energy', 'contrast', 'sharpness']
        
        return create_metrics_visualization(
            metrics_data, 
            output_path, 
            key_metrics,
            figsize=(15, 10)
        )
    
    def _result_to_dict(self, result: TransformResult) -> Dict[str, Any]:
        """Convert TransformResult to dictionary."""
        return {
            'transform_name': result.transform_name,
            'family': result.family,
            'param_key': result.param_key,
            'params': result.params,
            'output_path': result.output_path,
            'error': result.error,
            'metrics': result.metrics,
            'processing_time': result.processing_time
        }


if __name__ == "__main__":
    # Test runner with synthetic data
    print("Testing DiagnosticsRunner...")
    
    # Create synthetic test image
    test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    # Create test configuration
    from config import DiagnosticsConfig
    config = DiagnosticsConfig()
    
    # Disable most transforms for quick testing
    for transform_name in list(config.transforms.keys()):
        if not transform_name.startswith('frequency_gaussian_blur'):
            config.transforms[transform_name].enabled = False
    
    # Limit examples
    config.apply_max_examples(2)
    
    # Create runner
    runner = DiagnosticsRunner(config, seed=42, verbose=True)
    
    # Create temporary output directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "test_diagnostics"
        
        # Run diagnostics
        results = runner.run_diagnostics(
            input_image=test_image,
            input_path="test_image.png",
            outdir=outdir
        )
        
        print(f"Test completed. Results: {len(results['transform_results'])} transforms processed")
        print(f"Output directory: {outdir}")
        
        # List output files
        if outdir.exists():
            output_files = list(outdir.rglob("*"))
            print(f"Created {len(output_files)} output files")
    
    print("Runner test completed.")