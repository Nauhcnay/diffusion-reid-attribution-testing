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
            **get_morphology_transforms()
        }
        
        if self.verbose:
            print(f"Initialized runner with {len(self.transform_registries)} available transforms")
    
    def run_diagnostics(self, 
                       input_image: np.ndarray,
                       input_path: str,
                       outdir: Path,
                       mask: Optional[np.ndarray] = None,
                       bg_bank: Optional[str] = None) -> Dict[str, Any]:
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
            print(f"Starting diagnostics for image: {input_path}")
            print(f"Output directory: {outdir}")
        
        # Create output directory structure
        self._create_output_structure(outdir)
        
        # Load background images if provided
        background_images = []
        if bg_bank:
            background_images = load_background_images(bg_bank)
            if self.verbose:
                print(f"Loaded {len(background_images)} background images")
        
        # Get or generate foreground mask
        if mask is None:
            mask = get_foreground_mask(input_image)
            if mask is not None and self.verbose:
                print(f"Generated foreground mask with {np.sum(mask)} foreground pixels")
        
        # Save original image
        original_path = outdir / "images" / "original.png"
        save_image(input_image, str(original_path))
        
        # Run all transforms
        transform_results = self._run_all_transforms(
            input_image, outdir, mask, background_images
        )
        
        if self.verbose:
            successful_transforms = [r for r in transform_results if r.error is None]
            failed_transforms = [r for r in transform_results if r.error is not None]
            print(f"Completed transforms: {len(successful_transforms)} successful, {len(failed_transforms)} failed")
        
        # Create CSV index
        csv_path = outdir / self.config.output['csv_filename']
        self._create_csv_index(transform_results, input_path, str(csv_path))
        
        # Create contact sheets
        contact_sheet_paths = []
        if self.config.contact_sheet['enabled']:
            contact_sheet_paths = self._create_contact_sheets(
                input_image, transform_results, outdir
            )
            if self.verbose and contact_sheet_paths:
                print(f"Created {len(contact_sheet_paths)} contact sheet(s)")
        
        # Create metrics visualization
        metrics_viz_path = None
        successful_results = [r for r in transform_results if r.error is None and r.metrics]
        if successful_results:
            metrics_viz_path = outdir / "metrics_visualization.png"
            self._create_metrics_visualization(successful_results, str(metrics_viz_path))
        
        # Create summary report
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
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
            print(f"Diagnostics completed in {processing_time:.2f} seconds")
            print(f"Results saved to: {outdir}")
        
        return results_dict
    
    def _create_output_structure(self, outdir: Path) -> None:
        """Create output directory structure."""
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "images").mkdir(exist_ok=True)
        
        # Create family subdirectories
        families = set()
        for transform_name in self.config.get_enabled_transforms():
            family = get_family_from_transform_name(transform_name)
            families.add(family)
        
        for family in families:
            (outdir / "images" / family).mkdir(exist_ok=True)
    
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
            if transform_name not in self.transform_registries:
                if self.verbose:
                    print(f"Warning: Transform {transform_name} not found in registry")
                continue
            
            # Get transform info
            transform_info = self.transform_registries[transform_name]
            family = get_family_from_transform_name(transform_name)
            
            # Generate parameter combinations
            try:
                param_combinations = transform_info['param_combinations'](transform_config.params)
            except Exception as e:
                results.append(TransformResult(
                    transform_name=transform_name,
                    family=family,
                    param_key="",
                    params={},
                    error=f"param_generation_failed: {e}"
                ))
                continue
            
            # Limit number of examples if specified
            max_examples = transform_config.max_examples
            if max_examples and len(param_combinations) > max_examples:
                param_combinations = param_combinations[:max_examples]
            
            # Apply transform for each parameter combination
            for param_dict in param_combinations:
                result = self._apply_single_transform(
                    input_image, transform_name, transform_info, 
                    param_dict, outdir, mask, background_images
                )
                results.append(result)
        
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
                return TransformResult(
                    transform_name=transform_name,
                    family=family,
                    param_key=param_key,
                    params=param_dict,
                    error="mask_required_but_not_available"
                )
            
            if requires_backgrounds and not background_images:
                return TransformResult(
                    transform_name=transform_name,
                    family=family,
                    param_key=param_key,
                    params=param_dict,
                    error="backgrounds_required_but_not_available"
                )
            
            # Apply transform
            if transform_name.startswith('background_') and 'replace' in transform_name:
                # Special handling for background replacement
                result_image, actual_params = transform_func(
                    input_image, background_images, mask, self.seed
                )
            elif requires_mask:
                # Transform that requires mask
                result_image, actual_params = transform_func(
                    input_image, mask=mask, **param_dict
                )
            else:
                # Regular transform
                result_image, actual_params = transform_func(
                    input_image, **param_dict
                )
            
            # Check if transform failed
            if result_image is None:
                error_msg = actual_params.get('error', 'unknown_transform_error')
                return TransformResult(
                    transform_name=transform_name,
                    family=family,
                    param_key=param_key,
                    params=param_dict,
                    error=error_msg
                )
            
            # Save result image
            save_image(result_image, str(output_path))
            
            # Compute metrics
            metrics = None
            try:
                metrics = compute_all_metrics(result_image, include_expensive=False)
                # Add image hash
                metrics['image_hash'] = compute_image_hash(result_image)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to compute metrics for {transform_name}: {e}")
            
            processing_time = time.time() - start_time
            
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