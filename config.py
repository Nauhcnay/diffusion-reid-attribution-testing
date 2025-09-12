"""
Configuration management for ReID diagnostics.
Defines default parameter grids and handles YAML configuration loading.
"""

import os
import yaml
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass


@dataclass
class TransformConfig:
    """Configuration for a single transform family."""
    enabled: bool = True
    max_examples: Optional[int] = None
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


class DiagnosticsConfig:
    """Main configuration class for diagnostics runner."""
    
    def __init__(self):
        self.transforms = self._get_default_transforms()
        self.output = self._get_default_output_config()
        self.contact_sheet = self._get_default_contact_sheet_config()
        self.max_examples_per_family = None
    
    def _get_default_transforms(self) -> Dict[str, TransformConfig]:
        """Get default transform configurations."""
        return {
            # A) Frequency / contrast transforms
            'frequency_gaussian_blur': TransformConfig(params={
                'kernel_sizes': [5, 9, 15, 21],
                'sigmas': [1.0, 2.0, 3.0, 5.0]
            }),
            'frequency_bilateral': TransformConfig(params={
                'diameters': [9, 15, 21],
                'sigma_colors': [20, 50, 80],
                'sigma_spaces': [20, 50, 80]
            }),
            'frequency_median': TransformConfig(params={
                'kernel_sizes': [3, 5, 7, 9]
            }),
            'frequency_laplacian': TransformConfig(params={
                'kernel_sizes': [1, 3],
                'scales': [1.0, 2.0]
            }),
            'frequency_sobel': TransformConfig(params={
                'directions': ['x', 'y', 'magnitude'],
                'kernel_sizes': [3, 5]
            }),
            'frequency_unsharp': TransformConfig(params={
                'amounts': [0.5, 1.0, 2.0],
                'radii': [1.0, 2.0, 3.0],
                'thresholds': [0, 1]
            }),
            'frequency_fft_lowpass': TransformConfig(params={
                'cutoff_ratios': [0.1, 0.2, 0.3, 0.5]
            }),
            'frequency_fft_highpass': TransformConfig(params={
                'cutoff_ratios': [0.1, 0.2, 0.3]
            }),
            'frequency_fft_bandpass': TransformConfig(params={
                'low_ratios': [0.05, 0.1],
                'high_ratios': [0.3, 0.5]
            }),
            'frequency_fft_notch': TransformConfig(params={
                'center_ratios': [0.25, 0.5],
                'notch_radii': [0.02, 0.05, 0.1]
            }),
            'photometric_gamma': TransformConfig(params={
                'gammas': [0.5, 0.7, 1.4, 2.0, 2.5]
            }),
            'photometric_contrast_stretch': TransformConfig(params={
                'percentiles': [(1, 99), (2, 98), (5, 95)]
            }),
            'photometric_clahe': TransformConfig(params={
                'clip_limits': [2.0, 4.0, 8.0],
                'tile_grids': [(4, 4), (8, 8), (16, 16)]
            }),
            
            # B) Surveillance-like degradations
            'degradation_jpeg': TransformConfig(params={
                'qualities': [90, 70, 50, 30]
            }),
            'degradation_motion_blur': TransformConfig(params={
                'lengths': [5, 9, 15],
                'angles': [0, 30, 60, 90]
            }),
            'degradation_noise_poisson_gaussian': TransformConfig(params={
                'sigma_reads': [2, 5, 10],
                'gains': [0.8, 1.0, 1.2]
            }),
            'degradation_low_light': TransformConfig(params={
                'gains': [0.5, 0.7],
                'noise_sigmas': [0.01, 0.02]
            }),
            'degradation_downscale_upscale': TransformConfig(params={
                'factors': [0.75, 0.5]
            }),
            
            # C) Color / texture
            'color_desaturate': TransformConfig(params={}),
            'color_hsv_jitter': TransformConfig(params={
                'hue_deltas': [-5, 0, 5],
                'saturation_deltas': [-0.1, -0.05, 0.05, 0.1],
                'value_deltas': [-0.1, -0.05, 0.05, 0.1]
            }),
            'texture_nonlocal_means': TransformConfig(params={
                'h_values': [5, 10],
                'template_sizes': [7, 7],
                'search_sizes': [21, 21]
            }),
            'texture_bilateral_strong': TransformConfig(params={
                'diameters': [9, 15],
                'sigma_colors': [50, 75],
                'sigma_spaces': [9, 15]
            }),
            'texture_tv_denoise': TransformConfig(params={
                'weights': [0.05, 0.1]
            }),
            
            # D) Background / context
            'background_to_black': TransformConfig(),
            'background_to_mean': TransformConfig(),
            'background_to_blur': TransformConfig(params={
                'blur_sigmas': [10, 20, 30]
            }),
            'background_replace': TransformConfig(),
            'background_only': TransformConfig(),
            
            # E) Occlusion / missing data
            'occlusion_cutout': TransformConfig(params={
                'area_ratios': [0.1, 0.2, 0.3],
                'max_patches': [1, 2, 4],
                'fill_values': [0, 128]
            }),
            'occlusion_stripes': TransformConfig(params={
                'orientations': ['horizontal', 'vertical'],
                'bar_widths': [6, 12],
                'spacings': [24, 48]
            }),
            
            # F) Morphology
            'morphology_opening': TransformConfig(params={
                'kernel_shapes': ['ellipse', 'rectangle'],
                'kernel_sizes': [3, 5]
            }),
            'morphology_closing': TransformConfig(params={
                'kernel_shapes': ['ellipse', 'rectangle'],
                'kernel_sizes': [3, 5]
            }),
            'morphology_erode': TransformConfig(params={
                'kernel_shapes': ['ellipse', 'rectangle'],
                'kernel_sizes': [3, 5],
                'iterations': [1, 2]
            }),
            'morphology_dilate': TransformConfig(params={
                'kernel_shapes': ['ellipse', 'rectangle'],
                'kernel_sizes': [3, 5],
                'iterations': [1, 2]
            }),
            'morphology_fill_holes': TransformConfig()
        }
    
    def _get_default_output_config(self) -> Dict[str, Any]:
        """Get default output configuration."""
        return {
            'save_images': True,
            'save_csv': True,
            'image_format': 'PNG',
            'csv_filename': 'diagnostics_index.csv'
        }
    
    def _get_default_contact_sheet_config(self) -> Dict[str, Any]:
        """Get default contact sheet configuration."""
        return {
            'enabled': True,
            'thumbnail_size': 128,
            'grid_cols': 8,
            'max_images_per_sheet': 64,
            'filename_pattern': 'contact_sheet_{:02d}.png',
            'font_size': 10,
            'caption_height': 20,
            'padding': 5,
            'background_color': (240, 240, 240)
        }
    
    def apply_max_examples(self, max_examples: Optional[int]) -> None:
        """Apply global max examples limit to all transform families."""
        if max_examples is not None:
            self.max_examples_per_family = max_examples
            for transform_config in self.transforms.values():
                if transform_config.max_examples is None:
                    transform_config.max_examples = max_examples
    
    def get_enabled_transforms(self) -> Dict[str, TransformConfig]:
        """Get only enabled transform configurations."""
        return {name: config for name, config in self.transforms.items() if config.enabled}
    
    def disable_transform_family(self, family_name: str) -> None:
        """Disable a specific transform family."""
        if family_name in self.transforms:
            self.transforms[family_name].enabled = False
    
    def enable_transform_family(self, family_name: str) -> None:
        """Enable a specific transform family."""
        if family_name in self.transforms:
            self.transforms[family_name].enabled = True


def load_config(config_path: Optional[str] = None, max_examples_override: Optional[int] = None) -> DiagnosticsConfig:
    """
    Load configuration from YAML file or use defaults.
    
    Args:
        config_path: Path to YAML configuration file (optional)
        max_examples_override: Override max examples per family
        
    Returns:
        DiagnosticsConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config = DiagnosticsConfig()
    
    # Load YAML configuration if provided
    if config_path is not None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Apply YAML overrides
            _apply_yaml_config(config, yaml_config)
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML configuration: {e}")
    
    # Apply max examples override
    if max_examples_override is not None:
        config.apply_max_examples(max_examples_override)
    
    return config


def _apply_yaml_config(config: DiagnosticsConfig, yaml_config: Dict[str, Any]) -> None:
    """Apply YAML configuration to config object."""
    
    # Apply transform configurations
    if 'transforms' in yaml_config:
        for transform_name, transform_data in yaml_config['transforms'].items():
            if transform_name in config.transforms:
                if 'enabled' in transform_data:
                    config.transforms[transform_name].enabled = transform_data['enabled']
                
                if 'max_examples' in transform_data:
                    config.transforms[transform_name].max_examples = transform_data['max_examples']
                
                if 'params' in transform_data:
                    config.transforms[transform_name].params.update(transform_data['params'])
    
    # Apply output configuration
    if 'output' in yaml_config:
        config.output.update(yaml_config['output'])
    
    # Apply contact sheet configuration
    if 'contact_sheet' in yaml_config:
        config.contact_sheet.update(yaml_config['contact_sheet'])
    
    # Apply global max examples
    if 'max_examples_per_family' in yaml_config:
        config.apply_max_examples(yaml_config['max_examples_per_family'])


def save_example_config(output_path: str) -> None:
    """
    Save an example YAML configuration file.
    
    Args:
        output_path: Path to save example config
    """
    example_config = {
        'max_examples_per_family': 5,
        'transforms': {
            'frequency_gaussian_blur': {
                'enabled': True,
                'max_examples': 3,
                'params': {
                    'kernel_sizes': [5, 9, 15],
                    'sigmas': [1.0, 2.0, 3.0]
                }
            },
            'degradation_jpeg': {
                'enabled': True,
                'params': {
                    'qualities': [90, 50, 30]
                }
            },
            'background_to_black': {
                'enabled': False
            }
        },
        'output': {
            'save_images': True,
            'save_csv': True,
            'image_format': 'PNG'
        },
        'contact_sheet': {
            'enabled': True,
            'thumbnail_size': 128,
            'grid_cols': 8,
            'max_images_per_sheet': 64
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False, indent=2)


def get_transform_families() -> List[str]:
    """Get list of all available transform family names."""
    config = DiagnosticsConfig()
    return list(config.transforms.keys())


def get_family_from_transform_name(transform_name: str) -> str:
    """Extract family name from full transform name."""
    parts = transform_name.split('_', 1)
    if len(parts) >= 2:
        return parts[0]
    return 'unknown'


if __name__ == "__main__":
    # Example usage and testing
    config = load_config()
    print(f"Loaded {len(config.transforms)} transform configurations")
    print(f"Enabled transforms: {len(config.get_enabled_transforms())}")
    
    # Save example configuration
    save_example_config("example_config.yaml")
    print("Saved example configuration to example_config.yaml")