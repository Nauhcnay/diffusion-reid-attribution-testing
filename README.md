# ReID Diagnostics Tool

A production-quality CLI tool that runs a battery of image-domain diagnostics for person ReID attribution. This tool applies classical, deterministic image transforms and generates structured analysis of the results.

## Features

### Transform Categories (Category-1 diagnostics only — pure image-domain)

**A) Frequency / contrast transforms**
- Low-pass smoothing: Gaussian, Bilateral, Median filters
- High-pass / edge emphasis: Laplacian, Sobel, Unsharp masking  
- Band-pass / notch filters via FFT masks
- Photometric: Gamma correction, Global contrast stretch, CLAHE

**B) "Surveillance-like" degradations (domain-consistent)**
- JPEG compression sweep (quality ∈ {90, 70, 50, 30})
- Motion blur PSF (length ∈ {5, 9, 15}, angle ∈ {0°, 30°, 60°, 90°})
- Sensor noise: Poisson-Gaussian noise
- Low-light gain + clipping + noise
- Downscale→Upscale (factors ∈ {0.75, 0.5} with bicubic)

**C) Color / texture (lightweight, domain-safe)**
- Desaturate (grayscale conversion)
- HSV jitter (ΔH ∈ {±5°}, ΔS ∈ {±0.05, ±0.1}, ΔV ∈ {±0.05, ±0.1})
- Texture suppression: Non-Local Means, Bilateral strong, TV-denoise

**D) Background / context**
- Foreground keep, background to black / mean color / heavy Gaussian blur
- "Background only" control (foreground zeroed; expect terrible retrieval)
- Background replacement with provided bank images

**E) Occlusion / missing data**
- Random Cutout rectangles (area ratio ∈ {0.1, 0.2, 0.3}, up to 4 patches)
- Stripe occlusions (horizontal or vertical bars)

**F) Morphology on foreground mask**
- Opening/Closing (kernel shapes: ellipse/square; sizes ∈ {3,5})
- Light Erode/Dilate (iterations ∈ {1,2})
- Fill holes before compositing

**G) Cloth editing transforms (Optional)**
- AI-powered clothing modification using Qwen-Image
- Semantic cloth type analysis and substitution
- Multiple clothing variant generation for robustness testing

## Installation

### Basic Installation

1. Clone this repository
2. Install core dependencies:
```bash
pip install -r requirements.txt
```

### Advanced Segmentation Setup (Optional)

For state-of-the-art foreground segmentation using SAM2 or SAM:

```bash
python setup_advanced_segmentation.py
```

This will install PyTorch, SAM2/SAM packages, and download model checkpoints.

### Cloth Editing Setup (Optional)

For AI-powered clothing modification using Qwen-Image:

```bash
python setup_cloth_editing.py
```

**Requirements:**
- NVIDIA GPU with 8GB+ VRAM (tested with RTX 4090)
- CUDA-compatible PyTorch
- ~20GB of model downloads

**Quick requirements check:**
```bash
python check_cloth_editing_requirements.py
```

## Usage

Basic usage:
```bash
python reid_diagnostics.py --input path/to/image.jpg --outdir results/
```

Full command line options:
```bash
python reid_diagnostics.py \
    --input path/to/image.jpg \
    --outdir results/ \
    [--mask path/to/foreground_mask.png] \
    [--bg_bank path/to/background/images/] \
    [--config config.yaml] \
    [--seed 123] \
    [--max_examples_per_family N] \
    [--verbose]
```

### Arguments

- `--input, -i`: Path to input RGB image (required)
- `--outdir, -o`: Output directory for results (required)  
- `--mask, -m`: Path to foreground mask (binary PNG) (optional)
- `--bg_bank`: Directory of background images for replacement (optional)
- `--config, -c`: YAML configuration file to override defaults (optional)
- `--seed, -s`: Random seed for deterministic results (default: 123)
- `--max_examples_per_family, -n`: Maximum number of examples per transform family (optional)
- `--verbose, -v`: Enable verbose output (optional)
- `--segmentation`: Segmentation method for automatic mask generation (choices: auto, sam2, sam, grabcut, basic; default: auto)
- `--no_advanced_segmentation`: Disable advanced segmentation, use only basic methods (optional)
- `--enable-cloth-editing`: Enable AI-powered cloth editing transforms (requires GPU) (optional)

## Configuration

You can customize the transforms and parameters using a YAML configuration file:

```yaml
max_examples_per_family: 5

transforms:
  frequency_gaussian_blur:
    enabled: true
    max_examples: 3
    params:
      kernel_sizes: [5, 9, 15]
      sigmas: [1.0, 2.0, 3.0]
  
  degradation_jpeg:
    enabled: true
    params:
      qualities: [90, 50, 30]
  
  background_to_black:
    enabled: false

output:
  save_images: true
  save_csv: true
  image_format: "PNG"

contact_sheet:
  enabled: true
  thumbnail_size: 128
  grid_cols: 8
  max_images_per_sheet: 64
```

Generate an example configuration:
```bash
python -c "from config import save_example_config; save_example_config('example_config.yaml')"
```

## Advanced Segmentation

The tool supports multiple segmentation methods for automatic foreground mask generation:

### Segmentation Methods

1. **SAM2 (Segment Anything v2)** - State-of-the-art segmentation
   ```bash
   python reid_diagnostics.py --input image.jpg --outdir results/ --segmentation sam2
   ```

2. **SAM (original Segment Anything)** - High-quality segmentation  
   ```bash
   python reid_diagnostics.py --input image.jpg --outdir results/ --segmentation sam
   ```

3. **GrabCut** - Traditional computer vision method
   ```bash
   python reid_diagnostics.py --input image.jpg --outdir results/ --segmentation grabcut
   ```

4. **Auto mode** (default) - Tries SAM2 → SAM → GrabCut in order
   ```bash
   python reid_diagnostics.py --input image.jpg --outdir results/ --segmentation auto
   ```

5. **Basic methods only** - Disables SAM2/SAM, uses only GrabCut/thresholding
   ```bash
   python reid_diagnostics.py --input image.jpg --outdir results/ --no_advanced_segmentation
   ```

## Cloth Editing

The tool supports AI-powered clothing modification for robustness testing:

### Usage

```bash
# Enable cloth editing (requires GPU setup)
python reid_diagnostics.py --input image.jpg --outdir results/ --enable-cloth-editing
```

### Features

- **Automatic cloth analysis**: Extracts current clothing description from image
- **Smart cloth selection**: Chooses diverse clothing types for maximum variation
- **Multiple variants**: Generates 2-4 different clothing options per transform
- **Memory optimized**: Uses CPU offloading and VRAM management for efficient inference

### Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **GPU** | NVIDIA 8GB+ VRAM | Tested with RTX 4090 (24GB) |
| **CUDA** | CUDA-compatible PyTorch | Auto-installed by setup script |
| **Models** | Qwen-Image + Captioning | ~20GB download (one-time) |
| **Memory** | 23GB+ VRAM during inference | Uses sequential CPU offload |

### Performance

- **Setup time**: 10-30 minutes (model download)
- **First run**: ~2 minutes (model loading)
- **Per variant**: 60-80 seconds
- **Cached runs**: Faster model loading

### Segmentation Methods Comparison

| Method | Requires | Model Size | Quality | Speed |
|--------|----------|------------|---------|--------|
| **SAM2** | PyTorch + SAM2 models | 38MB-2.4GB | Excellent | Medium |
| **SAM** | PyTorch + SAM models | 91MB-2.6GB | Excellent | Slow |
| **GrabCut** | OpenCV only | - | Good | Fast |
| **Basic** | OpenCV only | - | Fair | Very Fast |

### Setup Scripts Summary

| Feature | Script | Requirements | Install Time |
|---------|--------|--------------|--------------|
| **Basic ReID** | `pip install -r requirements.txt` | Python 3.10+ | 1-2 minutes |
| **Advanced Segmentation** | `python setup_advanced_segmentation.py` | PyTorch | 5-10 minutes |
| **Cloth Editing** | `python setup_cloth_editing.py` | NVIDIA GPU 8GB+ | 10-30 minutes |

### Benefits of Advanced Segmentation

- **Higher accuracy**: SAM2/SAM provide much better person segmentation
- **No manual tuning**: Works automatically without parameter adjustment  
- **Handles complex scenes**: Performs well with cluttered backgrounds
- **Semantic understanding**: Recognizes person vs. other objects
- **Graceful fallback**: Automatically falls back to simpler methods if models unavailable

## Outputs

The tool creates the following outputs in the specified directory:

### 1. Transformed Images
- `images/original.png`: Original input image
- `images/<family>/<transform>__<params>.png`: Transformed images organized by family

### 2. Data Files
- `diagnostics_index.csv`: Complete index with metadata and metrics for all images
- `summary_report.txt`: Human-readable summary of results

### 3. Visualizations  
- `contact_sheet_01.png`, `contact_sheet_02.png`, etc.: Visual grids showing all variants
- `metrics_visualization.png`: Plots of key metrics across transforms

### 4. CSV Columns

The diagnostics index CSV contains:
- Basic info: `family`, `transform`, `param_key`, `params_json`, `input_path`, `output_path`
- Image info: `width`, `height`, `error`, `processing_time`
- Metrics (prefixed with `metrics_`): 
  - `edge_density`: Fraction of Canny edge pixels
  - `gradient_energy`: Mean Sobel magnitude  
  - `intensity_*`: Histogram statistics (mean, std, skewness, entropy)
  - `*_mean`, `*_std`: Per-channel color statistics
  - `contrast`, `sharpness`, `blur_metric`, `noise_metric`: Image quality measures
  - `jpeg_size_q90`: JPEG compressed size (texture complexity proxy)
  - `brisque_score`, `niqe_score`: Advanced quality metrics (if libraries available)

## Requirements

Core dependencies:
- numpy >= 1.21.0
- opencv-python >= 4.5.0  
- scikit-image >= 0.18.0
- pillow >= 8.0.0
- pandas >= 1.3.0
- tqdm >= 4.60.0
- matplotlib >= 3.3.0
- scipy >= 1.7.0
- pyyaml >= 5.4.0

Optional (for advanced metrics):
- piq >= 0.7.0 (for BRISQUE scores)
- imquality >= 1.2.0 (for NIQE scores)

## Testing

Run the included system test:
```bash
python test_system.py
```

This creates synthetic test data and runs the complete pipeline to verify functionality.

## Architecture

The codebase is organized into focused modules:

- `reid_diagnostics.py`: CLI entry point
- `config.py`: Configuration management and parameter grids
- `io_utils.py`: Image I/O, masking, and utility functions
- `transforms_*.py`: Transform implementations organized by category
- `metrics.py`: Image quality metric computations  
- `visualization.py`: Contact sheet and plot generation
- `runner.py`: Main orchestration and pipeline coordination

Each transform is implemented as a pure function returning `(result_image, params_dict)`, making the system modular and testable.

## Determinism and Performance

- All operations use fixed random seeds for reproducible results
- Vectorized NumPy/OpenCV operations avoid per-pixel Python loops
- Internal processing uses float32 in [0,1]; I/O converts to uint8 [0,255] safely
- Background operations gracefully skip when masks unavailable
- Failed transforms are logged but don't stop processing

## License

This code is provided as-is for research and educational purposes.