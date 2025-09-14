# Cloth Editing Setup Guide

This guide explains how to set up cloth editing functionality in the ReID diagnostics tool using Qwen-Image and DiffSynth-Studio.

## Prerequisites

- **NVIDIA GPU with at least 8GB VRAM** (recommended: 12GB+)
- **CUDA-compatible PyTorch installation**
- Python 3.10+
- Stable internet connection (for initial model downloads)

## Quick Requirements Check

Before enabling cloth editing, run the requirements checker:

```bash
python check_cloth_editing_requirements.py
```

This will verify:
- ✅ Required Python packages installed
- ✅ CUDA GPU availability and VRAM
- ✅ Internet connection for model downloads

## Installation Steps

### 1. Install Required Dependencies

```bash
# Install latest transformers
pip install transformers>=4.51.3

# Install latest diffusers from source (required for Qwen-Image)
pip install git+https://github.com/huggingface/diffusers

# Install DiffSynth-Studio for VRAM optimization
pip install diffsynth

# Install image captioning model dependencies
pip install torch torchvision torchaudio
```

### 2. Download Models

The models will be downloaded automatically on first use:
- Qwen/Qwen-Image-Edit (for cloth editing)
- microsoft/git-base-coco (for cloth description extraction)

### 3. Enable Cloth Editing in Configuration

You can enable cloth editing in two ways:

#### Option A: YAML Configuration File

Create a `config.yaml` file:

```yaml
transforms:
  cloth_edit_casual:
    enabled: true
    params:
      seeds: [42, 123, 456]

  cloth_edit_multiple:
    enabled: true
    params:
      num_variants: [2, 3]
      seeds: [42, 123]
```

#### Option B: Command Line

```bash
python reid_diagnostics.py --input input/image.jpg --outdir output/ --enable-cloth-editing
```

## Usage Examples

### Basic Cloth Editing

```bash
# Enable cloth editing with default settings
python reid_diagnostics.py --input input/person.jpg --outdir output/ --enable-cloth-editing --verbose
```

### Advanced Usage with Custom Configuration

```bash
# Use custom config file with cloth editing enabled
python reid_diagnostics.py --input input/person.jpg --outdir output/ --config config.yaml --verbose
```

## Features

### Cloth Description Extraction
- Automatically analyzes the person's current clothing using image captioning
- Identifies clothing types (tops, bottoms, dresses, outerwear)
- Generates descriptive text for current attire

### Intelligent Cloth Selection
- Maintains a pool of diverse clothing options across categories:
  - **Tops**: t-shirts, sweaters, hoodies, jackets, shirts, blazers
  - **Bottoms**: jeans, pants, shorts, skirts, leggings
  - **Dresses**: casual, formal, summer, cocktail, maxi dresses
  - **Outerwear**: coats, jackets, windbreakers, parkas
  - **Styles**: casual, formal, business, sporty, bohemian, vintage

### Memory Optimization
- Uses DiffSynth-Studio for VRAM management
- Automatically enables memory optimization when available
- Supports inference on GPUs with limited VRAM

### Multiple Variants
- Generates 2-4 clothing variants per transform
- Each variant represents a different clothing style/type
- Maintains person identity while changing attire

## Output Structure

Cloth editing results are saved to:
```
output/
├── images/
│   └── cloth_editing/
│       ├── cloth_edit_casual__seed=42.png
│       ├── cloth_edit_multiple__num_variants=3_seed=42.png
│       └── ...
├── diagnostics_index.csv  # Includes cloth editing results
└── summary_report.txt     # Cloth editing statistics
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce `num_variants` in configuration
   - Ensure DiffSynth-Studio is installed for VRAM optimization
   - Close other GPU applications

2. **Model Download Issues**
   - Ensure stable internet connection
   - Check Hugging Face Hub access
   - May require Hugging Face authentication for some models

3. **Transform Not Found Error**
   - Verify cloth editing transforms are in configuration
   - Check that `transforms_cloth_editing.py` is imported correctly
   - Enable verbose mode for detailed error messages

### Performance Tips

- **First Run**: Model downloading may take 10-15 minutes
- **Subsequent Runs**: Models are cached locally for faster startup
- **VRAM Usage**: ~6-8GB for cloth editing with optimization
- **Processing Time**: ~30-60 seconds per cloth variant

### Limitations

- Requires GPU with sufficient VRAM
- Best results on clear, well-lit images of people
- May struggle with complex poses or unusual clothing
- Disabled by default due to computational requirements

## Development Notes

The cloth editing system consists of:
- `transforms_cloth_editing.py`: Main transform implementation
- Qwen-Image integration for semantic cloth editing
- Image captioning for current cloth analysis
- DiffSynth-Studio integration for memory optimization
- Intelligent cloth candidate selection system

For technical details, see the source code documentation.