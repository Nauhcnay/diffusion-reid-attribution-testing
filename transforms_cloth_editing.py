"""
Cloth editing transforms using Qwen-Image for ReID diagnostics.

This module provides clothing transformation capabilities using Qwen-Image
with DiffSynth-Studio for memory optimization.
"""

from typing import Dict, List, Tuple, Any, Optional
import random
import warnings

import numpy as np
import torch
from PIL import Image
import cv2

try:
    from diffusers import QwenImageEditPipeline
    from transformers import AutoProcessor, AutoModelForCausalLM
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False

try:
    import diffsynth
    DIFFSYNTH_AVAILABLE = True
except ImportError:
    DIFFSYNTH_AVAILABLE = False


# Cloth candidate pool for various clothing types
CLOTH_CANDIDATES = {
    'tops': [
        'white t-shirt', 'black t-shirt', 'red sweater', 'blue hoodie', 'green jacket',
        'striped shirt', 'button-down shirt', 'tank top', 'polo shirt', 'cardigan',
        'blazer', 'leather jacket', 'denim jacket', 'winter coat', 'vest'
    ],
    'bottoms': [
        'blue jeans', 'black pants', 'khaki trousers', 'shorts', 'skirt',
        'dress pants', 'cargo pants', 'leggings', 'sweatpants', 'chinos',
        'denim skirt', 'pencil skirt', 'athletic shorts', 'formal trousers'
    ],
    'dresses': [
        'summer dress', 'cocktail dress', 'maxi dress', 'casual dress', 'formal dress',
        'sundress', 'wrap dress', 'shift dress', 'A-line dress', 'bodycon dress'
    ],
    'outerwear': [
        'trench coat', 'puffer jacket', 'windbreaker', 'raincoat', 'overcoat',
        'bomber jacket', 'pea coat', 'parka', 'cape', 'shawl'
    ],
    'styles': [
        'casual style', 'formal style', 'business casual', 'sporty style', 'bohemian style',
        'vintage style', 'modern style', 'elegant style', 'street style', 'minimalist style'
    ]
}


class ClothEditingPipeline:
    """Pipeline for cloth editing using Qwen-Image with memory optimization."""

    def __init__(self, use_vram_management: bool = True, device: str = "cuda"):
        self.device = device
        self.use_vram_management = use_vram_management
        self.editing_pipeline = None
        self.caption_model = None
        self.caption_processor = None
        self.initialized = False

    def _initialize_models(self):
        """Lazy initialization of models to save memory."""
        if self.initialized:
            return True

        if not QWEN_AVAILABLE:
            print("⚠️  Qwen-Image not available. Install with: pip install git+https://github.com/huggingface/diffusers")
            return False

        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available - cloth editing requires GPU")
            return False

        try:
            print("🔄 Initializing Qwen-Image editing pipeline with memory optimizations...")

            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 Cleared CUDA cache")

            # Initialize with maximum memory optimizations
            self.editing_pipeline = QwenImageEditPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )

            # Enable memory efficient attention if available
            if hasattr(self.editing_pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    self.editing_pipeline.enable_xformers_memory_efficient_attention()
                    print("✅ XFormers memory efficient attention enabled")
                except:
                    print("ℹ️  XFormers not available")

            # Enable sequential CPU offload for better memory management
            if hasattr(self.editing_pipeline, 'enable_sequential_cpu_offload'):
                self.editing_pipeline.enable_sequential_cpu_offload()
                print("✅ Sequential CPU offload enabled")
            elif hasattr(self.editing_pipeline, 'enable_model_cpu_offload'):
                # Fallback to model offload
                self.editing_pipeline.enable_model_cpu_offload()
                print("✅ Model CPU offload enabled")
            else:
                # Fallback: move to device
                self.editing_pipeline.to(self.device)
                print("ℹ️  Pipeline moved to GPU")

            # Enable VAE slicing and tiling for memory efficiency
            if hasattr(self.editing_pipeline, 'enable_vae_slicing'):
                self.editing_pipeline.enable_vae_slicing()
                print("✅ VAE slicing enabled")

            if hasattr(self.editing_pipeline, 'enable_vae_tiling'):
                self.editing_pipeline.enable_vae_tiling()
                print("✅ VAE tiling enabled")

            print("✅ Qwen-Image pipeline loaded with memory optimizations")

            # Initialize captioning model for cloth description
            try:
                print("🔄 Loading image captioning model...")
                self.caption_processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
                self.caption_model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/git-base-coco",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                ).to(self.device)
                print("✅ Image captioning model loaded")
            except Exception as e:
                print(f"⚠️  Captioning model failed to load: {e}")
                self.caption_processor = None
                self.caption_model = None

            self.initialized = True
            return True

        except Exception as e:
            print(f"❌ Failed to initialize cloth editing pipeline: {e}")
            if "out of memory" in str(e).lower():
                print("💡 Try reducing image size or closing other GPU applications")
            elif "connection" in str(e).lower() or "timeout" in str(e).lower():
                print("💡 Check internet connection - models need to be downloaded")
            return False

    def _extract_cloth_description(self, image: np.ndarray) -> str:
        """Extract cloth description from image using captioning model."""
        if self.caption_model is None or self.caption_processor is None:
            return "person wearing casual clothing"

        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Generate caption focusing on clothing
            prompt = "A photo of a person wearing"
            inputs = self.caption_processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.caption_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7
                )

            caption = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # Extract clothing-related parts
            clothing_desc = caption.replace(prompt, "").strip()

            return clothing_desc if clothing_desc else "casual clothing"

        except Exception as e:
            print(f"⚠️  Cloth description extraction failed: {e}")
            return "person wearing casual clothing"

    def _select_cloth_candidates(self, current_description: str, num_candidates: int = 4) -> List[str]:
        """Select cloth candidates based on current description."""
        candidates = []

        # Analyze current description to avoid similar items
        current_lower = current_description.lower()

        # Select from different categories
        for category, items in CLOTH_CANDIDATES.items():
            # Filter out similar items
            filtered_items = [
                item for item in items
                if not any(word in current_lower for word in item.split())
            ]
            if filtered_items:
                candidates.extend(random.sample(
                    filtered_items,
                    min(2, len(filtered_items))
                ))

        # Shuffle and select final candidates
        random.shuffle(candidates)
        return candidates[:num_candidates]

    def edit_clothing(self, image: np.ndarray, seed: Optional[int] = None) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Edit clothing in the image with multiple candidate options.

        Args:
            image: Input RGB image as numpy array
            seed: Random seed for reproducible results

        Returns:
            List of (edited_image, params) tuples
        """
        if not self._initialize_models():
            return []

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image)

            # Extract current clothing description
            current_description = self._extract_cloth_description(image)
            print(f"🔍 Current clothing: {current_description}")

            # Select cloth candidates
            cloth_candidates = self._select_cloth_candidates(current_description, num_candidates=4)
            print(f"🎯 Cloth candidates: {cloth_candidates}")

            results = []

            for i, cloth_item in enumerate(cloth_candidates):
                try:
                    # Create detailed editing prompt for better results
                    prompt = f"Change the person's clothing to {cloth_item}, keeping the person's face, pose, and background unchanged. Make the clothing change look natural and realistic."

                    # Apply cloth editing using correct Qwen-Image-Edit API
                    generator = torch.Generator(device="cpu")  # Generator should be on CPU
                    if seed is not None:
                        generator.manual_seed(seed + i)

                    with torch.no_grad():
                        # Clear cache before inference
                        torch.cuda.empty_cache()

                        # Use official Qwen-Image-Edit API parameters
                        inputs = {
                            "image": pil_image,
                            "prompt": prompt,
                            "generator": generator,
                            "true_cfg_scale": 4.0,  # Correct parameter name
                            "num_inference_steps": 50  # Higher quality with more steps
                        }

                        output = self.editing_pipeline(**inputs)
                        edited_image = output.images[0]

                        # Clear cache after inference
                        torch.cuda.empty_cache()

                    # Convert back to numpy array
                    edited_array = np.array(edited_image)

                    params = {
                        'original_description': current_description,
                        'target_clothing': cloth_item,
                        'prompt': prompt,
                        'candidate_index': i,
                        'cfg_scale': 4.0,
                        'seed': seed + i if seed else None
                    }

                    results.append((edited_array, params))
                    print(f"✅ Generated clothing variant {i+1}: {cloth_item}")

                except Exception as e:
                    print(f"⚠️  Failed to generate variant {i+1} ({cloth_item}): {e}")
                    continue

            return results

        except Exception as e:
            print(f"❌ Cloth editing failed: {e}")
            return []

    def cleanup(self):
        """Clean up models to free memory."""
        if self.editing_pipeline is not None:
            del self.editing_pipeline
            self.editing_pipeline = None
        if self.caption_model is not None:
            del self.caption_model
            self.caption_model = None
        if self.caption_processor is not None:
            del self.caption_processor
            self.caption_processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.initialized = False


# Global pipeline instance for reuse
_global_pipeline = None


def get_cloth_editing_pipeline() -> ClothEditingPipeline:
    """Get or create global cloth editing pipeline."""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = ClothEditingPipeline()
    return _global_pipeline


def apply_cloth_editing(image: np.ndarray, num_variants: int = 3, seed: Optional[int] = None) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Apply cloth editing transform with multiple variants.

    Args:
        image: Input RGB image (uint8)
        num_variants: Number of clothing variants to generate
        seed: Random seed for reproducible results

    Returns:
        List of (edited_image, params) tuples
    """
    pipeline = get_cloth_editing_pipeline()
    results = pipeline.edit_clothing(image, seed=seed)

    # Limit to requested number of variants
    return results[:num_variants]


# Transform registry for cloth editing
CLOTH_EDITING_TRANSFORMS = {
    'cloth_edit_casual': {
        'function': lambda image, seed=None: apply_cloth_editing(image, num_variants=1, seed=seed),
        'param_combinations': lambda params: [{'seed': seed} for seed in params.get('seeds', [42, 123, 456])],
        'requires_gpu': True,
        'family': 'cloth_editing'
    },
    'cloth_edit_multiple': {
        'function': lambda image, num_variants=3, seed=None: apply_cloth_editing(image, num_variants, seed),
        'param_combinations': lambda params: [
            {'num_variants': nv, 'seed': seed}
            for nv in params.get('num_variants', [2, 3])
            for seed in params.get('seeds', [42, 123])
        ],
        'requires_gpu': True,
        'family': 'cloth_editing'
    }
}


def get_cloth_editing_transforms() -> Dict[str, Any]:
    """Get all cloth editing transforms."""
    return CLOTH_EDITING_TRANSFORMS


def apply_cloth_editing_transform(transform_name: str, image: np.ndarray, params: Dict[str, Any]) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Apply a cloth editing transform by name with given parameters.

    Args:
        transform_name: Name of transform to apply
        image: Input RGB image (uint8)
        params: Parameters for the transform

    Returns:
        List of (transformed_image, actual_parameters) tuples, or empty list if transform failed
    """
    if transform_name not in CLOTH_EDITING_TRANSFORMS:
        return []

    if not QWEN_AVAILABLE:
        print("⚠️  Qwen-Image not available, skipping cloth editing")
        return []

    transform_info = CLOTH_EDITING_TRANSFORMS[transform_name]
    transform_func = transform_info['function']

    try:
        if transform_name == 'cloth_edit_multiple':
            return transform_func(image, **params)
        else:
            results = transform_func(image, **params)
            # Ensure we return a list of tuples
            if isinstance(results, list):
                return results
            else:
                return [results] if results else []
    except Exception as e:
        print(f"❌ Cloth editing transform failed: {e}")
        return []


if __name__ == "__main__":
    # Test cloth editing pipeline
    print("🧪 Testing Cloth Editing Pipeline")

    # Create test image
    test_image = np.random.randint(0, 256, (512, 256, 3), dtype=np.uint8)

    if QWEN_AVAILABLE:
        try:
            results = apply_cloth_editing(test_image, num_variants=2, seed=42)
            print(f"✅ Generated {len(results)} clothing variants")
            for i, (edited_img, params) in enumerate(results):
                print(f"   Variant {i+1}: {params['target_clothing']}")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    else:
        print("⚠️  Qwen-Image not available for testing")

    print(f"Available transforms: {list(CLOTH_EDITING_TRANSFORMS.keys())}")