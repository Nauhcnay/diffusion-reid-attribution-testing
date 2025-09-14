"""
Advanced segmentation utilities using Segment Anything v2 (SAM2) and other models.
Provides high-quality foreground mask generation for ReID diagnostics.
"""

import os
import warnings
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

# Graceful imports - these are optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    build_sam2 = None
    SAM2ImagePredictor = None

# Alternative: try segment-anything (original SAM)
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    sam_model_registry = None
    SamPredictor = None


class AdvancedSegmentationEngine:
    """Advanced segmentation engine with multiple model backends."""
    
    def __init__(self, model_type: str = "auto", device: str = "auto", verbose: bool = False):
        """
        Initialize segmentation engine.
        
        Args:
            model_type: "sam2", "sam", "grabcut", or "auto" (try in order)
            device: "cuda", "cpu", or "auto"
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.device = self._select_device(device)
        self.model = None
        self.predictor = None
        self.model_type = None
        
        # Initialize the best available model
        if model_type == "auto":
            self._init_auto()
        elif model_type == "sam2":
            self._init_sam2()
        elif model_type == "sam":
            self._init_sam()
        elif model_type == "grabcut":
            self.model_type = "grabcut"
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _select_device(self, device: str) -> str:
        """Select computation device."""
        if not TORCH_AVAILABLE:
            return "cpu"
        
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _init_auto(self):
        """Initialize best available model automatically."""
        if self._init_sam2():
            return
        if self._init_sam():
            return
        
        # Fallback to GrabCut
        self.model_type = "grabcut"
        if self.verbose:
            print("Using GrabCut fallback (no SAM models available)")
    
    def _init_sam2(self) -> bool:
        """Initialize SAM2 model."""
        if not SAM2_AVAILABLE or not TORCH_AVAILABLE:
            if self.verbose:
                print("SAM2 not available")
            return False
        
        try:
            # Try to find SAM2 model
            model_configs = [
                ("sam2_hiera_large.pt", "sam2_hiera_l.yaml"),
                ("sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml"),
                ("sam2_hiera_small.pt", "sam2_hiera_s.yaml"),
                ("sam2_hiera_tiny.pt", "sam2_hiera_t.yaml")
            ]
            
            for checkpoint_name, config_name in model_configs:
                try:
                    # Look for models in common locations
                    possible_paths = [
                        f"./checkpoints/{checkpoint_name}",
                        f"./models/{checkpoint_name}",
                        f"~/.cache/sam2/{checkpoint_name}",
                        checkpoint_name  # In current directory
                    ]
                    
                    checkpoint_path = None
                    for path in possible_paths:
                        expanded_path = os.path.expanduser(path)
                        if os.path.exists(expanded_path):
                            checkpoint_path = expanded_path
                            break
                    
                    if checkpoint_path is None:
                        continue
                    
                    # Build SAM2 model
                    sam2_model = build_sam2(config_name, checkpoint_path, device=self.device)
                    self.predictor = SAM2ImagePredictor(sam2_model)
                    self.model_type = "sam2"
                    
                    if self.verbose:
                        print(f"Initialized SAM2 with {checkpoint_name}")
                    return True
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to load SAM2 {checkpoint_name}: {e}")
                    continue
            
            if self.verbose:
                print("No SAM2 models found in common locations")
            return False
            
        except Exception as e:
            if self.verbose:
                print(f"SAM2 initialization failed: {e}")
            return False
    
    def _init_sam(self) -> bool:
        """Initialize original SAM model."""
        if not SAM_AVAILABLE or not TORCH_AVAILABLE:
            if self.verbose:
                print("Original SAM not available")
            return False
        
        try:
            # Try SAM model variants
            model_configs = [
                ("vit_h", "sam_vit_h_4b8939.pth"),
                ("vit_l", "sam_vit_l_0b3195.pth"),
                ("vit_b", "sam_vit_b_01ec64.pth")
            ]
            
            for model_type, checkpoint_name in model_configs:
                try:
                    # Look for models in common locations
                    possible_paths = [
                        f"./checkpoints/{checkpoint_name}",
                        f"./models/{checkpoint_name}",
                        f"~/.cache/sam/{checkpoint_name}",
                        checkpoint_name
                    ]
                    
                    checkpoint_path = None
                    for path in possible_paths:
                        expanded_path = os.path.expanduser(path)
                        if os.path.exists(expanded_path):
                            checkpoint_path = expanded_path
                            break
                    
                    if checkpoint_path is None:
                        continue
                    
                    # Build SAM model
                    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
                    sam.to(device=self.device)
                    self.predictor = SamPredictor(sam)
                    self.model_type = "sam"
                    
                    if self.verbose:
                        print(f"Initialized SAM with {checkpoint_name}")
                    return True
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to load SAM {checkpoint_name}: {e}")
                    continue
            
            if self.verbose:
                print("No SAM models found in common locations")
            return False
            
        except Exception as e:
            if self.verbose:
                print(f"SAM initialization failed: {e}")
            return False
    
    def segment_person(self, image: np.ndarray, method: str = "auto") -> Optional[np.ndarray]:
        """
        Segment person from image using the best available method.
        
        Args:
            image: RGB image as uint8 numpy array
            method: "auto", "center_prompt", "full_semantic", or "grabcut"
            
        Returns:
            Binary mask as bool array, or None if segmentation fails
        """
        if method == "auto":
            # Try methods in order of quality
            if self.model_type in ["sam2", "sam"]:
                # Try center prompt first (faster)
                mask = self._segment_with_center_prompt(image)
                if mask is not None:
                    return mask

                # Fall back to semantic segmentation
                mask = self._segment_semantic(image)
                if mask is not None:
                    return mask
            
            # Final fallback to GrabCut
            return self._segment_with_grabcut(image)
        
        elif method == "center_prompt":
            return self._segment_with_center_prompt(image)
        
        elif method == "full_semantic":
            return self._segment_semantic(image)
        
        elif method == "grabcut":
            return self._segment_with_grabcut(image)
        
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
    
    def _segment_with_center_prompt(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Segment using enhanced multi-point prompting strategy for person segmentation."""
        if self.predictor is None:
            return None

        try:
            # Set image for prediction
            if self.model_type == "sam2":
                self.predictor.set_image(image)
            else:  # original SAM
                self.predictor.set_image(image)

            h, w = image.shape[:2]

            # Enhanced prompting strategy with multiple positive points for person parts
            positive_points = []

            # Center torso area
            positive_points.append([w//2, h//2])

            # Head area (upper center)
            positive_points.append([w//2, h//4])

            # Upper body areas (chest/shoulders)
            positive_points.append([w//2 - w//8, h//3])  # Left shoulder area
            positive_points.append([w//2 + w//8, h//3])  # Right shoulder area

            # Lower body (hips/legs)
            positive_points.append([w//2, h*3//4])  # Hip area

            # Add negative prompts for background corners to improve segmentation
            negative_points = []
            # Corner points likely to be background
            margin = min(w, h) // 10  # 10% margin from edges
            negative_points.extend([
                [margin, margin],           # Top-left corner
                [w - margin, margin],       # Top-right corner
                [margin, h - margin],       # Bottom-left corner
                [w - margin, h - margin]    # Bottom-right corner
            ])

            # Combine positive and negative points
            all_points = positive_points + negative_points
            point_labels = [1] * len(positive_points) + [0] * len(negative_points)

            point_coords = np.array(all_points)
            point_labels = np.array(point_labels)

            if self.verbose:
                print(f"      Using {len(positive_points)} positive + {len(negative_points)} negative prompts")

            # Predict mask
            if self.model_type == "sam2":
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True
                )
            else:  # original SAM
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True
                )

            # Select best mask
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]

            if self.verbose:
                print(f"      Best mask score: {scores[best_mask_idx]:.3f}")

            # Validate mask quality
            if self._is_valid_person_mask(best_mask, image):
                # Enhance mask to include carried objects and accessories
                enhanced_mask = self._enhance_mask_with_carried_objects(best_mask, image)
                # Ensure mask is boolean type
                return enhanced_mask.astype(bool)

            return None
            
        except Exception as e:
            if self.verbose:
                print(f"Center prompt segmentation failed: {e}")
            return None
    
    def _segment_semantic(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Full semantic segmentation to find person objects."""
        if self.predictor is None:
            return None
        
        try:
            # Set image for prediction
            if self.model_type == "sam2":
                self.predictor.set_image(image)
            else:
                self.predictor.set_image(image)
            
            h, w = image.shape[:2]
            
            # Generate grid of points to sample
            grid_size = 32
            y_points = np.linspace(h//4, 3*h//4, grid_size//4)
            x_points = np.linspace(w//4, 3*w//4, grid_size//4)
            
            best_mask = None
            best_score = 0
            
            # Try multiple point prompts
            for y in y_points:
                for x in x_points:
                    try:
                        point_coords = np.array([[int(x), int(y)]])
                        point_labels = np.array([1])
                        
                        if self.model_type == "sam2":
                            masks, scores, _ = self.predictor.predict(
                                point_coords=point_coords,
                                point_labels=point_labels,
                                multimask_output=False
                            )
                        else:
                            masks, scores, _ = self.predictor.predict(
                                point_coords=point_coords,
                                point_labels=point_labels,
                                multimask_output=False
                            )
                        
                        mask = masks[0]
                        score = scores[0]
                        
                        # Check if this looks like a person
                        if self._is_valid_person_mask(mask, image) and score > best_score:
                            best_mask = mask
                            best_score = score

                    except Exception:
                        continue

            # Ensure mask is boolean type if found
            if best_mask is not None:
                # Enhance mask to include carried objects
                enhanced_mask = self._enhance_mask_with_carried_objects(best_mask, image)
                return enhanced_mask.astype(bool)
            return best_mask
            
        except Exception as e:
            if self.verbose:
                print(f"Semantic segmentation failed: {e}")
            return None
    
    def _segment_with_grabcut(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Fallback GrabCut segmentation."""
        try:
            h, w = image.shape[:2]
            
            # Create initial rectangle (10% border)
            border = min(h, w) // 10
            rect = (border, border, w - 2*border, h - 2*border)
            
            # Initialize masks
            mask = np.zeros((h, w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Convert to BGR for OpenCV
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Run GrabCut
            cv2.grabCut(bgr_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Extract foreground
            foreground_mask = np.where((mask == 1) | (mask == 3), True, False)
            
            return foreground_mask
            
        except Exception as e:
            if self.verbose:
                print(f"GrabCut segmentation failed: {e}")
            return None
    
    def _is_valid_person_mask(self, mask: np.ndarray, image: np.ndarray) -> bool:
        """Validate if mask looks like a person."""
        if mask is None:
            return False
        
        h, w = image.shape[:2]
        mask_area = np.sum(mask)
        total_area = h * w
        
        # Check mask area (person should be 5-80% of image)
        area_ratio = mask_area / total_area
        if area_ratio < 0.05 or area_ratio > 0.8:
            return False
        
        # Check aspect ratio of bounding box
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return False
        
        bbox_h = np.max(y_coords) - np.min(y_coords)
        bbox_w = np.max(x_coords) - np.min(x_coords)
        
        if bbox_h == 0 or bbox_w == 0:
            return False
        
        aspect_ratio = bbox_h / bbox_w
        
        # Person should be roughly vertical (aspect ratio > 1)
        if aspect_ratio < 0.8:
            return False
        
        # Check if mask is reasonably centered
        mask_center_y = np.mean(y_coords)
        mask_center_x = np.mean(x_coords)
        
        img_center_y = h / 2
        img_center_x = w / 2
        
        # Mask center shouldn't be too far from image center
        center_distance = np.sqrt((mask_center_x - img_center_x)**2 + (mask_center_y - img_center_y)**2)
        max_distance = min(h, w) * 0.4
        
        return center_distance <= max_distance

    def _enhance_mask_with_carried_objects(self, person_mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Enhance person mask to include carried objects, accessories, and attached items.

        Args:
            person_mask: Initial person segmentation mask
            image: Original RGB image

        Returns:
            Enhanced mask including carried objects
        """
        try:
            import cv2
            from scipy import ndimage

            # Ensure inputs are correct types
            enhanced_mask = person_mask.astype(bool)
            h, w = image.shape[:2]

            if self.verbose:
                original_pixels = np.sum(enhanced_mask)
                print(f"      Enhancing mask to include carried objects...")

            # Strategy 1: Morphological dilation to capture nearby objects
            # Create a kernel for moderate expansion around the person
            kernel_size = max(3, min(h, w) // 100)  # Adaptive kernel size
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            # Dilate the person mask to capture nearby objects
            dilated_mask = cv2.dilate(enhanced_mask.astype(np.uint8), kernel, iterations=2)

            # Strategy 2: Color and texture similarity expansion
            # Find regions similar to the person that might be carried objects
            person_region = image[enhanced_mask]
            if len(person_region) > 0:
                # Get color statistics of the person
                person_mean_color = np.mean(person_region, axis=0)
                person_std_color = np.std(person_region, axis=0) + 1e-6  # Avoid division by zero

                # Find pixels in dilated region that might belong to carried objects
                dilated_region = dilated_mask.astype(bool) & ~enhanced_mask
                dilated_coords = np.where(dilated_region)

                if len(dilated_coords[0]) > 0:
                    dilated_pixels = image[dilated_coords]

                    # Calculate color similarity (normalized distance)
                    color_diff = np.abs(dilated_pixels - person_mean_color)
                    normalized_diff = color_diff / person_std_color
                    color_similarity = np.exp(-np.mean(normalized_diff, axis=1))

                    # Include pixels that are reasonably similar to person colors
                    # This helps capture bags, accessories with similar tones
                    similarity_threshold = 0.3  # Adjustable threshold
                    similar_pixels = color_similarity > similarity_threshold

                    if np.any(similar_pixels):
                        similar_coords_y = dilated_coords[0][similar_pixels]
                        similar_coords_x = dilated_coords[1][similar_pixels]
                        enhanced_mask[similar_coords_y, similar_coords_x] = True

            # Strategy 3: Connected component analysis for attached objects
            # Find connected components in the dilated area that might be carried items
            dilated_region = dilated_mask.astype(bool) & ~enhanced_mask

            # Convert image to grayscale for better connected component analysis
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Apply adaptive threshold to find object boundaries
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Find connected components in the dilated region
            dilated_region_uint8 = dilated_region.astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(dilated_region_uint8)

            for label in range(1, num_labels):
                component_mask = labels == label
                component_size = np.sum(component_mask)

                # Include components that are reasonably sized (potential bags/accessories)
                min_size = max(20, h * w // 2000)  # At least 0.05% of image
                max_size = h * w // 20  # At most 5% of image

                if min_size <= component_size <= max_size:
                    # Check if component is close enough to person
                    component_coords = np.where(component_mask)
                    person_coords = np.where(enhanced_mask)

                    if len(person_coords[0]) > 0 and len(component_coords[0]) > 0:
                        # Calculate minimum distance from component to person
                        person_boundary = np.column_stack(person_coords)
                        component_boundary = np.column_stack(component_coords)

                        # Use a simplified distance calculation for efficiency
                        person_center = np.mean(person_boundary, axis=0)
                        component_center = np.mean(component_boundary, axis=0)
                        distance = np.linalg.norm(person_center - component_center)

                        # Include if component is close to person (potential carried object)
                        max_distance = min(h, w) * 0.15  # Within 15% of image dimension
                        if distance <= max_distance:
                            enhanced_mask[component_mask] = True

            # Strategy 4: Fill small holes that might disconnect carried objects
            # This helps connect bags that might be partially segmented
            enhanced_mask = ndimage.binary_fill_holes(enhanced_mask)

            if self.verbose:
                enhanced_pixels = np.sum(enhanced_mask)
                added_pixels = enhanced_pixels - original_pixels
                print(f"      Added {added_pixels} pixels for carried objects ({added_pixels/original_pixels*100:.1f}% increase)")

            return enhanced_mask

        except Exception as e:
            if self.verbose:
                print(f"      Warning: Failed to enhance mask with carried objects: {e}")
            return person_mask
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_type": self.model_type,
            "device": self.device,
            "torch_available": TORCH_AVAILABLE,
            "sam2_available": SAM2_AVAILABLE,
            "sam_available": SAM_AVAILABLE
        }


# Factory function for easy integration
def create_advanced_mask(image: np.ndarray, 
                        model_type: str = "auto",
                        method: str = "auto",
                        device: str = "auto",
                        verbose: bool = False) -> Optional[np.ndarray]:
    """
    Create foreground mask using advanced segmentation.
    
    Args:
        image: RGB image as uint8 numpy array
        model_type: "sam2", "sam", "grabcut", or "auto"
        method: "auto", "center_prompt", "full_semantic", or "grabcut"
        device: "cuda", "cpu", or "auto"
        verbose: Enable verbose logging
        
    Returns:
        Binary mask as bool array, or None if segmentation fails
    """
    try:
        engine = AdvancedSegmentationEngine(model_type=model_type, device=device, verbose=verbose)
        return engine.segment_person(image, method=method)
    except Exception as e:
        if verbose:
            print(f"Advanced segmentation failed: {e}")
        return None


# Integration function for backward compatibility
def get_advanced_foreground_mask(image: np.ndarray, 
                                provided_mask: Optional[np.ndarray] = None,
                                use_advanced: bool = True,
                                verbose: bool = False) -> Optional[np.ndarray]:
    """
    Get foreground mask with advanced segmentation support.
    
    Args:
        image: RGB image as uint8 numpy array
        provided_mask: Optional provided binary mask
        use_advanced: Whether to try advanced segmentation
        verbose: Enable verbose logging
        
    Returns:
        Binary mask as bool array, or None if no valid mask can be obtained
    """
    # Use provided mask if available and valid
    if provided_mask is not None:
        if provided_mask.shape[:2] == image.shape[:2]:
            return provided_mask
        else:
            if verbose:
                print("Provided mask dimensions don't match image")
    
    # Try advanced segmentation if requested
    if use_advanced:
        mask = create_advanced_mask(image, verbose=verbose)
        if mask is not None:
            return mask.astype(bool)
    
    # Fall back to original methods
    from io_utils import get_foreground_mask
    return get_foreground_mask(image, provided_mask)


if __name__ == "__main__":
    # Test advanced segmentation
    print("Testing Advanced Segmentation Engine")
    print("=" * 50)
    
    # Create synthetic test image
    test_image = np.random.randint(0, 256, (256, 128, 3), dtype=np.uint8)
    
    # Test engine initialization
    engine = AdvancedSegmentationEngine(verbose=True)
    print(f"Model info: {engine.get_model_info()}")
    
    # Test segmentation
    mask = engine.segment_person(test_image)
    if mask is not None:
        print(f"Segmentation successful: {np.sum(mask)} foreground pixels")
    else:
        print("Segmentation failed - using fallback methods")
    
    print("Advanced segmentation test completed.")