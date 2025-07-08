#!/usr/bin/env python3
"""
🎯 ULTIMATE SPRITE PROCESSING PIPELINE - WINDOWS OPTIMIZED
============================================================
Windows-optimized version with better fallback methods and improved quality.

Features:
- Advanced OpenCV background removal
- Improved pose detection without MediaPipe
- Better frame extraction algorithms
- Enhanced quality scoring
- Optimized for Windows systems
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import json
from PIL import Image, ImageEnhance, ImageFilter
from dataclasses import dataclass
from enum import Enum

# Import available libraries
try:
    import rembg
    REMBG_AVAILABLE = True
    print("✅ REMBG available")
except ImportError:
    REMBG_AVAILABLE = False
    print("⚠️ REMBG not available")

try:
    from transformers import pipeline
    HF_AVAILABLE = True
    print("✅ Hugging Face available")
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ Hugging Face not available")


class QualityPreset(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    PROFESSIONAL = "professional"
    ULTIMATE = "ultimate"


@dataclass
class ProcessingConfig:
    quality_preset: QualityPreset = QualityPreset.PROFESSIONAL
    device: str = "cpu"  # Windows default to CPU
    upscale_factor: int = 2
    output_formats: List[str] = None

    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ["frames_1x",
                                   "frames_2x", "gif_1x", "gif_2x"]


class WindowsBackgroundRemover:
    """Windows-optimized background removal"""

    def __init__(self):
        self.models = {}
        self._load_models()

    def _load_models(self):
        print("🔧 Loading Windows-optimized background removal...")

        # REMBG if available
        if REMBG_AVAILABLE:
            try:
                self.models['u2net'] = rembg.new_session('u2net')
                print("✅ U2Net loaded")
            except Exception as e:
                print(f"⚠️ U2Net failed: {e}")

        # OpenCV Background Subtractors
        try:
            self.models['mog2'] = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True)
            self.models['knn'] = cv2.createBackgroundSubtractorKNN(
                detectShadows=True)
            print("✅ OpenCV background subtractors loaded")
        except Exception as e:
            print(f"⚠️ OpenCV subtractors failed: {e}")

    def remove_background_advanced(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Advanced background removal for Windows"""
        print("🎭 Running Windows-optimized background removal...")

        # Try REMBG first if available
        if 'u2net' in self.models:
            try:
                image_pil = Image.fromarray(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                result = rembg.remove(image_pil, session=self.models['u2net'])
                result_np = np.array(result)

                if result_np.shape[2] == 4:  # Has alpha channel
                    alpha = result_np[:, :, 3]
                    rgb = result_np[:, :, :3]
                    confidence = self._calculate_quality(alpha)

                    if confidence > 0.4:  # Good quality
                        print(f"   ✓ REMBG U2Net: quality {confidence:.3f}")
                        return np.dstack([rgb, alpha]), confidence
            except Exception as e:
                print(f"   ⚠️ REMBG failed: {e}")

        # Fallback to advanced OpenCV method
        return self._opencv_advanced_removal(image)

    def _opencv_advanced_removal(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Advanced OpenCV-based background removal"""
        print("   🔄 Using advanced OpenCV background removal")

        # Method 1: Edge-based segmentation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Edge detection
        edges = cv2.Canny(filtered, 50, 150)

        # Morphological operations to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (likely the main subject)
            largest_contour = max(contours, key=cv2.contourArea)

            # Create mask
            mask = np.zeros(gray.shape, np.uint8)
            cv2.fillPoly(mask, [largest_contour], 255)

            # Improve mask with GrabCut
            mask = self._improve_mask_with_grabcut(image, mask)

            # Apply mask
            result = image.copy()
            alpha = mask.astype(np.uint8)

            confidence = self._calculate_quality(alpha)
            print(f"   ✓ OpenCV advanced: quality {confidence:.3f}")

            return np.dstack([result, alpha]), confidence

        # Fallback: simple thresholding
        return self._simple_threshold_removal(image)

    def _improve_mask_with_grabcut(self, image: np.ndarray, initial_mask: np.ndarray) -> np.ndarray:
        """Improve mask using GrabCut algorithm"""
        try:
            # Convert mask for GrabCut
            gc_mask = np.where((initial_mask == 0), 0, 1).astype('uint8')
            gc_mask[initial_mask == 255] = 1  # Foreground
            gc_mask[initial_mask == 0] = 0    # Background

            # Initialize background and foreground models
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            # Apply GrabCut
            cv2.grabCut(image, gc_mask, None, bgd_model,
                        fgd_model, 3, cv2.GC_INIT_WITH_MASK)

            # Extract foreground
            mask_out = np.where((gc_mask == 2) | (
                gc_mask == 0), 0, 255).astype('uint8')

            return mask_out

        except Exception as e:
            print(f"   ⚠️ GrabCut failed: {e}")
            return initial_mask

    def _simple_threshold_removal(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Simple threshold-based background removal"""
        print("   🔄 Using simple threshold removal")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Otsu's thresholding
        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        confidence = self._calculate_quality(mask)
        print(f"   ✓ Simple threshold: quality {confidence:.3f}")

        return np.dstack([image, mask]), confidence

    def _calculate_quality(self, mask: np.ndarray) -> float:
        """Calculate mask quality score"""
        if mask is None or mask.size == 0:
            return 0.0

        # Ensure mask is 2D
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        # Edge smoothness
        edges = cv2.Canny(mask, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size

        # Foreground ratio
        fg_ratio = np.sum(mask > 128) / mask.size

        # Avoid too much or too little foreground
        size_score = 1.0 - abs(0.3 - fg_ratio) / 0.3 if fg_ratio < 0.6 else 0.5
        edge_score = min(1.0, edge_ratio * 10)  # Reward good edges

        return (size_score * 0.6 + edge_score * 0.4)


class WindowsPoseAnalyzer:
    """Windows-compatible pose analysis without MediaPipe"""

    def __init__(self):
        self.initialized = True

    def analyze_pose_advanced(self, image: np.ndarray) -> Dict:
        """Advanced pose analysis using OpenCV"""
        print("🧠 Running Windows pose analysis...")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect potential head region (circular shapes)
        head_score = self._detect_head_region(gray)

        # Analyze body proportions
        body_score = self._analyze_body_proportions(gray)

        # Detect limb-like structures
        limb_score = self._detect_limbs(gray)

        # Motion potential based on asymmetry
        motion_score = self._calculate_motion_potential(gray)

        overall_confidence = (head_score + body_score + limb_score) / 3

        result = {
            "pose_detected": overall_confidence > 0.3,
            "confidence": overall_confidence,
            "head_score": head_score,
            "body_score": body_score,
            "limb_score": limb_score,
            "motion_potential": motion_score,
            "method": "windows_opencv"
        }

        print(f"   ✓ Pose analysis: confidence {overall_confidence:.3f}")
        return result

    def _detect_head_region(self, gray: np.ndarray) -> float:
        """Detect circular head-like regions"""
        # Focus on upper portion
        h, w = gray.shape
        head_region = gray[:h//3, :]

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            head_region, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=5, maxRadius=min(w, h)//4
        )

        if circles is not None and len(circles[0]) > 0:
            return min(1.0, len(circles[0]) * 0.5)

        return 0.1

    def _analyze_body_proportions(self, gray: np.ndarray) -> float:
        """Analyze if image has human-like proportions"""
        h, w = gray.shape
        aspect_ratio = h / w

        # Human figures typically have height > width
        if 1.2 <= aspect_ratio <= 3.0:
            return 0.8
        elif 1.0 <= aspect_ratio < 1.2:
            return 0.5
        else:
            return 0.2

    def _detect_limbs(self, gray: np.ndarray) -> float:
        """Detect limb-like elongated structures"""
        # Find contours
        contours, _ = cv2.findContours(
            gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.1

        # Analyze main contour
        main_contour = max(contours, key=cv2.contourArea)

        # Calculate contour complexity
        epsilon = 0.02 * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)

        # More complex shapes suggest limbs/details
        complexity = min(1.0, len(approx) / 15)

        return complexity

    def _calculate_motion_potential(self, gray: np.ndarray) -> float:
        """Calculate potential for motion based on asymmetry"""
        h, w = gray.shape

        # Check left-right symmetry
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]

        if left_half.shape != right_half.shape:
            return 0.7  # Different sizes suggest motion

        # Flip right half and compare
        right_flipped = np.fliplr(right_half)

        # Calculate difference
        diff = np.mean(np.abs(left_half.astype(
            float) - right_flipped.astype(float)))

        # More asymmetry = more motion potential
        asymmetry = min(1.0, diff / 100)

        return asymmetry


class WindowsFrameExtractor:
    """Advanced frame extraction for Windows"""

    def __init__(self):
        self.min_frame_area = 100

    def extract_frames_smart(self, image_rgba: np.ndarray, mask: np.ndarray) -> List[Dict]:
        """Smart frame extraction using advanced algorithms"""
        print("✂️ Running smart frame extraction...")

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)

        frames = []

        for i in range(1, num_labels):  # Skip background label 0
            area = stats[i, cv2.CC_STAT_AREA]

            if area < self.min_frame_area:
                continue

            # Extract bounding box
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Add padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image_rgba.shape[1] - x, w + 2*padding)
            h = min(image_rgba.shape[0] - y, h + 2*padding)

            # Extract frame
            frame_region = image_rgba[y:y+h, x:x+w]
            frame_mask = labels[y:y+h, x:x+w] == i

            # Create clean RGBA frame
            frame_rgba = frame_region.copy()
            if len(frame_rgba.shape) == 3:
                if frame_rgba.shape[2] == 3:
                    # Add alpha channel
                    alpha = np.where(frame_mask, 255, 0).astype(np.uint8)
                    frame_rgba = np.dstack([frame_rgba, alpha])
                else:
                    # Update existing alpha
                    frame_rgba[:, :, 3] = np.where(frame_mask, 255, 0)

            # Calculate quality
            quality = self._calculate_frame_quality(frame_rgba, area, w/h)

            frame_data = {
                "image": frame_rgba,
                "bbox": (x, y, w, h),
                "area": area,
                "quality": quality,
                "center": centroids[i]
            }

            frames.append(frame_data)

        # Sort by quality (extract overall score from quality dict)
        frames.sort(key=lambda f: f["quality"]["overall"] if isinstance(
            f["quality"], dict) else f["quality"], reverse=True)

        print(f"   ✓ Extracted {len(frames)} frames")
        return frames

    def _calculate_frame_quality(self, frame_rgba: np.ndarray, area: int, aspect_ratio: float) -> Dict:
        """Calculate comprehensive frame quality"""
        h, w = frame_rgba.shape[:2]

        # Size score
        size_score = min(1.0, area / 10000)  # Prefer larger frames

        # Aspect ratio score
        ideal_ratios = [1.0, 1.5, 0.67]  # Square, tall, wide
        ratio_scores = [1.0 / (1.0 + abs(aspect_ratio - ratio))
                        for ratio in ideal_ratios]
        aspect_score = max(ratio_scores)

        # Detail score (edge density)
        if len(frame_rgba.shape) > 2:
            gray = cv2.cvtColor(frame_rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            detail_score = min(1.0, np.sum(edges > 0) / (w * h) * 100)
        else:
            detail_score = 0.5

        overall_quality = (size_score * 0.4 + aspect_score *
                           0.3 + detail_score * 0.3)

        return {
            "overall": overall_quality,
            "size": size_score,
            "aspect": aspect_score,
            "detail": detail_score
        }


class WindowsSpriteProcessor:
    """Main Windows-optimized sprite processor"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.bg_remover = WindowsBackgroundRemover()
        self.pose_analyzer = WindowsPoseAnalyzer()
        self.frame_extractor = WindowsFrameExtractor()

    def process_sprite_optimized(self, image_path: Path) -> Dict:
        """Process sprite with Windows optimizations"""
        start_time = time.time()

        print(f"\n🎯 WINDOWS OPTIMIZED PROCESSING: {image_path.name}")
        print("=" * 60)

        # Load image
        print("📁 Loading image...")
        image = cv2.imread(str(image_path))
        if image is None:
            return {"success": False, "error": "Could not load image"}

        h, w = image.shape[:2]
        print(f"   ✓ Loaded: {w}x{h} pixels")

        # Stage 1: Background removal
        image_rgba, bg_confidence = self.bg_remover.remove_background_advanced(
            image)

        # Stage 2: Pose analysis
        pose_analysis = self.pose_analyzer.analyze_pose_advanced(image)

        # Stage 3: Frame extraction
        mask = image_rgba[:, :, 3] if image_rgba.shape[2] == 4 else np.ones(
            (h, w), dtype=np.uint8) * 255
        frames = self.frame_extractor.extract_frames_smart(image_rgba, mask)

        if not frames:
            # Fallback: use whole image as single frame
            frames = [{
                "image": image_rgba,
                "bbox": (0, 0, w, h),
                "area": w * h,
                "quality": {"overall": bg_confidence},
                "center": (w//2, h//2)
            }]

        # Stage 4: Generate outputs
        outputs = self._generate_outputs_optimized(frames, image_path)

        processing_time = time.time() - start_time

        # Calculate overall quality
        overall_quality = (
            bg_confidence * 0.4 +
            pose_analysis["confidence"] * 0.3 +
            (frames[0]["quality"]["overall"] if frames else 0) * 0.3
        )

        result = {
            "success": True,
            "processing_time": processing_time,
            "quality_metrics": {
                "overall_quality": overall_quality,
                "background_confidence": bg_confidence,
                "pose_confidence": pose_analysis["confidence"],
                "frames_extracted": len(frames)
            },
            "outputs": outputs,
            "pose_analysis": pose_analysis
        }

        print(f"\n✅ PROCESSING COMPLETE!")
        print(f"   ⏱️ Time: {processing_time:.2f}s")
        print(f"   🎯 Quality: {overall_quality:.3f}")
        print(f"   📦 Outputs: {len(outputs)} files generated")

        return result

    def _generate_outputs_optimized(self, frames: List[Dict], input_path: Path) -> Dict:
        """Generate optimized outputs"""
        outputs = {}
        base_name = input_path.stem
        output_dir = Path("output/ultimate_sprites")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Output 1: Single frames (1x)
        if "frames_1x" in self.config.output_formats:
            frame_dir = output_dir / f"{base_name}_frames_1x"
            frame_dir.mkdir(exist_ok=True)

            for i, frame_data in enumerate(frames):
                frame_path = frame_dir / f"frame_{i:03d}.png"
                frame_rgba = frame_data["image"]

                # Ensure zero background opacity
                if frame_rgba.shape[2] == 4:
                    frame_rgba[:, :, 3] = np.where(
                        frame_rgba[:, :, 3] > 128, 255, 0)

                # Save as PNG
                frame_pil = Image.fromarray(frame_rgba, 'RGBA')
                frame_pil.save(frame_path)

            outputs["frames_1x"] = {"path": frame_dir, "count": len(frames)}
            print(f"   ✓ Generated {len(frames)} single frames (1x)")

        # Output 2: Single frames upscaled (2x)
        if "frames_2x" in self.config.output_formats:
            frame_dir = output_dir / f"{base_name}_frames_2x"
            frame_dir.mkdir(exist_ok=True)

            for i, frame_data in enumerate(frames):
                frame_path = frame_dir / f"frame_{i:03d}_2x.png"
                frame_rgba = frame_data["image"]

                # Upscale
                h, w = frame_rgba.shape[:2]
                new_h, new_w = h * self.config.upscale_factor, w * self.config.upscale_factor
                frame_upscaled = cv2.resize(
                    frame_rgba, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

                # Ensure zero background opacity
                if frame_upscaled.shape[2] == 4:
                    frame_upscaled[:, :, 3] = np.where(
                        frame_upscaled[:, :, 3] > 128, 255, 0)

                # Save as PNG
                frame_pil = Image.fromarray(frame_upscaled, 'RGBA')
                frame_pil.save(frame_path)

            outputs["frames_2x"] = {"path": frame_dir, "count": len(frames)}
            print(f"   ✓ Generated {len(frames)} upscaled frames (2x)")

        # Output 3: Animated GIF (1x)
        if "gif_1x" in self.config.output_formats and len(frames) > 1:
            gif_path = output_dir / f"{base_name}_animated.gif"

            # Convert frames to PIL images
            pil_frames = []
            for frame_data in frames:
                frame_rgba = frame_data["image"]
                if frame_rgba.shape[2] == 4:
                    frame_rgba[:, :, 3] = np.where(
                        frame_rgba[:, :, 3] > 128, 255, 0)
                pil_frames.append(Image.fromarray(frame_rgba, 'RGBA'))

            # Save as GIF
            if pil_frames:
                pil_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=300,  # 300ms per frame
                    loop=0,
                    disposal=2  # Clear previous frame
                )

                outputs["gif_1x"] = {"path": gif_path, "frames": len(frames)}
                print(f"   ✓ Generated animated GIF (1x)")

        # Output 4: Animated GIF upscaled (2x)
        if "gif_2x" in self.config.output_formats and len(frames) > 1:
            gif_path = output_dir / f"{base_name}_animated_2x.gif"

            # Convert and upscale frames
            pil_frames = []
            for frame_data in frames:
                frame_rgba = frame_data["image"]

                # Upscale
                h, w = frame_rgba.shape[:2]
                new_h, new_w = h * self.config.upscale_factor, w * self.config.upscale_factor
                frame_upscaled = cv2.resize(
                    frame_rgba, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

                if frame_upscaled.shape[2] == 4:
                    frame_upscaled[:, :, 3] = np.where(
                        frame_upscaled[:, :, 3] > 128, 255, 0)

                pil_frames.append(Image.fromarray(frame_upscaled, 'RGBA'))

            # Save as GIF
            if pil_frames:
                pil_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=300,
                    loop=0,
                    disposal=2
                )

                outputs["gif_2x"] = {"path": gif_path, "frames": len(frames)}
                print(f"   ✓ Generated upscaled animated GIF (2x)")

        return outputs


def create_windows_config(quality_preset: str = "professional") -> ProcessingConfig:
    """Create Windows-optimized processing configuration"""
    preset_map = {
        "fast": QualityPreset.FAST,
        "balanced": QualityPreset.BALANCED,
        "professional": QualityPreset.PROFESSIONAL,
        "ultimate": QualityPreset.ULTIMATE
    }

    return ProcessingConfig(
        quality_preset=preset_map.get(
            quality_preset, QualityPreset.PROFESSIONAL),
        device="cpu",  # Windows default
        upscale_factor=2
    )


def process_sprites_windows(input_dir: str = "input", quality_preset: str = "professional"):
    """Process sprites with Windows optimizations"""
    print("🚀 WINDOWS-OPTIMIZED SPRITE PROCESSING")
    print("=" * 60)

    config = create_windows_config(quality_preset)
    processor = WindowsSpriteProcessor(config)

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return

    # Find image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [f for f in input_path.iterdir()
                   if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return

    print(f"📂 Found {len(image_files)} images to process")
    print(f"⚙️ Quality preset: {quality_preset}")
    print()

    successful = 0
    failed = 0

    for i, image_path in enumerate(image_files, 1):  # Process ALL images
        print(f"[{i}/{len(image_files)}] " + "=" * 50)

        result = processor.process_sprite_optimized(image_path)

        if result["success"]:
            successful += 1
            quality = result["quality_metrics"]["overall_quality"]
            print(f"✅ Success - Quality: {quality:.3f}")
        else:
            failed += 1
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")

    print(f"\n📊 PROCESSING SUMMARY")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Results saved to: output/ultimate_sprites/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Windows-Optimized Sprite Processing")
    parser.add_argument("--input", default="input", help="Input directory")
    parser.add_argument("--quality", default="professional",
                        choices=["fast", "balanced",
                                 "professional", "ultimate"],
                        help="Quality preset")

    args = parser.parse_args()
    process_sprites_windows(args.input, args.quality)
