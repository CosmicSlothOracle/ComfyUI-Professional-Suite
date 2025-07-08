#!/usr/bin/env python3
"""
STANDALONE BATCH SPRITESHEET PROCESSOR
Direkte Verarbeitung aller Spritesheets ohne ComfyUI API
Features: Hintergrundentfernung, Weißabgleich, Hochskalierung, Frame-Extraktion
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import time


class StandaloneBatchProcessor:
    """Standalone Enhanced Spritesheet Processor"""

    def __init__(self):
        self.input_dir = Path("input")
        self.output_dir = Path("output/intelligent_sprites_batch")
        self.processed_count = 0
        self.failed_count = 0

    def enhanced_background_detection(self, image: np.ndarray) -> Tuple[List[np.ndarray], float]:
        """Enhanced Multi-Zone Background Detection"""
        h, w = image.shape[:2]

        # ERWEITERTE SAMPLING-ZONEN
        corner_size = max(20, min(h, w) // 20)
        edge_width = max(15, min(h, w) // 30)

        zones = [
            # Größere Ecken
            image[:corner_size, :corner_size],
            image[:corner_size, -corner_size:],
            image[-corner_size:, :corner_size],
            image[-corner_size:, -corner_size:],

            # Kanten-Mitten
            image[:edge_width, w//4:3*w//4],
            image[-edge_width:, w//4:3*w//4],
            image[h//4:3*h//4, :edge_width],
            image[h//4:3*h//4, -edge_width:],

            # Border-Strips
            image[:edge_width//2, :],
            image[-edge_width//2:, :],
            image[:, :edge_width//2],
            image[:, -edge_width//2:]
        ]

        # MULTI-COLOR ANALYSIS
        bg_candidates = []
        for zone in zones:
            if zone.size > 0:
                # RGB Analysis
                pixels = zone.reshape(-1, 3).astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS +
                            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                k = min(2, len(np.unique(pixels.reshape(-1))))

                if k > 0:
                    _, _, centers = cv2.kmeans(
                        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    bg_candidates.extend([center.astype(np.uint8)
                                         for center in centers])

                # HSV Analysis
                hsv_zone = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
                hsv_pixels = hsv_zone.reshape(-1, 3).astype(np.float32)
                if len(hsv_pixels) > 0:
                    bg_candidates.append(
                        np.median(hsv_pixels, axis=0).astype(np.uint8))

        # FINAL CLUSTERING
        if bg_candidates:
            all_colors = np.array(bg_candidates, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
            k = min(3, len(all_colors))
            _, _, final_centers = cv2.kmeans(
                all_colors, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            final_bg_colors = [center.astype(np.uint8)
                               for center in final_centers]
        else:
            final_bg_colors = [np.array([255, 255, 255], dtype=np.uint8)]

        # ADAPTIVE TOLERANCE
        variances = []
        for zone in zones:
            if zone.size > 0:
                variance = np.var(zone.reshape(-1, 3), axis=0)
                variances.append(np.mean(variance))

        if variances:
            avg_variance = np.mean(variances)
            if avg_variance < 100:
                tolerance = 20.0
            elif avg_variance < 300:
                tolerance = 30.0
            elif avg_variance < 600:
                tolerance = 40.0
            else:
                tolerance = 50.0
        else:
            tolerance = 30.0

        return final_bg_colors, tolerance

    def create_enhanced_mask(self, image: np.ndarray, bg_colors: List[np.ndarray], tolerance: float) -> np.ndarray:
        """Enhanced Multi-Modal Background Masking"""
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=bool)

        # MULTI-COLOR MASKING
        for bg_color in bg_colors:
            # RGB Distance
            rgb_diff = np.abs(image.astype(int) - bg_color.astype(int))
            rgb_mask = np.all(rgb_diff <= tolerance, axis=2)

            # HSV Analysis
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_bg = cv2.cvtColor(bg_color.reshape(
                1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
            hsv_diff = np.abs(hsv_image.astype(int) - hsv_bg.astype(int))
            hsv_tolerance = [tolerance//3, tolerance, tolerance]
            hsv_mask = np.all(hsv_diff <= hsv_tolerance, axis=2)
            combined_mask |= (rgb_mask | hsv_mask)

        # MORPHOLOGICAL CLEANUP
        foreground_mask = ~combined_mask

        kernel_size = max(3, min(h, w) // 150)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        foreground_mask = cv2.morphologyEx(foreground_mask.astype(
            np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
        foreground_mask = cv2.morphologyEx(
            foreground_mask, cv2.MORPH_OPEN, kernel)

        # EDGE REFINEMENT
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        uncertain_pixels = cv2.morphologyEx(
            foreground_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
        edge_boost = edges_dilated & uncertain_pixels
        foreground_mask = foreground_mask.astype(np.uint8)
        foreground_mask[edge_boost] = 1

        return foreground_mask.astype(np.uint8)

    def extract_frames_enhanced(self, image: np.ndarray, foreground_mask: np.ndarray, min_area_factor: float = 2500.0) -> List[np.ndarray]:
        """Enhanced Aggressive Frame Extraction"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground_mask, connectivity=8)

        extracted_frames = []
        h, w = image.shape[:2]
        total_pixels = h * w

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]

            # AGGRESSIVE FILTERING
            aspect_ratio = width / height if height > 0 else 0

            if (area > min_area_factor and
                area < total_pixels * 0.8 and
                0.125 < aspect_ratio < 8 and
                    width > 32 and height > 32):

                # Extract component mask
                component_mask = (labels == label).astype(np.uint8)

                # Add padding
                padding = max(5, min(width, height) // 20)
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(w, x + width + padding)
                y_end = min(h, y + height + padding)

                # Extract frame
                frame_region = image[y_start:y_end, x_start:x_end]
                mask_region = component_mask[y_start:y_end, x_start:x_end]

                # Create RGBA frame
                frame_rgba = cv2.cvtColor(frame_region, cv2.COLOR_BGR2RGBA)
                frame_rgba[:, :, 3] = mask_region * 255

                extracted_frames.append(frame_rgba)

        return extracted_frames

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance individual frame with white balance and sharpening"""
        # Convert to PIL for enhancement
        pil_image = Image.fromarray(frame)

        # White Balance (simple version)
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(1.1)

        # Contrast enhancement
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.15)

        # Sharpness enhancement
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.2)

        return np.array(pil_image)

    def upscale_frame(self, frame: np.ndarray, scale_factor: int = 2) -> np.ndarray:
        """Upscale frame using INTER_CUBIC"""
        h, w = frame.shape[:2]
        new_h, new_w = h * scale_factor, w * scale_factor
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def process_single_spritesheet(self, image_path: Path) -> Dict[str, Any]:
        """Process a single spritesheet file"""
        print(f"\n🎮 Processing: {image_path.name}")

        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {"success": False, "error": "Could not load image"}

            print(f"📊 Image size: {image.shape[1]}x{image.shape[0]}")

            # Background detection
            print("🔍 Detecting background...")
            bg_colors, tolerance = self.enhanced_background_detection(image)
            print(
                f"📋 Found {len(bg_colors)} background colors, tolerance: {tolerance}")

            # Create mask
            print("🎭 Creating mask...")
            foreground_mask = self.create_enhanced_mask(
                image, bg_colors, tolerance)

            # Extract frames
            print("✂️ Extracting frames...")
            frames = self.extract_frames_enhanced(image, foreground_mask)
            print(f"🎯 Extracted {len(frames)} frames")

            if not frames:
                return {"success": False, "error": "No frames extracted"}

            # Save frames
            sprite_name = image_path.stem
            sprite_output_dir = self.output_dir / sprite_name
            sprite_output_dir.mkdir(parents=True, exist_ok=True)

            saved_frames = []
            for i, frame in enumerate(frames):
                # Enhance frame
                enhanced_frame = self.enhance_frame(frame)

                # Upscale frame
                upscaled_frame = self.upscale_frame(
                    enhanced_frame, scale_factor=2)

                # Save frame
                frame_filename = f"frame_{i:03d}.png"
                frame_path = sprite_output_dir / frame_filename

                # Convert RGBA to PIL Image and save
                pil_frame = Image.fromarray(upscaled_frame, 'RGBA')
                pil_frame.save(frame_path, 'PNG')
                saved_frames.append(str(frame_path))

            # Create animated GIF
            print("🎬 Creating animated GIF...")
            gif_path = sprite_output_dir / f"{sprite_name}_animated.gif"

            gif_frames = []
            for frame in frames:
                enhanced = self.enhance_frame(frame)
                pil_frame = Image.fromarray(enhanced, 'RGBA')
                gif_frames.append(pil_frame)

            if gif_frames:
                gif_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=gif_frames[1:],
                    duration=500,
                    loop=0,
                    disposal=2
                )

            print(
                f"✅ Successfully processed: {len(frames)} frames saved to {sprite_output_dir}")
            return {
                "success": True,
                "frames_count": len(frames),
                "output_dir": str(sprite_output_dir),
                "saved_frames": saved_frames,
                "gif_path": str(gif_path)
            }

        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
            return {"success": False, "error": str(e)}

    def get_spritesheet_files(self) -> List[Path]:
        """Get all spritesheet files from input directory"""
        sprite_files = []

        # Main input directory
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            sprite_files.extend(self.input_dir.glob(ext))

        # Sprite sheets subdirectory
        sprite_sheets_dir = self.input_dir / "sprite_sheets"
        if sprite_sheets_dir.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                sprite_files.extend(sprite_sheets_dir.glob(ext))

        # Filter out small files (likely thumbnails)
        valid_files = []
        for file in sprite_files:
            try:
                if file.stat().st_size > 100_000:  # > 100KB
                    valid_files.append(file)
                    print(
                        f"📋 Found spritesheet: {file.name} ({file.stat().st_size // 1024}KB)")
            except Exception:
                continue

        return valid_files

    def process_all_spritesheets(self):
        """Process all spritesheets in batch"""
        print("🚀 Starting Batch Spritesheet Processing...")
        print(f"📂 Input directory: {self.input_dir.absolute()}")
        print(f"📁 Output directory: {self.output_dir.absolute()}")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get all spritesheet files
        sprite_files = self.get_spritesheet_files()

        if not sprite_files:
            print("❌ No spritesheet files found in input directory!")
            return

        print(f"📊 Found {len(sprite_files)} spritesheet files to process")

        # Process each file
        results = []
        for i, sprite_file in enumerate(sprite_files, 1):
            print(f"\n[{i}/{len(sprite_files)}] " + "="*50)

            start_time = time.time()
            result = self.process_single_spritesheet(sprite_file)
            end_time = time.time()

            result["filename"] = sprite_file.name
            result["processing_time"] = end_time - start_time
            results.append(result)

            if result["success"]:
                self.processed_count += 1
                print(f"✅ Completed in {result['processing_time']:.2f}s")
            else:
                self.failed_count += 1
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")

        # Save processing report
        report_path = self.output_dir / "processing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Final statistics
        print("\n" + "="*70)
        print("🎉 BATCH PROCESSING COMPLETE!")
        print(f"✅ Successfully processed: {self.processed_count}")
        print(f"❌ Failed: {self.failed_count}")
        print(f"📊 Total files: {len(sprite_files)}")
        print(
            f"💯 Success rate: {(self.processed_count / len(sprite_files) * 100):.1f}%")
        print(f"📋 Report saved to: {report_path}")


def main():
    """Main execution function"""
    processor = StandaloneBatchProcessor()
    processor.process_all_spritesheets()


if __name__ == "__main__":
    print("🎮 STANDALONE BATCH SPRITESHEET PROCESSOR")
    print("=" * 70)
    print("Features:")
    print("• Enhanced background detection with multi-zone sampling")
    print("• Advanced HSV color analysis")
    print("• Morphological cleanup and edge refinement")
    print("• Aggressive frame extraction")
    print("• White balance and contrast enhancement")
    print("• 2x upscaling with cubic interpolation")
    print("• Animated GIF creation")
    print("• Transparent PNG output")
    print("=" * 70)

    main()
