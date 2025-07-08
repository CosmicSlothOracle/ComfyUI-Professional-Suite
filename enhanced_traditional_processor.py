#!/usr/bin/env python3
"""
ENHANCED TRADITIONAL SPRITESHEET PROCESSOR
Optimierte traditionelle CV mit verbesserter Hintergrundentfernung
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Tuple, List


class EnhancedTraditionalProcessor:
    """Optimierte traditionelle Verarbeitung - Speed + Quality"""

    def __init__(self, input_dir="input", output_base_dir="output/enhanced_traditional"):
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        self.session_dir = None
        self.processed_files = []
        self.total_frames_extracted = 0
        self.start_time = None

    def create_session_directory(self):
        """Erstellt Session-Verzeichnis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_base_dir / \
            f"enhanced_session_{timestamp}"

        (self.session_dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.session_dir / "gifs").mkdir(parents=True, exist_ok=True)

    def enhanced_background_detection(self, image: np.ndarray) -> Tuple[List[np.ndarray], float]:
        """ENHANCED: Multi-Zone Background Detection mit adaptiver Toleranz"""
        h, w = image.shape[:2]

        # 1. ERWEITERTE SAMPLING-ZONEN (nicht nur Ecken)
        corner_size = max(20, min(h, w) // 20)
        edge_width = max(15, min(h, w) // 30)

        zones = [
            # Größere Ecken
            image[:corner_size, :corner_size],
            image[:corner_size, -corner_size:],
            image[-corner_size:, :corner_size],
            image[-corner_size:, -corner_size:],

            # Kanten-Mitten
            image[:edge_width, w//4:3*w//4],      # Top center
            image[-edge_width:, w//4:3*w//4],     # Bottom center
            image[h//4:3*h//4, :edge_width],      # Left center
            image[h//4:3*h//4, -edge_width:],     # Right center

            # Border-Strips
            image[:edge_width//2, :],             # Complete top edge
            image[-edge_width//2:, :],            # Complete bottom edge
            image[:, :edge_width//2],             # Complete left edge
            image[:, -edge_width//2:]             # Complete right edge
        ]

        # 2. MULTI-COLOR ANALYSIS
        bg_candidates = []
        for zone in zones:
            if zone.size > 0:
                # RGB Dominant Colors
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

        # 3. FINAL CLUSTERING
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

        # 4. ADAPTIVE TOLERANCE
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

    def create_enhanced_mask(self, image: np.ndarray, bg_colors: List[np.ndarray],
                             tolerance: float) -> np.ndarray:
        """ENHANCED: Multi-Color + HSV Background Removal"""
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=bool)

        # 1. MULTI-COLOR MASKING
        for bg_color in bg_colors:
            # RGB Distance
            rgb_diff = np.abs(image.astype(int) - bg_color.astype(int))
            rgb_mask = np.all(rgb_diff <= tolerance, axis=2)

            # HSV Distance für bessere Farberkennung
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_bg = cv2.cvtColor(bg_color.reshape(
                1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
            hsv_diff = np.abs(hsv_image.astype(int) - hsv_bg.astype(int))
            hsv_tolerance = [tolerance//3, tolerance, tolerance]  # [H, S, V]
            hsv_mask = np.all(hsv_diff <= hsv_tolerance, axis=2)

            # Kombiniere beide Masken
            combined_mask |= (rgb_mask | hsv_mask)

        # 2. MORPHOLOGISCHE VERBESSERUNG
        foreground_mask = ~combined_mask
        kernel_size = max(3, min(h, w) // 150)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Schließe Löcher + Entferne Fragmente
        foreground_mask = cv2.morphologyEx(foreground_mask.astype(
            np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
        foreground_mask = cv2.morphologyEx(
            foreground_mask, cv2.MORPH_OPEN, kernel)

        # 3. EDGE REFINEMENT
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))

        # Bei starken Kanten: bevorzuge Vordergrund
        uncertain_pixels = cv2.morphologyEx(
            foreground_mask, cv2.MORPH_GRADIENT, kernel) > 0
        edge_boost = edges_dilated & uncertain_pixels
        foreground_mask[edge_boost] = 1

        return foreground_mask

    def extract_frames_aggressive(self, image: np.ndarray, foreground_mask: np.ndarray) -> List[np.ndarray]:
        """AGGRESSIVE: Frame-Extraktion wie im ursprünglichen System"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground_mask, connectivity=8)

        extracted_frames = []
        h, w = image.shape[:2]
        min_area = max(300, (w * h) // 3000)  # Aggressiverer Min-Area

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            # Aggressivere Filter
            if area < min_area:
                continue

            x, y, w_comp, h_comp = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

            # Weniger restriktive Aspect Ratio
            aspect_ratio = w_comp / h_comp if h_comp > 0 else 0
            if aspect_ratio < 0.05 or aspect_ratio > 20:  # Sehr permissiv
                continue

            # Extrahiere mit Padding
            padding = 8
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image.shape[1], x + w_comp + padding)
            y_end = min(image.shape[0], y + h_comp + padding)

            frame = image[y_start:y_end, x_start:x_end]

            if frame.size > 0:
                # Enhanced Background Removal für Frame
                frame_rgba = self._process_frame(frame)
                extracted_frames.append(frame_rgba)

        return extracted_frames

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhanced Frame Processing mit lokaler Hintergrunderkennung"""
        # Lokale Hintergrunderkennung
        frame_bg_colors, frame_tolerance = self.enhanced_background_detection(
            frame)
        frame_mask = self.create_enhanced_mask(
            frame, frame_bg_colors, frame_tolerance)

        # Zu RGBA konvertieren
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgba = np.dstack(
            [frame_rgb, np.full(frame_rgb.shape[:2], 255, dtype=np.uint8)])

        # Hintergrund transparent setzen
        frame_rgba[frame_mask == 0, 3] = 0

        # Sanfte Kanten
        alpha = frame_rgba[:, :, 3]
        alpha_smooth = cv2.GaussianBlur(alpha, (3, 3), 0.8)
        edges = cv2.morphologyEx(
            alpha, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        frame_rgba[edges > 0, 3] = alpha_smooth[edges > 0]

        return frame_rgba

    def process_spritesheet(self, image_path: Path) -> int:
        """Verarbeitet ein Spritesheet mit Enhanced Traditional Method"""
        try:
            print(f"🎮 Processing: {image_path.name}")

            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Could not load image")

            h, w = image.shape[:2]
            print(f"   📐 Size: {w}x{h}")

            # Enhanced Background Detection
            bg_colors, tolerance = self.enhanced_background_detection(image)
            print(
                f"   🔍 Background: {len(bg_colors)} colors, tolerance: {tolerance:.1f}")

            # Enhanced Masking
            foreground_mask = self.create_enhanced_mask(
                image, bg_colors, tolerance)

            # Aggressive Frame Extraction
            extracted_frames = self.extract_frames_aggressive(
                image, foreground_mask)
            print(f"   📦 Extracted: {len(extracted_frames)} frames")

            # Speichere Frames
            sprite_dir = self.session_dir / "frames" / image_path.stem
            sprite_dir.mkdir(exist_ok=True)

            for i, frame_rgba in enumerate(extracted_frames):
                frame_path = sprite_dir / f"frame_{i+1:03d}.png"
                Image.fromarray(frame_rgba, 'RGBA').save(frame_path)

            # Erstelle GIF
            if extracted_frames:
                gif_path = self.session_dir / "gifs" / \
                    f"{image_path.stem}_animation.gif"
                pil_frames = [Image.fromarray(frame, 'RGBA')
                              for frame in extracted_frames]
                pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                                   duration=500, loop=0, transparency=0, disposal=2)

            self.processed_files.append({
                'file': image_path.name,
                'frames': len(extracted_frames),
                'background_colors': len(bg_colors),
                'tolerance': tolerance
            })

            self.total_frames_extracted += len(extracted_frames)
            return len(extracted_frames)

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return 0

    def run_enhanced_batch(self):
        """Führt Enhanced Traditional Processing aus"""
        print("🚀 ENHANCED TRADITIONAL SPRITESHEET PROCESSING")
        print("=" * 60)

        self.start_time = time.time()
        self.create_session_directory()

        # Finde Spritesheets
        candidates = []
        for ext in ['.png', '.jpg', '.jpeg']:
            candidates.extend(self.input_dir.glob(f"*{ext}"))
        candidates = sorted(candidates)

        if not candidates:
            print("❌ No images found!")
            return

        print(f"🎯 Processing {len(candidates)} files...")

        # Verarbeite alle
        for i, image_path in enumerate(candidates, 1):
            print(f"\n[{i:>2}/{len(candidates)}] ", end="")
            frames = self.process_spritesheet(image_path)
            print(f"   ✅ {frames} frames" if frames > 0 else "   ❌ Failed")

        # Summary
        end_time = time.time()
        processing_time = end_time - self.start_time

        print(f"\n🎉 ENHANCED TRADITIONAL COMPLETE!")
        print(f"   ⏱️  Time: {processing_time:.1f} seconds")
        print(f"   📦 Total frames: {self.total_frames_extracted}")
        print(
            f"   ⚡ Speed: {self.total_frames_extracted/processing_time:.1f} frames/sec")
        print(f"   🏆 vs AI: ~{805/processing_time:.1f}x faster")

        # Save summary
        summary = {
            'method': 'Enhanced Traditional CV',
            'processing_time': processing_time,
            'total_frames': self.total_frames_extracted,
            'files_processed': len(self.processed_files),
            'frames_per_second': self.total_frames_extracted / processing_time,
            'improvements': [
                'Multi-zone background detection',
                'RGB + HSV color space analysis',
                'Adaptive tolerance calculation',
                'Edge-preserving refinement',
                'Aggressive frame extraction'
            ]
        }

        with open(self.session_dir / "ENHANCED_SUMMARY.json", 'w') as f:
            json.dump(summary, f, indent=2)

        return self.session_dir


if __name__ == "__main__":
    processor = EnhancedTraditionalProcessor()
    processor.run_enhanced_batch()
