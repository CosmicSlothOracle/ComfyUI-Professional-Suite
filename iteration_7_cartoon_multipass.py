#!/usr/bin/env python3
"""
ITERATION 7 - CARTOON MULTIPASS PROCESSOR
7-Stufen iterative Verbesserung:
1. ITERATION 1-3: Progressive Background Transparency Optimization
2. ITERATION 4-6: Linework Enhancement & Color Optimization
3. ITERATION 7: Final Sharpening & Optimal Frame Selection

Features:
- Cartoon-Effekt statt Vaporwave
- Doppelt so große Frame-Ausschnitte (64px padding)
- Überlappungen erlaubt für zentrale Frame-Positionierung
- Multi-Pass Quality Enhancement
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from collections import Counter
from pathlib import Path
import os
import json
import time
from datetime import datetime
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import multiprocessing
import statistics
from scipy import ndimage
from skimage import morphology, measure, filters


class Iteration7CartoonMultipass:
    """7-ITERATIONS CARTOON MULTIPASS: Progressive Quality Enhancement"""

    def __init__(self, input_dir="input", output_base_dir="output/iteration_7_cartoon"):
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        self.session_dir = None
        self.processed_files = []
        self.failed_files = []
        self.total_frames_extracted = 0
        self.total_frames_final = 0
        self.start_time = None
        self.supported_formats = {'.png', '.jpg',
                                  '.jpeg', '.bmp', '.tiff', '.tif'}

        # Multi-threading
        self.max_workers = min(multiprocessing.cpu_count(), 8)
        self.thread_lock = threading.Lock()

    def create_session_directory(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_base_dir / f"iter7_cartoon_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Iteration-spezifische Verzeichnisse
        for i in range(1, 8):
            (self.session_dir / f"iteration_{i}").mkdir(exist_ok=True)

        (self.session_dir / "individual_sprites").mkdir(exist_ok=True)
        (self.session_dir / "animations_final").mkdir(exist_ok=True)
        (self.session_dir / "iteration_reports").mkdir(exist_ok=True)
        (self.session_dir / "quality_analysis").mkdir(exist_ok=True)

        print(f"📁 ITERATION 7 CARTOON Session: {self.session_dir}")
        print(f"🔧 Using {self.max_workers} threads + 7-Pass Enhancement")

    # ITERATION 1-3: PROGRESSIVE BACKGROUND TRANSPARENCY
    def iteration_1_enhanced_background_detection(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ITERATION 1: Ultra-Enhanced Background Detection mit 32-Zonen-Sampling"""
        h, w = image.shape[:2]
        corner_size = max(2, int(min(h, w) / 15))  # Größere Sampling-Bereiche

        # 32-Zonen Grid für präzisere Background-Detection
        zones = []
        for i in range(8):
            for j in range(4):
                y_start = int(i * h / 8)
                x_start = int(j * w / 4)

                # Nur Border-Zonen sampeln
                if i == 0 or i == 7 or j == 0 or j == 3:
                    zone = image[y_start:y_start+corner_size,
                                 x_start:x_start+corner_size]
                    if zone.size > 0:
                        zones.append(zone)

        # Multi-Modal Background Detection
        all_pixels = []
        for zone in zones:
            if len(zone.shape) == 3:
                pixels = zone.reshape(-1, 3)
                all_pixels.extend([tuple(pixel) for pixel in pixels])

        # Top 3 häufigste Farben als Background-Kandidaten
        pixel_counts = Counter(all_pixels)
        top_candidates = pixel_counts.most_common(3)

        if top_candidates:
            bg_color = np.array(top_candidates[0][0], dtype=np.uint8)
        else:
            bg_color = np.array([255, 255, 255], dtype=np.uint8)

        return bg_color, zones

    def iteration_2_adaptive_masking(self, image: np.ndarray, bg_color: np.ndarray) -> np.ndarray:
        """ITERATION 2: Adaptive Multi-Tolerance Masking"""
        # Multi-Tolerance Approach
        tolerances = [15, 25, 35, 45]  # Progressive Toleranzen
        masks = []

        for tolerance in tolerances:
            diff = np.abs(image.astype(int) - bg_color.astype(int))
            background_mask = np.all(diff <= tolerance, axis=2)
            foreground_mask = ~background_mask
            masks.append(foreground_mask)

        # Kombiniere Masken durch Majority Voting
        mask_stack = np.stack(masks, axis=2)
        # Mindestens 2 von 4 Masken müssen zustimmen
        final_mask = np.sum(mask_stack, axis=2) >= 2

        return final_mask.astype(np.uint8)

    def iteration_3_morphological_refinement(self, mask: np.ndarray) -> np.ndarray:
        """ITERATION 3: Advanced Morphological Refinement"""
        # Progressive Morphological Operations
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (7, 7))  # Größerer Kernel
        ]

        refined_mask = mask.copy()

        # Multi-Pass Morphology
        for i, kernel in enumerate(kernels):
            # Closing -> Opening -> Closing Zyklus
            refined_mask = cv2.morphologyEx(
                refined_mask, cv2.MORPH_CLOSE, kernel)
            refined_mask = cv2.morphologyEx(
                refined_mask, cv2.MORPH_OPEN, kernel)
            if i >= 2:  # Für größere Kernels: zusätzliche Dilation
                refined_mask = cv2.morphologyEx(
                    refined_mask, cv2.MORPH_DILATE, kernel)

        # Finaler Gaussian Blur für weiche Kanten
        refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 1.5)

        return refined_mask

    # ITERATION 4-6: LINEWORK & COLOR ENHANCEMENT
    def iteration_4_linework_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """ITERATION 4: Cartoon Linework Enhancement"""
        # Bilateral Filter für Edge-Preservation
        smooth = cv2.bilateralFilter(frame, 15, 80, 80)

        # Edge Detection für Linework
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

        # Kombiniere Original mit verstärkten Edges
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.addWeighted(smooth, 0.8, 255 - edges_colored, 0.2, 0)

        return enhanced

    def iteration_5_cartoon_color_optimization(self, frame: np.ndarray) -> np.ndarray:
        """ITERATION 5: Cartoon Color Enhancement"""
        # K-Means Color Quantization für Cartoon-Look
        data = frame.reshape((-1, 3))
        data = np.float32(data)

        # Adaptive K basierend auf Frame-Größe
        frame_area = frame.shape[0] * frame.shape[1]
        k = min(max(8, frame_area // 5000), 16)  # 8-16 Farben

        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            data, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

        # Rekonstruiere quantisiertes Bild
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized = quantized_data.reshape(frame.shape)

        # Sanfte Mischung Original + Quantized
        cartoon = cv2.addWeighted(frame, 0.4, quantized, 0.6, 0)

        return cartoon

    def iteration_6_color_saturation_boost(self, frame: np.ndarray) -> np.ndarray:
        """ITERATION 6: Cartoon Color Saturation & Contrast"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)

        # Cartoon-spezifische Enhancements
        color_enhancer = ImageEnhance.Color(pil_frame)
        enhanced_frame = color_enhancer.enhance(1.3)  # Subtile Sättigung

        contrast_enhancer = ImageEnhance.Contrast(enhanced_frame)
        enhanced_frame = contrast_enhancer.enhance(1.15)  # Leichter Kontrast

        brightness_enhancer = ImageEnhance.Brightness(enhanced_frame)
        enhanced_frame = brightness_enhancer.enhance(
            1.05)  # Minimale Helligkeit

        return cv2.cvtColor(np.array(enhanced_frame), cv2.COLOR_RGB2BGR)

    # ITERATION 7: FINAL OPTIMIZATION
    def iteration_7_sharpening_and_selection(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """ITERATION 7: Final Sharpening & Optimal Frame Selection"""
        if not frames:
            return frames

        # Sharpen alle Frames
        sharpened_frames = []
        for frame in frames:
            # Unsharp Mask
            gaussian = cv2.GaussianBlur(frame, (0, 0), 1.5)
            sharpened = cv2.addWeighted(frame, 1.8, gaussian, -0.8, 0)

            # Adaptive Sharpening basierend auf Frame-Größe
            if frame.shape[0] > 100 or frame.shape[1] > 100:
                kernel = np.array(
                    [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
                sharpened = cv2.filter2D(sharpened, -1, kernel * 0.1)

            sharpened_frames.append(sharpened)

        # Quality-basierte Auswahl der besten Frames
        if len(sharpened_frames) > 8:  # Reduziere auf beste Frames
            frame_qualities = []
            for frame in sharpened_frames:
                # Qualitäts-Score basierend auf Edge-Density und Varianz
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                variance = np.var(gray)
                quality_score = edge_density * variance
                frame_qualities.append((quality_score, frame))

            # Sortiere nach Qualität und nimm die besten
            frame_qualities.sort(key=lambda x: x[0], reverse=True)
            best_count = min(8, len(frame_qualities))
            sharpened_frames = [frame for _,
                                frame in frame_qualities[:best_count]]

        return sharpened_frames

    def extract_large_frames_with_overlap(self, image: np.ndarray, mask: np.ndarray) -> List[Tuple[np.ndarray, dict]]:
        """Extrahiert Frames mit doppelter Größe (64px padding) und erlaubt Überlappungen"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)

        large_frames = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, w_frame, h_frame = stats[i,
                                           cv2.CC_STAT_LEFT:cv2.CC_STAT_TOP+3]

            # Basis-Filter
            aspect_ratio = w_frame / h_frame if h_frame > 0 else 0
            if area >= 300 and 0.1 <= aspect_ratio <= 15.0:

                # DOPPELTE PADDING-GRÖSSE: 64px statt 32px
                padding = 64
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(image.shape[1], x + w_frame + padding)
                y_end = min(image.shape[0], y + h_frame + padding)

                # Zentriere das gewünschte Frame
                center_x = x + w_frame // 2
                center_y = y + h_frame // 2

                # Größere Extraktion um das Zentrum
                half_width = (x_end - x_start) // 2
                half_height = (y_end - y_start) // 2

                # Stelle sicher, dass das Frame zentral und vollständig sichtbar ist
                final_x_start = max(0, center_x - half_width)
                final_y_start = max(0, center_y - half_height)
                final_x_end = min(image.shape[1], center_x + half_width)
                final_y_end = min(image.shape[0], center_y + half_height)

                # Extrahiere größeren Bereich
                component_mask = (labels == i)
                frame_region = image[final_y_start:final_y_end,
                                     final_x_start:final_x_end].copy()
                mask_region = component_mask[final_y_start:final_y_end,
                                             final_x_start:final_x_end]

                # Alpha-Channel mit verbesserter Edge-Behandlung
                alpha_region = np.zeros(
                    (final_y_end - final_y_start, final_x_end - final_x_start), dtype=np.uint8)
                alpha_region[mask_region] = 255

                # Mehrfacher Gaussian Blur für weichere Kanten
                alpha_region = cv2.GaussianBlur(alpha_region, (5, 5), 1.2)
                alpha_region = cv2.GaussianBlur(alpha_region, (3, 3), 0.8)

                frame_rgba = cv2.cvtColor(frame_region, cv2.COLOR_BGR2BGRA)
                frame_rgba[:, :, 3] = alpha_region

                frame_stats = {
                    'area': area,
                    'position': (x, y),
                    'center': (center_x, center_y),
                    'dimensions': (w_frame, h_frame),
                    'extracted_size': (final_x_end - final_x_start, final_y_end - final_y_start),
                    'padding_used': padding,
                    'aspect_ratio': aspect_ratio
                }

                large_frames.append((frame_rgba, frame_stats))

        return large_frames

    def process_single_spritesheet_7_iterations(self, image_path):
        """7-ITERATIONS CARTOON MULTIPASS Processing"""
        thread_id = threading.current_thread().ident
        print(f"\n🎭 ITER7 CARTOON [Thread {thread_id}]: {image_path.name}")

        try:
            start_time = time.time()

            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            original_size = (image.shape[1], image.shape[0])

            # Enhanced Upscaling (2x)
            h, w = image.shape[:2]
            upscaled = cv2.resize(image, (w * 2, h * 2),
                                  interpolation=cv2.INTER_CUBIC)
            denoised = cv2.bilateralFilter(upscaled, 9, 75, 75)

            print(f"   🔄 Starting 7-Iteration Processing...")

            # ===============================================
            # ITERATIONS 1-3: BACKGROUND TRANSPARENCY
            # ===============================================

            # ITERATION 1: Enhanced Background Detection
            bg_color, zones = self.iteration_1_enhanced_background_detection(
                denoised)
            print(f"   1️⃣ Background detected: {bg_color}")

            # ITERATION 2: Adaptive Masking
            initial_mask = self.iteration_2_adaptive_masking(
                denoised, bg_color)
            print(f"   2️⃣ Adaptive masking applied")

            # ITERATION 3: Morphological Refinement
            refined_mask = self.iteration_3_morphological_refinement(
                initial_mask)
            print(f"   3️⃣ Morphological refinement completed")

            # Frame-Extraktion mit großen Überlappungen
            large_frames = self.extract_large_frames_with_overlap(
                denoised, refined_mask)
            if not large_frames:
                return None

            print(
                f"   📦 Extracted {len(large_frames)} large frames with 64px padding")

            # ===============================================
            # ITERATIONS 4-6: LINEWORK & COLOR ENHANCEMENT
            # ===============================================

            enhanced_frames = []
            for frame, stats in large_frames:
                # ITERATION 4: Linework Enhancement
                line_enhanced = self.iteration_4_linework_enhancement(
                    frame[:, :, :3])

                # ITERATION 5: Cartoon Color Optimization
                color_optimized = self.iteration_5_cartoon_color_optimization(
                    line_enhanced)

                # ITERATION 6: Color Saturation Boost
                final_colored = self.iteration_6_color_saturation_boost(
                    color_optimized)

                # Alpha-Channel restaurieren
                final_rgba = cv2.cvtColor(final_colored, cv2.COLOR_BGR2BGRA)
                final_rgba[:, :, 3] = frame[:, :, 3]

                enhanced_frames.append((final_rgba, stats))

            print(
                f"   4️⃣ 5️⃣ 6️⃣ Linework & Color enhancement applied to {len(enhanced_frames)} frames")

            # ===============================================
            # ITERATION 7: FINAL SHARPENING & SELECTION
            # ===============================================

            # Nur die Frames für Sharpening
            frames_only = [frame for frame, _ in enhanced_frames]
            sharpened_frames = self.iteration_7_sharpening_and_selection(
                frames_only)

            # Statistiken zurück kombinieren
            final_frames_with_stats = []
            for i, sharpened in enumerate(sharpened_frames):
                if i < len(enhanced_frames):
                    stats = enhanced_frames[i][1]
                    final_frames_with_stats.append((sharpened, stats))

            print(
                f"   7️⃣ Final sharpening & selection: {len(final_frames_with_stats)} optimal frames")

            if not final_frames_with_stats:
                return None

            # ===============================================
            # SAVE RESULTS
            # ===============================================

            sprite_dir = self.session_dir / "individual_sprites" / image_path.stem
            sprite_dir.mkdir(exist_ok=True)

            # Save finale Frames
            pil_frames = []
            for i, (frame, stats) in enumerate(final_frames_with_stats):
                frame_path = sprite_dir / f"frame_{i:03d}_cartoon_7iter.png"
                cv2.imwrite(str(frame_path), frame)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                pil_frames.append(Image.fromarray(frame_rgb))

            # OPTIMIERTES CARTOON GIF
            if len(pil_frames) > 1:
                gif_path = self.session_dir / "animations_final" / \
                    f"{image_path.stem}_CARTOON_7ITER.gif"

                # Adaptive Duration für Cartoon-Feeling
                if len(pil_frames) <= 4:
                    duration = 300
                elif len(pil_frames) <= 8:
                    duration = 200
                else:
                    duration = 150

                pil_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=duration,
                    loop=0,
                    disposal=2
                )

            processing_time = time.time() - start_time

            # Thread-safe Updates
            with self.thread_lock:
                self.total_frames_extracted += len(large_frames)
                self.total_frames_final += len(final_frames_with_stats)

            # Report
            avg_frame_size = np.mean(
                [s['extracted_size'][0] * s['extracted_size'][1] for _, s in final_frames_with_stats])

            report = {
                "filename": image_path.name,
                "original_size": original_size,
                "upscaled_size": (w * 2, h * 2),
                "frames_extracted": len(large_frames),
                "frames_final": len(final_frames_with_stats),
                "avg_frame_size": int(avg_frame_size),
                "padding_used": 64,
                "processing_time": round(processing_time, 2),
                "iterations_completed": 7,
                "enhancements": [
                    "enhanced_background_detection",
                    "adaptive_masking",
                    "morphological_refinement",
                    "linework_enhancement",
                    "cartoon_color_optimization",
                    "color_saturation_boost",
                    "final_sharpening_selection"
                ],
                "thread_id": thread_id
            }

            report_path = self.session_dir / "iteration_reports" / \
                f"{image_path.stem}_7iter_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            print(
                f"   ✅ SUCCESS: {len(final_frames_with_stats)}/{len(large_frames)} frames (7-iter cartoon) in {processing_time:.1f}s")
            return report

        except Exception as e:
            print(f"   ❌ ERROR [Thread {thread_id}]: {str(e)}")
            return None

    def run_7_iteration_cartoon_batch(self):
        """7-ITERATION CARTOON MULTIPASS: Batch Processing"""
        print("🎭 STARTING ITERATION 7 - CARTOON MULTIPASS WORKFLOW")
        print("=" * 75)
        print("🔄 7-Stufen Progressive Enhancement:")
        print("   • Iterationen 1-3: Background Transparency Optimization")
        print("   • Iterationen 4-6: Linework Enhancement & Color Optimization")
        print("   • Iteration 7: Final Sharpening & Optimal Frame Selection")
        print("   • 64px Padding für doppelt so große Frame-Ausschnitte")
        print("   • Überlappungen erlaubt für zentrale Frame-Positionierung")
        print("=" * 75)

        self.start_time = time.time()
        self.create_session_directory()

        spritesheet_files = []
        for ext in self.supported_formats:
            spritesheet_files.extend(self.input_dir.glob(f"*{ext}"))

        print(f"📊 Found {len(spritesheet_files)} spritesheet files")
        print(f"🔧 Processing with {self.max_workers} parallel threads")
        print(f"🎭 Effect: Cartoon (Linework + Color Quantization + Saturation)")
        print(f"📐 Frame Size: Double padding (64px) with overlaps allowed")

        # Multi-threaded processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_single_spritesheet_7_iterations, file_path): file_path
                for file_path in spritesheet_files
            }

            completed = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                completed += 1

                try:
                    result = future.result()
                    if result:
                        with self.thread_lock:
                            self.processed_files.append(result)
                    else:
                        with self.thread_lock:
                            self.failed_files.append(file_path.name)
                except Exception as e:
                    print(f"❌ Exception: {file_path.name}: {str(e)}")
                    with self.thread_lock:
                        self.failed_files.append(file_path.name)

                if completed % 25 == 0:
                    print(
                        f"Progress: {completed}/{len(spritesheet_files)} completed")

        total_time = time.time() - self.start_time

        # Statistiken
        avg_iterations_per_file = 7  # Immer 7 Iterationen
        total_processing_operations = len(self.processed_files) * 7

        print("\n" + "=" * 75)
        print("🎯 ITERATION 7 - CARTOON MULTIPASS COMPLETE")
        print("=" * 75)
        print(
            f"📊 Files processed: {len(self.processed_files)}/{len(spritesheet_files)}")
        print(f"🎬 Total frames extracted: {self.total_frames_extracted}")
        print(f"✨ Final optimized frames: {self.total_frames_final}")
        print(f"🔄 Total processing operations: {total_processing_operations}")
        print(f"⏱️ Processing time: {total_time:.1f} seconds")
        print(
            f"⚡ Speed: {self.total_frames_extracted/total_time:.2f} frames/second")
        print(f"🎭 Effect: Cartoon (subtle, linework-focused)")
        print(f"📐 Frame extraction: 64px padding with overlaps")
        print(f"📁 Results: {self.session_dir}")

        # Master Report
        master_report = {
            "iteration": "7_cartoon_multipass",
            "timestamp": datetime.now().isoformat(),
            "total_files": len(spritesheet_files),
            "files_processed": len(self.processed_files),
            "files_failed": len(self.failed_files),
            "total_frames_extracted": self.total_frames_extracted,
            "total_frames_final": self.total_frames_final,
            "total_processing_operations": total_processing_operations,
            "processing_time_seconds": round(total_time, 2),
            "frames_per_second": round(self.total_frames_extracted/total_time, 2),
            "enhancements": {
                "iterations_1_3": "Progressive Background Transparency",
                "iterations_4_6": "Linework Enhancement & Color Optimization",
                "iteration_7": "Final Sharpening & Frame Selection",
                "frame_extraction": "64px padding with overlaps allowed",
                "effect_type": "Cartoon (subtle, linework-focused)",
                "multi_threading": f"{self.max_workers}x parallel processing"
            },
            "technical_details": {
                "background_detection": "32-zone multi-modal sampling",
                "masking": "Multi-tolerance adaptive masking",
                "morphology": "Progressive 4-kernel refinement",
                "linework": "Bilateral filter + adaptive threshold",
                "color_optimization": "K-means quantization (8-16 colors)",
                "final_processing": "Unsharp mask + quality selection"
            },
            "session_directory": str(self.session_dir),
            "detailed_reports": self.processed_files
        }

        master_path = self.session_dir / "ITERATION_7_CARTOON_MASTER_REPORT.json"
        with open(master_path, 'w') as f:
            json.dump(master_report, f, indent=2)

        return master_report


def main():
    processor = Iteration7CartoonMultipass()
    return processor.run_7_iteration_cartoon_batch()


if __name__ == "__main__":
    main()
