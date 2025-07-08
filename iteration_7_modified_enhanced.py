#!/usr/bin/env python3
"""
ITERATION 7 MODIFIED - ENHANCED UPSCALING & HOMOGENIZED WORKFLOW
Modifikationen:
1. REDUZIERTER CARTOON-EFFEKT: Subtilere Effekte, weniger aggressive Quantization
2. ERHÖHTES UPSCALING: 4x statt 2x für bessere Qualität
3. FARBTEMPERATUR-HOMOGENISIERUNG: Einheitliche Farbtemperatur über alle Animationen

Features:
- 4x Enhanced Upscaling mit LANCZOS/CUBIC Interpolation
- Subtile Cartoon-Effekte (reduced quantization, mild saturation)
- Global Color Temperature Analysis & Normalization
- Cross-Animation Color Harmony
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
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import multiprocessing
import statistics
from scipy import ndimage
from skimage import morphology, measure, filters, color
import colorsys


class Iteration7ModifiedEnhanced:
    """7-ITERATIONS MODIFIED: Reduced Cartoon + Enhanced Upscaling + Color Temperature Homogenization"""

    def __init__(self, input_dir="input", output_base_dir="output/iteration_7_enhanced_modified"):
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

        # Color Temperature Analysis Storage
        self.color_temperature_samples = []
        self.target_temperature = None

        # Multi-threading
        self.max_workers = min(multiprocessing.cpu_count(), 8)
        self.thread_lock = threading.Lock()

    def create_session_directory(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_base_dir / f"iter7_enhanced_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Iteration-spezifische Verzeichnisse
        for i in range(1, 8):
            (self.session_dir / f"iteration_{i}").mkdir(exist_ok=True)

        (self.session_dir / "individual_sprites").mkdir(exist_ok=True)
        (self.session_dir / "animations_final").mkdir(exist_ok=True)
        (self.session_dir / "iteration_reports").mkdir(exist_ok=True)
        (self.session_dir / "color_analysis").mkdir(exist_ok=True)

        print(f"📁 ITERATION 7 ENHANCED Session: {self.session_dir}")
        print(f"🔧 Using {self.max_workers} threads + Enhanced Processing")

    # ENHANCED UPSCALING METHODS
    def enhanced_upscaling_4x(self, image: np.ndarray) -> np.ndarray:
        """4x Enhanced Upscaling mit Multi-Stage Interpolation"""
        h, w = image.shape[:2]

        # Stage 1: 2x mit LANCZOS
        stage1 = cv2.resize(image, (w * 2, h * 2),
                            interpolation=cv2.INTER_LANCZOS4)

        # Stage 2: Denoising nach erstem Upscale
        denoised1 = cv2.bilateralFilter(stage1, 9, 75, 75)

        # Stage 3: Weiteres 2x mit CUBIC
        stage2 = cv2.resize(denoised1, (w * 4, h * 4),
                            interpolation=cv2.INTER_CUBIC)

        # Stage 4: Final denoising
        final_upscaled = cv2.bilateralFilter(stage2, 15, 80, 80)

        return final_upscaled

    # COLOR TEMPERATURE ANALYSIS
    def analyze_color_temperature(self, image: np.ndarray) -> float:
        """Analysiert die Farbtemperatur eines Bildes"""
        # Convert to LAB color space for better analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Extrahiere A und B Kanäle (Farbinformation)
        a_channel = lab[:, :, 1].astype(np.float32) - 128
        b_channel = lab[:, :, 2].astype(np.float32) - 128

        # Berechne durchschnittliche Farbtemperatur
        avg_a = np.mean(a_channel)
        avg_b = np.mean(b_channel)

        # Konvertiere zu Farbtemperatur-Approximation
        # Positive B-Werte = wärmer, negative = kälter
        temperature_factor = avg_b / (abs(avg_a) + 1)  # Normalisiert

        return temperature_factor

    def apply_color_temperature_correction(self, image: np.ndarray, target_temp: float, current_temp: float) -> np.ndarray:
        """Korrigiert die Farbtemperatur eines Bildes"""
        temp_diff = target_temp - current_temp

        if abs(temp_diff) < 0.1:  # Kleine Unterschiede ignorieren
            return image

        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Moderate Anpassung der B-Kanäle (Warm/Kalt)
        adjustment_factor = temp_diff * 0.3  # Reduzierte Stärke für subtile Korrektur
        lab[:, :, 2] = np.clip(lab[:, :, 2] + adjustment_factor, 0, 255)

        # Convert back
        corrected = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        return corrected

    # MODIFIED ITERATION METHODS (REDUCED CARTOON EFFECTS)
    def iteration_1_enhanced_background_detection(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ITERATION 1: Enhanced Background Detection (unverändert)"""
        h, w = image.shape[:2]
        corner_size = max(2, int(min(h, w) / 15))

        zones = []
        for i in range(8):
            for j in range(4):
                y_start = int(i * h / 8)
                x_start = int(j * w / 4)

                if i == 0 or i == 7 or j == 0 or j == 3:
                    zone = image[y_start:y_start+corner_size,
                                 x_start:x_start+corner_size]
                    if zone.size > 0:
                        zones.append(zone)

        all_pixels = []
        for zone in zones:
            if len(zone.shape) == 3:
                pixels = zone.reshape(-1, 3)
                all_pixels.extend([tuple(pixel) for pixel in pixels])

        pixel_counts = Counter(all_pixels)
        top_candidates = pixel_counts.most_common(3)

        if top_candidates:
            bg_color = np.array(top_candidates[0][0], dtype=np.uint8)
        else:
            bg_color = np.array([255, 255, 255], dtype=np.uint8)

        return bg_color, zones

    def iteration_2_adaptive_masking(self, image: np.ndarray, bg_color: np.ndarray) -> np.ndarray:
        """ITERATION 2: Adaptive Masking (unverändert)"""
        tolerances = [15, 25, 35, 45]
        masks = []

        for tolerance in tolerances:
            diff = np.abs(image.astype(int) - bg_color.astype(int))
            background_mask = np.all(diff <= tolerance, axis=2)
            foreground_mask = ~background_mask
            masks.append(foreground_mask)

        mask_stack = np.stack(masks, axis=2)
        final_mask = np.sum(mask_stack, axis=2) >= 2

        return final_mask.astype(np.uint8)

    def iteration_3_morphological_refinement(self, mask: np.ndarray) -> np.ndarray:
        """ITERATION 3: Morphological Refinement (unverändert)"""
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        ]

        refined_mask = mask.copy()

        for i, kernel in enumerate(kernels):
            refined_mask = cv2.morphologyEx(
                refined_mask, cv2.MORPH_CLOSE, kernel)
            refined_mask = cv2.morphologyEx(
                refined_mask, cv2.MORPH_OPEN, kernel)
            if i >= 2:
                refined_mask = cv2.morphologyEx(
                    refined_mask, cv2.MORPH_DILATE, kernel)

        refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 1.5)

        return refined_mask

    def iteration_4_subtle_linework_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """ITERATION 4: SUBTILE Linework Enhancement (REDUZIERT)"""
        # Reduzierte Parameter für subtileren Effekt
        # Weniger aggressive Glättung
        smooth = cv2.bilateralFilter(frame, 9, 50, 50)

        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        # Höhere Threshold für weniger starke Edges
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))

        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # Reduziertes Weight für subtileren Effekt
        enhanced = cv2.addWeighted(smooth, 0.9, 255 - edges_colored, 0.1, 0)

        return enhanced

    def iteration_5_mild_color_optimization(self, frame: np.ndarray) -> np.ndarray:
        """ITERATION 5: MILDE Color Optimization (REDUZIERT)"""
        data = frame.reshape((-1, 3))
        data = np.float32(data)

        # ERHÖHTE K-Werte für weniger aggressive Quantization
        frame_area = frame.shape[0] * frame.shape[1]
        k = min(max(16, frame_area // 3000), 32)  # 16-32 Farben statt 8-16

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
        _, labels, centers = cv2.kmeans(
            data, k, None, criteria, 2, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized = quantized_data.reshape(frame.shape)

        # REDUZIERTE Mischung für subtileren Effekt
        cartoon = cv2.addWeighted(
            frame, 0.7, quantized, 0.3, 0)  # Weniger Quantization

        return cartoon

    def iteration_6_gentle_saturation_boost(self, frame: np.ndarray) -> np.ndarray:
        """ITERATION 6: GENTLE Saturation Boost (REDUZIERT)"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)

        # REDUZIERTE Enhancement-Werte
        color_enhancer = ImageEnhance.Color(pil_frame)
        enhanced_frame = color_enhancer.enhance(1.1)  # 1.1 statt 1.3

        contrast_enhancer = ImageEnhance.Contrast(enhanced_frame)
        enhanced_frame = contrast_enhancer.enhance(1.08)  # 1.08 statt 1.15

        brightness_enhancer = ImageEnhance.Brightness(enhanced_frame)
        enhanced_frame = brightness_enhancer.enhance(1.02)  # 1.02 statt 1.05

        return cv2.cvtColor(np.array(enhanced_frame), cv2.COLOR_RGB2BGR)

    def iteration_7_sharpening_and_selection(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """ITERATION 7: Final Sharpening & Selection (enhanced)"""
        if not frames:
            return frames

        sharpened_frames = []
        for frame in frames:
            # Moderate Unsharp Mask
            gaussian = cv2.GaussianBlur(
                frame, (0, 0), 1.2)  # Reduziert von 1.5
            sharpened = cv2.addWeighted(
                frame, 1.5, gaussian, -0.5, 0)  # Reduziert

            # Conditional additional sharpening
            if frame.shape[0] > 150 or frame.shape[1] > 150:
                kernel = np.array(
                    [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
                sharpened = cv2.filter2D(
                    sharpened, -1, kernel * 0.05)  # Reduziert von 0.1

            sharpened_frames.append(sharpened)

        # Quality-basierte Auswahl
        if len(sharpened_frames) > 12:  # Erhöht von 8 für mehr Frames
            frame_qualities = []
            for frame in sharpened_frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                variance = np.var(gray)
                quality_score = edge_density * variance
                frame_qualities.append((quality_score, frame))

            frame_qualities.sort(key=lambda x: x[0], reverse=True)
            best_count = min(12, len(frame_qualities))  # Mehr Frames behalten
            sharpened_frames = [frame for _,
                                frame in frame_qualities[:best_count]]

        return sharpened_frames

    def extract_enhanced_frames_with_overlap(self, image: np.ndarray, mask: np.ndarray) -> List[Tuple[np.ndarray, dict]]:
        """Enhanced Frame Extraction mit größerem Padding für 4x Upscaling"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)

        large_frames = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, w_frame, h_frame = stats[i,
                                           cv2.CC_STAT_LEFT:cv2.CC_STAT_TOP+3]

            aspect_ratio = w_frame / h_frame if h_frame > 0 else 0
            if area >= 200 and 0.1 <= aspect_ratio <= 20.0:  # Entspanntere Filter

                # ERHÖHTES PADDING für 4x Upscaling
                padding = 96  # Erhöht von 64px
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(image.shape[1], x + w_frame + padding)
                y_end = min(image.shape[0], y + h_frame + padding)

                center_x = x + w_frame // 2
                center_y = y + h_frame // 2

                half_width = (x_end - x_start) // 2
                half_height = (y_end - y_start) // 2

                final_x_start = max(0, center_x - half_width)
                final_y_start = max(0, center_y - half_height)
                final_x_end = min(image.shape[1], center_x + half_width)
                final_y_end = min(image.shape[0], center_y + half_height)

                component_mask = (labels == i)
                frame_region = image[final_y_start:final_y_end,
                                     final_x_start:final_x_end].copy()
                mask_region = component_mask[final_y_start:final_y_end,
                                             final_x_start:final_x_end]

                alpha_region = np.zeros(
                    (final_y_end - final_y_start, final_x_end - final_x_start), dtype=np.uint8)
                alpha_region[mask_region] = 255

                # Enhanced Alpha Processing
                alpha_region = cv2.GaussianBlur(alpha_region, (7, 7), 1.5)
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

    def process_single_spritesheet_enhanced(self, image_path):
        """Enhanced 7-Iterations Processing mit Color Temperature Analysis"""
        thread_id = threading.current_thread().ident
        print(f"\n🚀 ENHANCED [Thread {thread_id}]: {image_path.name}")

        try:
            start_time = time.time()

            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            original_size = (image.shape[1], image.shape[0])

            # 4X ENHANCED UPSCALING
            upscaled = self.enhanced_upscaling_4x(image)
            h, w = upscaled.shape[:2]
            print(f"   📈 4x Upscaled: {original_size} → {(w, h)}")

            # Color Temperature Analysis
            current_temp = self.analyze_color_temperature(upscaled)
            with self.thread_lock:
                self.color_temperature_samples.append(current_temp)

            print(f"   🌡️ Color temperature analyzed: {current_temp:.3f}")

            # ITERATIONS 1-3: BACKGROUND TRANSPARENCY
            bg_color, zones = self.iteration_1_enhanced_background_detection(
                upscaled)
            print(f"   1️⃣ Background detected: {bg_color}")

            initial_mask = self.iteration_2_adaptive_masking(
                upscaled, bg_color)
            print(f"   2️⃣ Adaptive masking applied")

            refined_mask = self.iteration_3_morphological_refinement(
                initial_mask)
            print(f"   3️⃣ Morphological refinement completed")

            # Enhanced Frame Extraction
            large_frames = self.extract_enhanced_frames_with_overlap(
                upscaled, refined_mask)
            if not large_frames:
                return None

            print(
                f"   📦 Extracted {len(large_frames)} frames with 96px padding")

            # ITERATIONS 4-6: SUBTLE ENHANCEMENT
            enhanced_frames = []
            for frame, stats in large_frames:
                # ITERATION 4: Subtle Linework Enhancement
                line_enhanced = self.iteration_4_subtle_linework_enhancement(
                    frame[:, :, :3])

                # ITERATION 5: Mild Color Optimization
                color_optimized = self.iteration_5_mild_color_optimization(
                    line_enhanced)

                # ITERATION 6: Gentle Saturation Boost
                final_colored = self.iteration_6_gentle_saturation_boost(
                    color_optimized)

                final_rgba = cv2.cvtColor(final_colored, cv2.COLOR_BGR2BGRA)
                final_rgba[:, :, 3] = frame[:, :, 3]

                enhanced_frames.append((final_rgba, stats))

            print(
                f"   4️⃣ 5️⃣ 6️⃣ Subtle enhancement applied to {len(enhanced_frames)} frames")

            # ITERATION 7: FINAL OPTIMIZATION
            frames_only = [frame for frame, _ in enhanced_frames]
            sharpened_frames = self.iteration_7_sharpening_and_selection(
                frames_only)

            final_frames_with_stats = []
            for i, sharpened in enumerate(sharpened_frames):
                if i < len(enhanced_frames):
                    stats = enhanced_frames[i][1]
                    final_frames_with_stats.append((sharpened, stats))

            print(
                f"   7️⃣ Final optimization: {len(final_frames_with_stats)} frames")

            if not final_frames_with_stats:
                return None

            # SAVE RESULTS
            sprite_dir = self.session_dir / "individual_sprites" / image_path.stem
            sprite_dir.mkdir(exist_ok=True)

            pil_frames = []
            for i, (frame, stats) in enumerate(final_frames_with_stats):
                frame_path = sprite_dir / f"frame_{i:03d}_enhanced.png"
                cv2.imwrite(str(frame_path), frame)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                pil_frames.append(Image.fromarray(frame_rgb))

            # Enhanced GIF Generation
            if len(pil_frames) > 1:
                gif_path = self.session_dir / "animations_final" / \
                    f"{image_path.stem}_ENHANCED.gif"

                if len(pil_frames) <= 4:
                    duration = 400
                elif len(pil_frames) <= 8:
                    duration = 250
                else:
                    duration = 200

                pil_frames[0].save(
                    gif_path, save_all=True, append_images=pil_frames[1:],
                    duration=duration, loop=0, disposal=2, optimize=True
                )

            processing_time = time.time() - start_time

            with self.thread_lock:
                self.total_frames_extracted += len(large_frames)
                self.total_frames_final += len(final_frames_with_stats)

            # Enhanced Report
            avg_frame_size = np.mean(
                [s['extracted_size'][0] * s['extracted_size'][1] for _, s in final_frames_with_stats])

            report = {
                "filename": image_path.name,
                "original_size": original_size,
                "upscaled_size": (w, h),
                "upscale_factor": "4x_enhanced",
                "frames_extracted": len(large_frames),
                "frames_final": len(final_frames_with_stats),
                "avg_frame_size": int(avg_frame_size),
                "padding_used": 96,
                "processing_time": round(processing_time, 2),
                "color_temperature": round(current_temp, 3),
                "iterations_completed": 7,
                "enhancements": [
                    "enhanced_background_detection",
                    "adaptive_masking",
                    "morphological_refinement",
                    "subtle_linework_enhancement",
                    "mild_color_optimization",
                    "gentle_saturation_boost",
                    "enhanced_sharpening_selection"
                ],
                "modifications": [
                    "4x_upscaling_instead_of_2x",
                    "reduced_cartoon_effects",
                    "color_temperature_analysis",
                    "increased_padding_96px",
                    "more_frames_retained"
                ],
                "thread_id": thread_id
            }

            report_path = self.session_dir / "iteration_reports" / \
                f"{image_path.stem}_enhanced_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            print(
                f"   ✅ SUCCESS: {len(final_frames_with_stats)}/{len(large_frames)} frames in {processing_time:.1f}s")
            return report

        except Exception as e:
            print(f"   ❌ ERROR [Thread {thread_id}]: {str(e)}")
            return None

    def apply_global_color_temperature_correction(self):
        """Globale Farbtemperatur-Korrektur über alle Animationen"""
        if not self.color_temperature_samples:
            return

        # Berechne Ziel-Farbtemperatur (Median für Robustheit)
        self.target_temperature = statistics.median(
            self.color_temperature_samples)

        print(f"\n🌡️ GLOBAL COLOR TEMPERATURE CORRECTION")
        print(f"   📊 Analyzed {len(self.color_temperature_samples)} images")
        print(f"   🎯 Target temperature: {self.target_temperature:.3f}")

        # Korrigiere alle gespeicherten Animationen
        animations_dir = self.session_dir / "animations_final"
        corrected_count = 0

        for gif_path in animations_dir.glob("*.gif"):
            try:
                # Lade GIF
                gif = Image.open(gif_path)
                frames = []

                try:
                    while True:
                        frame = gif.copy()
                        frame_array = np.array(frame.convert('RGB'))
                        frame_bgr = cv2.cvtColor(
                            frame_array, cv2.COLOR_RGB2BGR)

                        # Analysiere Farbtemperatur dieses Frames
                        frame_temp = self.analyze_color_temperature(frame_bgr)

                        # Korrigiere falls nötig
                        corrected_bgr = self.apply_color_temperature_correction(
                            frame_bgr, self.target_temperature, frame_temp)

                        corrected_rgb = cv2.cvtColor(
                            corrected_bgr, cv2.COLOR_BGR2RGB)
                        corrected_pil = Image.fromarray(corrected_rgb)
                        frames.append(corrected_pil)

                        gif.seek(gif.tell() + 1)

                except EOFError:
                    pass

                # Speichere korrigierte Animation
                if frames:
                    corrected_path = animations_dir / \
                        f"{gif_path.stem}_TEMP_CORRECTED.gif"

                    frames[0].save(
                        corrected_path, save_all=True, append_images=frames[1:],
                        duration=250, loop=0, disposal=2, optimize=True
                    )
                    corrected_count += 1

            except Exception as e:
                print(f"   ❌ Error correcting {gif_path.name}: {str(e)}")

        print(f"   ✅ Corrected {corrected_count} animations")

        # Speichere Color Temperature Report
        temp_report = {
            "target_temperature": self.target_temperature,
            "sample_count": len(self.color_temperature_samples),
            "temperature_range": {
                "min": min(self.color_temperature_samples),
                "max": max(self.color_temperature_samples),
                "std": statistics.stdev(self.color_temperature_samples) if len(self.color_temperature_samples) > 1 else 0
            },
            "corrected_animations": corrected_count
        }

        temp_report_path = self.session_dir / "color_analysis" / \
            "temperature_correction_report.json"
        with open(temp_report_path, 'w') as f:
            json.dump(temp_report, f, indent=2)

    def run_enhanced_workflow_with_specific_files(self, file_paths: List[str]):
        """Enhanced Workflow für spezifische Dateien"""
        print("🚀 STARTING ITERATION 7 - ENHANCED MODIFIED WORKFLOW")
        print("=" * 75)
        print("🔧 Modifikationen:")
        print("   • 4x Enhanced Upscaling (statt 2x)")
        print("   • Reduzierte Cartoon-Effekte (subtil)")
        print("   • Farbtemperatur-Homogenisierung")
        print("   • 96px Padding mit erweiterten Frames")
        print("   • Cross-Animation Color Harmony")
        print("=" * 75)

        self.start_time = time.time()
        self.create_session_directory()

        # Filter nur existierende Dateien
        valid_files = []
        for file_path in file_paths:
            path_obj = Path(file_path)
            if path_obj.exists() and path_obj.suffix.lower() in self.supported_formats:
                valid_files.append(path_obj)
            else:
                print(f"⚠️ File not found or unsupported: {file_path}")

        print(f"📊 Processing {len(valid_files)} specified files")
        print(f"🔧 Using {self.max_workers} parallel threads")

        # Multi-threaded processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_single_spritesheet_enhanced, file_path): file_path
                for file_path in valid_files
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

                if completed % 10 == 0:
                    print(
                        f"Progress: {completed}/{len(valid_files)} completed")

        # GLOBAL COLOR TEMPERATURE CORRECTION
        self.apply_global_color_temperature_correction()

        total_time = time.time() - self.start_time

        # Final Statistics
        print("\n" + "=" * 75)
        print("🎯 ITERATION 7 - ENHANCED MODIFIED COMPLETE")
        print("=" * 75)
        print(
            f"📊 Files processed: {len(self.processed_files)}/{len(valid_files)}")
        print(f"🎬 Total frames extracted: {self.total_frames_extracted}")
        print(f"✨ Final optimized frames: {self.total_frames_final}")
        print(f"⏱️ Processing time: {total_time:.1f} seconds")
        print(
            f"⚡ Speed: {self.total_frames_extracted/total_time:.2f} frames/second")
        print(f"🚀 Upscaling: 4x Enhanced (LANCZOS + CUBIC)")
        print(f"🎭 Effects: Subtle (reduced cartoon)")
        print(f"🌡️ Color correction: Homogenized temperature")
        print(f"📁 Results: {self.session_dir}")

        # Master Report
        master_report = {
            "iteration": "7_enhanced_modified",
            "timestamp": datetime.now().isoformat(),
            "modifications": {
                "upscaling": "4x_enhanced_lanczos_cubic",
                "cartoon_effects": "reduced_subtle",
                "color_temperature": "global_homogenization",
                "padding": "96px_extended_frames",
                "frame_retention": "increased_up_to_12_frames"
            },
            "total_files": len(valid_files),
            "files_processed": len(self.processed_files),
            "files_failed": len(self.failed_files),
            "total_frames_extracted": self.total_frames_extracted,
            "total_frames_final": self.total_frames_final,
            "processing_time_seconds": round(total_time, 2),
            "frames_per_second": round(self.total_frames_extracted/total_time, 2),
            "color_temperature_correction": {
                "target_temperature": self.target_temperature,
                "samples_analyzed": len(self.color_temperature_samples)
            },
            "technical_details": {
                "upscaling_method": "2-stage_lanczos_cubic_with_bilateral_filtering",
                "cartoon_reduction": "increased_k_means_colors_reduced_weights",
                "color_homogenization": "lab_colorspace_temperature_correction",
                "frame_extraction": "96px_padding_relaxed_filters",
                "multi_threading": f"{self.max_workers}x_parallel_processing"
            },
            "session_directory": str(self.session_dir),
            "detailed_reports": self.processed_files
        }

        master_path = self.session_dir / "ENHANCED_MODIFIED_MASTER_REPORT.json"
        with open(master_path, 'w') as f:
            json.dump(master_report, f, indent=2)

        return master_report


def main():
    # Liste der spezifizierten Dateien
    specified_files = [
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_48.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_47.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_45.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_44.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_43.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_39.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_40.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_38.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_36.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_34.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_33.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_35.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_32.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_26.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_24.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_28.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_25.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_23.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_30.png",
        "C:/Users/Public/ComfyUI-master/input/ChatGPT Image 29. Juni 2025, 10_23_21.png"
    ]

    processor = Iteration7ModifiedEnhanced()
    return processor.run_enhanced_workflow_with_specific_files(specified_files)


if __name__ == "__main__":
    main()
