#!/usr/bin/env python3
"""
Enhanced Workflow Runner für spezifische Spritesheet-Dateien
Modifikationen des Iteration 7 Workflows:
1. Reduzierter Cartoon-Effekt
2. Erhöhtes Upscaling (4x)
3. Farbtemperatur-Homogenisierung
4. BRIGHTNESS CORRECTION - Löst das Darkness Problem
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import statistics
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the original iteration 7 processor
from iteration_7_cartoon_multipass import Iteration7CartoonMultipass

class EnhancedIteration7Processor(Iteration7CartoonMultipass):
    """Enhanced version with reduced cartoon effects, 4x upscaling, brightness correction, and color temperature homogenization"""

    def __init__(self, input_dir="input", output_base_dir="output/iteration_7_enhanced"):
        super().__init__(input_dir, output_base_dir)
        self.color_temperature_samples = []
        self.target_temperature = None
        self.brightness_samples = []
        self.target_brightness = None

    def enhanced_upscaling_4x(self, image: np.ndarray) -> np.ndarray:
        """4x Enhanced Upscaling instead of 2x"""
        h, w = image.shape[:2]

        # Stage 1: 2x with LANCZOS
        stage1 = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        denoised1 = cv2.bilateralFilter(stage1, 9, 75, 75)

        # Stage 2: Another 2x with CUBIC
        stage2 = cv2.resize(denoised1, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        final_upscaled = cv2.bilateralFilter(stage2, 15, 80, 80)

        return final_upscaled

    def analyze_brightness(self, image: np.ndarray) -> float:
        """Analyze average brightness of an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0  # Normalize to 0-1
        return brightness

    def apply_brightness_correction(self, image: np.ndarray, target_brightness: float, current_brightness: float) -> np.ndarray:
        """Apply brightness correction to match target"""
        if abs(target_brightness - current_brightness) < 0.05:
            return image

        # Calculate correction factor
        correction_factor = target_brightness / (current_brightness + 0.01)  # Avoid division by zero
        correction_factor = np.clip(correction_factor, 0.7, 1.8)  # Limit extreme corrections

        # Apply brightness correction
        corrected = image.astype(np.float32) * correction_factor
        corrected = np.clip(corrected, 0, 255)

        return corrected.astype(np.uint8)

    def analyze_color_temperature(self, image: np.ndarray) -> float:
        """Analyze color temperature of an image"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        a_channel = lab[:, :, 1].astype(np.float32) - 128
        b_channel = lab[:, :, 2].astype(np.float32) - 128

        avg_a = np.mean(a_channel)
        avg_b = np.mean(b_channel)
        temperature_factor = avg_b / (abs(avg_a) + 1)

        return temperature_factor

    def apply_color_temperature_correction(self, image: np.ndarray, target_temp: float, current_temp: float) -> np.ndarray:
        """Apply color temperature correction"""
        temp_diff = target_temp - current_temp

        if abs(temp_diff) < 0.1:
            return image

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        adjustment_factor = temp_diff * 0.15  # Very mild adjustment
        lab[:, :, 2] = np.clip(lab[:, :, 2] + adjustment_factor, 0, 255)

        corrected = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return corrected

    def iteration_4_gentle_linework_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """VERY GENTLE Linework Enhancement - Prevents Darkening"""
        # Much gentler parameters to prevent darkening
        smooth = cv2.bilateralFilter(frame, 5, 30, 30)  # Very reduced parameters

        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        # Much higher threshold for minimal edge enhancement
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 15)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))

        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # Very minimal effect to prevent darkening
        enhanced = cv2.addWeighted(smooth, 0.98, 255 - edges_colored, 0.02, 0)

        return enhanced

    def iteration_5_minimal_color_optimization(self, frame: np.ndarray) -> np.ndarray:
        """MINIMAL Color Quantization - Prevents Darkening"""
        data = frame.reshape((-1, 3))
        data = np.float32(data)

        # Very high K values for minimal quantization
        frame_area = frame.shape[0] * frame.shape[1]
        k = min(max(32, frame_area // 1500), 128)  # Much higher color count

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 2, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized = quantized_data.reshape(frame.shape)

        # Minimal mixing to preserve brightness
        cartoon = cv2.addWeighted(frame, 0.9, quantized, 0.1, 0)

        return cartoon

    def iteration_6_brightness_preserving_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """BRIGHTNESS PRESERVING Enhancement - Prevents Darkening"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)

        # Brightness-preserving enhancements
        color_enhancer = ImageEnhance.Color(pil_frame)
        enhanced_frame = color_enhancer.enhance(1.08)  # Mild saturation

        contrast_enhancer = ImageEnhance.Contrast(enhanced_frame)
        enhanced_frame = contrast_enhancer.enhance(1.05)  # Very mild contrast

        # BRIGHTNESS BOOST to counter any darkening
        brightness_enhancer = ImageEnhance.Brightness(enhanced_frame)
        enhanced_frame = brightness_enhancer.enhance(1.15)  # Brightness boost

        return cv2.cvtColor(np.array(enhanced_frame), cv2.COLOR_RGB2BGR)

    def process_single_spritesheet_enhanced(self, image_path):
        """Enhanced processing with brightness correction and reduced cartoon effects"""
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

            # BRIGHTNESS AND COLOR TEMPERATURE ANALYSIS
            current_brightness = self.analyze_brightness(upscaled)
            current_temp = self.analyze_color_temperature(upscaled)

            with self.thread_lock:
                self.brightness_samples.append(current_brightness)
                self.color_temperature_samples.append(current_temp)

            print(f"   💡 Brightness: {current_brightness:.3f}, Temperature: {current_temp:.3f}")

            # ITERATIONS 1-3: Background processing (unchanged)
            bg_color, zones = self.iteration_1_enhanced_background_detection(upscaled)
            initial_mask = self.iteration_2_adaptive_masking(upscaled, bg_color)
            refined_mask = self.iteration_3_morphological_refinement(initial_mask)

            # Frame extraction with larger padding
            large_frames = self.extract_large_frames_with_overlap(upscaled, refined_mask)
            if not large_frames:
                return None

            print(f"   📦 Extracted {len(large_frames)} frames")

            # ITERATIONS 4-6: BRIGHTNESS-PRESERVING Enhancement
            enhanced_frames = []
            for frame, stats in large_frames:
                # Use brightness-preserving methods
                line_enhanced = self.iteration_4_gentle_linework_enhancement(frame[:, :, :3])
                color_optimized = self.iteration_5_minimal_color_optimization(line_enhanced)
                final_colored = self.iteration_6_brightness_preserving_enhancement(color_optimized)

                final_rgba = cv2.cvtColor(final_colored, cv2.COLOR_BGR2BGRA)
                final_rgba[:, :, 3] = frame[:, :, 3]

                enhanced_frames.append((final_rgba, stats))

            # ITERATION 7: Final processing with brightness preservation
            frames_only = [frame for frame, _ in enhanced_frames]
            sharpened_frames = self.iteration_7_sharpening_and_selection(frames_only)

            final_frames_with_stats = []
            for i, sharpened in enumerate(sharpened_frames):
                if i < len(enhanced_frames):
                    stats = enhanced_frames[i][1]
                    final_frames_with_stats.append((sharpened, stats))

            if not final_frames_with_stats:
                return None

            # Save results
            sprite_dir = self.session_dir / "individual_sprites" / image_path.stem
            sprite_dir.mkdir(exist_ok=True)

            pil_frames = []
            for i, (frame, stats) in enumerate(final_frames_with_stats):
                frame_path = sprite_dir / f"frame_{i:03d}_enhanced.png"
                cv2.imwrite(str(frame_path), frame)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                pil_frames.append(Image.fromarray(frame_rgb))

            # Create enhanced GIF
            if len(pil_frames) > 1:
                gif_path = self.session_dir / "animations_final" / f"{image_path.stem}_ENHANCED.gif"

                duration = 300 if len(pil_frames) <= 4 else 200

                pil_frames[0].save(
                    gif_path, save_all=True, append_images=pil_frames[1:],
                    duration=duration, loop=0, disposal=2, optimize=True
                )

            processing_time = time.time() - start_time

            with self.thread_lock:
                self.total_frames_extracted += len(large_frames)
                self.total_frames_final += len(final_frames_with_stats)

            # Create report
            report = {
                "filename": image_path.name,
                "original_size": original_size,
                "upscaled_size": (w, h),
                "upscale_factor": "4x_enhanced",
                "frames_extracted": len(large_frames),
                "frames_final": len(final_frames_with_stats),
                "processing_time": round(processing_time, 2),
                "brightness": round(current_brightness, 3),
                "color_temperature": round(current_temp, 3),
                "enhancements": "brightness_preserving_minimal_cartoon",
                "thread_id": thread_id
            }

            print(f"   ✅ SUCCESS: {len(final_frames_with_stats)} frames in {processing_time:.1f}s")
            return report

        except Exception as e:
            print(f"   ❌ ERROR [Thread {thread_id}]: {str(e)}")
            return None

    def apply_global_corrections(self):
        """Apply global brightness and color temperature corrections"""
        if not self.brightness_samples or not self.color_temperature_samples:
            return

        # Calculate target values (prefer brighter images)
        self.target_brightness = min(0.7, statistics.median(self.brightness_samples) * 1.2)  # Boost brightness
        self.target_temperature = statistics.median(self.color_temperature_samples)

        print(f"\n🌡️ GLOBAL CORRECTIONS")
        print(f"   💡 Target brightness: {self.target_brightness:.3f}")
        print(f"   🌡️ Target temperature: {self.target_temperature:.3f}")

        # Correct all saved animations
        animations_dir = self.session_dir / "animations_final"
        corrected_count = 0

        for gif_path in animations_dir.glob("*_ENHANCED.gif"):
            try:
                gif = Image.open(gif_path)
                frames = []

                try:
                    while True:
                        frame = gif.copy()
                        frame_array = np.array(frame.convert('RGB'))
                        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)

                        # Apply brightness correction
                        frame_brightness = self.analyze_brightness(frame_bgr)
                        brightness_corrected = self.apply_brightness_correction(
                            frame_bgr, self.target_brightness, frame_brightness)

                        # Apply color temperature correction
                        frame_temp = self.analyze_color_temperature(brightness_corrected)
                        final_corrected = self.apply_color_temperature_correction(
                            brightness_corrected, self.target_temperature, frame_temp)

                        corrected_rgb = cv2.cvtColor(final_corrected, cv2.COLOR_BGR2RGB)
                        corrected_pil = Image.fromarray(corrected_rgb)
                        frames.append(corrected_pil)

                        gif.seek(gif.tell() + 1)

                except EOFError:
                    pass

                if frames:
                    corrected_path = animations_dir / f"{gif_path.stem}_CORRECTED.gif"

                    frames[0].save(
                        corrected_path, save_all=True, append_images=frames[1:],
                        duration=250, loop=0, disposal=2, optimize=True
                    )
                    corrected_count += 1

            except Exception as e:
                print(f"   ❌ Error correcting {gif_path.name}: {str(e)}")

        print(f"   ✅ Corrected {corrected_count} animations")

    def run_enhanced_workflow_with_specific_files(self, file_paths):
        """Run enhanced workflow with specific files"""
        print("🚀 STARTING ENHANCED ITERATION 7 WORKFLOW")
        print("=" * 75)
        print("🔧 Modifikationen:")
        print("   • 4x Enhanced Upscaling (statt 2x)")
        print("   • Reduzierte Cartoon-Effekte")
        print("   • BRIGHTNESS CORRECTION (löst Darkness Problem)")
        print("   • Farbtemperatur-Homogenisierung")
        print("=" * 75)

        self.start_time = time.time()
        self.create_session_directory()

        # Filter valid files
        valid_files = []
        for file_path in file_paths:
            path_obj = Path(file_path)
            if path_obj.exists() and path_obj.suffix.lower() in self.supported_formats:
                valid_files.append(path_obj)
            else:
                print(f"⚠️ File not found: {file_path}")

        print(f"📊 Processing {len(valid_files)} files")

        # Process files
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_single_spritesheet_enhanced, file_path): file_path
                for file_path in valid_files
            }

            completed = 0
            for future in as_completed(future_to_file):
                try:
                    result = future.result()
                    if result:
                        self.processed_files.append(result)
                    completed += 1

                    if completed % 5 == 0:
                        print(f"Progress: {completed}/{len(valid_files)} completed")

                except Exception as e:
                    print(f"❌ Processing error: {str(e)}")

        # Apply global corrections
        self.apply_global_corrections()

        total_time = time.time() - self.start_time

        print(f"\n✅ ENHANCED WORKFLOW COMPLETE")
        print(f"📊 Files processed: {len(self.processed_files)}")
        print(f"💡 Brightness optimized, darkness issue resolved")
        print(f"⏱️ Total time: {total_time:.1f} seconds")
        print(f"📁 Results: {self.session_dir}")

        return self.session_dir


def main():
    """Main function to run enhanced workflow with specified files"""

    # Liste der spezifizierten Dateien (erste 20 für Test)
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

    processor = EnhancedIteration7Processor()
    result_dir = processor.run_enhanced_workflow_with_specific_files(specified_files)

    print(f"\n🎯 Enhanced workflow completed!")
    print(f"📁 Results saved to: {result_dir}")
    print(f"💡 Brightness issue resolved - images should be much lighter!")

    return result_dir


if __name__ == "__main__":
    main()
