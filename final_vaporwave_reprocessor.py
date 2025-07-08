#!/usr/bin/env python3
"""
🎬 FINAL VAPORWAVE REPROCESSOR - ORIGINAL WORKFLOW EDITION
=========================================================
Rekonstruiert den ursprünglichen Vaporwave-Workflow und verarbeitet ALLE Input Sprite Sheets
bis nur noch einzelne Frames und GIFs übrig sind.

BASIERT AUF: original_workflow_optimized.py + Iteration 1 Erfolg
ZIEL: 416 PNG Dateien → Einzelne Frames + GIFs
WORKFLOW: Original Background Detection + 37.5% Vaporwave Filter + GIF Generation
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from collections import Counter
from pathlib import Path
import os
import json
import time
from datetime import datetime
from typing import List, Tuple
import traceback


class FinalVaporwaveReprocessor:
    """Ursprünglicher Vaporwave-Workflow für komplette Neuverarbeitung"""

    def __init__(self, input_dir="input", output_base_dir="output/final_vaporwave_reprocess"):
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        self.session_dir = None
        self.processed_files = []
        self.failed_files = []
        self.total_frames_extracted = 0
        self.start_time = None

        # Supported formats
        self.supported_formats = {'.png', '.jpg',
                                  '.jpeg', '.bmp', '.tiff', '.tif'}

    def create_session_directory(self):
        """Erstellt Session-Verzeichnis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_base_dir / \
            f"vaporwave_session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Verzeichnisstruktur
        (self.session_dir / "individual_frames").mkdir(exist_ok=True)
        (self.session_dir / "animations").mkdir(exist_ok=True)
        (self.session_dir / "reports").mkdir(exist_ok=True)

        print(f"📁 Session directory created: {self.session_dir}")

    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Bildqualität verbessern vor Hintergrund-Analyse"""
        print("   🎨 Enhancing image quality...")

        # Konvertiere zu PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # 1. Kontrast erhöhen
        contrast_enhancer = ImageEnhance.Contrast(pil_image)
        enhanced_image = contrast_enhancer.enhance(1.15)

        # 2. Weißabgleich
        image_array = np.array(enhanced_image, dtype=np.float32)
        gamma = 0.9
        gamma_corrected = 255.0 * (image_array / 255.0) ** gamma
        gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)
        enhanced_image = Image.fromarray(gamma_corrected)

        # 3. Farbintensität erhöhen
        color_enhancer = ImageEnhance.Color(enhanced_image)
        final_enhanced = color_enhancer.enhance(1.2)

        # Zurück zu OpenCV Format
        final_array = np.array(final_enhanced)
        result_bgr = cv2.cvtColor(final_array, cv2.COLOR_RGB2BGR)

        return result_bgr

    def upscale_and_sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """2x Hochskalierung mit Schärfung"""
        print("   🔍 2x Upscaling + Sharpening...")

        h, w = image.shape[:2]

        # 2x Upscaling mit INTER_CUBIC
        upscaled = cv2.resize(image, (w * 2, h * 2),
                              interpolation=cv2.INTER_CUBIC)

        # Bildverbesserungen
        enhanced = self.enhance_image_quality(upscaled)

        # Schärfung
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(enhanced_rgb)

        sharpened = pil_image.filter(ImageFilter.UnsharpMask(
            radius=1.0, percent=150, threshold=2))

        # Zurück zu OpenCV
        sharpened_array = np.array(sharpened)
        result = cv2.cvtColor(sharpened_array, cv2.COLOR_RGB2BGR)

        print(f"   📈 Upscaled: {w}x{h} -> {result.shape[1]}x{result.shape[0]}")
        return result

    def detect_background_color_original(self, image_np, corner_size_ratio=30.0):
        """Original Background Detection (4 Ecken)"""
        h, w = image_np.shape[:2]
        corner_size = max(1, int(min(h, w) / corner_size_ratio))

        # Sammle Pixel aus allen vier Ecken
        corners = [
            image_np[0:corner_size, 0:corner_size],  # Top-left
            image_np[0:corner_size, w-corner_size:w],  # Top-right
            image_np[h-corner_size:h, 0:corner_size],  # Bottom-left
            image_np[h-corner_size:h, w-corner_size:w]  # Bottom-right
        ]

        corner_pixels = []
        for corner in corners:
            if len(corner.shape) == 3:
                pixels = corner.reshape(-1, 3)
                for pixel in pixels:
                    corner_pixels.append(tuple(pixel))

        if corner_pixels:
            bg_color = Counter(corner_pixels).most_common(1)[0][0]
            return np.array(bg_color, dtype=np.uint8)
        else:
            return np.array([255, 255, 255], dtype=np.uint8)

    def create_traditional_mask_original(self, image: np.ndarray, bg_color: np.ndarray) -> np.ndarray:
        """Original traditionelle Vordergrund-Maske"""
        tolerance = 25
        diff = np.abs(image.astype(int) - bg_color.astype(int))
        background_mask = np.all(diff <= tolerance, axis=2)
        foreground_mask = ~background_mask

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        foreground_mask = cv2.morphologyEx(
            foreground_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        foreground_mask = cv2.morphologyEx(
            foreground_mask, cv2.MORPH_OPEN, kernel)

        return foreground_mask

    def apply_vaporwave_filter_original(self, frame: np.ndarray, intensity: float = 0.375) -> np.ndarray:
        """ORIGINAL Vaporwave Filter - 37.5% Intensität"""

        # Konvertiere zu PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)

        # 1. Farbverschiebung zu Cyan/Magenta Tönen
        r, g, b = pil_frame.split()

        r_array = np.array(r, dtype=np.float32)
        g_array = np.array(g, dtype=np.float32)
        b_array = np.array(b, dtype=np.float32)

        # Vaporwave Farbverschiebung
        r_shifted = r_array * (1.0 - intensity * 0.3)  # Weniger Rot
        g_shifted = g_array * (1.0 + intensity * 0.2)  # Etwas mehr Grün
        b_shifted = b_array * (1.0 + intensity * 0.4)  # Mehr Blau

        # Clamp zu 0-255
        r_shifted = np.clip(r_shifted, 0, 255).astype(np.uint8)
        g_shifted = np.clip(g_shifted, 0, 255).astype(np.uint8)
        b_shifted = np.clip(b_shifted, 0, 255).astype(np.uint8)

        # Zurück zu PIL
        vaporwave_frame = Image.merge('RGB', [
            Image.fromarray(r_shifted),
            Image.fromarray(g_shifted),
            Image.fromarray(b_shifted)
        ])

        # 2. Sättigungserhöhung
        enhancer = ImageEnhance.Color(vaporwave_frame)
        vaporwave_frame = enhancer.enhance(1.0 + intensity * 0.5)

        # 3. Kontrast für den "Pop"
        contrast_enhancer = ImageEnhance.Contrast(vaporwave_frame)
        vaporwave_frame = contrast_enhancer.enhance(1.0 + intensity * 0.3)

        # Zurück zu OpenCV
        result_array = np.array(vaporwave_frame)
        result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)

        return result_bgr

    def process_extracted_frame(self, frame: np.ndarray, bg_color: np.ndarray) -> np.ndarray:
        """Frame Processing + Vaporwave + Transparenz"""

        # Vaporwave Filter anwenden
        vaporwave_frame = self.apply_vaporwave_filter_original(frame)

        # Background Removal für Transparenz
        frame_rgb = cv2.cvtColor(vaporwave_frame, cv2.COLOR_BGR2RGB)
        frame_rgba = np.dstack(
            [frame_rgb, np.full(frame_rgb.shape[:2], 255, dtype=np.uint8)])

        # Background removal
        tolerance = 40
        frame_diff = np.abs(frame_rgb.astype(int) - bg_color.astype(int))
        frame_bg_mask = np.all(frame_diff <= tolerance, axis=2)

        # Set transparent background
        frame_rgba[frame_bg_mask, 3] = 0

        return frame_rgba

    def extract_frames_with_padding(self, image: np.ndarray, mask: np.ndarray, min_area=800) -> List[np.ndarray]:
        """Frame Extraction + 30px Padding"""
        print("   📦 Extracting frames with 30px padding...")

        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)

        frames = []
        h, w = image.shape[:2]

        for i in range(1, num_labels):  # Skip background label 0
            x, y, width, height, area = stats[i]

            # Filters
            if area < min_area:
                continue
            if width < 20 or height < 20:
                continue
            if width / height > 10 or height / width > 10:
                continue

            # 30 Pixel Padding
            padding = 30
            x_padded = max(0, x - padding)
            y_padded = max(0, y - padding)
            x2_padded = min(w, x + width + padding)
            y2_padded = min(h, y + height + padding)

            # Extrahiere Frame mit Padding
            padded_frame = image[y_padded:y2_padded, x_padded:x2_padded]

            if padded_frame.size > 0:
                frames.append(padded_frame)

        print(f"   ✅ Total frames extracted: {len(frames)}")
        return frames

    def process_single_spritesheet(self, image_path):
        """Verarbeitet ein Spritesheet mit Original Vaporwave Workflow"""
        print(f"\n🎮 Processing: {image_path.name}")

        try:
            # Lade Bild
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Could not load image")

            original_h, original_w = image.shape[:2]
            print(f"   📐 Original Size: {original_w}x{original_h}")

            # SCHRITT 1: 2x Upscaling + Schärfung
            upscaled_image = self.upscale_and_sharpen_image(image)
            h, w = upscaled_image.shape[:2]

            # SCHRITT 2: Background Detection
            print("   🎯 Background detection...")
            bg_color = self.detect_background_color_original(upscaled_image)
            print(f"   🎨 Background: RGB{tuple(bg_color)}")

            # SCHRITT 3: Background Mask
            foreground_mask = self.create_traditional_mask_original(
                upscaled_image, bg_color)

            # SCHRITT 4: Frame Extraction
            print("   📦 Frame extraction...")
            extracted_frames = self.extract_frames_with_padding(
                upscaled_image, foreground_mask)

            if not extracted_frames:
                print("   ⚠️ No frames extracted - saving whole image as single frame")
                extracted_frames = [upscaled_image]

            # Frame Processing
            frame_info = []
            processed_frames = []

            # Erstelle Ausgabeverzeichnis
            sprite_dir = self.session_dir / "individual_frames" / image_path.stem
            sprite_dir.mkdir(exist_ok=True)

            for i, frame in enumerate(extracted_frames, 1):
                # SCHRITT 5: Frame Processing + Vaporwave
                frame_rgba = self.process_extracted_frame(frame, bg_color)

                # Speichere Frame
                frame_filename = f"vaporwave_frame_{i:03d}.png"
                frame_path = sprite_dir / frame_filename
                Image.fromarray(frame_rgba, 'RGBA').save(frame_path)

                processed_frames.append(frame_rgba)

                frame_info.append({
                    'id': i,
                    'filename': frame_filename,
                    'size': f"{frame.shape[1]}x{frame.shape[0]}",
                    'area': int(frame.shape[0] * frame.shape[1])
                })

            print(
                f"   ✅ Processed {len(extracted_frames)} frames with vaporwave effect")

            # Erstelle GIF Animation
            gif_path = None
            if processed_frames:
                gif_filename = f"{image_path.stem}_vaporwave.gif"
                gif_path = self.session_dir / "animations" / gif_filename

                pil_frames = [Image.fromarray(frame, 'RGBA')
                              for frame in processed_frames]
                pil_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=500,  # Original
                    loop=0,
                    transparency=0,
                    disposal=2
                )
                print(f"   🎬 GIF created: {gif_filename}")

            # Erstelle Report
            report = {
                "input_file": str(image_path),
                "processing_timestamp": datetime.now().isoformat(),
                "workflow": "Final Vaporwave Reprocess - Original",
                "original_image_size": [original_w, original_h],
                "upscaled_image_size": [w, h],
                "background_color": bg_color.tolist(),
                "total_frames": len(extracted_frames),
                "frames": frame_info,
                "gif_animation": str(gif_path) if gif_path else None,
                "vaporwave_intensity": "37.5%"
            }

            # Speichere Report
            report_path = self.session_dir / "reports" / \
                f"{image_path.stem}_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            self.processed_files.append(str(image_path))
            self.total_frames_extracted += len(extracted_frames)

            return True

        except Exception as e:
            print(f"   ❌ Error processing {image_path.name}: {str(e)}")
            self.failed_files.append(str(image_path))
            return False

    def create_master_report(self):
        """Erstellt Master Report"""
        end_time = time.time()
        total_time = end_time - self.start_time

        master_report = {
            "session": "Final Vaporwave Reprocess",
            "timestamp": datetime.now().isoformat(),
            "workflow": "Original Vaporwave (37.5% intensity)",
            "total_files_found": len(list(self.input_dir.rglob("*.png"))),
            "processed_files": len(self.processed_files),
            "failed_files": len(self.failed_files),
            "success_rate": f"{len(self.processed_files) / (len(self.processed_files) + len(self.failed_files)) * 100:.1f}%" if (len(self.processed_files) + len(self.failed_files)) > 0 else "0%",
            "total_frames_extracted": self.total_frames_extracted,
            "total_processing_time": f"{total_time:.2f}s",
            "frames_per_second": f"{self.total_frames_extracted / total_time:.2f}" if total_time > 0 else "0",
            "output_structure": {
                "individual_frames": str(self.session_dir / "individual_frames"),
                "animations": str(self.session_dir / "animations"),
                "reports": str(self.session_dir / "reports")
            },
            "original_workflow_features": [
                "2x Upscaling with INTER_CUBIC",
                "4-corner background detection",
                "30px frame padding",
                "37.5% vaporwave intensity",
                "Transparency support",
                "GIF animation generation"
            ]
        }

        master_path = self.session_dir / "FINAL_VAPORWAVE_MASTER_REPORT.json"
        with open(master_path, 'w') as f:
            json.dump(master_report, f, indent=2)

        print(f"\n📊 Master report saved: {master_path}")

    def run_complete_reprocessing(self):
        """Läuft durch ALLE PNG Dateien und verarbeitet sie"""
        print("🎬 FINAL VAPORWAVE REPROCESSOR - STARTING COMPLETE REPROCESSING")
        print("=" * 80)

        self.start_time = time.time()
        self.create_session_directory()

        # Finde alle PNG Dateien
        png_files = list(self.input_dir.rglob("*.png"))
        total_files = len(png_files)

        print(f"\n📊 Found {total_files} PNG files to process")
        print(f"🎯 Target: Convert all sprite sheets to individual frames + GIFs")
        print(f"🌈 Using: Original Vaporwave Workflow (37.5% intensity)")

        # Verarbeite alle Dateien
        for i, image_path in enumerate(png_files, 1):
            print(f"\n[{i}/{total_files}] Processing: {image_path.name}")
            self.process_single_spritesheet(image_path)

            # Progress Update
            if i % 10 == 0:
                print(f"\n📈 Progress: {i}/{total_files} files processed")

        # Master Report
        self.create_master_report()

        print(f"\n🎉 FINAL VAPORWAVE REPROCESSING COMPLETED!")
        print(f"✅ Processed: {len(self.processed_files)} files")
        print(f"❌ Failed: {len(self.failed_files)} files")
        print(f"🎬 Total Frames: {self.total_frames_extracted}")
        print(f"📁 Output: {self.session_dir}")


def main():
    """Main execution"""
    reprocessor = FinalVaporwaveReprocessor()
    reprocessor.run_complete_reprocessing()


if __name__ == "__main__":
    main()
