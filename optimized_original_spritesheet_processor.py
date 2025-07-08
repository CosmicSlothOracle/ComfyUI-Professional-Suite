#!/usr/bin/env python3
"""
OPTIMIZED ORIGINAL SPRITESHEET PROCESSOR
Zurück zum ursprünglichen erfolgreichen Workflow + Optimierungen:
1. 2x Upscaling, Schärfung, dann original background removal, dann Vaporwave Filter 25%.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import os
import time
from datetime import datetime
from typing import List, Tuple


class OptimizedOriginalProcessor:
    """Zurück zum ursprünglichen Workflow mit Optimierungen"""

    def __init__(self, input_dir="input", output_base_dir="output/optimized_original"):
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
            f"original_optimized_{timestamp}"

        (self.session_dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.session_dir / "gifs").mkdir(parents=True, exist_ok=True)

    def upscale_and_sharpen(self, image: np.ndarray) -> np.ndarray:
        """SCHRITT 1: 2x Hochskalierung mit leichter Schärfung"""
        print("   🔍 2x Upscaling + Sharpening...")

        # 2x Upscaling mit INTER_CUBIC (beste Qualität)
        h, w = image.shape[:2]
        upscaled = cv2.resize(image, (w * 2, h * 2),
                              interpolation=cv2.INTER_CUBIC)

        # Konvertiere zu PIL für Schärfung
        upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(upscaled_rgb)

        # Leichte Schärfung (nicht zu stark für Pixel Art)
        sharpened = pil_image.filter(ImageFilter.UnsharpMask(
            radius=1.0, percent=150, threshold=2))

        # Zurück zu OpenCV Format
        sharpened_array = np.array(sharpened)
        result = cv2.cvtColor(sharpened_array, cv2.COLOR_RGB2BGR)

        print(f"   📈 Upscaled: {w}x{h} -> {result.shape[1]}x{result.shape[0]}")
        return result

    def original_background_detection(self, image: np.ndarray) -> np.ndarray:
        """SCHRITT 2: Original Background Detection (4 Ecken, 20x20)"""
        h, w = image.shape[:2]

        # Original 4-Corner Sampling (20x20 pixels)
        corner_size = 20
        corners = [
            image[:corner_size, :corner_size],           # Top-left
            image[:corner_size, -corner_size:],          # Top-right
            image[-corner_size:, :corner_size],          # Bottom-left
            image[-corner_size:, -corner_size:]          # Bottom-right
        ]

        bg_colors = []
        for corner in corners:
            if corner.size > 0:
                # Einfache Durchschnittsfarbe
                avg_color = np.mean(corner.reshape(-1, 3), axis=0)
                bg_colors.append(avg_color.astype(np.uint8))

        # Haupthintergrundfarbe
        if bg_colors:
            main_bg_color = np.mean(bg_colors, axis=0).astype(np.uint8)
        else:
            main_bg_color = np.array([255, 255, 255], dtype=np.uint8)

        return main_bg_color

    def original_background_removal(self, image: np.ndarray, bg_color: np.ndarray,
                                    tolerance: int = 25) -> np.ndarray:
        """SCHRITT 2: Original Background Removal Method"""
        # Original simple color distance
        diff = np.abs(image.astype(int) - bg_color.astype(int))
        mask = np.all(diff <= tolerance, axis=2)

        # Foreground mask (inverse)
        foreground_mask = ~mask

        # Original simple morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foreground_mask = cv2.morphologyEx(
            foreground_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        foreground_mask = cv2.morphologyEx(
            foreground_mask, cv2.MORPH_OPEN, kernel)

        return foreground_mask

    def original_frame_extraction(self, image: np.ndarray, foreground_mask: np.ndarray) -> List[np.ndarray]:
        """SCHRITT 3: Original Frame Extraction Method"""
        # Original Connected Components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            foreground_mask, connectivity=8)

        extracted_frames = []
        h, w = image.shape[:2]

        # Original parameters
        min_area = max(500, (w * h) // 2000)  # Original threshold

        for i in range(1, num_labels):  # Skip background label 0
            area = stats[i, cv2.CC_STAT_AREA]

            if area < min_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w_comp = stats[i, cv2.CC_STAT_WIDTH]
            h_comp = stats[i, cv2.CC_STAT_HEIGHT]

            # Original aspect ratio filter
            aspect_ratio = w_comp / h_comp if h_comp > 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:  # Original values
                continue

            # Extract frame with padding
            padding = 5  # Original padding
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image.shape[1], x + w_comp + padding)
            y_end = min(image.shape[0], y + h_comp + padding)

            frame = image[y_start:y_end, x_start:x_end]

            if frame.size > 0:
                extracted_frames.append(frame)

        return extracted_frames

    def apply_vaporwave_filter(self, frame: np.ndarray, intensity: float = 0.25) -> np.ndarray:
        """SCHRITT 4: Klassischer Vaporwave Filter auf 25%"""

        # Konvertiere zu PIL für bessere Filter-Kontrolle
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)

        # VAPORWAVE COLOR PALETTE (klassisch)
        # Cyan, Magenta, Pink, Purple Tones

        # 1. Leichte Farbverschiebung zu Cyan/Magenta
        r, g, b = pil_frame.split()

        # Erhöhe Cyan (weniger Rot, mehr Grün+Blau)
        r_array = np.array(r, dtype=np.float32)
        g_array = np.array(g, dtype=np.float32)
        b_array = np.array(b, dtype=np.float32)

        # Vaporwave Farbverschiebung (sanft bei 25%)
        r_shifted = r_array * (1.0 - intensity * 0.3)  # Weniger Rot
        g_shifted = g_array * (1.0 + intensity * 0.2)  # Mehr Grün
        # Mehr Blau (Cyan-Effekt)
        b_shifted = b_array * (1.0 + intensity * 0.4)

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

        # 2. Leichte Sättigung erhöhen für Vaporwave Look
        enhancer = ImageEnhance.Color(vaporwave_frame)
        vaporwave_frame = enhancer.enhance(1.0 + intensity * 0.5)

        # 3. Leichter Kontrast für Pop
        enhancer = ImageEnhance.Contrast(vaporwave_frame)
        vaporwave_frame = enhancer.enhance(1.0 + intensity * 0.3)

        # Zurück zu OpenCV Format
        result_array = np.array(vaporwave_frame)
        result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)

        return result_bgr

    def create_transparent_frame(self, frame: np.ndarray, bg_color: np.ndarray,
                                 tolerance: int = 25) -> np.ndarray:
        """Erstellt transparenten Frame für GIF"""
        # Background mask erstellen
        diff = np.abs(frame.astype(int) - bg_color.astype(int))
        bg_mask = np.all(diff <= tolerance, axis=2)

        # Zu RGBA konvertieren
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgba = np.dstack(
            [frame_rgb, np.full(frame_rgb.shape[:2], 255, dtype=np.uint8)])

        # Hintergrund transparent setzen
        frame_rgba[bg_mask, 3] = 0

        return frame_rgba

    def process_single_spritesheet(self, image_path: Path) -> int:
        """Verarbeitet ein Spritesheet mit Original + Optimized Workflow"""
        try:
            print(f"🎮 Processing: {image_path.name}")

            # Lade Bild
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Could not load image")

            original_h, original_w = image.shape[:2]
            print(f"   📐 Original Size: {original_w}x{original_h}")

            # SCHRITT 1: 2x Upscaling + Schärfung
            upscaled_image = self.upscale_and_sharpen(image)

            # SCHRITT 2: Original Background Detection
            print("   🔍 Original background detection...")
            bg_color = self.original_background_detection(upscaled_image)
            print(f"   🎯 Background color: {bg_color}")

            # SCHRITT 2: Original Background Removal
            foreground_mask = self.original_background_removal(
                upscaled_image, bg_color)

            # SCHRITT 3: Original Frame Extraction
            print("   📦 Original frame extraction...")
            extracted_frames = self.original_frame_extraction(
                upscaled_image, foreground_mask)
            print(f"   📦 Extracted: {len(extracted_frames)} frames")

            # SCHRITT 4: Vaporwave Filter auf jeden Frame
            print("   🌈 Applying Vaporwave filter (25%)...")
            vaporwave_frames = []
            for frame in extracted_frames:
                vaporwave_frame = self.apply_vaporwave_filter(
                    frame, intensity=0.25)
                vaporwave_frames.append(vaporwave_frame)

            # Speichere Frames mit Transparenz
            sprite_dir = self.session_dir / "frames" / image_path.stem
            sprite_dir.mkdir(exist_ok=True)

            transparent_frames = []
            for i, frame in enumerate(vaporwave_frames):
                frame_filename = f"frame_{i+1:03d}.png"
                frame_path = sprite_dir / frame_filename

                # Erstelle transparenten Frame
                transparent_frame = self.create_transparent_frame(
                    frame, bg_color)
                Image.fromarray(transparent_frame, 'RGBA').save(frame_path)
                transparent_frames.append(transparent_frame)

            # Erstelle GIF
            if transparent_frames:
                gif_path = self.session_dir / "gifs" / \
                    f"{image_path.stem}_vaporwave.gif"

                pil_frames = [Image.fromarray(frame, 'RGBA')
                              for frame in transparent_frames]
                pil_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=500,
                    loop=0,
                    transparency=0,
                    disposal=2
                )
                print(f"   🎬 Vaporwave GIF created: {gif_path.name}")

            # Erfolgreich verarbeitet
            self.processed_files.append({
                'file': image_path.name,
                'original_size': f"{original_w}x{original_h}",
                'upscaled_size': f"{upscaled_image.shape[1]}x{upscaled_image.shape[0]}",
                'frames': len(vaporwave_frames),
                'background_color': [int(c) for c in bg_color],
                'vaporwave_applied': True
            })

            self.total_frames_extracted += len(vaporwave_frames)
            return len(vaporwave_frames)

        except Exception as e:
            print(f"   ❌ Error processing {image_path.name}: {e}")
            self.failed_files.append({
                'file': image_path.name,
                'error': str(e)
            })
            return 0

    def run_optimized_original_batch(self):
        """Führt Original + Optimized Batch Processing aus"""
        print("🚀 OPTIMIZED ORIGINAL SPRITESHEET PROCESSING")
        print("=" * 60)
        print("📋 WORKFLOW:")
        print("   1. 🔍 2x Upscaling + Sharpening")
        print("   2. 🎯 Original Background Detection")
        print("   3. 📦 Original Frame Extraction")
        print("   4. 🌈 Vaporwave Filter (25%)")
        print("   5. 🎬 GIF Generation")
        print("=" * 60)

        self.start_time = time.time()
        self.create_session_directory()

        # Finde alle Spritesheets
        candidates = []
        for ext in self.supported_formats:
            candidates.extend(self.input_dir.glob(f"*{ext}"))

        candidates = sorted(candidates)

        if not candidates:
            print("❌ No image files found!")
            return

        print(f"\n🎯 Processing {len(candidates)} image files...")

        # Verarbeite alle Kandidaten
        for i, image_path in enumerate(candidates, 1):
            print(f"\n[{i:>2}/{len(candidates)}] ", end="")
            frames_count = self.process_single_spritesheet(image_path)

            if frames_count > 0:
                print(f"   ✅ Success: {frames_count} vaporwave frames")
            else:
                print(f"   ❌ Failed or no frames found")

        # Finale Zusammenfassung
        end_time = time.time()
        processing_time = end_time - self.start_time

        print(f"\n🎉 OPTIMIZED ORIGINAL PROCESSING COMPLETE!")
        print(f"   📁 Session: {self.session_dir}")
        print(f"   ⏱️  Time: {processing_time:.1f} seconds")
        print(f"   📦 Frames: {self.total_frames_extracted}")
        print(
            f"   ⚡ Speed: {self.total_frames_extracted/processing_time:.1f} frames/sec")
        print(
            f"   🎯 Success: {len(self.processed_files)}/{len(candidates)} files")
        print(f"   🌈 All frames processed with Vaporwave filter!")

        # Speichere Zusammenfassung
        summary = {
            'workflow': 'Optimized Original',
            'steps': [
                '2x Upscaling + Sharpening',
                'Original Background Detection (4 corners)',
                'Original Frame Extraction',
                'Vaporwave Filter (25%)',
                'GIF Generation'
            ],
            'processing_time': processing_time,
            'total_frames': self.total_frames_extracted,
            'files_processed': len(self.processed_files),
            'files_failed': len(self.failed_files),
            'frames_per_second': self.total_frames_extracted / processing_time,
            'processed_files': self.processed_files,
            'failed_files': self.failed_files
        }

        summary_path = self.session_dir / "OPTIMIZED_ORIGINAL_SUMMARY.json"
        import json
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return self.session_dir


def main():
    """Hauptfunktion für Optimized Original Processing"""
    processor = OptimizedOriginalProcessor()
    session_dir = processor.run_optimized_original_batch()

    if session_dir:
        print(f"\n🎮 Optimized Original Processing Complete!")
        print(f"🌈 Zurück zum ursprünglichen Workflow + Vaporwave!")
        print(f"📁 Check results: {session_dir}")


if __name__ == "__main__":
    main()
