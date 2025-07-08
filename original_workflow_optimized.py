#!/usr/bin/env python3
"""
ORIGINAL WORKFLOW OPTIMIZED - Enhanced Version
Zurück zum ursprünglichen erfolgreichen Workflow + verbesserte Optimierungen:
1. 2x Upscaling + Schärfung + Bildverbesserungen (VOR Hintergrund-Analyse)
2. Original Background Detection (4 Ecken)
3. Original Frame Extraction + 30px Padding
4. Verstärkter Vaporwave Filter 37.5% auf jeden Frame
5. GIF Generation

Neue Verbesserungen:
- Kontrast und Schwarzwert erhöht
- Weißabgleich durchgeführt
- Farben intensiviert
- Frame-Padding 30px auf alle Seiten
- Vaporwave-Effekt um 50% verstärkt

Basiert auf dem ursprünglichen 51-Sekunden Workflow
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


class OriginalWorkflowOptimized:
    """Zurück zum ursprünglichen erfolgreichen Workflow mit erweiterten Optimierungen"""

    def __init__(self, input_dir="input", output_base_dir="output/original_optimized"):
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        self.session_dir = None
        self.processed_files = []
        self.failed_files = []
        self.total_frames_extracted = 0
        self.start_time = None

        # Supported formats (original)
        self.supported_formats = {'.png', '.jpg',
                                  '.jpeg', '.bmp', '.tiff', '.tif'}

    def create_session_directory(self):
        """Erstellt Session-Verzeichnis (original)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_base_dir / \
            f"original_session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Original structure
        (self.session_dir / "individual_sprites").mkdir(exist_ok=True)
        (self.session_dir / "animations").mkdir(exist_ok=True)
        (self.session_dir / "reports").mkdir(exist_ok=True)

        print(f"📁 Session directory created: {self.session_dir}")

    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """NEUE FUNKTION: Bildqualität vor Hintergrund-Analyse verbessern"""
        print("   🎨 Enhancing image quality (contrast, levels, white balance, colors)...")

        # Konvertiere zu PIL für erweiterte Bearbeitung
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # 1. KONTRAST ERHÖHEN (leicht)
        contrast_enhancer = ImageEnhance.Contrast(pil_image)
        enhanced_image = contrast_enhancer.enhance(1.15)  # 15% mehr Kontrast

        # 2. SCHWARZWERT ANPASSEN (Shadows lift)
        image_array = np.array(enhanced_image, dtype=np.float32)

        # Gamma-Korrektur für Schwarzwert-Anhebung
        gamma = 0.9  # Leichte Aufhellung der dunklen Bereiche
        gamma_corrected = 255.0 * (image_array / 255.0) ** gamma
        gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

        enhanced_image = Image.fromarray(gamma_corrected)

        # 3. WEIßABGLEICH (Automatisch)
        # Grauwelt-Annahme für Weißabgleich
        r, g, b = enhanced_image.split()
        r_avg = np.mean(np.array(r))
        g_avg = np.mean(np.array(g))
        b_avg = np.mean(np.array(b))

        # Durchschnittswert aller Kanäle
        avg_gray = (r_avg + g_avg + b_avg) / 3

        # Korrekturfaktoren berechnen
        r_factor = avg_gray / r_avg if r_avg > 0 else 1.0
        g_factor = avg_gray / g_avg if g_avg > 0 else 1.0
        b_factor = avg_gray / b_avg if b_avg > 0 else 1.0

        # Sanfte Anwendung der Korrekturfaktoren (nur 30% der Korrektur)
        r_factor = 1.0 + (r_factor - 1.0) * 0.3
        g_factor = 1.0 + (g_factor - 1.0) * 0.3
        b_factor = 1.0 + (b_factor - 1.0) * 0.3

        # Weißabgleich anwenden
        r_corrected = np.clip(np.array(r) * r_factor, 0, 255).astype(np.uint8)
        g_corrected = np.clip(np.array(g) * g_factor, 0, 255).astype(np.uint8)
        b_corrected = np.clip(np.array(b) * b_factor, 0, 255).astype(np.uint8)

        wb_corrected = Image.merge('RGB', [
            Image.fromarray(r_corrected),
            Image.fromarray(g_corrected),
            Image.fromarray(b_corrected)
        ])

        # 4. FARBINTENSITÄT ERHÖHEN (Sättigung)
        color_enhancer = ImageEnhance.Color(wb_corrected)
        final_enhanced = color_enhancer.enhance(1.2)  # 20% mehr Farbsättigung

        # Zurück zu OpenCV Format
        final_array = np.array(final_enhanced)
        result_bgr = cv2.cvtColor(final_array, cv2.COLOR_RGB2BGR)

        print("   ✨ Image quality enhanced: contrast +15%, gamma correction, white balance, saturation +20%")
        return result_bgr

    def upscale_and_sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """SCHRITT 1: 2x Hochskalierung mit Schärfung + Qualitätsverbesserungen"""
        print("   🔍 2x Upscaling + Sharpening + Quality Enhancement...")

        h, w = image.shape[:2]

        # 2x Upscaling mit INTER_CUBIC (beste Qualität für Pixel Art)
        upscaled = cv2.resize(image, (w * 2, h * 2),
                              interpolation=cv2.INTER_CUBIC)

        # NEUE BILDVERBESSERUNGEN VOR HINTERGRUND-ANALYSE
        enhanced = self.enhance_image_quality(upscaled)

        # Konvertiere zu PIL für Schärfung
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(enhanced_rgb)

        # Leichte Schärfung (angepasst für Pixel Art)
        sharpened = pil_image.filter(ImageFilter.UnsharpMask(
            radius=1.0,    # Kleiner Radius für Pixel Art
            percent=150,   # Moderate Schärfung
            threshold=2    # Threshold um Artefakte zu vermeiden
        ))

        # Zurück zu OpenCV Format
        sharpened_array = np.array(sharpened)
        result = cv2.cvtColor(sharpened_array, cv2.COLOR_RGB2BGR)

        print(
            f"   📈 Enhanced & Upscaled: {w}x{h} -> {result.shape[1]}x{result.shape[0]}")
        return result

    def detect_background_color_original(self, image_np, corner_size_ratio=30.0):
        """SCHRITT 2: Original Background Detection (4 Ecken)"""
        h, w = image_np.shape[:2]
        corner_size = max(1, int(min(h, w) / corner_size_ratio))

        # Original: Sammle Pixel aus allen vier Ecken
        corners = [
            # Top-left
            image_np[0:corner_size, 0:corner_size],
            # Top-right
            image_np[0:corner_size, w-corner_size:w],
            # Bottom-left
            image_np[h-corner_size:h, 0:corner_size],
            # Bottom-right
            image_np[h-corner_size:h, w-corner_size:w]
        ]

        corner_pixels = []
        for corner in corners:
            if len(corner.shape) == 3:
                pixels = corner.reshape(-1, 3)
                for pixel in pixels:
                    corner_pixels.append(tuple(pixel))

        if corner_pixels:
            # Original: Most common color
            bg_color = Counter(corner_pixels).most_common(1)[0][0]
            return np.array(bg_color, dtype=np.uint8)
        else:
            return np.array([255, 255, 255], dtype=np.uint8)  # Default white

    def create_traditional_mask_original(self, image: np.ndarray, bg_color: np.ndarray) -> np.ndarray:
        """SCHRITT 2: Original traditionelle Vordergrund-Maske"""
        # Original settings
        tolerance = 25
        diff = np.abs(image.astype(int) - bg_color.astype(int))
        background_mask = np.all(diff <= tolerance, axis=2)
        foreground_mask = ~background_mask

        # Original morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        foreground_mask = cv2.morphologyEx(
            foreground_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel
        )
        foreground_mask = cv2.morphologyEx(
            foreground_mask, cv2.MORPH_OPEN, kernel
        )

        return foreground_mask

    def apply_vaporwave_filter_enhanced(self, frame: np.ndarray) -> np.ndarray:
        """SCHRITT 4: VERSTÄRKTER Vaporwave Filter auf 37.5% (50% mehr als vorher)"""

        # Konvertiere zu PIL für bessere Kontrolle
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)

        # VAPORWAVE VERSTÄRKT - 37.5% Intensität (50% mehr als 25%)
        intensity = 0.375

        # 1. Verstärkte Farbverschiebung zu Cyan/Magenta Tönen
        r, g, b = pil_frame.split()

        r_array = np.array(r, dtype=np.float32)
        g_array = np.array(g, dtype=np.float32)
        b_array = np.array(b, dtype=np.float32)

        # Verstärkte Vaporwave Farbverschiebung
        r_shifted = r_array * (1.0 - intensity * 0.3)  # Weniger Rot
        g_shifted = g_array * (1.0 + intensity * 0.2)  # Etwas mehr Grün
        # Mehr Blau (verstärkter Cyan-Effekt)
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

        # 2. Verstärkte Sättigungserhöhung (Vaporwave-typisch)
        enhancer = ImageEnhance.Color(vaporwave_frame)
        vaporwave_frame = enhancer.enhance(1.0 + intensity * 0.5)

        # 3. Verstärkter Kontrast für den "Pop"
        contrast_enhancer = ImageEnhance.Contrast(vaporwave_frame)
        vaporwave_frame = contrast_enhancer.enhance(1.0 + intensity * 0.3)

        # Zurück zu OpenCV
        result_array = np.array(vaporwave_frame)
        result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)

        return result_bgr

    def process_extracted_frame_original(self, frame: np.ndarray, bg_color: np.ndarray) -> np.ndarray:
        """SCHRITT 4: Original Frame Processing + Verstärkter Vaporwave"""

        # VERSTÄRKTER VAPORWAVE FILTER ANWENDEN
        vaporwave_frame = self.apply_vaporwave_filter_enhanced(frame)

        # Dann original Background Removal für Transparenz
        frame_rgb = cv2.cvtColor(vaporwave_frame, cv2.COLOR_BGR2RGB)
        frame_rgba = np.dstack(
            [frame_rgb, np.full(frame_rgb.shape[:2], 255, dtype=np.uint8)])

        # Original background removal mit etwas höherer Toleranz für Vaporwave
        tolerance = 40  # Etwas höher wegen verstärkter Farbverschiebung
        frame_diff = np.abs(frame_rgb.astype(int) - bg_color.astype(int))
        frame_bg_mask = np.all(frame_diff <= tolerance, axis=2)

        # Set transparent background
        frame_rgba[frame_bg_mask, 3] = 0

        return frame_rgba

    def extract_frames_with_padding(self, image: np.ndarray, mask: np.ndarray, min_area=800) -> List[np.ndarray]:
        """SCHRITT 3: Original Frame Extraction + 30px Padding auf alle Seiten"""
        print("   📦 Extracting frames with 30px padding...")

        # Connected components (original method)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)

        frames = []
        h, w = image.shape[:2]

        for i in range(1, num_labels):  # Skip background label 0
            x, y, width, height, area = stats[i]

            # Original filters
            if area < min_area:
                continue
            if width < 20 or height < 20:
                continue
            if width / height > 10 or height / width > 10:  # Skip very elongated
                continue

            # NEUE FEATURE: +30 Pixel Padding auf alle Seiten
            padding = 30

            # Berechne erweiterte Bounding Box mit Padding
            x_padded = max(0, x - padding)
            y_padded = max(0, y - padding)
            x2_padded = min(w, x + width + padding)
            y2_padded = min(h, y + height + padding)

            # Extrahiere Frame mit Padding
            padded_frame = image[y_padded:y2_padded, x_padded:x2_padded]

            if padded_frame.size > 0:
                frames.append(padded_frame)
                print(
                    f"   📦 Frame extracted: {x}x{y} ({width}x{height}) -> with padding: ({x2_padded-x_padded}x{y2_padded-y_padded})")

        print(f"   ✅ Total frames extracted with 30px padding: {len(frames)}")
        return frames

    def process_single_spritesheet(self, image_path):
        """Verarbeitet ein Spritesheet mit Original + Optimized Workflow"""
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

            # SCHRITT 2: Original Background Detection
            print("   🎯 Original background detection...")
            bg_color = self.detect_background_color_original(upscaled_image)
            print(f"   🎨 Background: RGB{tuple(bg_color)}")

            # SCHRITT 2: Original Background Mask
            foreground_mask = self.create_traditional_mask_original(
                upscaled_image, bg_color)

            # SCHRITT 3: Original Connected Components Analysis
            print("   📦 Original frame extraction...")
            extracted_frames = self.extract_frames_with_padding(
                upscaled_image, foreground_mask)

            # Original Frame-Extraktion
            frame_info = []
            processed_frames = []

            # Erstelle Ausgabeverzeichnis für dieses Spritesheet
            sprite_dir = self.session_dir / "individual_sprites" / image_path.stem
            sprite_dir.mkdir(exist_ok=True)

            total_pixels = h * w
            min_area = 500  # Original

            for i, frame in enumerate(extracted_frames, 1):
                # SCHRITT 4: Frame Processing + Verstärkter Vaporwave Filter
                frame_rgba = self.process_extracted_frame_original(
                    frame, bg_color)

                # Speichere Frame
                frame_filename = f"frame_{i:03d}.png"
                frame_path = sprite_dir / frame_filename
                Image.fromarray(frame_rgba, 'RGBA').save(frame_path)

                processed_frames.append(frame_rgba)

                frame_info.append({
                    'id': i,
                    'filename': frame_filename,
                    'bbox': [0, 0, frame.shape[1], frame.shape[0]],
                    'area': int(frame.shape[0] * frame.shape[1]),
                    'size': f"{frame.shape[1]}x{frame.shape[0]}",
                    'vaporwave_applied': True
                })

            print(
                f"   ✅ Extracted {len(extracted_frames)} frames with enhanced Vaporwave filter")

            # Erstelle GIF Animation
            gif_path = None
            if processed_frames:
                gif_filename = f"{image_path.stem}_vaporwave_animation.gif"
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
                print(f"   🎬 Enhanced Vaporwave GIF created: {gif_filename}")

            # Report für dieses Spritesheet erstellen
            report = {
                'input_file': str(image_path),
                'processing_timestamp': datetime.now().isoformat(),
                'workflow': 'Original + Optimized',
                'steps': [
                    '2x Upscaling + Sharpening',
                    'Original Background Detection (4 corners)',
                    'Original Frame Extraction',
                    'Vaporwave Filter (37.5%)',
                    'GIF Generation'
                ],
                'original_image_size': [original_w, original_h],
                'upscaled_image_size': [w, h],
                'background_color': [int(c) for c in bg_color],
                'total_frames': len(extracted_frames),
                'frames': frame_info,
                'output_directory': str(sprite_dir),
                'gif_animation': str(gif_path) if gif_path else None
            }

            report_path = self.session_dir / "reports" / \
                f"{image_path.stem}_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            # Erfolgreich verarbeitet
            self.processed_files.append({
                'file': image_path.name,
                'original_size': f"{original_w}x{original_h}",
                'upscaled_size': f"{w}x{h}",
                'frames': len(extracted_frames),
                'background_color': [int(c) for c in bg_color],
                'gif_path': gif_filename if gif_path else None,
                'vaporwave_applied': True
            })

            self.total_frames_extracted += len(extracted_frames)
            return len(extracted_frames)

        except Exception as e:
            print(f"   ❌ Error processing {image_path.name}: {e}")
            self.failed_files.append({
                'file': image_path.name,
                'error': str(e)
            })
            return 0

    def create_master_report(self):
        """Erstellt Master-Report (original style)"""
        end_time = time.time()
        processing_time = end_time - self.start_time

        master_report = {
            'batch_session': {
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': round(processing_time, 2),
                'session_directory': str(self.session_dir),
                'workflow_type': 'Original + Optimized'
            },
            'summary': {
                'total_files_processed': len(self.processed_files),
                'total_files_failed': len(self.failed_files),
                'total_frames_extracted': self.total_frames_extracted,
                'success_rate': round(len(self.processed_files) / (len(self.processed_files) + len(self.failed_files)) * 100, 1) if (self.processed_files or self.failed_files) else 0,
                'frames_per_second': round(self.total_frames_extracted / processing_time, 2)
            },
            'optimizations': [
                '2x Upscaling with INTER_CUBIC',
                'UnsharpMask Sharpening (radius=1.0, percent=150)',
                'Vaporwave Filter (37.5% intensity)',
                'Enhanced tolerance for color-shifted backgrounds'
            ],
            'processed_files': self.processed_files,
            'failed_files': self.failed_files
        }

        master_report_path = self.session_dir / "MASTER_REPORT.json"
        with open(master_report_path, 'w', encoding='utf-8') as f:
            json.dump(master_report, f, indent=2, ensure_ascii=False)

        # Lesbare Zusammenfassung
        summary_lines = [
            "🎮 ORIGINAL WORKFLOW OPTIMIZED - FINAL REPORT",
            "=" * 70,
            f"📅 Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"⏱️  Processing Time: {processing_time:.1f} seconds",
            f"🚀 Workflow: Original + Optimizations",
            f"📁 Output Directory: {self.session_dir}",
            "",
            "🔧 OPTIMIZATIONS APPLIED:",
            "   📈 2x Upscaling + Sharpening",
            "   🎯 Original 4-Corner Background Detection",
            "   📦 Original Frame Extraction Method",
            "   �� Vaporwave Filter (37.5% intensity)",
            "   🎬 Enhanced GIF Generation",
            "",
            "📊 PERFORMANCE:",
            f"   ✅ Successfully processed: {len(self.processed_files)} files",
            f"   ❌ Failed: {len(self.failed_files)} files",
            f"   📦 Total frames extracted: {self.total_frames_extracted}",
            f"   ⚡ Processing speed: {master_report['summary']['frames_per_second']} frames/sec",
            f"   🎯 Success rate: {master_report['summary']['success_rate']}%",
            "",
            "📋 PROCESSED FILES:",
            "-" * 50
        ]

        for file_info in self.processed_files:
            summary_lines.append(
                f"✅ {file_info['file']:<35} → {file_info['frames']:>3} frames | "
                f"BG: RGB{file_info['background_color']} | Vaporwave: ✅ | GIF: {file_info['gif_path'] or 'None'}"
            )

        if self.failed_files:
            summary_lines.extend([
                "",
                "❌ FAILED FILES:",
                "-" * 30
            ])
            for file_info in self.failed_files:
                summary_lines.append(
                    f"❌ {file_info['file']}: {file_info['error']}")

        summary_lines.extend([
            "",
            "🎉 ORIGINAL WORKFLOW OPTIMIZED COMPLETE!",
            f"🌈 All frames processed with 37.5% Vaporwave filter",
            f"📁 All results saved to: {self.session_dir}"
        ])

        summary_path = self.session_dir / "BATCH_SUMMARY.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))

        return master_report_path, summary_path

    def run_original_optimized_batch(self):
        """Führt den Original + Optimized Workflow aus"""
        print("🚀 STARTING ORIGINAL WORKFLOW OPTIMIZED")
        print("=" * 60)
        print("📋 WORKFLOW STEPS:")
        print("   1. 🔍 2x Upscaling + Sharpening")
        print("   2. 🎯 Original Background Detection (4 corners)")
        print("   3. 📦 Original Frame Extraction")
        print("   4. �� Vaporwave Filter (37.5% intensity)")
        print("   5. 🎬 GIF Generation")
        print("=" * 60)

        self.start_time = time.time()

        # Session-Verzeichnis erstellen
        self.create_session_directory()

        # Spritesheet-Kandidaten finden
        candidates = []
        for ext in self.supported_formats:
            candidates.extend(self.input_dir.glob(f"*{ext}"))
        candidates = sorted(candidates)

        if not candidates:
            print("❌ No image files found in input directory!")
            return

        print(f"\n🎯 Processing {len(candidates)} image files...")

        # Verarbeite jeden Kandidaten
        for i, image_path in enumerate(candidates, 1):
            print(f"\n[{i:>2}/{len(candidates)}] ", end="")
            frames_count = self.process_single_spritesheet(image_path)

            if frames_count > 0:
                print(f"   ✅ Success: {frames_count} vaporwave frames")
            else:
                print(f"   ❌ Failed or no frames found")

        # Master-Report erstellen
        print(f"\n📋 Creating master report...")
        master_report_path, summary_path = self.create_master_report()

        # Finale Zusammenfassung
        end_time = time.time()
        processing_time = end_time - self.start_time

        print(f"\n🎉 ORIGINAL WORKFLOW OPTIMIZED COMPLETE!")
        print(f"   📁 Session: {self.session_dir}")
        print(f"   ⏱️  Time: {processing_time:.1f} seconds")
        print(f"   📦 Frames: {self.total_frames_extracted}")
        print(f"   🌈 All frames enhanced with Vaporwave filter!")
        print(f"   📊 Report: {master_report_path}")

        return self.session_dir


def main():
    """Hauptfunktion für Original Workflow Optimized"""
    processor = OriginalWorkflowOptimized()
    session_dir = processor.run_original_optimized_batch()

    if session_dir:
        print(f"\n🎮 Original Workflow Optimized Complete!")
        print(f"🌈 Zurück zum ursprünglichen Erfolg + Vaporwave!")
        print(f"📁 Check results: {session_dir}")


if __name__ == "__main__":
    main()
