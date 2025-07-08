#!/usr/bin/env python3
"""
AUTOMATED SPRITESHEET BATCH PROCESSOR - AI ENHANCED
Vollautomatische Verarbeitung aller Spritesheets mit KI-basierter Hintergrundentfernung
"""

import cv2
import numpy as np
from PIL import Image
from collections import Counter
from pathlib import Path
import os
import json
import time
from datetime import datetime

# Importiere KI-basierte Hintergrundentfernung
try:
    from ai_background_remover import AIBackgroundRemover
    AI_AVAILABLE = True
    print("✅ AI Background Removal available")
except ImportError:
    AI_AVAILABLE = False
    print("⚠️  AI Background Removal not available - using fallback methods")


class AutomatedSpritesheetBatchProcessor:
    """
    Vollautomatisches Batch-Processing für alle Spritesheets mit KI-Enhancement
    """

    def __init__(self, input_dir="input", output_base_dir="output/automated_batch"):
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        self.session_dir = None
        self.processed_files = []
        self.failed_files = []
        self.total_frames_extracted = 0
        self.start_time = None

        # Supported image formats
        self.supported_formats = {'.png', '.jpg',
                                  '.jpeg', '.bmp', '.tiff', '.tif'}

        # Initialize AI Background Remover if available
        self.ai_remover = None
        if AI_AVAILABLE:
            try:
                print("🤖 Initializing AI Background Remover...")
                self.ai_remover = AIBackgroundRemover()
                print("✅ AI Background Remover ready")
            except Exception as e:
                print(f"⚠️  AI initialization failed: {e}")
                self.ai_remover = None

    def create_session_directory(self):
        """Erstellt ein eindeutiges Session-Verzeichnis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_base_dir / f"batch_session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Erstelle Unterverzeichnisse
        (self.session_dir / "individual_sprites").mkdir(exist_ok=True)
        (self.session_dir / "animations").mkdir(exist_ok=True)
        (self.session_dir / "reports").mkdir(exist_ok=True)

        print(f"📁 Session directory created: {self.session_dir}")

    def find_spritesheet_candidates(self):
        """Findet alle potentiellen Spritesheets im input-Verzeichnis"""
        candidates = []

        if not self.input_dir.exists():
            print(f"❌ Input directory not found: {self.input_dir}")
            return candidates

        for file_path in self.input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                candidates.append(file_path)

        print(f"🔍 Found {len(candidates)} image files to process")
        return candidates

    def detect_background_color(self, image_np, corner_size_ratio=30.0):
        """Intelligente Hintergrundfarb-Erkennung"""
        h, w = image_np.shape[:2]
        corner_size = max(1, int(min(h, w) / corner_size_ratio))

        # Sammle Pixel aus allen vier Ecken
        corners = [
            image_np[0:corner_size, 0:corner_size],
            image_np[0:corner_size, w-corner_size:w],
            image_np[h-corner_size:h, 0:corner_size],
            image_np[h-corner_size:h, w-corner_size:w]
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
            return np.array([255, 255, 255], dtype=np.uint8)  # Default white

    def process_single_spritesheet(self, image_path):
        """Verarbeitet ein einzelnes Spritesheet"""
        print(f"\n🎮 Processing: {image_path.name}")

        try:
            # Lade Bild
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Could not load image")

            h, w = image.shape[:2]
            print(f"   📐 Size: {w}x{h}")

            # KI-basierte oder Fallback Hintergrundentfernung
            if self.ai_remover is not None:
                print("   🤖 Using AI-powered background removal...")
                # Nutze KI für bessere Hintergrundentfernung
                try:
                    full_image_rgba, ai_quality = self.ai_remover.remove_background_ai(
                        image)
                    print(f"   🎯 AI Quality Score: {ai_quality:.3f}")

                    # Konvertiere AI-Ergebnis zu Maske für Connected Components
                    if len(full_image_rgba.shape) == 3 and full_image_rgba.shape[2] == 4:
                        alpha_channel = full_image_rgba[:, :, 3]
                        foreground_mask = (
                            alpha_channel > 128).astype(np.uint8)
                        print(f"   ✅ AI background removal successful")
                    else:
                        raise ValueError("AI result has wrong format")

                except Exception as e:
                    print(f"   ⚠️  AI failed ({e}), using fallback...")
                    # Fallback zu traditioneller Methode
                    bg_color = self.detect_background_color(image)
                    print(f"   🎯 Fallback Background: RGB{tuple(bg_color)}")
                    self._last_detected_bg_color = bg_color  # Für Report speichern
                    foreground_mask = self._create_traditional_mask(
                        image, bg_color)
            else:
                print("   🔄 Using traditional background detection...")
                # Traditionelle Hintergrundfarb-Erkennung
                bg_color = self.detect_background_color(image)
                print(f"   🎯 Background: RGB{tuple(bg_color)}")
                self._last_detected_bg_color = bg_color  # Für Report speichern
                foreground_mask = self._create_traditional_mask(
                    image, bg_color)

            # Connected Components Analysis
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                foreground_mask, connectivity=8
            )

            # Frame-Extraktion
            extracted_frames = []
            frame_info = []

            # Erstelle Ausgabeverzeichnis für dieses Spritesheet
            sprite_dir = self.session_dir / "individual_sprites" / image_path.stem
            sprite_dir.mkdir(exist_ok=True)

            total_pixels = h * w
            min_area = 500  # Dynamisch anpassbar

            for i in range(1, num_labels):  # Skip background label 0
                x, y, w_comp, h_comp, area = stats[i]

                # Intelligente Filterung
                if area < min_area or area > total_pixels * 0.8:
                    continue

                aspect_ratio = w_comp / h_comp if h_comp > 0 else float('inf')
                if aspect_ratio > 10 or aspect_ratio < 0.1:
                    continue

                # Frame mit Padding extrahieren
                padding = 5
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(w, x + w_comp + padding)
                y_end = min(h, y + h_comp + padding)

                frame = image[y_start:y_end, x_start:x_end]

                if frame.size == 0:
                    continue

                # Verbesserte Frame-Hintergrundentfernung
                frame_rgba = self._process_extracted_frame(
                    frame, x_start, y_start, x_end, y_end)

                # Speichere Frame
                frame_filename = f"frame_{len(extracted_frames)+1:03d}.png"
                frame_path = sprite_dir / frame_filename

                pil_frame = Image.fromarray(frame_rgba, 'RGBA')
                pil_frame.save(frame_path)

                extracted_frames.append(frame_rgba)

                frame_info.append({
                    'id': len(extracted_frames),
                    'filename': frame_filename,
                    'bbox': (int(x), int(y), int(w_comp), int(h_comp)),
                    'area': int(area),
                    'aspect_ratio': float(aspect_ratio),
                    'size': f"{w_comp}x{h_comp}"
                })

            print(f"   📦 Extracted: {len(extracted_frames)} frames")

            # Erstelle GIF-Animation
            gif_path = None
            if extracted_frames:
                gif_path = self.session_dir / "animations" / \
                    f"{image_path.stem}_animation.gif"

                pil_frames = [Image.fromarray(frame, 'RGBA')
                              for frame in extracted_frames]

                pil_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=500,  # 500ms per frame
                    loop=0,
                    transparency=0,
                    disposal=2
                )
                print(f"   🎬 GIF created: {gif_path.name}")

            # Erstelle JSON-Report für dieses Spritesheet
            # Bestimme Hintergrundfarbe für Report (falls traditionelle Methode verwendet)
            report_bg_color = getattr(
                self, '_last_detected_bg_color', [0, 0, 0])

            report = {
                'input_file': str(image_path),
                'processing_timestamp': datetime.now().isoformat(),
                'image_size': [w, h],
                'background_method': 'AI-powered' if self.ai_remover else 'traditional',
                'background_color': tuple(int(c) for c in report_bg_color),
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
                'frames': len(extracted_frames),
                'method': 'AI-powered' if self.ai_remover else 'traditional',
                'background_color': tuple(int(c) for c in report_bg_color),
                'gif_path': gif_path.name if gif_path else None
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

    def _create_traditional_mask(self, image: np.ndarray, bg_color: np.ndarray) -> np.ndarray:
        """Erstellt traditionelle Vordergrund-Maske basierend auf Hintergrundfarbe"""
        # Verbesserte traditionelle Methode mit adaptiver Toleranz
        tolerance = 25
        diff = np.abs(image.astype(int) - bg_color.astype(int))
        background_mask = np.all(diff <= tolerance, axis=2)
        foreground_mask = ~background_mask

        # Morphologische Bereinigung
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        foreground_mask = cv2.morphologyEx(
            foreground_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel
        )
        foreground_mask = cv2.morphologyEx(
            foreground_mask, cv2.MORPH_OPEN, kernel)

        return foreground_mask

    def _process_extracted_frame(self, frame: np.ndarray, x_start: int, y_start: int,
                                 x_end: int, y_end: int) -> np.ndarray:
        """Verarbeitet einen extrahierten Frame mit verbesserter Hintergrundentfernung"""

        # Wenn KI verfügbar ist, nutze sie für individuelle Frame-Bearbeitung
        if self.ai_remover is not None:
            try:
                # KI-basierte Hintergrundentfernung für diesen Frame
                frame_rgba, frame_quality = self.ai_remover.remove_background_ai(
                    frame)

                # Qualitätsprüfung
                if frame_quality > 0.1:  # Mindestqualität erreicht
                    return frame_rgba
                else:
                    print(
                        f"      ⚠️  AI quality too low ({frame_quality:.2f}), using fallback")

            except Exception as e:
                print(f"      ⚠️  AI frame processing failed: {e}")

        # Fallback: Nutze traditionelle Methode mit Corner-Detection
        # Erkenne Hintergrundfarbe aus diesem spezifischen Frame
        frame_bg_color = self._detect_frame_background(frame)

        # Konvertiere zu RGBA
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgba = np.dstack(
            [frame_rgb, np.full(frame_rgb.shape[:2], 255, dtype=np.uint8)])

        # Verbesserte Hintergrundentfernung
        tolerance = 30  # Etwas höhere Toleranz für Einzelframes
        frame_diff = np.abs(frame_rgb.astype(int) - frame_bg_color.astype(int))
        frame_bg_mask = np.all(frame_diff <= tolerance, axis=2)

        # Morphologische Verbesserung der Maske
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        frame_bg_mask = cv2.morphologyEx(
            frame_bg_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        frame_bg_mask = frame_bg_mask.astype(bool)

        # Setze Hintergrund transparent
        frame_rgba[frame_bg_mask, 3] = 0

        return frame_rgba

    def _detect_frame_background(self, frame: np.ndarray) -> np.ndarray:
        """Erkennt Hintergrundfarbe eines einzelnen Frames"""
        h, w = frame.shape[:2]

        # Nutze Rand-Pixel für Hintergrunderkennung (nicht nur Ecken)
        border_width = max(1, min(h, w) // 20)

        border_pixels = []

        # Oberer und unterer Rand
        border_pixels.extend(frame[:border_width, :].reshape(-1, 3))
        border_pixels.extend(frame[-border_width:, :].reshape(-1, 3))

        # Linker und rechter Rand
        border_pixels.extend(frame[:, :border_width].reshape(-1, 3))
        border_pixels.extend(frame[:, -border_width:].reshape(-1, 3))

        if border_pixels:
            border_pixels = np.array(border_pixels)
            # Nutze Median statt Mean für robustere Hintergrunderkennung
            bg_color = np.median(border_pixels, axis=0).astype(np.uint8)
        else:
            bg_color = np.array(
                [255, 255, 255], dtype=np.uint8)  # Default white

        return bg_color

    def create_master_report(self):
        """Erstellt einen Master-Report für die gesamte Batch-Session"""

        end_time = time.time()
        processing_time = end_time - self.start_time

        master_report = {
            'batch_session': {
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': round(processing_time, 2),
                'session_directory': str(self.session_dir)
            },
            'summary': {
                'total_files_processed': len(self.processed_files),
                'total_files_failed': len(self.failed_files),
                'total_frames_extracted': self.total_frames_extracted,
                'success_rate': round(len(self.processed_files) / (len(self.processed_files) + len(self.failed_files)) * 100, 1) if (self.processed_files or self.failed_files) else 0
            },
            'processed_files': self.processed_files,
            'failed_files': self.failed_files
        }

        master_report_path = self.session_dir / "MASTER_REPORT.json"
        with open(master_report_path, 'w', encoding='utf-8') as f:
            json.dump(master_report, f, indent=2, ensure_ascii=False)

        # Erstelle auch eine lesbare Zusammenfassung
        summary_lines = [
            "🎮 AUTOMATED SPRITESHEET BATCH PROCESSING - FINAL REPORT",
            "=" * 70,
            f"📅 Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"⏱️  Processing Time: {processing_time:.1f} seconds",
            f"📁 Output Directory: {self.session_dir}",
            "",
            "📊 SUMMARY:",
            f"   ✅ Successfully processed: {len(self.processed_files)} files",
            f"   ❌ Failed: {len(self.failed_files)} files",
            f"   📦 Total frames extracted: {self.total_frames_extracted}",
            f"   🎯 Success rate: {master_report['summary']['success_rate']}%",
            "",
            "📋 PROCESSED FILES:",
            "-" * 50
        ]

        for file_info in self.processed_files:
            summary_lines.append(
                f"✅ {file_info['file']:<35} → {file_info['frames']:>3} frames | "
                f"BG: RGB{file_info['background_color']} | GIF: {file_info['gif_path'] or 'None'}"
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
            "🎉 BATCH PROCESSING COMPLETE!",
            f"📁 All results saved to: {self.session_dir}"
        ])

        summary_path = self.session_dir / "BATCH_SUMMARY.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))

        return master_report_path, summary_path

    def run_automated_batch(self):
        """Führt die vollautomatische Batch-Verarbeitung aus"""
        print("🚀 STARTING AUTOMATED SPRITESHEET BATCH PROCESSING")
        print("=" * 60)

        self.start_time = time.time()

        # Session-Verzeichnis erstellen
        self.create_session_directory()

        # Spritesheet-Kandidaten finden
        candidates = self.find_spritesheet_candidates()

        if not candidates:
            print("❌ No image files found in input directory!")
            return

        print(f"\n🎯 Processing {len(candidates)} image files...")

        # Verarbeite jeden Kandidaten
        for i, image_path in enumerate(candidates, 1):
            print(f"\n[{i:>2}/{len(candidates)}] ", end="")
            frames_count = self.process_single_spritesheet(image_path)

            if frames_count > 0:
                print(f"   ✅ Success: {frames_count} frames")
            else:
                print(f"   ❌ Failed or no frames found")

        # Master-Report erstellen
        print(f"\n📋 Creating master report...")
        master_report_path, summary_path = self.create_master_report()

        # Finale Zusammenfassung
        print(f"\n🎉 AUTOMATED BATCH PROCESSING COMPLETE!")
        print(f"   📁 Session directory: {self.session_dir}")
        print(f"   📊 Master report: {master_report_path.name}")
        print(f"   📝 Summary: {summary_path.name}")
        print(f"   ✅ Processed: {len(self.processed_files)} files")
        print(f"   📦 Total frames: {self.total_frames_extracted}")

        return self.session_dir


def main():
    """Hauptfunktion für automatisierte Verarbeitung"""
    processor = AutomatedSpritesheetBatchProcessor()
    session_dir = processor.run_automated_batch()

    if session_dir:
        print(f"\n🎮 All your spritesheets have been automatically processed!")
        print(f"Check the results in: {session_dir}")


if __name__ == "__main__":
    main()
