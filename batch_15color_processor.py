#!/usr/bin/env python3
"""
🎨 BATCH 15-FARBEN-PALETTE PROCESSOR
===================================
KORRIGIERTE VERSION - Behält ALLE Frames!

Verarbeitet viele GIFs mit der exakten 15-Farben-Palette
Input: Liste von GIF-Dateien
Output: Batch-verarbeitete GIFs in output/batch_15color_processed/
"""

import cv2
import numpy as np
from PIL import Image, ImageSequence
from pathlib import Path
import time
import os


class Batch15ColorProcessor:
    def __init__(self):
        # Exakte 15-Farben-Palette aus der Analyse
        self.extracted_palette = [
            # Dunkel & Monochrom (1-2)
            (8, 12, 16),      # #1: Tiefschwarz mit Blaustich
            (25, 15, 35),     # #2: Sehr dunkles Violett

            # Warme Töne (3-5)
            (85, 25, 35),     # #3: Dunkelrot/Weinrot
            (200, 100, 45),   # #4: Sattes Orange
            (235, 220, 180),  # #5: Blasses Gelb/Beige

            # Grüntöne (6-7)
            (145, 220, 85),   # #6: Helles Apfelgrün
            (45, 160, 95),    # #7: Satteres Smaragdgrün

            # Blautöne (8-11)
            (35, 85, 95),     # #8: Dunkles Grünblau
            (25, 45, 85),     # #9: Marineblau
            (65, 115, 180),   # #10: Mittleres Blau
            (85, 195, 215),   # #11: Cyan/Türkis

            # Kühle Grautöne (12-14)
            (95, 105, 125),   # #12: Graublau
            (85, 75, 95),     # #13: Graulila
            (45, 55, 75),     # #14: Graphitblau

            # Letzter Ton (15)
            (12, 8, 8),       # #15: Fast Schwarz, leicht abweichend von #1
        ]

        self.palette_array = np.array(self.extracted_palette)

    def apply_extracted_palette_optimized(self, image):
        """Super-optimierte Palette-Anwendung mit Vektorisierung"""
        height, width, channels = image.shape

        # Reshape für Batch-Verarbeitung
        flat_image = image.reshape(-1, channels).astype(np.float32)

        # Expandiere Dimensionen für Broadcasting
        pixels = flat_image[:, np.newaxis, :]  # (num_pixels, 1, 3)
        palette = self.palette_array[np.newaxis, :, :]  # (1, 15, 3)

        # Berechne alle Distanzen auf einmal
        distances = np.sum((pixels - palette) ** 2, axis=2)  # (num_pixels, 15)

        # Finde nächste Farben
        closest_indices = np.argmin(distances, axis=1)  # (num_pixels,)

        # Wende Palette an
        result = self.palette_array[closest_indices]

        return result.reshape(height, width, channels).astype(np.uint8)

    def pixelize(self, image, block_size=4):
        """Pixelize-Effekt"""
        height, width = image.shape[:2]
        small_height = height // block_size
        small_width = width // block_size
        small_image = cv2.resize(image, (small_width, small_height),
                                 interpolation=cv2.INTER_AREA)
        pixelized = cv2.resize(small_image, (width, height),
                               interpolation=cv2.INTER_NEAREST)
        return pixelized

    def process_single_gif_batch(self, input_path, output_path):
        """Verarbeite eine GIF - KORRIGIERTE VERSION mit ALLEN Frames und TRANSPARENZ"""
        try:
            gif = Image.open(input_path)

            # Konfiguration
            config = {
                "resolution": (512, 512),
                "pixelize": 4,
                "apply_palette": True
            }

            processed_frames = []
            frame_durations = []

            # Prüfe ob ursprüngliches GIF Transparenz hat
            has_transparency = False
            if hasattr(gif, 'info') and 'transparency' in gif.info:
                has_transparency = True
            elif gif.mode in ('RGBA', 'LA'):
                has_transparency = True

            # Verarbeite ALLE Frames ohne Limit!
            try:
                while True:
                    frame = gif.copy()

                    # Transparenz-Behandlung
                    if has_transparency:
                        # Konvertiere zu RGBA um Alpha-Kanal zu erhalten
                        if frame.mode != 'RGBA':
                            if frame.mode == 'P' and 'transparency' in frame.info:
                                # Palette-Mode mit Transparenz
                                frame = frame.convert('RGBA')
                            elif frame.mode in ('RGBA', 'LA'):
                                frame = frame.convert('RGBA')
                            else:
                                # Fallback: Erstelle Alpha-Kanal
                                frame = frame.convert('RGBA')
                    else:
                        # Kein transparenter Hintergrund
                        if frame.mode != 'RGB':
                            frame = frame.convert('RGB')

                    # Resize
                    frame = frame.resize(
                        config["resolution"], Image.Resampling.LANCZOS)

                    # Zu NumPy
                    frame_np = np.array(frame)

                    # Pixelize
                    if config["pixelize"] > 1:
                        frame_np = self.pixelize(frame_np, config["pixelize"])

                    # Palette anwenden nur auf RGB-Kanäle
                    if config["apply_palette"]:
                        if has_transparency and frame_np.shape[2] == 4:  # RGBA
                            # Verarbeite nur RGB, behalte Alpha
                            rgb_channels = frame_np[:, :, :3]
                            alpha_channel = frame_np[:, :, 3]

                            # Wende Palette nur auf RGB an
                            processed_rgb = self.apply_extracted_palette_optimized(
                                rgb_channels)

                            # Kombiniere RGB mit Alpha
                            frame_np = np.dstack(
                                [processed_rgb, alpha_channel])
                        else:
                            # Normale RGB-Verarbeitung
                            frame_np = self.apply_extracted_palette_optimized(
                                frame_np)

                    # Zurück zu PIL
                    if has_transparency and frame_np.shape[2] == 4:
                        processed_frame = Image.fromarray(frame_np, 'RGBA')
                    else:
                        processed_frame = Image.fromarray(frame_np, 'RGB')

                    processed_frames.append(processed_frame)

                    # Duration für diesen Frame
                    duration = gif.info.get('duration', 100)
                    frame_durations.append(duration)

                    # Nächster Frame
                    gif.seek(gif.tell() + 1)

            except EOFError:
                # Ende der Frames erreicht
                pass

            # Speichere GIF mit ALLEN Frames und TRANSPARENZ
            if processed_frames:
                # Verwende durchschnittliche Duration wenn unterschiedlich
                avg_duration = sum(
                    frame_durations) // len(frame_durations) if frame_durations else 100
                loop = gif.info.get('loop', 0)

                # Speicher-Parameter je nach Transparenz
                save_params = {
                    'save_all': True,
                    'append_images': processed_frames[1:],
                    'duration': avg_duration,
                    'loop': loop,
                    'optimize': False  # Wichtig: optimize=False für bessere Kompatibilität
                }

                # Transparenz-Einstellungen
                if has_transparency:
                    # Für transparente GIFs: Konvertiere zu Palette mit Transparenz
                    save_params['transparency'] = 0
                    save_params['disposal'] = 2  # Für saubere Transparenz

                processed_frames[0].save(output_path, **save_params)

                return True, len(processed_frames)

            return False, 0

        except Exception as e:
            print(f"❌ Fehler bei {Path(input_path).name}: {str(e)[:50]}...")
            return False, 0

    def process_batch(self, file_list):
        """Verarbeite eine Liste von GIF-Dateien"""
        print("🎨 BATCH 15-FARBEN-PALETTE PROCESSOR (KORRIGIERT)")
        print("=" * 60)
        print(f"📁 {len(file_list)} GIF-Dateien zu verarbeiten")
        print(f"📂 Output: output/batch_15color_processed/")
        print("=" * 60)

        # Erstelle Output-Verzeichnis
        output_dir = Path("output/batch_15color_processed")
        output_dir.mkdir(parents=True, exist_ok=True)

        successful = 0
        failed = 0
        start_time = time.time()

        for i, input_file in enumerate(file_list):
            input_path = Path(input_file.strip('"'))  # Remove quotes

            if not input_path.exists():
                print(f"❌ Datei nicht gefunden: {input_path.name}")
                failed += 1
                continue

            # Erstelle Output-Pfad
            output_name = f"FIXED_15color_{input_path.stem}.gif"
            output_path = output_dir / output_name

            print(f"\n🔄 [{i+1:3d}/{len(file_list)}] {input_path.name}")

            success, frame_count = self.process_single_gif_batch(
                input_path, output_path)

            if success:
                successful += 1
                print(f"   ✅ Erstellt: {frame_count} Frames (ALLE erhalten!)")
            else:
                failed += 1

            # Progress update
            progress = ((i + 1) / len(file_list)) * 100
            elapsed = time.time() - start_time
            if i > 0:
                estimated_total = elapsed / (i + 1) * len(file_list)
                remaining = estimated_total - elapsed
                print(
                    f"   📊 Fortschritt: {progress:.1f}% | ✅{successful} ❌{failed} | ~{remaining/60:.1f}min verbleibend")

        # Zusammenfassung
        total_time = time.time() - start_time
        print(f"\n" + "=" * 60)
        print(f"🎯 BATCH VERARBEITUNG ABGESCHLOSSEN")
        print(f"   ✅ Erfolgreich: {successful}")
        print(f"   ❌ Fehlgeschlagen: {failed}")
        print(f"   ⏱️  Gesamtzeit: {total_time/60:.1f} Minuten")
        print(f"   📁 Output: {output_dir}")
        print("=" * 60)


def main():
    # Test mit ersten 5 Dateien
    test_files = [
        "input/0ea53a3cdbfdcf14caf1c8cccdb60143_fast_transparent_converted.gif",
        "input/spinning_vinyl_clean_fast_transparent_converted.gif",
        "input/dodo_1_fast_transparent_converted.gif",
        "input/rick-and-morty-fortnite_fast_transparent_converted.gif",
        "input/final_dance_pingpong_transparent_fast_transparent_converted.gif"
    ]

    processor = Batch15ColorProcessor()
    processor.process_batch(test_files)


if __name__ == "__main__":
    main()
