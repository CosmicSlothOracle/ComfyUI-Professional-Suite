#!/usr/bin/env python3
"""
🎬 Single GIF Pixel-Art Processor
Verarbeitet genau 1 GIF mit der analysierten 15-Farben-Palette
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageSequence
import numpy as np
import cv2


class SingleGifProcessor:
    def __init__(self):
        # Exakte 15-Farben-Palette aus der Analyse
        self.extracted_palette = [
            # Dunkel & Monochrom (1-2)
            (8, 12, 16),      # #1: Tiefes Schwarz mit leichtem Blaustich
            (25, 15, 35),     # #2: Sehr dunkles Lila

            # Warme Töne (3-5)
            (85, 25, 35),     # #3: Dunkles Rot (weinrot)
            (200, 100, 45),   # #4: Sattes Orange
            (235, 220, 180),  # #5: Blasses Gelb/Beige

            # Grüntöne (6-7)
            (145, 220, 85),   # #6: Helles Apfelgrün
            (45, 160, 95),    # #7: Satteres Smaragdgrün

            # Blautöne (8-11)
            (35, 85, 95),     # #8: Dunkles Grün-Blau
            (25, 45, 85),     # #9: Navyblau
            (65, 115, 180),   # #10: Mittelblau
            (85, 195, 215),   # #11: Cyan / Türkis

            # Kühle Grautöne (12-14)
            (95, 105, 125),   # #12: Graublau
            (85, 75, 95),     # #13: Graulila
            (45, 55, 75),     # #14: Graphitblau

            # Letzter Ton (15)
            (12, 8, 8),       # #15: Fast Schwarz, leicht abweichend von #1
        ]

    def apply_extracted_palette(self, image):
        """Wende die extrahierte 15-Farben-Palette an - OPTIMIERT"""
        height, width, channels = image.shape

        # Reshape für Vektor-Verarbeitung
        pixels = image.reshape(-1, channels)
        palette_array = np.array(self.extracted_palette)

        # Initialisiere Output
        result_pixels = np.zeros_like(pixels)

        # Batch-Verarbeitung aller Pixel
        for i, pixel in enumerate(pixels):
            # Berechne Distanzen zu allen Palette-Farben
            distances = np.sum((palette_array - pixel) ** 2, axis=1)
            # Finde nächste Farbe
            closest_idx = np.argmin(distances)
            result_pixels[i] = palette_array[closest_idx]

            # Progress update every 10000 pixels
            if i % 10000 == 0:
                progress = (i / len(pixels)) * 100
                print(f"   🎨 Palette-Anwendung: {progress:.1f}%", end='\r')

        print(f"   ✅ Palette-Anwendung: 100.0%")
        return result_pixels.reshape(height, width, channels)

    def pixelize(self, image, block_size=4):
        """Pixelize-Effekt"""
        height, width = image.shape[:2]

        # Verkleinere das Bild
        small_height = height // block_size
        small_width = width // block_size
        small_image = cv2.resize(image, (small_width, small_height),
                                 interpolation=cv2.INTER_AREA)

        # Vergrößere es wieder mit Nearest-Neighbor
        pixelized = cv2.resize(small_image, (width, height),
                               interpolation=cv2.INTER_NEAREST)

        return pixelized

    def process_single_gif(self, input_path, output_path):
        """Verarbeite eine einzelne GIF-Datei mit TRANSPARENZ-Erhaltung"""
        try:
            gif = Image.open(input_path)

            # Konfiguration
            config = {
                "resolution": (512, 512),
                "pixelize": 4,
                "apply_palette": True
            }

            processed_frames = []

            # Prüfe ob ursprüngliches GIF Transparenz hat
            has_transparency = False
            if hasattr(gif, 'info') and 'transparency' in gif.info:
                has_transparency = True
            elif gif.mode in ('RGBA', 'LA'):
                has_transparency = True

            # Verarbeite alle Frames
            for frame_index, frame in enumerate(ImageSequence.Iterator(gif)):
                print(
                    f"   ✅ Palette-Anwendung: {((frame_index + 1) / gif.n_frames * 100):.1f}%")

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
                        processed_rgb = self.apply_extracted_palette(
                            rgb_channels)

                        # Kombiniere RGB mit Alpha
                        frame_np = np.dstack([processed_rgb, alpha_channel])
                    else:
                        # Normale RGB-Verarbeitung
                        frame_np = self.apply_extracted_palette(frame_np)

                # Zurück zu PIL
                if has_transparency and frame_np.shape[2] == 4:
                    processed_frame = Image.fromarray(frame_np, 'RGBA')
                else:
                    processed_frame = Image.fromarray(frame_np, 'RGB')

                processed_frames.append(processed_frame)

            # Speichere GIF mit TRANSPARENZ
            if processed_frames:
                duration = gif.info.get('duration', 100)
                loop = gif.info.get('loop', 0)

                # Speicher-Parameter je nach Transparenz
                save_params = {
                    'save_all': True,
                    'append_images': processed_frames[1:],
                    'duration': duration,
                    'loop': loop,
                    'optimize': False
                }

                # Transparenz-Einstellungen
                if has_transparency:
                    # Für transparente GIFs: Konvertiere zu Palette mit Transparenz
                    save_params['transparency'] = 0
                    save_params['disposal'] = 2  # Für saubere Transparenz

                processed_frames[0].save(output_path, **save_params)

            print(
                f"✅ GIF erstellt: {output_path} ({len(processed_frames)} Frames)")
            if has_transparency:
                print("🎨 15-Farben-Palette angewendet mit TRANSPARENZ erhalten")
            else:
                print("🎨 15-Farben-Palette angewendet")

            return True

        except Exception as e:
            print(f"❌ Fehler: {str(e)}")
            return False

    def show_palette_info(self):
        """Zeige Informationen über die verwendete Palette"""
        print("🎨 EXTRAHIERTE 15-FARBEN-PALETTE")
        print("=" * 50)

        categories = [
            ("Dunkel & Monochrom", [0, 1]),
            ("Warme Töne", [2, 3, 4]),
            ("Grüntöne", [5, 6]),
            ("Blautöne", [7, 8, 9, 10]),
            ("Kühle Grautöne", [11, 12, 13]),
            ("Letzter Ton", [14])
        ]

        for category, indices in categories:
            print(f"\n{category}:")
            for i in indices:
                color = self.extracted_palette[i]
                print(f"   #{i+1:2d}: RGB{color}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python single_gif_processor.py <input_gif> <output_gif>")
        print("Example: python single_gif_processor.py input/example.gif output/result.gif")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Prüfe Input-Datei
    if not os.path.exists(input_file):
        print(f"❌ Input-Datei nicht gefunden: {input_file}")
        return

    # Erstelle Output-Verzeichnis falls nötig
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verarbeitung
    processor = SingleGifProcessor()
    processor.show_palette_info()

    print(f"\n🔄 Starte Verarbeitung...")
    print(f"📁 Input:  {input_file}")
    print(f"📁 Output: {output_file}")

    success = processor.process_single_gif(input_file, output_file)

    if success:
        print(f"\n✅ Erfolgreich verarbeitet!")
        print(f"🎯 Einzelne GIF erstellt mit exakter 15-Farben-Palette")
    else:
        print(f"\n❌ Verarbeitung fehlgeschlagen")


if __name__ == "__main__":
    main()
