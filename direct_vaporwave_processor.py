#!/usr/bin/env python3
"""
🎬 DIRECT VAPORWAVE PROCESSOR
============================
Verarbeitet MP4 → Vaporwave GIFs OHNE ComfyUI
Nutzt: FFmpeg + OpenCV + Pillow
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import subprocess
from pathlib import Path
import sys


class VaporwaveProcessor:
    def __init__(self):
        self.input_file = "ComfyUI_engine/input/comica1750462002773.mp4"
        self.output_dir = Path("output/direct_vaporwave")
        self.temp_dir = Path("temp_frames")

        # Vaporwave-Farben (HSV-Format)
        self.vaporwave_colors = [
            (300, 255, 255),  # Magenta
            (180, 255, 255),  # Cyan
            (270, 255, 255),  # Violett
            (330, 255, 180),  # Pink
            (240, 255, 255),  # Blau
        ]

        print("🎬 DIRECT VAPORWAVE PROCESSOR")
        print("=" * 40)

    def create_directories(self):
        """Erstelle Output-Verzeichnisse"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Verzeichnisse erstellt: {self.output_dir}")

    def extract_frames(self):
        """Extrahiere Frames aus MP4"""
        print(f"📹 Extrahiere Frames aus: {self.input_file}")

        if not Path(self.input_file).exists():
            print(f"❌ Video nicht gefunden: {self.input_file}")
            return False

        # Lösche alte Frames
        for frame in self.temp_dir.glob("*.png"):
            frame.unlink()

        # OpenCV für Frame-Extraktion
        cap = cv2.VideoCapture(self.input_file)

        if not cap.isOpened():
            print("❌ Kann Video nicht öffnen")
            return False

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"   📊 Frames: {frame_count}, FPS: {fps:.1f}")

        frames_extracted = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Speichere Frame
            frame_path = self.temp_dir / f"frame_{frames_extracted:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            frames_extracted += 1

            if frames_extracted % 10 == 0:
                print(f"   Frame {frames_extracted}/{frame_count}", end="\r")

        cap.release()
        print(f"\n✅ {frames_extracted} Frames extrahiert")
        return frames_extracted > 0

    def apply_vaporwave_effect(self, image_path):
        """Wende Vaporwave-Effekt auf ein Bild an"""
        try:
            # Lade Bild
            img = Image.open(image_path)
            original_size = img.size

            # 1. Upscaling (2x mit Lanczos)
            img = img.resize(
                (original_size[0] * 2, original_size[1] * 2), Image.Resampling.LANCZOS)

            # 2. Konvertiere zu HSV für Farbmanipulation
            img_hsv = img.convert('HSV')
            hsv_array = np.array(img_hsv)

            # 3. Vaporwave-Farbshift
            # Erhöhe Sättigung
            hsv_array[:, :, 1] = np.clip(hsv_array[:, :, 1] * 1.4, 0, 255)

            # Shift zu Magenta/Cyan-Tönen
            hue = hsv_array[:, :, 0].astype(float)
            hue = (hue + 30) % 180  # Shift Richtung Magenta
            hsv_array[:, :, 0] = hue.astype(np.uint8)

            # 4. Zurück zu RGB
            img_vaporwave = Image.fromarray(hsv_array, 'HSV').convert('RGB')

            # 5. Kontrast und Helligkeit anpassen
            enhancer = ImageEnhance.Contrast(img_vaporwave)
            img_vaporwave = enhancer.enhance(1.3)

            enhancer = ImageEnhance.Brightness(img_vaporwave)
            img_vaporwave = enhancer.enhance(1.1)

            # 6. Leichter Blur für VHS-Effekt
            img_vaporwave = img_vaporwave.filter(
                ImageFilter.GaussianBlur(radius=0.5))

            # 7. Film-Grain simulieren
            grain = np.random.randint(-10, 10,
                                      img_vaporwave.size[::-1] + (3,), dtype=np.int16)
            img_array = np.array(img_vaporwave).astype(np.int16)
            img_array = np.clip(img_array + grain, 0, 255).astype(np.uint8)
            img_vaporwave = Image.fromarray(img_array)

            return img_vaporwave

        except Exception as e:
            print(f"❌ Fehler bei Effekt-Anwendung: {e}")
            return None

    def process_frames(self):
        """Verarbeite alle Frames mit Vaporwave-Effekt"""
        print("🎨 Wende Vaporwave-Effekte an...")

        frame_files = sorted(list(self.temp_dir.glob("frame_*.png")))
        total_frames = len(frame_files)

        if total_frames == 0:
            print("❌ Keine Frames gefunden")
            return False

        processed_frames = []

        for i, frame_path in enumerate(frame_files):
            # Effekt anwenden
            vaporwave_img = self.apply_vaporwave_effect(frame_path)

            if vaporwave_img:
                # Speichere verarbeitetes Frame
                output_path = self.output_dir / f"vaporwave_{i:06d}.png"
                vaporwave_img.save(output_path, "PNG")
                processed_frames.append(output_path)

                if (i + 1) % 5 == 0:
                    print(f"   Verarbeitet: {i + 1}/{total_frames}", end="\r")

        print(f"\n✅ {len(processed_frames)} Frames verarbeitet")
        return processed_frames

    def create_gifs(self, processed_frames, frames_per_gif=30):
        """Erstelle GIFs aus verarbeiteten Frames"""
        print("🎬 Erstelle Vaporwave-GIFs...")

        if not processed_frames:
            print("❌ Keine verarbeiteten Frames")
            return []

        gif_files = []
        gif_count = 0

        # Teile Frames in Sequenzen
        for start_idx in range(0, len(processed_frames), frames_per_gif):
            end_idx = min(start_idx + frames_per_gif, len(processed_frames))
            sequence_frames = processed_frames[start_idx:end_idx]

            if len(sequence_frames) < 5:  # Skip zu kurze Sequenzen
                continue

            # Lade Frames für GIF
            gif_frames = []
            for frame_path in sequence_frames:
                img = Image.open(frame_path)
                # Behalte upscaled Größe für bessere Qualität
                gif_frames.append(img)

            # Speichere GIF
            gif_path = self.output_dir / \
                f"vaporwave_sequence_{gif_count + 1}.gif"
            gif_frames[0].save(
                gif_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=83,  # ~12 FPS
                loop=0,
                optimize=True
            )

            gif_files.append(gif_path)
            gif_count += 1
            print(
                f"   GIF {gif_count}: {len(sequence_frames)} frames → {gif_path.name}")

        print(f"✅ {len(gif_files)} GIFs erstellt")
        return gif_files

    def cleanup(self):
        """Räume temporäre Dateien auf"""
        print("🧹 Räume auf...")
        for frame in self.temp_dir.glob("*.png"):
            frame.unlink()

        if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
            self.temp_dir.rmdir()

    def show_results(self, gif_files):
        """Zeige Ergebnisse"""
        print("\n" + "=" * 50)
        print("🎉 VAPORWAVE-TRANSFORMATION ABGESCHLOSSEN!")
        print("=" * 50)

        total_size = 0
        for gif_file in gif_files:
            size_mb = gif_file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"🎬 {gif_file.name}: {size_mb:.1f} MB")

        print(f"\n📊 GESAMT: {len(gif_files)} GIFs, {total_size:.1f} MB")
        print(f"📁 Ausgabe: {self.output_dir}")
        print(f"🚀 Erfolg! Ihre Vaporwave-GIFs sind bereit!")

    def run(self):
        """Hauptausführung"""
        try:
            # 1. Verzeichnisse erstellen
            self.create_directories()

            # 2. Frames extrahieren
            if not self.extract_frames():
                return False

            # 3. Vaporwave-Effekte anwenden
            processed_frames = self.process_frames()
            if not processed_frames:
                return False

            # 4. GIFs erstellen
            gif_files = self.create_gifs(processed_frames)
            if not gif_files:
                return False

            # 5. Ergebnisse anzeigen
            self.show_results(gif_files)

            # 6. Aufräumen
            self.cleanup()

            return True

        except KeyboardInterrupt:
            print("\n⏹️ Abgebrochen")
            return False
        except Exception as e:
            print(f"\n💥 Fehler: {e}")
            return False


def main():
    """Hauptfunktion"""
    print("""
██╗   ██╗ █████╗ ██████╗  ██████╗ ██████╗ ██╗    ██╗ █████╗ ██╗   ██╗███████╗
██║   ██║██╔══██╗██╔══██╗██╔═══██╗██╔══██╗██║    ██║██╔══██╗██║   ██║██╔════╝
██║   ██║███████║██████╔╝██║   ██║██████╔╝██║ █╗ ██║███████║██║   ██║█████╗
╚██╗ ██╔╝██╔══██║██╔═══╝ ██║   ██║██╔══██╗██║███╗██║██╔══██║╚██╗ ██╔╝██╔══╝
 ╚████╔╝ ██║  ██║██║     ╚██████╔╝██║  ██║╚███╔███╔╝██║  ██║ ╚████╔╝ ███████╗
  ╚═══╝  ╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝  ╚═══╝  ╚══════╝

           🚀 DIRECT EDITION - Keine ComfyUI-Abhängigkeiten! 🚀
    """)

    processor = VaporwaveProcessor()
    success = processor.run()

    if success:
        print("\n🎊 MISSION ERFOLGREICH!")
        print("🎬 Ihre Vaporwave-GIFs sind bereit!")
    else:
        print("\n💥 MISSION FEHLGESCHLAGEN!")

    input("\nDrücken Sie Enter zum Beenden...")


if __name__ == "__main__":
    main()
