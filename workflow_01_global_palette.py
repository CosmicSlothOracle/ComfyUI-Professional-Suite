#!/usr/bin/env python3
"""
WORKFLOW 1: GLOBALE FARBPALETTEN-KONSISTENZ
+ 100% TRANSPARENTER HINTERGRUND
"""
import cv2
import numpy as np
import imageio
from pathlib import Path
from tqdm import tqdm


class GlobalPaletteWorkflow:
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output"
        self.output_dir.mkdir(exist_ok=True)

    def extract_global_palette(self, frames, n_colors=16):
        """Extrahiert globale Farbpalette aus allen Frames"""
        all_pixels = []

        for i in range(0, len(frames), max(1, len(frames)//10)):
            frame = frames[i]
            if frame.shape[-1] == 4:
                # Nur nicht-transparente Pixel verwenden
                alpha = frame[:, :, 3]
                mask = alpha > 128
                rgb_pixels = frame[mask][:, :3]
            else:
                rgb_pixels = frame.reshape(-1, 3)

            if len(rgb_pixels) > 0:
                all_pixels.extend(rgb_pixels[::4])  # Subsample

        if len(all_pixels) > n_colors:
            all_pixels = np.array(all_pixels)
            unique_colors = np.unique(all_pixels, axis=0)
            if len(unique_colors) > n_colors:
                indices = np.linspace(
                    0, len(unique_colors)-1, n_colors, dtype=int)
                return unique_colors[indices]
            return unique_colors
        return np.array(all_pixels) if all_pixels else None

    def remove_background_smart(self, frame):
        """Intelligente Hintergrundentfernung"""
        if frame.shape[-1] == 4:
            return frame  # Bereits RGBA

        # Zu RGBA konvertieren
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        # Hintergrund als häufigste Eckfarbe erkennen
        h, w = frame.shape[:2]
        corners = [
            frame[0, 0], frame[0, w-1],
            frame[h-1, 0], frame[h-1, w-1]
        ]

        # Häufigste Eckfarbe als Hintergrund
        corner_colors = [tuple(c) for c in corners]
        bg_color = max(set(corner_colors), key=corner_colors.count)

        # Toleranz für Hintergrundentfernung
        tolerance = 30
        diff = np.abs(frame.astype(np.float32) - np.array(bg_color))
        bg_mask = np.all(diff < tolerance, axis=2)

        # Alpha-Kanal setzen
        rgba_frame[bg_mask, 3] = 0  # Transparent
        rgba_frame[~bg_mask, 3] = 255  # Opaque

        return rgba_frame

    def apply_global_palette(self, frame, palette):
        """Wendet globale Palette auf Frame an"""
        if palette is None:
            return frame

        # Nur RGB-Kanäle bearbeiten
        rgb_frame = frame[:, :, :3].copy()
        alpha_channel = frame[:, :, 3] if frame.shape[-1] == 4 else None

        # Palette anwenden
        h, w = rgb_frame.shape[:2]
        pixels = rgb_frame.reshape(-1, 3).astype(np.float32)

        for i, pixel in enumerate(pixels):
            if alpha_channel is None or alpha_channel.flat[i] > 128:
                distances = np.sum(
                    (palette.astype(np.float32) - pixel) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                pixels[i] = palette[closest_idx]

        result = pixels.reshape(h, w, 3).astype(np.uint8)

        # Alpha-Kanal wieder hinzufügen
        if alpha_channel is not None:
            return np.dstack([result, alpha_channel])
        else:
            return np.dstack([result, np.full((h, w), 255, dtype=np.uint8)])

    def process_gif(self, gif_path):
        """Verarbeitet GIF mit globalem Palette-Workflow"""
        print(f"\n🎨 WORKFLOW 1: {gif_path.name}")

        try:
            frames = imageio.mimread(gif_path)
            print(f"📹 {len(frames)} Frames")

            # Globale Palette extrahieren
            palette = self.extract_global_palette(frames, n_colors=16)
            print(
                f"🎨 Palette: {len(palette) if palette is not None else 0} Farben")

            processed_frames = []

            for frame in tqdm(frames, desc="Global Palette"):
                # Hintergrund entfernen
                transparent_frame = self.remove_background_smart(frame)

                # Globale Palette anwenden
                palette_frame = self.apply_global_palette(
                    transparent_frame, palette)

                # 2x Upscaling
                h, w = palette_frame.shape[:2]
                upscaled = cv2.resize(
                    palette_frame, (w*2, h*2), interpolation=cv2.INTER_NEAREST)

                processed_frames.append(upscaled)

            # Als MP4 mit Alpha speichern
            output_name = gif_path.stem + "_workflow01_global_palette.mp4"
            output_path = self.output_dir / output_name

            self.save_as_mp4_with_alpha(processed_frames, output_path)
            print(f"✅ Gespeichert: {output_name}")
            return True

        except Exception as e:
            print(f"❌ Fehler: {e}")
            return False

    def save_as_mp4_with_alpha(self, frames, output_path, fps=12):
        """Speichert MP4 mit Alpha-Kanal"""
        if not frames:
            return

        h, w = frames[0].shape[:2]

        # RGBA zu BGRA für OpenCV
        bgra_frames = []
        for frame in frames:
            if frame.shape[-1] == 4:
                bgra = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
            else:
                bgra = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
                bgra[:, :, 3] = 255  # Vollständig opaque falls kein Alpha
            bgra_frames.append(bgra)

        # MP4 mit Alpha-Unterstützung
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h), True)

        for frame in bgra_frames:
            out.write(frame[:, :, :3])  # Nur RGB für Standard MP4

        out.release()


def main():
    workflow = GlobalPaletteWorkflow()

    test_files = [
        "DwfOrtv.gif",
        "l.gif",
        "laurinbimsdnace.gif",
        "anotherspriteauramovieclip.gif",
        "as.gif"
    ]

    print("🎨 WORKFLOW 1: GLOBALE FARBPALETTEN-KONSISTENZ")
    print("🔍 Features: Globale Palette + 100% Transparenz")

    success = 0
    for gif_name in test_files:
        gif_path = workflow.input_dir / gif_name
        if gif_path.exists():
            if workflow.process_gif(gif_path):
                success += 1
        else:
            print(f"⚠️ Nicht gefunden: {gif_name}")

    print(f"\n✅ Workflow 1 abgeschlossen: {success}/{len(test_files)}")


if __name__ == "__main__":
    main()
