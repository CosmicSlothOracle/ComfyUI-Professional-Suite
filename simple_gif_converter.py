#!/usr/bin/env python3
"""
Einfacher GIF zu MP4 Konverter mit Pixelart-Effekt
"""
import os
import sys
from pathlib import Path
import imageio
import cv2
import numpy as np
from tqdm import tqdm


def pixelart_effect(frame, factor=3):
    """Pixelart-Effekt durch Downsampling"""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w//factor, h//factor),
                       interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def process_gif(gif_path, output_dir):
    """Verarbeite ein einzelnes GIF"""
    print(f"🎬 Verarbeite: {gif_path.name}")

    try:
        # GIF laden
        frames = imageio.mimread(gif_path)
        print(f"📹 {len(frames)} Frames geladen")

        processed_frames = []

        for frame in tqdm(frames, desc="Verarbeitung"):
            # RGBA -> RGB falls nötig
            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]

            # Zu BGR für OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Pixelart-Effekt
            pixel_frame = pixelart_effect(bgr_frame)

            # 2x Upscaling
            h, w = pixel_frame.shape[:2]
            upscaled = cv2.resize(pixel_frame, (w*2, h*2),
                                  interpolation=cv2.INTER_LANCZOS4)

            processed_frames.append(upscaled)

        # Als MP4 speichern
        output_name = gif_path.stem + "_pixelart.mp4"
        output_path = output_dir / output_name

        if processed_frames:
            h, w = processed_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, 10.0, (w, h))

            for frame in processed_frames:
                out.write(frame)
            out.release()

            print(f"✅ Gespeichert: {output_name}")
            return True

    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False


def main():
    """Hauptfunktion"""
    base_dir = Path(__file__).parent
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"

    output_dir.mkdir(exist_ok=True)

    # Alle GIFs finden
    gif_files = list(input_dir.glob("*.gif"))
    print(f"Gefunden: {len(gif_files)} GIF-Dateien")

    if not gif_files:
        print("❌ Keine GIFs gefunden!")
        return

    # Verarbeitung
    success = 0
    for gif_file in gif_files:
        if process_gif(gif_file, output_dir):
            success += 1

    print(f"\n📊 Fertig! {success}/{len(gif_files)} erfolgreich verarbeitet")


if __name__ == "__main__":
    main()
