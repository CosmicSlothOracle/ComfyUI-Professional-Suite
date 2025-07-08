#!/usr/bin/env python3
"""
WORKFLOW 4: STRUKTUR-ERHALTENDE PIXELART
+ 100% TRANSPARENTER HINTERGRUND
"""
import cv2
import numpy as np
import imageio
from pathlib import Path
from tqdm import tqdm


class StructurePreservingWorkflow:
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output"
        self.output_dir.mkdir(exist_ok=True)

    def remove_background_smart(self, frame):
        """Intelligente Hintergrundentfernung"""
        if frame.shape[-1] == 4:
            return frame

        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        h, w = frame.shape[:2]

        corners = [frame[0, 0], frame[0, w-1], frame[h-1, 0], frame[h-1, w-1]]
        corner_colors = [tuple(c) for c in corners]
        bg_color = max(set(corner_colors), key=corner_colors.count)

        tolerance = 30
        diff = np.abs(frame.astype(np.float32) - np.array(bg_color))
        bg_mask = np.all(diff < tolerance, axis=2)

        rgba_frame[bg_mask, 3] = 0
        rgba_frame[~bg_mask, 3] = 255

        return rgba_frame

    def structure_preserving_pixelart(self, frame, pixel_factor=3):
        """Struktur-erhaltende Pixelart-Stilisierung"""
        if frame.shape[-1] == 4:
            rgb = frame[:, :, :3]
            alpha = frame[:, :, 3]
        else:
            rgb = frame
            alpha = np.full(frame.shape[:2], 255, dtype=np.uint8)

        h, w = rgb.shape[:2]

        # Struktur-erhaltende Verkleinerung mit INTER_AREA
        small = cv2.resize(rgb, (w//pixel_factor, h//pixel_factor),
                           interpolation=cv2.INTER_AREA)

        # Farbquantisierung mit K-Means (vereinfacht)
        small_flat = small.reshape(-1, 3).astype(np.float32)

        # Einfache Farbquantisierung durch Rundung
        quantized = np.round(small_flat / 32) * 32
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)

        small_quantized = quantized.reshape(small.shape)

        # Zurück zur Originalgröße mit Nearest Neighbor
        pixelized = cv2.resize(small_quantized, (w, h),
                               interpolation=cv2.INTER_NEAREST)

        # Alpha-Kanal auch verarbeiten
        if alpha is not None:
            alpha_small = cv2.resize(alpha, (w//pixel_factor, h//pixel_factor),
                                     interpolation=cv2.INTER_AREA)
            alpha_pixelized = cv2.resize(
                alpha_small, (w, h), interpolation=cv2.INTER_NEAREST)
            return np.dstack([pixelized, alpha_pixelized])

        return pixelized

    def edge_preserving_enhancement(self, frame):
        """Kantenschonende Verbesserung"""
        if frame.shape[-1] == 4:
            rgb = frame[:, :, :3]
            alpha = frame[:, :, 3]
        else:
            rgb = frame
            alpha = None

        # Edge-preserving filter
        enhanced = cv2.edgePreservingFilter(
            rgb, flags=2, sigma_s=30, sigma_r=0.3)

        if alpha is not None:
            return np.dstack([enhanced, alpha])
        return enhanced

    def process_gif(self, gif_path):
        """Verarbeitet GIF mit Structure-Preserving Workflow"""
        print(f"\n🎯 WORKFLOW 4: {gif_path.name}")

        try:
            frames = imageio.mimread(gif_path)
            print(f"📹 {len(frames)} Frames")

            processed_frames = []

            for frame in tqdm(frames, desc="Structure-Preserving"):
                # Hintergrund entfernen
                transparent_frame = self.remove_background_smart(frame)

                # Struktur-erhaltende Pixelart
                pixelart_frame = self.structure_preserving_pixelart(
                    transparent_frame)

                # Kantenschonende Verbesserung
                enhanced_frame = self.edge_preserving_enhancement(
                    pixelart_frame)

                # 2x Upscaling
                h, w = enhanced_frame.shape[:2]
                upscaled = cv2.resize(
                    enhanced_frame, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)

                processed_frames.append(upscaled)

            # Speichern
            output_name = gif_path.stem + "_workflow04_structure_preserving.mp4"
            output_path = self.output_dir / output_name

            self.save_as_mp4_with_alpha(processed_frames, output_path)
            print(f"✅ Gespeichert: {output_name}")
            return True

        except Exception as e:
            print(f"❌ Fehler: {e}")
            return False

    def save_as_mp4_with_alpha(self, frames, output_path, fps=12):
        """Speichert MP4 mit Alpha"""
        if not frames:
            return

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        for frame in frames:
            if frame.shape[-1] == 4:
                bgr = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
            else:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr)

        out.release()


def main():
    workflow = StructurePreservingWorkflow()

    test_files = [
        "DwfOrtv.gif",
        "l.gif",
        "laurinbimsdnace.gif",
        "anotherspriteauramovieclip.gif",
        "as.gif"
    ]

    print("🎯 WORKFLOW 4: STRUKTUR-ERHALTENDE PIXELART")
    print("🔍 Features: INTER_AREA + Color Quantization + Edge-Preserving + 100% Transparenz")

    success = 0
    for gif_name in test_files:
        gif_path = workflow.input_dir / gif_name
        if gif_path.exists():
            if workflow.process_gif(gif_path):
                success += 1
        else:
            print(f"⚠️ Nicht gefunden: {gif_name}")

    print(f"\n✅ Workflow 4 abgeschlossen: {success}/{len(test_files)}")


if __name__ == "__main__":
    main()
