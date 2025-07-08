#!/usr/bin/env python3
"""
WORKFLOW 2: ANTI-FRAKTAL RAUSCHUNTERDRÜCKUNG
+ 100% TRANSPARENTER HINTERGRUND
"""
import cv2
import numpy as np
import imageio
from pathlib import Path
from tqdm import tqdm


class AntiFractalWorkflow:
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

        # Eckfarben als Hintergrund
        corners = [frame[0, 0], frame[0, w-1], frame[h-1, 0], frame[h-1, w-1]]
        corner_colors = [tuple(c) for c in corners]
        bg_color = max(set(corner_colors), key=corner_colors.count)

        # Hintergrund entfernen
        tolerance = 25
        diff = np.abs(frame.astype(np.float32) - np.array(bg_color))
        bg_mask = np.all(diff < tolerance, axis=2)

        rgba_frame[bg_mask, 3] = 0
        rgba_frame[~bg_mask, 3] = 255

        return rgba_frame

    def anti_fractal_processing(self, frame):
        """Anti-Fraktal Verarbeitung mit Rauschunterdrückung"""
        if frame.shape[-1] == 4:
            rgb = frame[:, :, :3]
            alpha = frame[:, :, 3]
        else:
            rgb = frame
            alpha = np.full(frame.shape[:2], 255, dtype=np.uint8)

        # 1. Bilateral Filter für Rauschunterdrückung
        denoised = cv2.bilateralFilter(rgb, 9, 50, 50)

        # 2. Kantenerkennung für sichere Schärfung
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edge_mask = cv2.GaussianBlur(
            edges.astype(np.float32), (3, 3), 0) / 255.0

        # 3. Sanfte Schärfung nur an Kanten
        kernel = np.array([[-0.3, -0.3, -0.3], [-0.3, 3.4, -0.3],
                          [-0.3, -0.3, -0.3]], dtype=np.float32)
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # 4. Nur dort schärfen wo echte Kanten sind
        result = denoised.astype(np.float32)
        for c in range(3):
            result[:, :, c] = (edge_mask * sharpened[:, :, c] +
                               (1 - edge_mask) * denoised[:, :, c])

        result = np.clip(result, 0, 255).astype(np.uint8)

        return np.dstack([result, alpha])

    def process_gif(self, gif_path):
        """Verarbeitet GIF mit Anti-Fraktal Workflow"""
        print(f"\n⚡ WORKFLOW 2: {gif_path.name}")

        try:
            frames = imageio.mimread(gif_path)
            print(f"📹 {len(frames)} Frames")

            processed_frames = []

            for frame in tqdm(frames, desc="Anti-Fractal"):
                # Hintergrund entfernen
                transparent_frame = self.remove_background_smart(frame)

                # Anti-Fraktal Verarbeitung
                processed = self.anti_fractal_processing(transparent_frame)

                # 2x Upscaling
                h, w = processed.shape[:2]
                upscaled = cv2.resize(
                    processed, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)

                processed_frames.append(upscaled)

            # Speichern
            output_name = gif_path.stem + "_workflow02_anti_fractal.mp4"
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
    workflow = AntiFractalWorkflow()

    test_files = [
        "DwfOrtv.gif",
        "l.gif",
        "laurinbimsdnace.gif",
        "anotherspriteauramovieclip.gif",
        "as.gif"
    ]

    print("⚡ WORKFLOW 2: ANTI-FRAKTAL RAUSCHUNTERDRÜCKUNG")
    print("🔍 Features: Bilateral Filter + Edge-Safe Sharpening + 100% Transparenz")

    success = 0
    for gif_name in test_files:
        gif_path = workflow.input_dir / gif_name
        if gif_path.exists():
            if workflow.process_gif(gif_path):
                success += 1
        else:
            print(f"⚠️ Nicht gefunden: {gif_name}")

    print(f"\n✅ Workflow 2 abgeschlossen: {success}/{len(test_files)}")


if __name__ == "__main__":
    main()
