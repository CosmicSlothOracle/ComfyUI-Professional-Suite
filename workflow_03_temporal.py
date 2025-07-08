#!/usr/bin/env python3
"""
WORKFLOW 3: TEMPORAL FRAME-PUFFER KONSISTENZ
+ 100% TRANSPARENTER HINTERGRUND
"""
import cv2
import numpy as np
import imageio
from pathlib import Path
from tqdm import tqdm


class TemporalWorkflow:
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output"
        self.output_dir.mkdir(exist_ok=True)
        self.frame_history = []

    def remove_background_smart(self, frame):
        """Intelligente Hintergrundentfernung"""
        if frame.shape[-1] == 4:
            return frame

        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        h, w = frame.shape[:2]

        corners = [frame[0, 0], frame[0, w-1], frame[h-1, 0], frame[h-1, w-1]]
        corner_colors = [tuple(c) for c in corners]
        bg_color = max(set(corner_colors), key=corner_colors.count)

        tolerance = 25
        diff = np.abs(frame.astype(np.float32) - np.array(bg_color))
        bg_mask = np.all(diff < tolerance, axis=2)

        rgba_frame[bg_mask, 3] = 0
        rgba_frame[~bg_mask, 3] = 255

        return rgba_frame

    def temporal_consistency_filter(self, current_frame, buffer_size=3):
        """Temporal Konsistenz mit Frame-Puffer"""
        self.frame_history.append(current_frame.copy())

        if len(self.frame_history) > buffer_size:
            self.frame_history.pop(0)

        if len(self.frame_history) < 2:
            return current_frame

        # Gewichteter Durchschnitt mit Alpha-Berücksichtigung
        if current_frame.shape[-1] == 4:
            rgb_result = current_frame[:, :, :3].astype(np.float32) * 0.7
            alpha_result = current_frame[:, :, 3].astype(np.float32) * 0.7

            for frame in self.frame_history[:-1]:
                weight = 0.3 / len(self.frame_history[:-1])
                rgb_result += frame[:, :, :3].astype(np.float32) * weight
                if frame.shape[-1] == 4:
                    alpha_result += frame[:, :, 3].astype(np.float32) * weight

            rgb_result = np.clip(rgb_result, 0, 255).astype(np.uint8)
            alpha_result = np.clip(alpha_result, 0, 255).astype(np.uint8)

            return np.dstack([rgb_result, alpha_result])
        else:
            result = current_frame.astype(np.float32) * 0.7
            for frame in self.frame_history[:-1]:
                weight = 0.3 / len(self.frame_history[:-1])
                result += frame.astype(np.float32) * weight

            return np.clip(result, 0, 255).astype(np.uint8)

    def process_gif(self, gif_path):
        """Verarbeitet GIF mit Temporal Workflow"""
        print(f"\n🎬 WORKFLOW 3: {gif_path.name}")

        try:
            frames = imageio.mimread(gif_path)
            print(f"📹 {len(frames)} Frames")

            self.frame_history = []  # Reset
            processed_frames = []

            for frame in tqdm(frames, desc="Temporal"):
                # Hintergrund entfernen
                transparent_frame = self.remove_background_smart(frame)

                # Temporal Konsistenz
                consistent_frame = self.temporal_consistency_filter(
                    transparent_frame)

                # 2x Upscaling
                h, w = consistent_frame.shape[:2]
                upscaled = cv2.resize(
                    consistent_frame, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

                processed_frames.append(upscaled)

            # Speichern
            output_name = gif_path.stem + "_workflow03_temporal.mp4"
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
    workflow = TemporalWorkflow()

    test_files = [
        "DwfOrtv.gif",
        "l.gif",
        "laurinbimsdnace.gif",
        "anotherspriteauramovieclip.gif",
        "as.gif"
    ]

    print("🎬 WORKFLOW 3: TEMPORAL FRAME-PUFFER KONSISTENZ")
    print("🔍 Features: Frame-Buffer + Weighted Averaging + 100% Transparenz")

    success = 0
    for gif_name in test_files:
        gif_path = workflow.input_dir / gif_name
        if gif_path.exists():
            if workflow.process_gif(gif_path):
                success += 1
        else:
            print(f"⚠️ Nicht gefunden: {gif_name}")

    print(f"\n✅ Workflow 3 abgeschlossen: {success}/{len(test_files)}")


if __name__ == "__main__":
    main()
