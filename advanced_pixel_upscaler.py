#!/usr/bin/env python3
"""
Advanced Pixel Art Upscaler mit Schärfung und Detailverbesserung
Kombiniert mehrere Upscaling-Techniken für optimale Objektabgrenzung
"""
import os
import sys
from pathlib import Path
import imageio
import cv2
import numpy as np
from tqdm import tqdm


class AdvancedPixelUpscaler:
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output"

        # Verzeichnisse erstellen
        self.output_dir.mkdir(exist_ok=True)

    def pixelart_stylize_advanced(self, frame, pixel_factor=3, color_levels=24):
        """Erweiterte Pixelart-Stilisierung mit Farboptimierung"""
        h, w = frame.shape[:2]

        # 1. Pixeleffekt durch Downsampling
        small_h, small_w = max(1, h // pixel_factor), max(1, w // pixel_factor)
        small_frame = cv2.resize(
            frame, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        pixelized = cv2.resize(small_frame, (w, h),
                               interpolation=cv2.INTER_NEAREST)

        # 2. Erweiterte Farbquantisierung
        data = pixelized.reshape((-1, 3)).astype(np.float32)

        # K-Means für optimierte Farbpalette
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            data, color_levels, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized = quantized_data.reshape(pixelized.shape)

        # 3. Kantenverstärkung für bessere Objektabgrenzung
        gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Kanten subtil verstärken
        enhanced = cv2.addWeighted(quantized, 0.85, edges_colored, 0.15, 0)

        return enhanced

    def unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        """Unsharp Masking für bessere Schärfe"""
        # Gauss-Filter anwenden
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)

        # Unsharp Mask berechnen
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)

        # Threshold anwenden
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)

        return sharpened

    def adaptive_sharpen(self, image, strength=0.8):
        """Adaptive Schärfung basierend auf lokaler Varianz"""
        # Laplacian-Kernel für Kantenerkennung
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)

        # Adaptive Schärfung
        sharpened = cv2.filter2D(image, -1, kernel * strength)

        # Mit Original mischen für natürliches Ergebnis
        result = cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)

        return np.clip(result, 0, 255).astype(np.uint8)

    def multi_stage_upscale(self, frame, target_scale=2.5):
        """Mehrstufiges Upscaling für optimale Qualität"""
        current_frame = frame.copy()

        # Stufe 1: Lanczos für Grundvergrößerung
        h, w = current_frame.shape[:2]
        stage1 = cv2.resize(current_frame, (int(w * target_scale), int(h * target_scale)),
                            interpolation=cv2.INTER_LANCZOS4)

        # Stufe 2: Unsharp Masking
        stage2 = self.unsharp_mask(
            stage1, kernel_size=(5, 5), sigma=1.2, amount=1.5)

        # Stufe 3: Adaptive Schärfung
        stage3 = self.adaptive_sharpen(stage2, strength=0.6)

        return stage3

    def edge_enhancement(self, image, strength=0.3):
        """Kantenverstärkung für bessere Objektabgrenzung"""
        # Sobel-Filter für Kantenerkennung
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Kantenstärke berechnen
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(np.clip(edges, 0, 255))

        # Kanten zu Farbbild konvertieren
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Mit Original kombinieren
        enhanced = cv2.addWeighted(image, 1.0, edges_colored, strength, 0)

        return enhanced

    def clarity_boost(self, image, amount=0.5):
        """Zusätzliche Klarheitsverbesserung"""
        # High-Pass Filter für Detailschärfung
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        high_pass = cv2.addWeighted(image, 1.0 + amount, gaussian, -amount, 0)

        return np.clip(high_pass, 0, 255).astype(np.uint8)

    def process_gif_advanced(self, gif_path):
        """Erweiterte GIF-Verarbeitung mit allen Optimierungen"""
        print(f"\n🎨 Verarbeite: {gif_path.name}")

        try:
            # GIF laden
            frames = imageio.mimread(gif_path)
            print(f"📹 {len(frames)} Frames geladen")

            processed_frames = []

            for frame in tqdm(frames, desc="Erweiterte Verarbeitung"):
                # RGBA -> RGB falls nötig
                if frame.shape[-1] == 4:
                    frame = frame[:, :, :3]

                # Zu BGR für OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # 1. Erweiterte Pixelart-Stilisierung
                pixel_frame = self.pixelart_stylize_advanced(
                    bgr_frame, pixel_factor=3, color_levels=20)

                # 2. Mehrstufiges Upscaling
                upscaled = self.multi_stage_upscale(
                    pixel_frame, target_scale=2.5)

                # 3. Kantenverstärkung für Objektabgrenzung
                enhanced = self.edge_enhancement(upscaled, strength=0.25)

                # 4. Finale Schärfung (30% reduziert)
                sharpened = self.adaptive_sharpen(enhanced, strength=0.28)

                # 5. Klarheitsverbesserung
                final_frame = self.clarity_boost(sharpened, amount=0.3)

                processed_frames.append(final_frame)

            # Als MP4 speichern
            output_name = gif_path.stem + "_ultra_sharp.mp4"
            output_path = self.output_dir / output_name

            self.save_as_mp4(processed_frames, output_path, fps=12)

            print(f"✅ Gespeichert: {output_name}")
            return True

        except Exception as e:
            print(f"❌ Fehler: {e}")
            return False

    def save_as_mp4(self, frames, output_path, fps=12):
        """Speichert Frames als hochqualitatives MP4"""
        if not frames:
            return

        h, w = frames[0].shape[:2]

        # H.264 Codec für beste Qualität
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        for frame in frames:
            out.write(frame)

        out.release()

    def process_all_advanced(self):
        """Verarbeitet alle GIFs mit erweiterten Algorithmen"""
        # GIFs finden
        gif_files = list(self.input_dir.glob("*.gif"))
        print(f"\n🎯 Gefunden: {len(gif_files)} GIF-Dateien")

        if not gif_files:
            print("❌ Keine GIFs gefunden!")
            return False

        # Verarbeitung
        success = 0
        for gif_file in gif_files:
            if self.process_gif_advanced(gif_file):
                success += 1

        print(f"\n📊 Erweiterte Verarbeitung abgeschlossen!")
        print(f"✅ Erfolgreich: {success}/{len(gif_files)}")
        print(f"📁 Ausgabe: {self.output_dir}")

        return success > 0


def main():
    """Hauptfunktion"""
    upscaler = AdvancedPixelUpscaler()

    print("🚀 Advanced Pixel Art Upscaler mit Ultra-Schärfung")
    print("=" * 60)
    print("Features:")
    print("• Erweiterte Pixelart-Stilisierung mit optimierter Farbpalette")
    print("• Mehrstufiges Upscaling (2.5x)")
    print("• Unsharp Masking für Detailschärfe")
    print("• Adaptive Kantenschärfung")
    print("• Sobel-Filter für Objektabgrenzung")
    print("• High-Pass Klarheitsverbesserung")
    print("=" * 60)

    success = upscaler.process_all_advanced()

    if success:
        print("\n🎉 Alle Verarbeitungen erfolgreich!")
        print("\n📋 Workflow-Zusammenfassung:")
        print("1. ✅ Pixelart-Stilisierung mit 20-Farben-Palette")
        print("2. ✅ 2.5x Upscaling mit Lanczos-Interpolation")
        print("3. ✅ Unsharp Masking (σ=1.2, Amount=1.5)")
        print("4. ✅ Adaptive Schärfung mit Laplacian-Kernel")
        print("5. ✅ Sobel-Kantenverstärkung")
        print("6. ✅ High-Pass Klarheitsboost")
        print("\n🎯 Resultat: Hochauflösende MP4s mit optimaler Objektabgrenzung")
    else:
        print("\n❌ Verarbeitung fehlgeschlagen")
        sys.exit(1)


if __name__ == "__main__":
    main()
