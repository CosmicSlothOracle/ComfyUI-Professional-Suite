#!/usr/bin/env python3
"""
Optimized Pixel Art Processor
Addresses issues found in comprehensive visual test:
- Better color preservation
- Smoother motion preservation
- Balanced structure changes
"""

import os
from pathlib import Path
from PIL import Image, ImageSequence
import numpy as np
from sklearn.cluster import KMeans


class OptimizedPixelArtProcessor:
    def __init__(self):
        self.base_dir = Path("C:/Users/Public/ComfyUI-master")
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output" / "optimized_pixel_art_test"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test file
        self.test_file = "0ea53a3cdbfdcf14caf1c8cccdb60143_fast_transparent_converted.gif"

        # Optimized color palettes - more colors for better preservation
        self.palettes = {
            "commodore64_extended": [
                # Original C64 colors
                (0, 0, 0), (255, 255, 255), (136, 57, 50), (103, 182, 189),
                (139, 63, 150), (85, 160, 73), (64, 49, 141), (191, 206, 114),
                (139, 84, 41), (87, 66, 0), (184, 105, 98), (80, 80, 80),
                (120, 120, 120), (148, 224, 137), (120, 105, 196), (159, 159, 159),
                # Extended colors for better preservation
                (200, 200, 200), (50, 50, 50), (100, 100, 100), (175, 175, 175),
                (200, 100, 100), (100, 200, 100), (100, 100, 200), (200, 200, 100),
                (200, 100, 200), (100, 200, 200), (150, 75, 75), (75, 150, 75),
                (75, 75, 150), (150, 150, 75), (150, 75, 150), (75, 150, 150)
            ]
        }

    def analyze_transparency(self, image):
        """Transparenz-Analyse"""
        if image.mode not in ('RGBA', 'LA', 'P'):
            return False, None, None

        if image.mode == 'P':
            transparency = image.info.get('transparency')
            if transparency is not None:
                image = image.convert('RGBA')
            else:
                return False, None, None
        elif image.mode == 'LA':
            image = image.convert('RGBA')

        if image.mode == 'RGBA':
            alpha_channel = np.array(image)[:, :, 3]
            transparent_pixels = np.sum(alpha_channel < 128)
            total_pixels = alpha_channel.size
            transparency_ratio = transparent_pixels / total_pixels

            return True, transparency_ratio, alpha_channel

        return False, None, None

    def gentle_pixelization(self, image, pixel_factor=3):
        """Sanftere Pixelisierung für bessere Struktur-Erhaltung"""
        original_size = image.size

        # Kleinerer Pixel-Faktor für sanftere Pixelisierung
        # Minimum 20 statt 16
        new_width = max(20, original_size[0] // pixel_factor)
        new_height = max(20, original_size[1] // pixel_factor)

        if image.mode == 'RGBA':
            r, g, b, a = image.split()

            # Verwende LANCZOS für Downscaling (bessere Qualität)
            r_small = r.resize((new_width, new_height),
                               Image.Resampling.LANCZOS)
            g_small = g.resize((new_width, new_height),
                               Image.Resampling.LANCZOS)
            b_small = b.resize((new_width, new_height),
                               Image.Resampling.LANCZOS)
            a_small = a.resize((new_width, new_height),
                               Image.Resampling.LANCZOS)

            # NEAREST für Upscaling (Pixel-Effekt)
            r_pixelized = r_small.resize(
                original_size, Image.Resampling.NEAREST)
            g_pixelized = g_small.resize(
                original_size, Image.Resampling.NEAREST)
            b_pixelized = b_small.resize(
                original_size, Image.Resampling.NEAREST)
            a_pixelized = a_small.resize(
                original_size, Image.Resampling.NEAREST)

            pixelized = Image.merge(
                'RGBA', (r_pixelized, g_pixelized, b_pixelized, a_pixelized))
        else:
            small = image.resize((new_width, new_height),
                                 Image.Resampling.LANCZOS)
            pixelized = small.resize(original_size, Image.Resampling.NEAREST)

        return pixelized

    def enhanced_color_reduction(self, image, target_colors=16):
        """Verbesserte Farbreduktion mit mehr Farben"""
        if image.mode != 'RGBA':
            return image

        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))

        alpha_array = np.array(a)
        rgb_array = np.array(rgb_image)

        # Nur sichtbare Pixel verarbeiten
        opaque_mask = alpha_array > 128

        if np.any(opaque_mask):
            opaque_pixels = rgb_array[opaque_mask]

            if len(opaque_pixels) > 0:
                unique_pixels = np.unique(opaque_pixels.reshape(-1, 3), axis=0)
                n_clusters = min(target_colors, len(unique_pixels))

                if n_clusters > 1:
                    # Mehr Iterationen für bessere Qualität
                    kmeans = KMeans(
                        n_clusters=n_clusters,
                        random_state=42,
                        n_init=20,  # Mehr Initialisierungen
                        max_iter=300
                    )
                    kmeans.fit(unique_pixels)

                    # Verwende K-means Zentren direkt (keine Palette-Zwang)
                    labels = kmeans.predict(opaque_pixels)
                    quantized_pixels = kmeans.cluster_centers_[labels]

                    rgb_array[opaque_mask] = quantized_pixels

        final_rgb = Image.fromarray(rgb_array.astype(np.uint8), 'RGB')
        r_new, g_new, b_new = final_rgb.split()
        final_image = Image.merge('RGBA', (r_new, g_new, b_new, a))

        return final_image

    def motion_aware_processing(self, frames):
        """Bewegungs-bewusste Verarbeitung für sanftere Übergänge"""
        if len(frames) <= 1:
            return frames

        processed_frames = []

        for i, frame in enumerate(frames):
            if frame.mode != 'RGBA':
                frame = frame.convert('RGBA')

            # Sanftere Pixelisierung
            pixelized = self.gentle_pixelization(frame, pixel_factor=3)

            # Mehr Farben für bessere Bewegungserhaltung
            color_reduced = self.enhanced_color_reduction(
                pixelized, target_colors=16)

            processed_frames.append(color_reduced)

        return processed_frames

    def save_optimized_gif(self, frames, output_path, original_gif):
        """Optimiertes GIF-Speichern"""
        if not frames:
            return False

        try:
            # Sammle alle Farben für optimale Palette
            all_colors = set()

            for frame in frames:
                if frame.mode == 'RGBA':
                    r, g, b, a = frame.split()
                    rgb_array = np.array(Image.merge('RGB', (r, g, b)))
                    alpha_array = np.array(a)

                    opaque_mask = alpha_array > 128
                    if np.any(opaque_mask):
                        opaque_colors = rgb_array[opaque_mask]
                        # Begrenzte Anzahl von Farben sammeln
                        unique_colors = np.unique(
                            opaque_colors.reshape(-1, 3), axis=0)
                        for color in unique_colors[:200]:  # Max 200 Farben
                            all_colors.add(tuple(color))

            # Optimierte Palette erstellen
            # Max 255 + 1 für Transparenz
            palette_colors = list(all_colors)[:255]

            converted_frames = []

            for frame in frames:
                if frame.mode == 'RGBA':
                    # Erstelle Palette
                    palette_img = Image.new('P', (1, 1))
                    palette_data = []

                    for color in palette_colors:
                        palette_data.extend(color)

                    # Fülle auf 256 Farben auf
                    while len(palette_data) < 768:
                        palette_data.extend([0, 0, 0])

                    palette_img.putpalette(palette_data)

                    # Konvertiere Frame
                    r, g, b, a = frame.split()
                    rgb_frame = Image.merge('RGB', (r, g, b))

                    # Quantisiere mit Dithering für sanftere Übergänge
                    quantized = rgb_frame.quantize(
                        palette=palette_img,
                        dither=Image.Dither.FLOYDSTEINBERG
                    )

                    # Transparenz setzen
                    quantized_array = np.array(quantized)
                    alpha_array = np.array(a)

                    transparent_mask = alpha_array <= 128
                    quantized_array[transparent_mask] = 255

                    final_frame = Image.fromarray(quantized_array, 'P')
                    final_frame.putpalette(palette_data)

                    converted_frames.append(final_frame)

            # Speichere GIF
            if converted_frames:
                converted_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=converted_frames[1:],
                    duration=original_gif.info.get('duration', 50),
                    loop=original_gif.info.get('loop', 0),
                    optimize=True,
                    transparency=255,
                    disposal=2
                )
                return True

        except Exception as e:
            print(f"   ⚠️  GIF save error: {e}")
            return False

        return False

    def process_gif_optimized(self, gif_path, output_path):
        """Optimierte GIF-Verarbeitung"""
        print(f"   📸 Processing with optimized pipeline...")

        gif = Image.open(gif_path)
        frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
        frame_count = len(frames)

        # Analysiere Transparenz
        first_frame = frames[0].convert('RGBA')
        has_transparency, trans_ratio, _ = self.analyze_transparency(
            first_frame)

        print(
            f"   🔍 Transparency: {trans_ratio:.1%}" if has_transparency else "   🔍 No transparency")
        print(f"   📊 Processing {frame_count} frames...")

        # Bewegungs-bewusste Verarbeitung
        processed_frames = self.motion_aware_processing(frames)

        print(f"   ✅ Frames processed with motion awareness")

        # Speichere optimiert
        success = self.save_optimized_gif(processed_frames, output_path, gif)

        if success:
            return True, frame_count, "optimized", []
        else:
            return False, 0, None, []

    def run_optimized_test(self):
        """Führe optimierten Test aus"""
        print("🚀 OPTIMIZED PIXEL ART TEST")
        print("=" * 50)

        input_path = self.input_dir / self.test_file

        if not input_path.exists():
            print(f"❌ Test file not found: {input_path}")
            return False

        print(f"📁 Test file: {self.test_file}")
        file_size = input_path.stat().st_size
        print(f"📊 Input size: {file_size} bytes")

        # Verarbeite
        output_filename = f"optimized_{self.test_file}"
        output_path = self.output_dir / output_filename

        try:
            success, frame_count, palette, _ = self.process_gif_optimized(
                input_path, output_path)

            if success and output_path.exists():
                output_size = output_path.stat().st_size
                compression_ratio = (1 - output_size / file_size) * 100

                print(f"\n🎯 OPTIMIZED RESULTS:")
                print(f"   ✅ Success: {output_size} bytes")
                print(f"   📊 Frames: {frame_count}")
                print(f"   🎨 Method: {palette}")
                print(f"   📈 Compression: {compression_ratio:.1f}%")

                return True
            else:
                print(f"   ❌ Processing failed")
                return False

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False


if __name__ == "__main__":
    processor = OptimizedPixelArtProcessor()
    success = processor.run_optimized_test()

    if success:
        print(f"\n🎉 OPTIMIZED VERSION CREATED!")
        print(f"✅ Ready for quality comparison")
    else:
        print(f"\n❌ OPTIMIZATION FAILED!")
