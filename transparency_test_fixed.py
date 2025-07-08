#!/usr/bin/env python3
"""
Fixed Transparency Test - Proper GIF saving with transparency
Test with: 0ea53a3cdbfdcf14caf1c8cccdb60143_fast_transparent_converted.gif
"""

import os
from pathlib import Path
from PIL import Image, ImageSequence
import numpy as np
from sklearn.cluster import KMeans


class FixedTransparencyPixelArt:
    def __init__(self):
        self.base_dir = Path("C:/Users/Public/ComfyUI-master")
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output" / "transparency_test_fixed"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test file
        self.test_file = "0ea53a3cdbfdcf14caf1c8cccdb60143_fast_transparent_converted.gif"

        # Enhanced color palettes
        self.palettes = {
            "commodore64": [
                (0, 0, 0), (255, 255, 255), (136, 57, 50), (103, 182, 189),
                (139, 63, 150), (85, 160, 73), (64, 49, 141), (191, 206, 114),
                (139, 84, 41), (87, 66, 0), (184, 105, 98), (80, 80, 80),
                (120, 120, 120), (148, 224, 137), (120, 105, 196), (159, 159, 159)
            ]
        }

    def analyze_transparency(self, image):
        """Analysiere Transparenz im Bild"""
        if image.mode not in ('RGBA', 'LA', 'P'):
            return False, None, None

        # Konvertiere zu RGBA für einheitliche Behandlung
        if image.mode == 'P':
            transparency = image.info.get('transparency')
            if transparency is not None:
                image = image.convert('RGBA')
            else:
                return False, None, None
        elif image.mode == 'LA':
            image = image.convert('RGBA')

        # Analysiere Alpha-Kanal
        if image.mode == 'RGBA':
            alpha_channel = np.array(image)[:, :, 3]
            transparent_pixels = np.sum(alpha_channel < 128)
            total_pixels = alpha_channel.size
            transparency_ratio = transparent_pixels / total_pixels

            return True, transparency_ratio, alpha_channel

        return False, None, None

    def pixelize_with_transparency(self, image, pixel_factor=4):
        """Pixelisierung unter Beibehaltung der Transparenz"""
        original_size = image.size

        # Berechne neue Größe
        new_width = max(16, original_size[0] // pixel_factor)
        new_height = max(16, original_size[1] // pixel_factor)

        if image.mode == 'RGBA':
            # Separate RGB und Alpha-Kanäle
            r, g, b, a = image.split()

            # Pixelisiere jeden Kanal separat
            r_small = r.resize((new_width, new_height),
                               Image.Resampling.LANCZOS)
            g_small = g.resize((new_width, new_height),
                               Image.Resampling.LANCZOS)
            b_small = b.resize((new_width, new_height),
                               Image.Resampling.LANCZOS)
            a_small = a.resize((new_width, new_height),
                               Image.Resampling.LANCZOS)

            # Upscale mit Nearest Neighbor für Pixel-Look
            r_pixelized = r_small.resize(
                original_size, Image.Resampling.NEAREST)
            g_pixelized = g_small.resize(
                original_size, Image.Resampling.NEAREST)
            b_pixelized = b_small.resize(
                original_size, Image.Resampling.NEAREST)
            a_pixelized = a_small.resize(
                original_size, Image.Resampling.NEAREST)

            # Kombiniere Kanäle
            pixelized = Image.merge(
                'RGBA', (r_pixelized, g_pixelized, b_pixelized, a_pixelized))
        else:
            # Fallback für andere Modi
            small = image.resize((new_width, new_height),
                                 Image.Resampling.LANCZOS)
            pixelized = small.resize(original_size, Image.Resampling.NEAREST)

        return pixelized

    def apply_palette_with_transparency(self, image, palette_name):
        """Palette-Anwendung unter Beibehaltung der Transparenz"""
        if image.mode != 'RGBA':
            return image

        palette = self.palettes[palette_name]

        # Trenne RGB und Alpha
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))

        # Konvertiere zu Arrays
        alpha_array = np.array(a)
        rgb_array = np.array(rgb_image)

        # Maske für nicht-transparente Pixel (Alpha > 128)
        opaque_mask = alpha_array > 128

        if np.any(opaque_mask):
            # Extrahiere nur sichtbare Pixel
            opaque_pixels = rgb_array[opaque_mask]

            if len(opaque_pixels) > 0:
                # K-means für Farbreduktion
                unique_pixels = np.unique(opaque_pixels.reshape(-1, 3), axis=0)
                # Begrenzt auf 8 Cluster
                n_clusters = min(len(palette), len(unique_pixels), 8)

                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters,
                                    random_state=42, n_init=10)
                    kmeans.fit(unique_pixels)

                    # Verwende Palette-Farben
                    palette_colors = np.array(palette[:n_clusters])

                    # Quantisiere sichtbare Pixel
                    labels = kmeans.predict(opaque_pixels)
                    quantized_pixels = palette_colors[labels]

                    # Setze quantisierte Pixel zurück
                    rgb_array[opaque_mask] = quantized_pixels

        # Erstelle finales RGBA-Bild
        final_rgb = Image.fromarray(rgb_array, 'RGB')
        r_new, g_new, b_new = final_rgb.split()
        final_image = Image.merge('RGBA', (r_new, g_new, b_new, a))

        return final_image

    def save_gif_with_transparency(self, frames, output_path, original_gif):
        """Speichere GIF mit korrekter Transparenz-Behandlung"""
        if not frames:
            return False

        try:
            # Erstelle Palette mit Transparenz
            # Sammle alle einzigartigen Farben aus allen Frames
            all_colors = set()

            for frame in frames:
                if frame.mode == 'RGBA':
                    # Konvertiere zu RGB für nicht-transparente Pixel
                    r, g, b, a = frame.split()
                    rgb_array = np.array(Image.merge('RGB', (r, g, b)))
                    alpha_array = np.array(a)

                    # Nur sichtbare Pixel berücksichtigen
                    opaque_mask = alpha_array > 128
                    if np.any(opaque_mask):
                        opaque_colors = rgb_array[opaque_mask]
                        for color in opaque_colors:
                            all_colors.add(tuple(color))

            # Erstelle optimierte Palette
            unique_colors = list(all_colors)
            if len(unique_colors) > 255:
                # Reduziere auf 255 Farben (1 für Transparenz reserviert)
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=255, random_state=42, n_init=10)
                kmeans.fit(unique_colors)
                unique_colors = [tuple(map(int, color))
                                 for color in kmeans.cluster_centers_]

            # Konvertiere Frames zu P-Modus mit Transparenz
            converted_frames = []

            for frame in frames:
                if frame.mode == 'RGBA':
                    # Erstelle Palette-Bild
                    palette_img = Image.new('P', (1, 1))

                    # Palette erstellen (255 Farben + 1 für Transparenz)
                    palette_data = []
                    for color in unique_colors[:255]:
                        palette_data.extend(color)

                    # Fülle auf 256 Farben auf
                    while len(palette_data) < 768:  # 256 * 3
                        palette_data.extend([0, 0, 0])

                    palette_img.putpalette(palette_data)

                    # Konvertiere Frame mit Transparenz
                    # Verwende Index 255 für Transparenz
                    r, g, b, a = frame.split()
                    rgb_frame = Image.merge('RGB', (r, g, b))

                    # Quantisiere zu Palette
                    quantized = rgb_frame.quantize(
                        palette=palette_img, dither=Image.Dither.NONE)

                    # Setze transparente Pixel auf Index 255
                    quantized_array = np.array(quantized)
                    alpha_array = np.array(a)

                    # Transparente Bereiche auf Palette-Index 255 setzen
                    transparent_mask = alpha_array <= 128
                    quantized_array[transparent_mask] = 255

                    # Erstelle finales P-Bild
                    final_frame = Image.fromarray(quantized_array, 'P')
                    final_frame.putpalette(palette_data)

                    converted_frames.append(final_frame)
                else:
                    converted_frames.append(frame)

            # Speichere GIF
            if converted_frames:
                converted_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=converted_frames[1:],
                    duration=original_gif.info.get('duration', 100),
                    loop=original_gif.info.get('loop', 0),
                    optimize=True,
                    transparency=255,  # Transparenz-Index
                    disposal=2
                )
                return True

        except Exception as e:
            print(f"   ⚠️  GIF save error: {e}")
            # Fallback: Speichere als RGBA-Frames in separaten PNGs
            try:
                for i, frame in enumerate(frames):
                    frame_path = output_path.parent / \
                        f"{output_path.stem}_frame_{i:03d}.png"
                    frame.save(frame_path, 'PNG')
                print(f"   💾 Saved as PNG sequence instead")
                return True
            except Exception as e2:
                print(f"   ❌ PNG fallback failed: {e2}")
                return False

        return False

    def process_gif_with_transparency(self, gif_path, output_path):
        """GIF-Verarbeitung mit Transparenz-Erhaltung"""
        print(f"   📸 Processing GIF with transparency preservation...")

        gif = Image.open(gif_path)
        processed_frames = []
        frame_count = 0
        transparency_info = []

        # Analysiere erstes Frame
        first_frame = gif.convert('RGBA')
        has_transparency, trans_ratio, alpha_data = self.analyze_transparency(
            first_frame)

        print(f"   🔍 Transparency analysis:")
        print(f"      • Has transparency: {has_transparency}")
        if has_transparency and trans_ratio is not None:
            print(f"      • Transparency ratio: {trans_ratio:.1%}")

        # Wähle Palette
        selected_palette = "commodore64"
        print(f"   🎨 Selected palette: {selected_palette}")

        for frame in ImageSequence.Iterator(gif):
            frame_count += 1

            # Konvertiere zu RGBA
            if frame.mode != 'RGBA':
                frame = frame.convert('RGBA')

            # Analysiere Frame-Transparenz
            has_trans, trans_ratio, _ = self.analyze_transparency(frame)
            transparency_info.append((frame_count, has_trans, trans_ratio))

            # Verarbeitung mit Transparenz-Erhaltung
            # 1. Pixelisierung mit Transparenz
            pixelized = self.pixelize_with_transparency(frame, pixel_factor=4)

            # 2. Palette-Anwendung mit Transparenz
            final_frame = self.apply_palette_with_transparency(
                pixelized, selected_palette)

            processed_frames.append(final_frame)

            if frame_count % 10 == 0:
                print(f"   📊 Processed {frame_count} frames...")

        print(
            f"   ✅ Completed {frame_count} frames with transparency preservation")

        # Speichere mit korrekter Transparenz-Behandlung
        success = self.save_gif_with_transparency(
            processed_frames, output_path, gif)

        if success:
            return True, frame_count, selected_palette, transparency_info
        else:
            return False, 0, None, []

    def run_transparency_test(self):
        """Führe Transparenz-Test aus"""
        print("🔍 FIXED TRANSPARENCY PRESERVATION TEST")
        print("=" * 50)

        input_path = self.input_dir / self.test_file

        if not input_path.exists():
            print(f"❌ Test file not found: {input_path}")
            return False

        print(f"📁 Test file: {self.test_file}")
        file_size = input_path.stat().st_size
        print(f"📊 Input size: {file_size} bytes")

        # Analysiere Original
        original_gif = Image.open(input_path)
        original_frame = original_gif.convert('RGBA')
        has_orig_trans, orig_trans_ratio, _ = self.analyze_transparency(
            original_frame)

        print(f"\n🔍 ORIGINAL ANALYSIS:")
        print(f"   • Has transparency: {has_orig_trans}")
        if has_orig_trans and orig_trans_ratio is not None:
            print(f"   • Transparency ratio: {orig_trans_ratio:.1%}")

        # Verarbeite
        output_filename = f"transparency_fixed_{self.test_file}"
        output_path = self.output_dir / output_filename

        try:
            success, frame_count, palette, transparency_info = self.process_gif_with_transparency(
                input_path, output_path
            )

            if success:
                print(f"\n🎯 PROCESSING RESULTS:")
                print(f"   ✅ Success!")
                print(f"   📊 Frames processed: {frame_count}")
                print(f"   🎨 Palette used: {palette}")

                # Prüfe ob Output-Datei existiert
                if output_path.exists():
                    output_size = output_path.stat().st_size
                    print(f"   📁 Output size: {output_size} bytes")

                    # Analysiere Ergebnis
                    try:
                        result_gif = Image.open(output_path)
                        result_frame = result_gif.convert('RGBA')
                        has_result_trans, result_trans_ratio, _ = self.analyze_transparency(
                            result_frame)

                        print(f"\n🔍 RESULT ANALYSIS:")
                        print(f"   • Has transparency: {has_result_trans}")
                        if has_result_trans and result_trans_ratio is not None:
                            print(
                                f"   • Transparency ratio: {result_trans_ratio:.1%}")

                        # Vergleich
                        print(f"\n📈 TRANSPARENCY COMPARISON:")
                        print(
                            f"   • Original: {orig_trans_ratio:.1%} transparent")
                        print(
                            f"   • Result:   {result_trans_ratio:.1%} transparent")

                        if has_result_trans and result_trans_ratio > 0.01:  # > 1% transparent
                            print(f"   ✅ TRANSPARENCY PRESERVED!")
                            return True
                        else:
                            print(f"   ❌ TRANSPARENCY LOST!")
                            return False
                    except Exception as e:
                        print(f"   ⚠️  Could not analyze result: {e}")
                        # Prüfe PNG-Sequenz
                        png_files = list(self.output_dir.glob(
                            f"{output_path.stem}_frame_*.png"))
                        if png_files:
                            print(
                                f"   📁 Created {len(png_files)} PNG frames with transparency")
                            # Teste erstes PNG
                            first_png = Image.open(png_files[0])
                            has_png_trans, png_trans_ratio, _ = self.analyze_transparency(
                                first_png)
                            if has_png_trans and png_trans_ratio > 0.01:
                                print(
                                    f"   ✅ TRANSPARENCY PRESERVED IN PNG SEQUENCE!")
                                return True
                        return False
                else:
                    print(f"   ❌ Output file not created")
                    return False
            else:
                print(f"   ❌ Processing failed")
                return False

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False


if __name__ == "__main__":
    tester = FixedTransparencyPixelArt()
    success = tester.run_transparency_test()

    if success:
        print(f"\n🎉 TRANSPARENCY TEST PASSED!")
        print(f"✅ Ready for batch processing with transparency preservation")
    else:
        print(f"\n❌ TRANSPARENCY TEST FAILED!")
        print(f"⚠️  Need to fix transparency handling before batch processing")
