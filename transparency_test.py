#!/usr/bin/env python3
"""
Transparency Test - Pixel Art with Preserved Transparency
Test with: 0ea53a3cdbfdcf14caf1c8cccdb60143_fast_transparent_converted.gif
"""

import os
from pathlib import Path
from PIL import Image, ImageSequence
import numpy as np
from sklearn.cluster import KMeans


class TransparencyPreservingPixelArt:
    def __init__(self):
        self.base_dir = Path("C:/Users/Public/ComfyUI-master")
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output" / "transparency_test"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test file
        self.test_file = "0ea53a3cdbfdcf14caf1c8cccdb60143_fast_transparent_converted.gif"

        # Enhanced color palettes (same as before)
        self.palettes = {
            "nes": [
                (124, 124, 124), (0, 0, 252), (0, 0, 188), (68, 40, 188),
                (148, 0, 132), (168, 0, 32), (168, 16, 0), (136, 20, 0),
                (80, 48, 0), (0, 120, 0), (0, 104, 0), (0, 88, 0),
                (0, 64, 88), (0, 0, 0), (188, 188, 188), (0, 120, 248),
                (0, 88, 248), (104, 68, 252), (216, 0, 204), (228, 0, 88),
                (248, 56, 0), (228, 92, 16), (172, 124, 0), (0, 184, 0),
                (0, 168, 0), (0, 168, 68), (0, 136, 136), (248, 248, 248),
                (60, 188, 252), (104, 136, 252), (152, 120, 248), (248, 120, 248),
                (248, 88, 152), (248, 120, 88), (252, 160, 68), (248, 184, 0),
                (184, 248, 24), (88, 216, 84), (88, 248, 152), (0, 232, 216),
                (120, 120, 120), (252, 252, 252), (164, 228, 252), (184, 184, 248),
                (216, 184, 248), (248, 184, 248), (248, 164, 192), (240, 208, 176),
                (252, 224, 168), (248, 216, 120), (216, 248, 120), (184, 248, 184),
                (184, 248, 216), (0, 252, 252), (248, 216, 248), (159, 159, 159)
            ],
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
            # Palette-Modus - prüfe auf Transparenz-Index
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
            transparent_pixels = np.sum(alpha_channel == 0)
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

        # Separate RGB und Alpha-Kanäle
        if image.mode == 'RGBA':
            # RGB-Kanäle pixelisieren
            rgb_image = Image.new('RGB', image.size, (0, 0, 0))
            # Verwende Alpha als Maske
            rgb_image.paste(image, mask=image.split()[3])

            # Pixelisiere RGB
            rgb_small = rgb_image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS)
            rgb_pixelized = rgb_small.resize(
                original_size, Image.Resampling.NEAREST)

            # Alpha-Kanal separat pixelisieren
            alpha_channel = image.split()[3]
            alpha_small = alpha_channel.resize(
                (new_width, new_height), Image.Resampling.LANCZOS)
            alpha_pixelized = alpha_small.resize(
                original_size, Image.Resampling.NEAREST)

            # Kombiniere RGB und Alpha
            pixelized = Image.merge(
                'RGBA', rgb_pixelized.split() + (alpha_pixelized,))

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
        rgb_channels = image.split()[:3]
        alpha_channel = image.split()[3]

        # Erstelle RGB-Bild für Palette-Anwendung
        rgb_image = Image.merge('RGB', rgb_channels)

        # Wende Palette nur auf nicht-transparente Bereiche an
        alpha_array = np.array(alpha_channel)
        rgb_array = np.array(rgb_image)

        # Maske für nicht-transparente Pixel
        opaque_mask = alpha_array > 128  # Schwellwert für Transparenz

        if np.any(opaque_mask):
            # Extrahiere nur nicht-transparente Pixel
            opaque_pixels = rgb_array[opaque_mask]

            if len(opaque_pixels) > 0:
                # K-means nur für sichtbare Pixel
                unique_pixels = np.unique(opaque_pixels.reshape(-1, 3), axis=0)
                n_clusters = min(len(palette), len(unique_pixels))

                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters,
                                    random_state=42, n_init=10)
                    kmeans.fit(unique_pixels)

                    # Ersetze Clusterzentren durch Palette-Farben
                    palette_colors = np.array(palette[:n_clusters])

                    # Quantisiere nur sichtbare Pixel
                    labels = kmeans.predict(opaque_pixels)
                    quantized_pixels = palette_colors[labels]

                    # Setze quantisierte Pixel zurück ins Bild
                    rgb_array[opaque_mask] = quantized_pixels

        # Erstelle finales RGBA-Bild
        final_rgb = Image.fromarray(rgb_array, 'RGB')
        final_image = Image.merge('RGBA', final_rgb.split() + (alpha_channel,))

        return final_image

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
        selected_palette = "commodore64"  # Für Test verwenden wir Commodore64
        print(f"   🎨 Selected palette: {selected_palette}")

        for frame in ImageSequence.Iterator(gif):
            frame_count += 1

            # Konvertiere zu RGBA für einheitliche Behandlung
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

        # Speichere als GIF mit Transparenz
        if processed_frames:
            # Konvertiere zurück zu P-Modus mit Transparenz für optimale GIF-Speicherung
            final_frames = []
            for frame in processed_frames:
                # Konvertiere RGBA zu P mit Transparenz
                frame_p = frame.quantize(method=Image.Quantize.MEDIANCUT)
                frame_p.info['transparency'] = 0  # Setze Transparenz-Index
                final_frames.append(frame_p)

            final_frames[0].save(
                output_path,
                save_all=True,
                append_images=final_frames[1:],
                duration=gif.info.get('duration', 100),
                loop=gif.info.get('loop', 0),
                optimize=True,
                transparency=0,
                disposal=2  # Restore background
            )

            return True, frame_count, selected_palette, transparency_info

        return False, 0, None, []

    def run_transparency_test(self):
        """Führe Transparenz-Test aus"""
        print("🔍 TRANSPARENCY PRESERVATION TEST")
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
        output_filename = f"transparency_preserved_{self.test_file}"
        output_path = self.output_dir / output_filename

        try:
            success, frame_count, palette, transparency_info = self.process_gif_with_transparency(
                input_path, output_path
            )

            if success and output_path.exists():
                output_size = output_path.stat().st_size

                # Analysiere Ergebnis
                result_gif = Image.open(output_path)
                result_frame = result_gif.convert('RGBA')
                has_result_trans, result_trans_ratio, _ = self.analyze_transparency(
                    result_frame)

                print(f"\n🎯 PROCESSING RESULTS:")
                print(f"   ✅ Success: {output_size} bytes")
                print(f"   📊 Frames processed: {frame_count}")
                print(f"   🎨 Palette used: {palette}")

                print(f"\n🔍 RESULT ANALYSIS:")
                print(f"   • Has transparency: {has_result_trans}")
                if has_result_trans and result_trans_ratio is not None:
                    print(f"   • Transparency ratio: {result_trans_ratio:.1%}")

                # Vergleich
                print(f"\n📈 TRANSPARENCY COMPARISON:")
                print(f"   • Original: {orig_trans_ratio:.1%} transparent")
                print(f"   • Result:   {result_trans_ratio:.1%} transparent")

                if has_result_trans and result_trans_ratio > 0.01:  # > 1% transparent
                    print(f"   ✅ TRANSPARENCY PRESERVED!")
                    return True
                else:
                    print(f"   ❌ TRANSPARENCY LOST!")
                    return False
            else:
                print(f"   ❌ Processing failed")
                return False

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False


if __name__ == "__main__":
    tester = TransparencyPreservingPixelArt()
    success = tester.run_transparency_test()

    if success:
        print(f"\n🎉 TRANSPARENCY TEST PASSED!")
        print(f"✅ Ready for batch processing with transparency preservation")
    else:
        print(f"\n❌ TRANSPARENCY TEST FAILED!")
        print(f"⚠️  Need to fix transparency handling before batch processing")
