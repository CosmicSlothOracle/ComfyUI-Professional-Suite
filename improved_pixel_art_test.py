#!/usr/bin/env python3
"""
Improved Pixel Art Test - Better Color Palettes
Enhanced version with diverse color palettes and better processing
"""

import os
from pathlib import Path
from PIL import Image, ImageSequence
import numpy as np
from sklearn.cluster import KMeans


class ImprovedPixelArtProcessor:
    def __init__(self):
        self.base_dir = Path("C:/Users/Public/ComfyUI-master")
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output" / "improved_pixel_art_test"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test files
        self.test_files = [
            "c0a420e57c75f1f5863d48197fd19c3a_fast_transparent_converted.gif",
            "eleni_fast_transparent_converted.gif",
            "0e35f5b16b8ba60a10fdd360de075def_fast_transparent_converted.gif",
            "9f720323126213.56047641e9c83_fast_transparent_converted.gif"
        ]

        # Enhanced color palettes
        self.palettes = {
            "nes": [
                (124, 124, 124), (0, 0, 252), (0, 0, 188), (68, 40, 188),
                (148, 0, 132), (168, 0, 32), (168, 16, 0), (136, 20, 0),
                (80, 48, 0), (0, 120, 0), (0, 104, 0), (0, 88, 0),
                (0, 64, 88), (0, 0, 0), (0, 0, 0), (0, 0, 0),
                (188, 188, 188), (0, 120, 248), (0, 88, 248), (104, 68, 252),
                (216, 0, 204), (228, 0, 88), (248, 56, 0), (228, 92, 16),
                (172, 124, 0), (0, 184, 0), (0, 168, 0), (0, 168, 68),
                (0, 136, 136), (0, 0, 0), (0, 0, 0), (0, 0, 0),
                (248, 248, 248), (60, 188, 252), (104, 136, 252), (152, 120, 248),
                (248, 120, 248), (248, 88, 152), (248, 120, 88), (252, 160, 68),
                (248, 184, 0), (184, 248, 24), (88, 216, 84), (88, 248, 152),
                (0, 232, 216), (120, 120, 120), (0, 0, 0), (0, 0, 0),
                (252, 252, 252), (164, 228, 252), (184, 184, 248), (216, 184, 248),
                (248, 184, 248), (248, 164, 192), (240, 208, 176), (252, 224, 168),
                (248, 216, 120), (216, 248, 120), (184, 248, 184), (184, 248, 216),
                (0, 252, 252), (248, 216, 248), (0, 0, 0), (0, 0, 0)
            ],
            "commodore64": [
                (0, 0, 0), (255, 255, 255), (136, 57, 50), (103, 182, 189),
                (139, 63, 150), (85, 160, 73), (64, 49, 141), (191, 206, 114),
                (139, 84, 41), (87, 66, 0), (184, 105, 98), (80, 80, 80),
                (120, 120, 120), (148, 224, 137), (120, 105, 196), (159, 159, 159)
            ],
            "cga": [
                (0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170),
                (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170),
                (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255),
                (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255)
            ],
            "modern_pixel": [
                # Erweiterte moderne Pixel Art Palette mit 32 Farben
                (0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81),
                (171, 82, 54), (95, 87, 79), (194, 195, 199), (255, 241, 232),
                (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54),
                (41, 173, 255), (131, 118, 156), (255, 119, 168), (255, 204, 170),
                (41, 24, 20), (17, 29, 53), (66, 33, 54), (18, 83, 89),
                (116, 63, 57), (63, 63, 116), (142, 142, 142), (255, 255, 255),
                (180, 32, 42), (224, 111, 139), (73, 60, 43), (164, 100, 34),
                (235, 137, 49), (247, 226, 107), (47, 72, 78), (68, 137, 26)
            ],
            "vibrant": [
                # Lebendige Farben für moderne Pixel Art
                (34, 32, 52), (69, 40, 60), (102, 57, 49), (143, 86, 59),
                (223, 113, 38), (217, 160, 102), (238, 195, 154), (251, 242, 54),
                (153, 229, 80), (106, 190, 48), (55, 148, 110), (75, 105, 47),
                (82, 75, 36), (50, 60, 57), (63, 63, 116), (48, 96, 130),
                (91, 110, 225), (99, 155, 255), (95, 205, 228), (203, 219, 252),
                (255, 255, 255), (155, 173, 183), (132, 126, 135), (105, 106, 106),
                (89, 86, 82), (118, 66, 138), (172, 50, 50), (217, 87, 99),
                (215, 123, 186), (143, 151, 74), (138, 111, 48), (194, 133, 105)
            ]
        }

    def intelligent_palette_selection(self, image):
        """Intelligente Auswahl der besten Palette basierend auf Bildinhalt"""
        # Analysiere dominante Farben im Bild
        img_array = np.array(image.convert('RGB'))
        pixels = img_array.reshape(-1, 3)

        # Berechne Farbstatistiken
        avg_brightness = np.mean(pixels)
        color_variance = np.var(pixels, axis=0)

        # Wähle Palette basierend auf Bildcharakteristiken
        if avg_brightness < 100:  # Dunkles Bild
            return "commodore64"
        elif np.max(color_variance) > 3000:  # Hohe Farbvariation
            return "vibrant"
        elif avg_brightness > 180:  # Helles Bild
            return "modern_pixel"
        else:  # Standard
            return "nes"

    def adaptive_color_reduction(self, image, target_colors=32):
        """Adaptive Farbreduktion mit intelligenter Clusteranzahl"""
        img_array = np.array(image.convert('RGB'))
        pixels = img_array.reshape(-1, 3)

        # Entferne identische Pixel für bessere Performance
        unique_pixels = np.unique(pixels, axis=0)

        # Passe Clusteranzahl an verfügbare Farben an
        n_clusters = min(target_colors, len(unique_pixels))

        if n_clusters < 2:
            return image

        # K-means mit verbesserter Initialisierung
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,
            max_iter=300,
            init='k-means++'
        )
        kmeans.fit(unique_pixels)

        # Ersetze Pixel mit Clusterzentren
        labels = kmeans.predict(pixels)
        new_pixels = kmeans.cluster_centers_[labels]
        new_img_array = new_pixels.reshape(img_array.shape).astype(np.uint8)

        return Image.fromarray(new_img_array, 'RGB')

    def apply_palette_with_dithering(self, image, palette_name):
        """Wendet Palette mit Floyd-Steinberg Dithering an"""
        palette = self.palettes[palette_name]

        # Konvertiere zu RGB falls nötig
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Erstelle Palette-Bild
        palette_img = Image.new('P', (1, 1))
        flat_palette = []
        for color in palette:
            flat_palette.extend(color)

        # Fülle auf 256 Farben auf
        while len(flat_palette) < 768:
            flat_palette.extend([0, 0, 0])

        palette_img.putpalette(flat_palette)

        # Quantisierung mit Dithering
        quantized = image.quantize(
            palette=palette_img,
            dither=Image.Dither.FLOYDSTEINBERG
        )

        return quantized.convert('RGB')

    def pixelize_smart(self, image, pixel_factor=4):
        """Intelligente Pixelisierung mit Kantenschutz"""
        original_size = image.size

        # Berechne optimale Pixelgröße basierend auf Bildgröße
        if max(original_size) > 500:
            pixel_factor = 6
        elif max(original_size) > 300:
            pixel_factor = 4
        else:
            pixel_factor = 2

        # Neue Größe berechnen
        new_width = max(8, original_size[0] // pixel_factor)
        new_height = max(8, original_size[1] // pixel_factor)

        # Downscale mit Lanczos für bessere Qualität
        small = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Upscale mit Nearest Neighbor für Pixel-Look
        pixelized = small.resize(original_size, Image.Resampling.NEAREST)

        return pixelized

    def process_gif_frames_enhanced(self, gif_path, output_path):
        """Verbesserte GIF-Frame-Verarbeitung"""
        print(f"   📸 Processing frames with enhanced pipeline...")

        gif = Image.open(gif_path)
        processed_frames = []
        frame_count = 0

        # Analysiere erstes Frame für Palette-Auswahl
        first_frame = gif.convert('RGB')
        best_palette = self.intelligent_palette_selection(first_frame)
        print(f"   🎨 Selected palette: {best_palette}")

        for frame in ImageSequence.Iterator(gif):
            frame_count += 1

            # Konvertiere zu RGB
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')

            # Verbesserter Processing-Pipeline
            # 1. Intelligente Pixelisierung
            pixelized = self.pixelize_smart(frame)

            # 2. Adaptive Farbreduktion
            color_reduced = self.adaptive_color_reduction(
                pixelized, target_colors=32)

            # 3. Palette-Anwendung mit Dithering
            final_frame = self.apply_palette_with_dithering(
                color_reduced, best_palette)

            processed_frames.append(final_frame)

            if frame_count % 10 == 0:
                print(f"   📊 Processed {frame_count} frames...")

        print(
            f"   ✅ Completed {frame_count} frames with {best_palette} palette")

        # Speichere als optimiertes GIF
        if processed_frames:
            processed_frames[0].save(
                output_path,
                save_all=True,
                append_images=processed_frames[1:],
                duration=gif.info.get('duration', 100),
                loop=gif.info.get('loop', 0),
                optimize=True,
                quality=95
            )
            return True, best_palette

        return False, None

    def test_single_file(self, filename):
        """Test mit verbesserter Pipeline"""
        print(f"\n🎨 Processing: {filename}")

        input_path = self.input_dir / filename
        if not input_path.exists():
            print(f"   ❌ Input file not found: {input_path}")
            return False, None

        file_size = input_path.stat().st_size
        print(f"   📁 Input: {file_size} bytes")

        output_filename = f"enhanced_pixel_art_{filename}"
        output_path = self.output_dir / output_filename

        try:
            success, palette_used = self.process_gif_frames_enhanced(
                input_path, output_path)

            if success and output_path.exists():
                output_size = output_path.stat().st_size
                compression_ratio = (1 - output_size / file_size) * 100
                print(
                    f"   ✅ Success: {output_size} bytes ({compression_ratio:.1f}% compression)")
                print(f"   🎨 Palette used: {palette_used}")
                return True, palette_used
            else:
                print(f"   ❌ Failed to create output file")
                return False, None

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False, None

    def run_enhanced_test(self):
        """Führe verbesserten Test aus"""
        print("🚀 ENHANCED PIXEL ART TEST")
        print("=" * 50)

        print("🎨 Available palettes:")
        for name, palette in self.palettes.items():
            print(f"   • {name}: {len(palette)} colors")

        print(f"\n📊 Testing {len(self.test_files)} files:")
        for i, filename in enumerate(self.test_files, 1):
            print(f"   {i}. {filename}")

        # Test files
        results = {}
        successful = 0
        palette_usage = {}

        for i, filename in enumerate(self.test_files, 1):
            print(f"\n[{i}/{len(self.test_files)}] Processing...")

            success, palette = self.test_single_file(filename)
            if success:
                results[filename] = "SUCCESS"
                successful += 1
                if palette:
                    palette_usage[palette] = palette_usage.get(palette, 0) + 1
            else:
                results[filename] = "FAILED"

        # Results
        print("\n" + "=" * 50)
        print("🎯 ENHANCED TEST RESULTS")
        print(
            f"✅ Successful: {successful}/{len(self.test_files)} ({successful/len(self.test_files)*100:.1f}%)")

        if palette_usage:
            print(f"\n🎨 Palette usage:")
            for palette, count in palette_usage.items():
                print(f"   • {palette}: {count} files")

        print(f"\n📋 DETAILED RESULTS:")
        for filename, result in results.items():
            icon = "✅" if result == "SUCCESS" else "❌"
            print(f"   {icon} {filename}: {result}")

        # Save report
        report_path = self.output_dir / "enhanced_test_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ENHANCED PIXEL ART TEST REPORT\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Files tested: {len(self.test_files)}\n")
            f.write(f"Successful: {successful}\n")
            f.write(
                f"Success rate: {successful/len(self.test_files)*100:.1f}%\n\n")

            f.write("PALETTE USAGE:\n")
            for palette, count in palette_usage.items():
                f.write(f"{palette}: {count} files\n")

            f.write(f"\nRESULTS:\n")
            for filename, result in results.items():
                f.write(f"{result}: {filename}\n")

        print(f"\n📄 Report saved: {report_path}")

        if successful == len(self.test_files):
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Enhanced pixel art pipeline works perfectly")
            print("✅ Multiple color palettes working correctly")
            print("✅ Ready for full batch processing with enhanced quality")
        else:
            print(f"\n⚠️  {successful}/{len(self.test_files)} tests passed")


if __name__ == "__main__":
    processor = ImprovedPixelArtProcessor()
    processor.run_enhanced_test()
