#!/usr/bin/env python3
"""
Final Pixel Art Batch Processor
Enhanced version with intelligent palette selection for all 249 GIF files
"""

import os
import time
from pathlib import Path
from PIL import Image, ImageSequence
import numpy as np
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class FinalPixelArtBatchProcessor:
    def __init__(self):
        self.base_dir = Path("C:/Users/Public/ComfyUI-master")
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output" / "final_pixel_art_batch"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.processed_count = 0
        self.failed_count = 0

        # Enhanced color palettes
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
            ],
            "modern_pixel": [
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
        """Intelligente Palette-Auswahl basierend auf Bildanalyse"""
        img_array = np.array(image.convert('RGB'))
        pixels = img_array.reshape(-1, 3)

        # Farbstatistiken
        avg_brightness = np.mean(pixels)
        color_variance = np.var(pixels, axis=0)
        saturation = np.std(pixels, axis=0)

        # Intelligente Auswahl
        if avg_brightness < 80:  # Sehr dunkle Bilder
            return "commodore64"
        elif np.max(saturation) > 60:  # Hohe Sättigung
            return "vibrant"
        elif avg_brightness > 170:  # Helle Bilder
            return "modern_pixel"
        else:  # Standard für die meisten Fälle
            return "nes"

    def adaptive_color_reduction(self, image, target_colors=24):
        """Optimierte Farbreduktion"""
        img_array = np.array(image.convert('RGB'))
        pixels = img_array.reshape(-1, 3)

        unique_pixels = np.unique(pixels, axis=0)
        n_clusters = min(target_colors, len(unique_pixels))

        if n_clusters < 2:
            return image

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=15,
            max_iter=200,
            init='k-means++'
        )
        kmeans.fit(unique_pixels)

        labels = kmeans.predict(pixels)
        new_pixels = kmeans.cluster_centers_[labels]
        new_img_array = new_pixels.reshape(img_array.shape).astype(np.uint8)

        return Image.fromarray(new_img_array, 'RGB')

    def apply_palette_with_dithering(self, image, palette_name):
        """Palette-Anwendung mit Floyd-Steinberg Dithering"""
        palette = self.palettes[palette_name]

        if image.mode != 'RGB':
            image = image.convert('RGB')

        palette_img = Image.new('P', (1, 1))
        flat_palette = []
        for color in palette:
            flat_palette.extend(color)

        while len(flat_palette) < 768:
            flat_palette.extend([0, 0, 0])

        palette_img.putpalette(flat_palette)

        quantized = image.quantize(
            palette=palette_img,
            dither=Image.Dither.FLOYDSTEINBERG
        )

        return quantized.convert('RGB')

    def pixelize_adaptive(self, image):
        """Adaptive Pixelisierung basierend auf Bildgröße"""
        original_size = image.size

        # Dynamische Pixel-Faktor-Berechnung
        max_dimension = max(original_size)
        if max_dimension > 800:
            pixel_factor = 8
        elif max_dimension > 500:
            pixel_factor = 6
        elif max_dimension > 300:
            pixel_factor = 4
        else:
            pixel_factor = 3

        new_width = max(16, original_size[0] // pixel_factor)
        new_height = max(16, original_size[1] // pixel_factor)

        small = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        pixelized = small.resize(original_size, Image.Resampling.NEAREST)

        return pixelized

    def process_gif_optimized(self, gif_path, output_path):
        """Optimierte GIF-Verarbeitung"""
        try:
            gif = Image.open(gif_path)
            processed_frames = []

            # Analysiere erstes Frame
            first_frame = gif.convert('RGB')
            selected_palette = self.intelligent_palette_selection(first_frame)

            frame_count = 0
            for frame in ImageSequence.Iterator(gif):
                frame_count += 1

                if frame.mode != 'RGB':
                    frame = frame.convert('RGB')

                # Processing-Pipeline
                pixelized = self.pixelize_adaptive(frame)
                color_reduced = self.adaptive_color_reduction(
                    pixelized, target_colors=24)
                final_frame = self.apply_palette_with_dithering(
                    color_reduced, selected_palette)

                processed_frames.append(final_frame)

            # Speichere optimiertes GIF
            if processed_frames:
                processed_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=processed_frames[1:],
                    duration=gif.info.get('duration', 100),
                    loop=gif.info.get('loop', 0),
                    optimize=True
                )
                return True, frame_count, selected_palette

            return False, 0, None

        except Exception as e:
            return False, 0, None

    def process_single_gif(self, gif_filename):
        """Verarbeite einzelne GIF-Datei"""
        input_path = self.input_dir / gif_filename

        if not input_path.exists():
            with self.lock:
                self.failed_count += 1
            return False, f"File not found: {gif_filename}"

        output_filename = f"pixel_art_{gif_filename}"
        output_path = self.output_dir / output_filename

        try:
            file_size = input_path.stat().st_size
            success, frame_count, palette = self.process_gif_optimized(
                input_path, output_path)

            if success and output_path.exists():
                output_size = output_path.stat().st_size

                with self.lock:
                    self.processed_count += 1

                return True, {
                    'filename': gif_filename,
                    'frames': frame_count,
                    'palette': palette,
                    'input_size': file_size,
                    'output_size': output_size,
                    'compression': (1 - output_size / file_size) * 100
                }
            else:
                with self.lock:
                    self.failed_count += 1
                return False, f"Processing failed: {gif_filename}"

        except Exception as e:
            with self.lock:
                self.failed_count += 1
            return False, f"Error processing {gif_filename}: {str(e)}"

    def run_batch_processing(self, max_workers=4):
        """Führe Batch-Verarbeitung aus"""
        print("🚀 FINAL PIXEL ART BATCH PROCESSING")
        print("=" * 60)

        # Finde alle GIF-Dateien
        gif_files = [f for f in os.listdir(self.input_dir)
                     if f.endswith('_fast_transparent_converted.gif')]

        total_files = len(gif_files)
        print(f"📊 Found {total_files} GIF files to process")
        print(f"🔧 Using {max_workers} parallel workers")
        print(f"📁 Output directory: {self.output_dir}")

        print(f"\n🎨 Available palettes:")
        for name, palette in self.palettes.items():
            print(f"   • {name}: {len(palette)} colors")

        # Verarbeitung mit ThreadPool
        start_time = time.time()
        results = []
        palette_stats = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit alle Jobs
            future_to_file = {
                executor.submit(self.process_single_gif, gif_file): gif_file
                for gif_file in gif_files
            }

            # Verarbeite Ergebnisse
            for i, future in enumerate(as_completed(future_to_file), 1):
                gif_file = future_to_file[future]

                try:
                    success, result = future.result()

                    if success:
                        results.append(result)
                        palette = result['palette']
                        if palette:
                            palette_stats[palette] = palette_stats.get(
                                palette, 0) + 1

                        print(
                            f"[{i}/{total_files}] ✅ {gif_file} -> {result['frames']} frames, {result['palette']} palette")
                    else:
                        print(f"[{i}/{total_files}] ❌ {result}")

                except Exception as e:
                    print(f"[{i}/{total_files}] ❌ {gif_file}: {str(e)}")

                # Progress update
                if i % 25 == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (total_files - i) / rate if rate > 0 else 0

                    print(
                        f"\n📈 Progress: {i}/{total_files} ({i/total_files*100:.1f}%)")
                    print(
                        f"   ⏱️  Rate: {rate:.1f} files/sec, ETA: {eta/60:.1f} min")
                    print(
                        f"   ✅ Success: {self.processed_count}, ❌ Failed: {self.failed_count}")

        # Final results
        elapsed_time = time.time() - start_time
        success_rate = (self.processed_count / total_files) * \
            100 if total_files > 0 else 0

        print(f"\n" + "=" * 60)
        print("🎯 FINAL BATCH PROCESSING RESULTS")
        print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
        print(
            f"📊 Processed: {self.processed_count}/{total_files} ({success_rate:.1f}%)")
        print(f"❌ Failed: {self.failed_count}")
        print(f"📈 Average rate: {total_files/elapsed_time:.1f} files/sec")

        if palette_stats:
            print(f"\n🎨 Palette usage statistics:")
            for palette, count in sorted(palette_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / self.processed_count) * 100
                print(f"   • {palette}: {count} files ({percentage:.1f}%)")

        # Speichere detaillierten Report
        self.save_detailed_report(results, elapsed_time, palette_stats)

        if self.processed_count == total_files:
            print(f"\n🎉 ALL FILES PROCESSED SUCCESSFULLY!")
            print(f"✅ {total_files} pixel art GIFs created")
            print(f"📁 Results saved in: {self.output_dir}")
        else:
            print(
                f"\n⚠️  {self.processed_count}/{total_files} files processed successfully")

    def save_detailed_report(self, results, elapsed_time, palette_stats):
        """Speichere detaillierten Report"""
        report_path = self.output_dir / "final_batch_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("FINAL PIXEL ART BATCH PROCESSING REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing time: {elapsed_time/60:.1f} minutes\n")
            f.write(f"Total files: {len(results) + self.failed_count}\n")
            f.write(f"Successful: {len(results)}\n")
            f.write(f"Failed: {self.failed_count}\n")
            f.write(
                f"Success rate: {len(results)/(len(results) + self.failed_count)*100:.1f}%\n\n")

            f.write("PALETTE USAGE:\n")
            for palette, count in palette_stats.items():
                f.write(f"{palette}: {count} files\n")

            f.write(f"\nDETAILED RESULTS:\n")
            f.write("-" * 30 + "\n")
            for result in results:
                f.write(
                    f"SUCCESS: {result['filename']} -> {result['frames']} frames, {result['palette']}\n")

        print(f"📄 Detailed report saved: {report_path}")


if __name__ == "__main__":
    processor = FinalPixelArtBatchProcessor()
    processor.run_batch_processing(max_workers=3)
