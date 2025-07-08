#!/usr/bin/env python3
"""
Final Batch Processing - 8-Color Fixed Version
Process all 249 GIF files with the winning transparency-fixed approach
"""

import os
import time
from pathlib import Path
from PIL import Image, ImageSequence
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans
import threading


class FinalBatchProcessor:
    def __init__(self):
        self.base_dir = Path("C:/Users/Public/ComfyUI-master")
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output" / "final_8color_batch"

        # Erstelle Output-Verzeichnis
        self.output_dir.mkdir(exist_ok=True)

        # Threading lock für sichere Ausgabe
        self.print_lock = threading.Lock()

        # Statistiken
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'failed': 0,
            'start_time': time.time(),
            'total_frames': 0,
            'total_colors_reduced': 0
        }

    def safe_print(self, message):
        """Thread-sichere Ausgabe"""
        with self.print_lock:
            print(message)

    def get_gif_files(self):
        """Sammle alle GIF-Dateien"""
        pattern = "*_fast_transparent_converted.gif"
        gif_files = list(self.input_dir.glob(pattern))
        return sorted(gif_files)

    def reduce_colors_kmeans(self, image_array, n_colors=8):
        """Reduziere Farben mit K-Means (nur sichtbare Pixel)"""
        height, width = image_array.shape[:2]

        # Separiere Alpha-Kanal
        if image_array.shape[2] == 4:
            rgb_data = image_array[:, :, :3]
            alpha_data = image_array[:, :, 3]

            # Nur sichtbare Pixel für Clustering
            opaque_mask = alpha_data >= 128
            if not np.any(opaque_mask):
                return image_array  # Komplett transparent

            opaque_pixels = rgb_data[opaque_mask]

            if len(opaque_pixels) < n_colors:
                return image_array  # Zu wenige Pixel

            # K-Means Clustering
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(opaque_pixels)

            # Neue Farben zuweisen
            new_colors = kmeans.predict(opaque_pixels)
            palette = kmeans.cluster_centers_.astype(np.uint8)

            # Rekonstruiere Bild
            new_rgb = rgb_data.copy()
            new_rgb[opaque_mask] = palette[new_colors]

            # Kombiniere mit Alpha
            result = np.dstack([new_rgb, alpha_data])
            return result
        else:
            # Kein Alpha-Kanal
            pixels = image_array.reshape(-1, 3)
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)

            new_colors = kmeans.predict(pixels)
            palette = kmeans.cluster_centers_.astype(np.uint8)

            result = palette[new_colors].reshape(height, width, 3)
            return result

    def process_single_gif(self, input_path):
        """Verarbeite eine einzelne GIF-Datei"""
        try:
            # Öffne GIF
            gif = Image.open(input_path)
            frames = []
            frame_durations = []

            original_colors = set()
            total_frames = 0

            # Verarbeite alle Frames
            for frame in ImageSequence.Iterator(gif):
                total_frames += 1

                # Konvertiere zu RGBA
                if frame.mode != 'RGBA':
                    frame = frame.convert('RGBA')

                # Sammle ursprüngliche Farben (Sample)
                if total_frames <= 5:  # Nur erste 5 Frames sampeln
                    frame_array = np.array(frame)
                    alpha_channel = frame_array[:, :, 3]
                    opaque_mask = alpha_channel >= 128
                    if np.any(opaque_mask):
                        rgb_values = frame_array[opaque_mask][:, :3]
                        for rgb in rgb_values[::10]:  # Jedes 10. Pixel
                            original_colors.add(tuple(rgb))

                # Reduziere Farben auf 8
                frame_array = np.array(frame)
                reduced_array = self.reduce_colors_kmeans(
                    frame_array, n_colors=8)

                # Zurück zu PIL Image
                reduced_frame = Image.fromarray(
                    reduced_array.astype(np.uint8), 'RGBA')
                frames.append(reduced_frame)

                # Frame-Dauer beibehalten
                duration = getattr(frame, 'info', {}).get('duration', 100)
                frame_durations.append(duration)

            # Speichere als GIF
            output_filename = f"8color_{input_path.name}"
            output_path = self.output_dir / output_filename

            # Optimiere GIF-Speicherung
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=frame_durations,
                loop=0,
                transparency=0,
                disposal=2,  # Restore background
                optimize=True
            )

            # Statistiken
            final_colors = 8  # Wir reduzieren immer auf 8
            original_count = len(original_colors)

            return {
                'success': True,
                'input_file': input_path.name,
                'output_file': output_filename,
                'frames': total_frames,
                'original_colors': original_count,
                'final_colors': final_colors,
                'reduction': (1 - final_colors/max(original_count, 1)) * 100,
                'file_size': output_path.stat().st_size if output_path.exists() else 0
            }

        except Exception as e:
            return {
                'success': False,
                'input_file': input_path.name,
                'error': str(e),
                'frames': 0,
                'original_colors': 0,
                'final_colors': 0,
                'reduction': 0,
                'file_size': 0
            }

    def run_batch_processing(self, max_workers=4):
        """Führe Batch-Processing aus"""
        gif_files = self.get_gif_files()
        self.stats['total_files'] = len(gif_files)

        if not gif_files:
            self.safe_print("❌ No GIF files found!")
            return

        self.safe_print("🚀 FINAL 8-COLOR BATCH PROCESSING")
        self.safe_print("=" * 60)
        self.safe_print(f"📊 Found {len(gif_files)} GIF files to process")
        self.safe_print(f"🔧 Using {max_workers} parallel workers")
        self.safe_print(f"📁 Output directory: {self.output_dir}")
        self.safe_print(f"🎨 Target palette: 8 colors (K-Means optimized)")
        self.safe_print("")

        results = []

        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Starte alle Tasks
            future_to_file = {
                executor.submit(self.process_single_gif, gif_file): gif_file
                for gif_file in gif_files
            }

            # Verarbeite Ergebnisse
            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)

                if result['success']:
                    self.stats['processed'] += 1
                    self.stats['total_frames'] += result['frames']
                    self.stats['total_colors_reduced'] += result['original_colors'] - \
                        result['final_colors']

                    # Progress-Ausgabe
                    progress = (
                        self.stats['processed'] + self.stats['failed']) / self.stats['total_files'] * 100
                    elapsed = time.time() - self.stats['start_time']
                    rate = (
                        self.stats['processed'] + self.stats['failed']) / elapsed if elapsed > 0 else 0
                    eta = (self.stats['total_files'] - self.stats['processed'] -
                           self.stats['failed']) / rate if rate > 0 else 0

                    self.safe_print(
                        f"[{self.stats['processed'] + self.stats['failed']}/{self.stats['total_files']}] "
                        f"✅ {result['input_file'][:50]}... -> "
                        f"{result['frames']} frames, "
                        f"{result['original_colors']}→{result['final_colors']} colors "
                        f"({result['reduction']:.1f}% reduction)"
                    )

                    # Progress-Update alle 25 Dateien
                    if (self.stats['processed'] + self.stats['failed']) % 25 == 0:
                        self.safe_print(
                            f"📈 Progress: {self.stats['processed'] + self.stats['failed']}/{self.stats['total_files']} ({progress:.1f}%)")
                        self.safe_print(
                            f"   ⏱️  Rate: {rate:.1f} files/sec, ETA: {eta/60:.1f} min")
                        self.safe_print(
                            f"   ✅ Success: {self.stats['processed']}, ❌ Failed: {self.stats['failed']}")
                        self.safe_print("")

                else:
                    self.stats['failed'] += 1
                    self.safe_print(
                        f"[{self.stats['processed'] + self.stats['failed']}/{self.stats['total_files']}] ❌ {result['input_file']}: {result['error']}")

        # Finale Statistiken
        self.print_final_report(results)

        return results

    def print_final_report(self, results):
        """Drucke finalen Report"""
        elapsed = time.time() - self.stats['start_time']
        successful_results = [r for r in results if r['success']]

        self.safe_print("\n" + "=" * 60)
        self.safe_print("🎯 FINAL PROCESSING REPORT")
        self.safe_print("=" * 60)

        self.safe_print(f"⏱️  TIMING:")
        self.safe_print(f"   • Total time: {elapsed/60:.1f} minutes")
        self.safe_print(
            f"   • Average rate: {len(results)/elapsed:.2f} files/sec")

        self.safe_print(f"\n📊 PROCESSING RESULTS:")
        self.safe_print(f"   • Total files: {self.stats['total_files']}")
        self.safe_print(f"   • ✅ Successful: {self.stats['processed']}")
        self.safe_print(f"   • ❌ Failed: {self.stats['failed']}")
        self.safe_print(
            f"   • Success rate: {self.stats['processed']/self.stats['total_files']*100:.1f}%")

        if successful_results:
            total_original_size = sum(Path(self.input_dir / r['input_file']).stat(
            ).st_size for r in successful_results if Path(self.input_dir / r['input_file']).exists())
            total_final_size = sum(r['file_size'] for r in successful_results)

            avg_frames = sum(r['frames']
                             for r in successful_results) / len(successful_results)
            avg_original_colors = sum(
                r['original_colors'] for r in successful_results) / len(successful_results)
            avg_reduction = sum(r['reduction']
                                for r in successful_results) / len(successful_results)

            self.safe_print(f"\n🎨 COLOR PROCESSING:")
            self.safe_print(
                f"   • Total frames processed: {self.stats['total_frames']:,}")
            self.safe_print(f"   • Average frames per GIF: {avg_frames:.1f}")
            self.safe_print(
                f"   • Average original colors: {avg_original_colors:.1f}")
            self.safe_print(f"   • Target colors: 8 (fixed)")
            self.safe_print(
                f"   • Average color reduction: {avg_reduction:.1f}%")

            self.safe_print(f"\n💾 FILE SIZES:")
            self.safe_print(
                f"   • Original total: {total_original_size/1024/1024:.1f} MB")
            self.safe_print(
                f"   • Final total: {total_final_size/1024/1024:.1f} MB")
            if total_original_size > 0:
                compression = (1 - total_final_size/total_original_size) * 100
                self.safe_print(f"   • Compression: {compression:.1f}%")

        self.safe_print(f"\n📁 OUTPUT LOCATION:")
        self.safe_print(f"   {self.output_dir}")

        if self.stats['processed'] > 0:
            self.safe_print(
                f"\n🚀 STATUS: BATCH PROCESSING COMPLETED SUCCESSFULLY!")
            self.safe_print(
                f"✅ {self.stats['processed']} pixel art videos ready!")
        else:
            self.safe_print(f"\n❌ STATUS: BATCH PROCESSING FAILED!")


if __name__ == "__main__":
    processor = FinalBatchProcessor()
    processor.run_batch_processing(max_workers=4)
