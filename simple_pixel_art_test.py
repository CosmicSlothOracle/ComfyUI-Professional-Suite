#!/usr/bin/env python3
"""
Simple Pixel Art Test - Direct Processing
Tests the 4 GIF files using available Python libraries
"""

import os
from pathlib import Path
from PIL import Image, ImageSequence
import numpy as np
from sklearn.cluster import KMeans
import cv2


class SimplePixelArtProcessor:
    def __init__(self):
        self.base_dir = Path("C:/Users/Public/ComfyUI-master")
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output" / "simple_pixel_art_test"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test files
        self.test_files = [
            "c0a420e57c75f1f5863d48197fd19c3a_fast_transparent_converted.gif",
            "eleni_fast_transparent_converted.gif",
            "0e35f5b16b8ba60a10fdd360de075def_fast_transparent_converted.gif",
            "9f720323126213.56047641e9c83_fast_transparent_converted.gif"
        ]

        # Gameboy palette (classic pixel art)
        self.gameboy_palette = [
            (15, 56, 15),      # Dark green
            (48, 98, 48),      # Medium green
            (139, 172, 15),    # Light green
            (155, 188, 15)     # Lightest green
        ]

    def quantize_to_palette(self, image, palette, dither=True):
        """Quantize image to specific palette"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Create palette image
        palette_img = Image.new('P', (1, 1))
        flat_palette = []
        for color in palette:
            flat_palette.extend(color)

        # Pad palette to 256 colors
        while len(flat_palette) < 768:
            flat_palette.extend([0, 0, 0])

        palette_img.putpalette(flat_palette)

        # Quantize image
        if dither:
            quantized = image.quantize(
                palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
        else:
            quantized = image.quantize(
                palette=palette_img, dither=Image.Dither.NONE)

        return quantized.convert('RGB')

    def pixelize_image(self, image, pixel_size=8):
        """Create pixel art effect by downscaling and upscaling"""
        # Get original size
        original_size = image.size

        # Calculate new size for pixelization
        new_width = max(1, original_size[0] // pixel_size)
        new_height = max(1, original_size[1] // pixel_size)

        # Downscale with nearest neighbor
        small = image.resize((new_width, new_height), Image.Resampling.NEAREST)

        # Upscale back to original size
        pixelized = small.resize(original_size, Image.Resampling.NEAREST)

        return pixelized

    def reduce_colors_kmeans(self, image, n_colors=16):
        """Reduce colors using K-means clustering"""
        # Convert to numpy array
        img_array = np.array(image)

        # Handle transparency
        if img_array.shape[2] == 4:  # RGBA
            alpha = img_array[:, :, 3]
            img_array = img_array[:, :, :3]
            has_alpha = True
        else:
            has_alpha = False

        # Reshape for clustering
        pixels = img_array.reshape(-1, 3)

        # Apply K-means
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        # Replace pixels with cluster centers
        new_pixels = kmeans.cluster_centers_[kmeans.labels_]
        new_img_array = new_pixels.reshape(img_array.shape).astype(np.uint8)

        # Convert back to PIL
        result = Image.fromarray(new_img_array, 'RGB')

        return result

    def process_gif_frames(self, gif_path, output_path):
        """Process all frames of a GIF"""
        print(f"   📸 Processing frames...")

        # Open GIF
        gif = Image.open(gif_path)

        # Process each frame
        processed_frames = []
        frame_count = 0

        for frame in ImageSequence.Iterator(gif):
            frame_count += 1

            # Convert frame to RGB
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')

            # Apply pixel art processing pipeline
            # 1. Pixelize
            pixelized = self.pixelize_image(frame, pixel_size=4)

            # 2. Reduce colors
            color_reduced = self.reduce_colors_kmeans(pixelized, n_colors=16)

            # 3. Apply gameboy palette
            final_frame = self.quantize_to_palette(
                color_reduced, self.gameboy_palette)

            processed_frames.append(final_frame)

        print(f"   📊 Processed {frame_count} frames")

        # Save as GIF
        if processed_frames:
            processed_frames[0].save(
                output_path,
                save_all=True,
                append_images=processed_frames[1:],
                duration=gif.info.get('duration', 100),
                loop=gif.info.get('loop', 0),
                optimize=True
            )
            return True

        return False

    def test_single_file(self, filename):
        """Test processing a single GIF file"""
        print(f"\n🎨 Processing: {filename}")

        # Check input file
        input_path = self.input_dir / filename
        if not input_path.exists():
            print(f"   ❌ Input file not found: {input_path}")
            return False

        file_size = input_path.stat().st_size
        print(f"   📁 Input: {file_size} bytes")

        # Set output path
        output_filename = f"pixel_art_{filename}"
        output_path = self.output_dir / output_filename

        try:
            # Process the GIF
            success = self.process_gif_frames(input_path, output_path)

            if success and output_path.exists():
                output_size = output_path.stat().st_size
                print(
                    f"   ✅ Success: {output_size} bytes -> {output_path.name}")
                return True
            else:
                print(f"   ❌ Failed to create output file")
                return False

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

    def run_test(self):
        """Run the complete test"""
        print("🚀 SIMPLE PIXEL ART TEST")
        print("=" * 40)

        print(f"📊 Testing {len(self.test_files)} files:")
        for i, filename in enumerate(self.test_files, 1):
            print(f"   {i}. {filename}")

        # Test each file
        results = {}
        successful = 0

        for i, filename in enumerate(self.test_files, 1):
            print(f"\n[{i}/{len(self.test_files)}] Processing...")

            if self.test_single_file(filename):
                results[filename] = "SUCCESS"
                successful += 1
            else:
                results[filename] = "FAILED"

        # Results
        print("\n" + "=" * 40)
        print("🎯 TEST RESULTS")
        print(
            f"✅ Successful: {successful}/{len(self.test_files)} ({successful/len(self.test_files)*100:.1f}%)")

        print("\n📋 DETAILED RESULTS:")
        for filename, result in results.items():
            icon = "✅" if result == "SUCCESS" else "❌"
            print(f"   {icon} {filename}: {result}")

        # Save report
        report_path = self.output_dir / "test_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("SIMPLE PIXEL ART TEST REPORT\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Files tested: {len(self.test_files)}\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {len(self.test_files) - successful}\n")
            f.write(
                f"Success rate: {successful/len(self.test_files)*100:.1f}%\n\n")

            f.write("RESULTS:\n")
            for filename, result in results.items():
                f.write(f"{result}: {filename}\n")

        print(f"\n📄 Report saved: {report_path}")

        if successful == len(self.test_files):
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Pixel art processing pipeline works correctly")
            print("✅ Ready to proceed with full batch processing")
        else:
            print(f"\n⚠️  {successful}/{len(self.test_files)} tests passed")


if __name__ == "__main__":
    processor = SimplePixelArtProcessor()
    processor.run_test()
