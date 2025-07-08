#!/usr/bin/env python3
"""
Analyze the specific GIF color palette and extract workflow configuration
"""

import os
from pathlib import Path
from PIL import Image, ImageSequence
import numpy as np
from collections import Counter
import json


class SpecificGifPaletteAnalyzer:
    def __init__(self):
        self.base_dir = Path("C:/Users/Public/ComfyUI-master")

        # Target GIF file
        self.target_file = self.base_dir / "output" / "pixel_art_gifs_ultimate" / \
            "pixel_modern_0ea53a3cdbfdcf14caf1c8cccdb60143.gif"

    def extract_dominant_colors(self, gif_path, num_colors=64):
        """Extract the most dominant colors from the GIF"""
        print(f"🎨 Analyzing color palette: {gif_path.name}")

        if not gif_path.exists():
            print(f"❌ File not found: {gif_path}")
            return None

        gif = Image.open(gif_path)
        frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]

        # Sample every 3rd frame for performance
        sampled_frames = frames[::3]
        print(
            f"📸 Analyzing {len(sampled_frames)} frames from {len(frames)} total")

        all_colors = []

        for frame in sampled_frames:
            # Convert to RGB
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')

            # Get all pixel colors
            frame_array = np.array(frame)
            pixels = frame_array.reshape(-1, 3)

            # Add to color collection
            for pixel in pixels:
                all_colors.append(tuple(pixel))

        # Count color frequencies
        color_counts = Counter(all_colors)

        # Get most common colors
        dominant_colors = color_counts.most_common(num_colors)

        return dominant_colors, color_counts

    def create_palette_config(self, dominant_colors):
        """Create a palette configuration from dominant colors"""
        palette = [color for color, count in dominant_colors]

        # Analyze the palette characteristics
        palette_array = np.array(palette)

        # Calculate statistics
        brightness_avg = np.mean(np.mean(palette_array, axis=1))
        saturation_range = np.max(
            palette_array, axis=1) - np.min(palette_array, axis=1)
        saturation_avg = np.mean(saturation_range)

        # RGB channel statistics
        r_range = (int(np.min(palette_array[:, 0])), int(
            np.max(palette_array[:, 0])))
        g_range = (int(np.min(palette_array[:, 1])), int(
            np.max(palette_array[:, 1])))
        b_range = (int(np.min(palette_array[:, 2])), int(
            np.max(palette_array[:, 2])))

        config = {
            "name": "modern_extracted",
            "palette": palette[:32],  # Limit to 32 most dominant colors
            "statistics": {
                "total_unique_colors": len(palette),
                "average_brightness": float(brightness_avg),
                "average_saturation": float(saturation_avg),
                "rgb_ranges": {
                    "red": r_range,
                    "green": g_range,
                    "blue": b_range
                }
            },
            "workflow_config": {
                "resolution": (512, 512),
                "colors": 32,
                "pixelize": 4,
                "style": "modern_extracted"
            }
        }

        return config

    def analyze_and_save(self):
        """Analyze the GIF and save the palette configuration"""
        print("🔍 ANALYZING SPECIFIC GIF PALETTE")
        print("=" * 50)

        # Extract colors
        dominant_colors, color_counts = self.extract_dominant_colors(
            self.target_file, 64)

        if not dominant_colors:
            print("❌ Could not extract colors")
            return None

        print(f"🎨 Found {len(dominant_colors)} dominant colors")
        print(f"📊 Total color instances: {sum(color_counts.values())}")

        # Show top 10 colors
        print("\n🌈 TOP 10 DOMINANT COLORS:")
        for i, (color, count) in enumerate(dominant_colors[:10]):
            percentage = (count / sum(color_counts.values())) * 100
            print(
                f"   {i+1:2d}. RGB{color} - {percentage:.2f}% ({count:,} pixels)")

        # Create configuration
        config = self.create_palette_config(dominant_colors)

        # Save configuration
        config_path = self.base_dir / "configs" / "extracted_modern_palette.json"
        config_path.parent.mkdir(exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n💾 Configuration saved to: {config_path}")

        # Display configuration summary
        stats = config["statistics"]
        print(f"\n📈 PALETTE ANALYSIS SUMMARY:")
        print(f"   • Unique colors: {stats['total_unique_colors']}")
        print(f"   • Average brightness: {stats['average_brightness']:.1f}")
        print(f"   • Average saturation: {stats['average_saturation']:.1f}")
        print(f"   • Red range: {stats['rgb_ranges']['red']}")
        print(f"   • Green range: {stats['rgb_ranges']['green']}")
        print(f"   • Blue range: {stats['rgb_ranges']['blue']}")

        return config


def main():
    analyzer = SpecificGifPaletteAnalyzer()
    config = analyzer.analyze_and_save()

    if config:
        print("\n✅ Palette analysis complete!")
        print("🔧 Ready to create single GIF workflow with this palette")
    else:
        print("\n❌ Analysis failed")


if __name__ == "__main__":
    main()
