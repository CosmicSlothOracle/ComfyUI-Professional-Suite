#!/usr/bin/env python3
"""
Palette Analysis - Analyze current color palette of the winning version
"""

import os
from pathlib import Path
from PIL import Image, ImageSequence
import numpy as np
from collections import Counter


class PaletteAnalyzer:
    def __init__(self):
        self.base_dir = Path("C:/Users/Public/ComfyUI-master")

        # Fixed version (winner)
        self.fixed_file = self.base_dir / "output" / "transparency_test_fixed" / \
            "transparency_fixed_0ea53a3cdbfdcf14caf1c8cccdb60143_fast_transparent_converted.gif"

        # Original for comparison
        self.original_file = self.base_dir / "input" / \
            "0ea53a3cdbfdcf14caf1c8cccdb60143_fast_transparent_converted.gif"

    def extract_colors_from_gif(self, gif_path, sample_frames=5):
        """Extrahiere alle Farben aus einem GIF"""
        print(f"📊 Analyzing: {gif_path.name}")

        if not gif_path.exists():
            print(f"❌ File not found: {gif_path}")
            return None, None, None

        gif = Image.open(gif_path)
        frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
        total_frames = len(frames)

        print(f"   📸 Total frames: {total_frames}")

        # Sample frames für Analyse
        if total_frames > sample_frames:
            step = total_frames // sample_frames
            sampled_frames = [frames[i] for i in range(
                0, total_frames, step)][:sample_frames]
            print(f"   🎯 Sampling {len(sampled_frames)} frames")
        else:
            sampled_frames = frames
            print(f"   🎯 Using all {len(sampled_frames)} frames")

        all_colors = set()
        transparent_pixels = 0
        total_pixels = 0

        for i, frame in enumerate(sampled_frames):
            # Konvertiere zu RGBA
            if frame.mode != 'RGBA':
                frame = frame.convert('RGBA')

            frame_array = np.array(frame)
            height, width = frame_array.shape[:2]
            total_pixels += height * width

            # Analysiere Alpha-Kanal
            alpha_channel = frame_array[:, :, 3]
            transparent_count = np.sum(alpha_channel < 128)
            transparent_pixels += transparent_count

            # Sammle nur sichtbare Farben
            opaque_mask = alpha_channel >= 128
            if np.any(opaque_mask):
                # Nur RGB, kein Alpha
                rgb_values = frame_array[opaque_mask][:, :3]
                for rgb in rgb_values:
                    all_colors.add(tuple(rgb))

        transparency_ratio = transparent_pixels / \
            total_pixels if total_pixels > 0 else 0
        unique_colors = len(all_colors)

        print(f"   🔍 Transparency: {transparency_ratio:.1%}")
        print(f"   🎨 Unique colors: {unique_colors}")

        return all_colors, unique_colors, transparency_ratio

    def analyze_color_distribution(self, colors):
        """Analysiere Farbverteilung"""
        if not colors:
            return None

        colors_list = list(colors)
        colors_array = np.array(colors_list)

        # RGB-Statistiken
        r_values = colors_array[:, 0]
        g_values = colors_array[:, 1]
        b_values = colors_array[:, 2]

        stats = {
            'total_colors': len(colors),
            'r_range': (int(np.min(r_values)), int(np.max(r_values))),
            'g_range': (int(np.min(g_values)), int(np.max(g_values))),
            'b_range': (int(np.min(b_values)), int(np.max(b_values))),
            'r_mean': float(np.mean(r_values)),
            'g_mean': float(np.mean(g_values)),
            'b_mean': float(np.mean(b_values)),
            'brightness_range': (
                int(np.min(np.mean(colors_array, axis=1))),
                int(np.max(np.mean(colors_array, axis=1)))
            )
        }

        return stats

    def categorize_colors(self, colors):
        """Kategorisiere Farben nach Helligkeit und Sättigung"""
        if not colors:
            return {}

        categories = {
            'very_dark': [],      # < 50
            'dark': [],           # 50-100
            'medium': [],         # 100-150
            'bright': [],         # 150-200
            'very_bright': [],    # > 200
            'grayscale': [],      # R≈G≈B
            'colorful': []        # Hohe Sättigung
        }

        for color in colors:
            r, g, b = color
            brightness = (r + g + b) / 3

            # Helligkeit kategorisieren
            if brightness < 50:
                categories['very_dark'].append(color)
            elif brightness < 100:
                categories['dark'].append(color)
            elif brightness < 150:
                categories['medium'].append(color)
            elif brightness < 200:
                categories['bright'].append(color)
            else:
                categories['very_bright'].append(color)

            # Sättigung prüfen
            color_range = max(r, g, b) - min(r, g, b)
            if color_range < 20:  # Geringe Sättigung = Graustufen
                categories['grayscale'].append(color)
            else:
                categories['colorful'].append(color)

        return categories

    def run_palette_analysis(self):
        """Führe komplette Palette-Analyse aus"""
        print("🎨 COLOR PALETTE ANALYSIS")
        print("=" * 60)

        # Analysiere beide Versionen
        print("\n🔍 ORIGINAL VERSION:")
        print("-" * 30)
        orig_colors, orig_count, orig_transparency = self.extract_colors_from_gif(
            self.original_file)

        print("\n🔍 FIXED VERSION (WINNER):")
        print("-" * 30)
        fixed_colors, fixed_count, fixed_transparency = self.extract_colors_from_gif(
            self.fixed_file)

        if not orig_colors or not fixed_colors:
            print("❌ Could not analyze colors")
            return

        # Detaillierte Analyse
        print("\n" + "=" * 60)
        print("📊 DETAILED PALETTE COMPARISON")
        print("=" * 60)

        print(f"\n📈 COLOR COUNT:")
        print(f"   • Original:  {orig_count:3d} unique colors")
        print(f"   • Fixed:     {fixed_count:3d} unique colors")
        print(f"   • Reduction: {(1 - fixed_count/orig_count)*100:.1f}%")

        # Statistiken
        orig_stats = self.analyze_color_distribution(orig_colors)
        fixed_stats = self.analyze_color_distribution(fixed_colors)

        print(f"\n🌈 COLOR RANGES:")
        print(f"   Original RGB ranges:")
        print(
            f"     • Red:   {orig_stats['r_range'][0]}-{orig_stats['r_range'][1]}")
        print(
            f"     • Green: {orig_stats['g_range'][0]}-{orig_stats['g_range'][1]}")
        print(
            f"     • Blue:  {orig_stats['b_range'][0]}-{orig_stats['b_range'][1]}")

        print(f"   Fixed RGB ranges:")
        print(
            f"     • Red:   {fixed_stats['r_range'][0]}-{fixed_stats['r_range'][1]}")
        print(
            f"     • Green: {fixed_stats['g_range'][0]}-{fixed_stats['g_range'][1]}")
        print(
            f"     • Blue:  {fixed_stats['b_range'][0]}-{fixed_stats['b_range'][1]}")

        # Kategorisierung
        orig_categories = self.categorize_colors(orig_colors)
        fixed_categories = self.categorize_colors(fixed_colors)

        print(f"\n🎯 COLOR CATEGORIES (Fixed Version):")
        print(
            f"   • Very Dark:   {len(fixed_categories['very_dark']):2d} colors")
        print(f"   • Dark:        {len(fixed_categories['dark']):2d} colors")
        print(f"   • Medium:      {len(fixed_categories['medium']):2d} colors")
        print(f"   • Bright:      {len(fixed_categories['bright']):2d} colors")
        print(
            f"   • Very Bright: {len(fixed_categories['very_bright']):2d} colors")
        print(
            f"   • Grayscale:   {len(fixed_categories['grayscale']):2d} colors")
        print(
            f"   • Colorful:    {len(fixed_categories['colorful']):2d} colors")

        # Aktuelle Palette anzeigen
        print(f"\n🎨 CURRENT FIXED VERSION PALETTE ({fixed_count} colors):")
        print("-" * 50)

        # Sortiere Farben nach Helligkeit
        sorted_colors = sorted(list(fixed_colors), key=lambda c: sum(c)/3)

        for i, color in enumerate(sorted_colors, 1):
            r, g, b = color
            brightness = (r + g + b) / 3
            hex_color = f"#{r:02x}{g:02x}{b:02x}"

            # Kategorisiere Farbe
            if brightness < 50:
                category = "Very Dark"
            elif brightness < 100:
                category = "Dark"
            elif brightness < 150:
                category = "Medium"
            elif brightness < 200:
                category = "Bright"
            else:
                category = "Very Bright"

            print(
                f"   {i:2d}. RGB({r:3d},{g:3d},{b:3d}) {hex_color} - {category}")

        # Empfehlung
        print(f"\n" + "=" * 60)
        print("🎯 PALETTE ASSESSMENT")
        print("=" * 60)

        if fixed_count <= 16:
            palette_quality = "EXCELLENT"
            icon = "✅"
        elif fixed_count <= 32:
            palette_quality = "GOOD"
            icon = "🟡"
        elif fixed_count <= 64:
            palette_quality = "FAIR"
            icon = "🟠"
        else:
            palette_quality = "POOR"
            icon = "❌"

        print(f"{icon} PALETTE SIZE: {fixed_count} colors - {palette_quality}")
        print(f"✅ TRANSPARENCY: {fixed_transparency:.1%} preserved")

        # Batch-Processing Empfehlung
        if fixed_count <= 32 and fixed_transparency > 0.8:
            print(f"\n🚀 BATCH PROCESSING RECOMMENDATION:")
            print(f"✅ PROCEED - Palette size is optimal for pixel art")
            print(f"✅ Transparency preservation is excellent")
            print(f"✅ Ready for all 249 files")
        else:
            print(f"\n⚠️  BATCH PROCESSING RECOMMENDATION:")
            print(f"🔧 Consider further optimization")


if __name__ == "__main__":
    analyzer = PaletteAnalyzer()
    analyzer.run_palette_analysis()
