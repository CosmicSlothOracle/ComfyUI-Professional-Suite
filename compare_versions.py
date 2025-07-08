#!/usr/bin/env python3
"""
Version Comparison Test
Compares original fixed version vs optimized version
Determines which is better for batch processing
"""

import os
from pathlib import Path
from PIL import Image, ImageSequence
import numpy as np


class VersionComparison:
    def __init__(self):
        self.base_dir = Path("C:/Users/Public/ComfyUI-master")
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output" / "version_comparison"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test file
        self.test_file = "0ea53a3cdbfdcf14caf1c8cccdb60143_fast_transparent_converted.gif"

        # Version paths
        self.original_path = self.input_dir / self.test_file
        self.fixed_path = self.base_dir / "output" / \
            "transparency_test_fixed" / f"transparency_fixed_{self.test_file}"
        self.optimized_path = self.base_dir / "output" / \
            "optimized_pixel_art_test" / f"optimized_{self.test_file}"

    def analyze_single_version(self, version_name, file_path, original_path):
        """Analysiere eine Version"""
        print(f"\n📊 ANALYZING {version_name.upper()}")
        print("-" * 40)

        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return None

        # Lade Bilder
        original_gif = Image.open(original_path)
        processed_gif = Image.open(file_path)

        orig_frame = original_gif.convert('RGBA')
        proc_frame = processed_gif.convert('RGBA')

        results = {}

        # 1. Transparenz-Analyse
        orig_alpha = np.array(orig_frame)[:, :, 3]
        proc_alpha = np.array(proc_frame)[:, :, 3]

        orig_transparency = np.sum(orig_alpha < 128) / orig_alpha.size
        proc_transparency = np.sum(proc_alpha < 128) / proc_alpha.size
        transparency_diff = abs(orig_transparency - proc_transparency)

        print(f"🔍 Transparency:")
        print(f"   • Original: {orig_transparency:.1%}")
        print(f"   • Processed: {proc_transparency:.1%}")
        print(f"   • Difference: {transparency_diff:.1%}")

        # 2. Farbanalyse
        orig_rgb = orig_frame.convert('RGB')
        proc_rgb = proc_frame.convert('RGB')

        orig_colors = orig_rgb.getcolors(maxcolors=256*256*256)
        proc_colors = proc_rgb.getcolors(maxcolors=256*256*256)

        orig_unique = len(orig_colors) if orig_colors else "256+"
        proc_unique = len(proc_colors) if proc_colors else "256+"

        print(f"🎨 Colors:")
        print(f"   • Original: {orig_unique} unique colors")
        print(f"   • Processed: {proc_unique} unique colors")

        # 3. Animations-Analyse
        orig_frames = [f.copy() for f in ImageSequence.Iterator(original_gif)]
        proc_frames = [f.copy() for f in ImageSequence.Iterator(processed_gif)]

        orig_frame_count = len(orig_frames)
        proc_frame_count = len(proc_frames)

        orig_duration = original_gif.info.get('duration', 100)
        proc_duration = processed_gif.info.get('duration', 100)

        print(f"🎬 Animation:")
        print(f"   • Original frames: {orig_frame_count}")
        print(f"   • Processed frames: {proc_frame_count}")
        print(f"   • Duration: {orig_duration}ms → {proc_duration}ms")

        # 4. Bewegungsanalyse
        def calculate_motion_smoothness(frames):
            if len(frames) <= 1:
                return 0

            differences = []
            for i in range(1, len(frames)):
                prev = np.array(frames[i-1].convert('RGB'))
                curr = np.array(frames[i].convert('RGB'))
                diff = np.mean(np.abs(prev.astype(float) - curr.astype(float)))
                differences.append(diff)

            return np.mean(differences), np.std(differences)

        orig_motion_mean, orig_motion_std = calculate_motion_smoothness(
            orig_frames)
        proc_motion_mean, proc_motion_std = calculate_motion_smoothness(
            proc_frames)

        print(f"📈 Motion Analysis:")
        print(
            f"   • Original: mean={orig_motion_mean:.2f}, std={orig_motion_std:.2f}")
        print(
            f"   • Processed: mean={proc_motion_mean:.2f}, std={proc_motion_std:.2f}")

        # 5. Dateigröße
        orig_size = original_path.stat().st_size
        proc_size = file_path.stat().st_size
        compression = (orig_size - proc_size) / orig_size

        print(f"📁 File Size:")
        print(f"   • Original: {orig_size:,} bytes")
        print(f"   • Processed: {proc_size:,} bytes")
        print(f"   • Compression: {compression:.1%}")

        # Bewertung
        transparency_score = 100 if transparency_diff < 0.02 else max(
            0, 100 - transparency_diff * 5000)

        # Farb-Score (mehr Farben = besser, aber nicht zu viele)
        if isinstance(proc_unique, int) and isinstance(orig_unique, int):
            color_ratio = proc_unique / orig_unique
            if 0.2 <= color_ratio <= 0.8:  # 20-80% der Original-Farben ist gut
                color_score = 100
            elif color_ratio < 0.2:
                color_score = color_ratio * 500  # Zu wenige Farben
            else:
                color_score = max(0, 100 - (color_ratio - 0.8)
                                  * 200)  # Zu viele Farben
        else:
            color_score = 50  # Fallback

        # Animations-Score
        frame_score = 100 if orig_frame_count == proc_frame_count else 0
        timing_score = 100 if abs(orig_duration - proc_duration) <= 10 else 50

        # Bewegungs-Score (sanftere Bewegung = besser)
        motion_change = abs(proc_motion_mean - orig_motion_mean) / \
            orig_motion_mean if orig_motion_mean > 0 else 0
        motion_score = max(0, 100 - motion_change * 100)

        # Dateigröße-Score
        size_score = 100 if proc_size <= orig_size * \
            1.5 else max(0, 100 - (proc_size / orig_size - 1.5) * 100)

        overall_score = (transparency_score + color_score +
                         frame_score + timing_score + motion_score + size_score) / 6

        print(f"\n✅ SCORES:")
        print(f"   • Transparency: {transparency_score:.0f}%")
        print(f"   • Colors: {color_score:.0f}%")
        print(f"   • Frames: {frame_score:.0f}%")
        print(f"   • Timing: {timing_score:.0f}%")
        print(f"   • Motion: {motion_score:.0f}%")
        print(f"   • File Size: {size_score:.0f}%")
        print(f"   • OVERALL: {overall_score:.0f}%")

        return {
            'version': version_name,
            'transparency_score': transparency_score,
            'color_score': color_score,
            'frame_score': frame_score,
            'timing_score': timing_score,
            'motion_score': motion_score,
            'size_score': size_score,
            'overall_score': overall_score,
            'file_size': proc_size,
            'unique_colors': proc_unique,
            'transparency_diff': transparency_diff,
            'motion_mean': proc_motion_mean
        }

    def run_comparison(self):
        """Führe Versions-Vergleich aus"""
        print("🔍 VERSION COMPARISON TEST")
        print("=" * 60)

        if not self.original_path.exists():
            print(f"❌ Original file not found: {self.original_path}")
            return False

        print(f"📁 Test file: {self.test_file}")

        # Analysiere beide Versionen
        fixed_results = self.analyze_single_version(
            "FIXED VERSION", self.fixed_path, self.original_path)
        optimized_results = self.analyze_single_version(
            "OPTIMIZED VERSION", self.optimized_path, self.original_path)

        if not fixed_results or not optimized_results:
            print("❌ Could not analyze both versions")
            return False

        # Vergleich
        print("\n" + "=" * 60)
        print("🏆 FINAL COMPARISON")
        print("=" * 60)

        categories = [
            ('Transparency', 'transparency_score'),
            ('Colors', 'color_score'),
            ('Frames', 'frame_score'),
            ('Timing', 'timing_score'),
            ('Motion', 'motion_score'),
            ('File Size', 'size_score'),
            ('OVERALL', 'overall_score')
        ]

        print(f"{'Category':<15} {'Fixed':<10} {'Optimized':<10} {'Winner'}")
        print("-" * 50)

        fixed_wins = 0
        optimized_wins = 0

        for category, key in categories:
            fixed_score = fixed_results[key]
            optimized_score = optimized_results[key]

            if fixed_score > optimized_score:
                winner = "Fixed ✅"
                if category != 'OVERALL':
                    fixed_wins += 1
            elif optimized_score > fixed_score:
                winner = "Optimized ✅"
                if category != 'OVERALL':
                    optimized_wins += 1
            else:
                winner = "Tie"

            print(
                f"{category:<15} {fixed_score:<10.0f} {optimized_score:<10.0f} {winner}")

        print("\n📊 SUMMARY:")
        print(f"   • Fixed version wins: {fixed_wins} categories")
        print(f"   • Optimized version wins: {optimized_wins} categories")

        # Empfehlung
        if fixed_results['overall_score'] > optimized_results['overall_score']:
            winner = "FIXED VERSION"
            winner_score = fixed_results['overall_score']
            recommended_path = self.fixed_path
        else:
            winner = "OPTIMIZED VERSION"
            winner_score = optimized_results['overall_score']
            recommended_path = self.optimized_path

        print(f"\n🏆 WINNER: {winner} ({winner_score:.0f}%)")

        # Finale Empfehlung
        if winner_score >= 80:
            print(f"\n🎉 RECOMMENDATION: PROCEED WITH BATCH PROCESSING")
            print(f"✅ Use {winner.lower()} for all 249 files")
            print(f"📁 Template file: {recommended_path.name}")
            return True
        elif winner_score >= 70:
            print(f"\n🟡 RECOMMENDATION: ACCEPTABLE QUALITY")
            print(f"⚠️  Proceed with caution using {winner.lower()}")
            return True
        else:
            print(f"\n❌ RECOMMENDATION: FURTHER IMPROVEMENTS NEEDED")
            print(f"🛠️  Both versions need optimization before batch processing")
            return False


if __name__ == "__main__":
    comparator = VersionComparison()
    success = comparator.run_comparison()

    if success:
        print(f"\n🚀 READY FOR BATCH PROCESSING!")
    else:
        print(f"\n🔧 MORE WORK NEEDED!")
