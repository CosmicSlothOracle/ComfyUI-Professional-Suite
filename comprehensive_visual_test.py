#!/usr/bin/env python3
"""
Comprehensive Visual Quality Test
Tests all visual aspects: transparency, colors, pixelization, quality preservation
Test file: 0ea53a3cdbfdcf14caf1c8cccdb60143_fast_transparent_converted.gif
"""

import os
from pathlib import Path
from PIL import Image, ImageSequence, ImageStat
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class ComprehensiveVisualTest:
    def __init__(self):
        self.base_dir = Path("C:/Users/Public/ComfyUI-master")
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output" / "comprehensive_visual_test"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test files
        self.test_file = "0ea53a3cdbfdcf14caf1c8cccdb60143_fast_transparent_converted.gif"
        self.processed_file = "transparency_fixed_0ea53a3cdbfdcf14caf1c8cccdb60143_fast_transparent_converted.gif"
        self.processed_path = self.base_dir / "output" / \
            "transparency_test_fixed" / self.processed_file

    def analyze_transparency_quality(self, original, processed):
        """Detaillierte Transparenz-Analyse"""
        print("🔍 TRANSPARENCY ANALYSIS")
        print("-" * 30)

        results = {}

        # Original analysieren
        orig_frame = original.convert('RGBA')
        orig_alpha = np.array(orig_frame)[:, :, 3]
        orig_transparent_pixels = np.sum(orig_alpha < 128)
        orig_total_pixels = orig_alpha.size
        orig_transparency_ratio = orig_transparent_pixels / orig_total_pixels

        print(f"📊 Original:")
        print(f"   • Total pixels: {orig_total_pixels:,}")
        print(f"   • Transparent pixels: {orig_transparent_pixels:,}")
        print(f"   • Transparency ratio: {orig_transparency_ratio:.1%}")

        # Processed analysieren
        proc_frame = processed.convert('RGBA')
        proc_alpha = np.array(proc_frame)[:, :, 3]
        proc_transparent_pixels = np.sum(proc_alpha < 128)
        proc_total_pixels = proc_alpha.size
        proc_transparency_ratio = proc_transparent_pixels / proc_total_pixels

        print(f"📊 Processed:")
        print(f"   • Total pixels: {proc_total_pixels:,}")
        print(f"   • Transparent pixels: {proc_transparent_pixels:,}")
        print(f"   • Transparency ratio: {proc_transparency_ratio:.1%}")

        # Transparenz-Erhaltung bewerten
        transparency_preservation = abs(
            orig_transparency_ratio - proc_transparency_ratio) < 0.05  # 5% Toleranz

        print(f"📈 Transparency Comparison:")
        print(
            f"   • Difference: {abs(orig_transparency_ratio - proc_transparency_ratio):.1%}")
        print(
            f"   • Preservation: {'✅ GOOD' if transparency_preservation else '❌ POOR'}")

        results['transparency'] = {
            'original_ratio': orig_transparency_ratio,
            'processed_ratio': proc_transparency_ratio,
            'preservation_good': transparency_preservation,
            'difference': abs(orig_transparency_ratio - proc_transparency_ratio)
        }

        return results

    def analyze_color_quality(self, original, processed):
        """Detaillierte Farbqualitäts-Analyse"""
        print("\n🎨 COLOR QUALITY ANALYSIS")
        print("-" * 30)

        results = {}

        # Original Farbanalyse
        orig_rgb = original.convert('RGB')
        orig_colors = orig_rgb.getcolors(maxcolors=256*256*256)
        orig_unique_colors = len(orig_colors) if orig_colors else "256+"

        # Processed Farbanalyse
        proc_rgb = processed.convert('RGB')
        proc_colors = proc_rgb.getcolors(maxcolors=256*256*256)
        proc_unique_colors = len(proc_colors) if proc_colors else "256+"

        print(f"🌈 Color Count:")
        print(f"   • Original: {orig_unique_colors} unique colors")
        print(f"   • Processed: {proc_unique_colors} unique colors")

        # Farbverteilung analysieren
        orig_array = np.array(orig_rgb)
        proc_array = np.array(proc_rgb)

        # Durchschnittliche Farbwerte
        orig_mean_rgb = np.mean(orig_array, axis=(0, 1))
        proc_mean_rgb = np.mean(proc_array, axis=(0, 1))

        print(f"📊 Average RGB Values:")
        print(
            f"   • Original: R={orig_mean_rgb[0]:.1f}, G={orig_mean_rgb[1]:.1f}, B={orig_mean_rgb[2]:.1f}")
        print(
            f"   • Processed: R={proc_mean_rgb[0]:.1f}, G={proc_mean_rgb[1]:.1f}, B={proc_mean_rgb[2]:.1f}")

        # Farbbereich (Dynamik)
        orig_color_range = [np.ptp(orig_array[:, :, i]) for i in range(3)]
        proc_color_range = [np.ptp(proc_array[:, :, i]) for i in range(3)]

        print(f"📈 Color Range (Dynamic):")
        print(
            f"   • Original: R={orig_color_range[0]}, G={orig_color_range[1]}, B={orig_color_range[2]}")
        print(
            f"   • Processed: R={proc_color_range[0]}, G={proc_color_range[1]}, B={proc_color_range[2]}")

        # Bewerte Farberhaltung
        color_preservation = np.mean(
            [abs(orig_mean_rgb[i] - proc_mean_rgb[i]) for i in range(3)]) < 50
        range_preservation = np.mean(
            [abs(orig_color_range[i] - proc_color_range[i]) for i in range(3)]) < 100

        print(f"✅ Color Quality Assessment:")
        print(
            f"   • Color preservation: {'✅ GOOD' if color_preservation else '❌ POOR'}")
        print(
            f"   • Range preservation: {'✅ GOOD' if range_preservation else '❌ POOR'}")

        results['colors'] = {
            'original_unique': orig_unique_colors,
            'processed_unique': proc_unique_colors,
            'color_preservation_good': color_preservation,
            'range_preservation_good': range_preservation,
            'original_mean_rgb': orig_mean_rgb.tolist(),
            'processed_mean_rgb': proc_mean_rgb.tolist()
        }

        return results

    def analyze_pixelization_quality(self, original, processed):
        """Analyse der Pixelisierungs-Qualität"""
        print("\n🔲 PIXELIZATION QUALITY ANALYSIS")
        print("-" * 30)

        results = {}

        orig_array = np.array(original.convert('RGB'))
        proc_array = np.array(processed.convert('RGB'))

        # Kantenschärfe messen (Gradient)
        orig_gradient = np.gradient(np.mean(orig_array, axis=2))
        proc_gradient = np.gradient(np.mean(proc_array, axis=2))

        orig_edge_strength = np.mean(
            np.sqrt(orig_gradient[0]**2 + orig_gradient[1]**2))
        proc_edge_strength = np.mean(
            np.sqrt(proc_gradient[0]**2 + proc_gradient[1]**2))

        print(f"📊 Edge Strength (Pixelization Effect):")
        print(f"   • Original: {orig_edge_strength:.2f}")
        print(f"   • Processed: {proc_edge_strength:.2f}")

        # Pixelisierung sollte Kanten schärfer machen
        pixelization_effective = proc_edge_strength > orig_edge_strength * 0.8

        # Strukturerhaltung prüfen
        orig_structure = np.std(orig_array)
        proc_structure = np.std(proc_array)

        print(f"📈 Structure Preservation:")
        print(f"   • Original variance: {orig_structure:.2f}")
        print(f"   • Processed variance: {proc_structure:.2f}")

        structure_preserved = abs(
            orig_structure - proc_structure) / orig_structure < 0.3  # 30% Toleranz

        print(f"✅ Pixelization Quality:")
        print(
            f"   • Effective pixelization: {'✅ GOOD' if pixelization_effective else '❌ POOR'}")
        print(
            f"   • Structure preserved: {'✅ GOOD' if structure_preserved else '❌ POOR'}")

        results['pixelization'] = {
            'original_edge_strength': orig_edge_strength,
            'processed_edge_strength': proc_edge_strength,
            'pixelization_effective': pixelization_effective,
            'structure_preserved': structure_preserved
        }

        return results

    def analyze_animation_quality(self, original_path, processed_path):
        """Analyse der Animations-Qualität"""
        print("\n🎬 ANIMATION QUALITY ANALYSIS")
        print("-" * 30)

        results = {}

        # Original GIF
        orig_gif = Image.open(original_path)
        orig_frames = [frame.copy()
                       for frame in ImageSequence.Iterator(orig_gif)]
        orig_frame_count = len(orig_frames)
        orig_duration = orig_gif.info.get('duration', 100)

        # Processed GIF
        proc_gif = Image.open(processed_path)
        proc_frames = [frame.copy()
                       for frame in ImageSequence.Iterator(proc_gif)]
        proc_frame_count = len(proc_frames)
        proc_duration = proc_gif.info.get('duration', 100)

        print(f"📊 Animation Properties:")
        print(f"   • Original frames: {orig_frame_count}")
        print(f"   • Processed frames: {proc_frame_count}")
        print(f"   • Original duration: {orig_duration}ms per frame")
        print(f"   • Processed duration: {proc_duration}ms per frame")

        # Frame-Erhaltung
        frame_preservation = orig_frame_count == proc_frame_count
        timing_preservation = abs(
            orig_duration - proc_duration) <= 10  # 10ms Toleranz

        # Bewegungsfluss analysieren (Frame-zu-Frame Unterschiede)
        def calculate_frame_differences(frames):
            differences = []
            for i in range(1, len(frames)):
                prev_array = np.array(frames[i-1].convert('RGB'))
                curr_array = np.array(frames[i].convert('RGB'))
                diff = np.mean(np.abs(prev_array.astype(
                    float) - curr_array.astype(float)))
                differences.append(diff)
            return differences

        motion_preserved = True
        if orig_frame_count > 1 and proc_frame_count > 1:
            orig_diffs = calculate_frame_differences(orig_frames)
            proc_diffs = calculate_frame_differences(proc_frames)

            orig_avg_diff = np.mean(orig_diffs)
            proc_avg_diff = np.mean(proc_diffs)

            print(f"📈 Motion Analysis:")
            print(f"   • Original avg frame difference: {orig_avg_diff:.2f}")
            print(f"   • Processed avg frame difference: {proc_avg_diff:.2f}")

            motion_preserved = abs(
                orig_avg_diff - proc_avg_diff) / orig_avg_diff < 0.5  # 50% Toleranz
        else:
            motion_preserved = True  # Single frame

        print(f"✅ Animation Quality:")
        print(
            f"   • Frame count preserved: {'✅ GOOD' if frame_preservation else '❌ POOR'}")
        print(
            f"   • Timing preserved: {'✅ GOOD' if timing_preservation else '❌ POOR'}")
        print(
            f"   • Motion preserved: {'✅ GOOD' if motion_preserved else '❌ POOR'}")

        results['animation'] = {
            'original_frames': orig_frame_count,
            'processed_frames': proc_frame_count,
            'frame_preservation': frame_preservation,
            'timing_preservation': timing_preservation,
            'motion_preserved': motion_preserved
        }

        return results

    def analyze_file_quality(self, original_path, processed_path):
        """Analyse der Datei-Qualität"""
        print("\n📁 FILE QUALITY ANALYSIS")
        print("-" * 30)

        results = {}

        # Dateigrößen
        orig_size = original_path.stat().st_size
        proc_size = processed_path.stat().st_size

        compression_ratio = (orig_size - proc_size) / orig_size

        print(f"📊 File Sizes:")
        print(
            f"   • Original: {orig_size:,} bytes ({orig_size/1024/1024:.1f} MB)")
        print(
            f"   • Processed: {proc_size:,} bytes ({proc_size/1024/1024:.1f} MB)")
        print(f"   • Compression: {compression_ratio:.1%}")

        # Qualitäts-Effizienz bewerten
        size_efficient = proc_size <= orig_size * 2  # Nicht mehr als doppelt so groß

        print(f"✅ File Quality:")
        print(
            f"   • Size efficient: {'✅ GOOD' if size_efficient else '❌ POOR'}")

        results['file'] = {
            'original_size': orig_size,
            'processed_size': proc_size,
            'compression_ratio': compression_ratio,
            'size_efficient': size_efficient
        }

        return results

    def generate_visual_comparison(self, original, processed):
        """Erstelle visuelle Vergleichsbilder"""
        print("\n📸 GENERATING VISUAL COMPARISON")
        print("-" * 30)

        try:
            # Erstelle Vergleichsbild
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(
                'Pixel Art Conversion - Visual Comparison', fontsize=16)

            # Original
            axes[0, 0].imshow(original.convert('RGB'))
            axes[0, 0].set_title('Original')
            axes[0, 0].axis('off')

            # Processed
            axes[0, 1].imshow(processed.convert('RGB'))
            axes[0, 1].set_title('Processed (Pixel Art)')
            axes[0, 1].axis('off')

            # Transparenz-Vergleich
            if original.mode == 'RGBA' and processed.mode == 'RGBA':
                orig_alpha = np.array(original)[:, :, 3]
                proc_alpha = np.array(processed)[:, :, 3]

                axes[1, 0].imshow(orig_alpha, cmap='gray')
                axes[1, 0].set_title('Original Alpha Channel')
                axes[1, 0].axis('off')

                axes[1, 1].imshow(proc_alpha, cmap='gray')
                axes[1, 1].set_title('Processed Alpha Channel')
                axes[1, 1].axis('off')

            plt.tight_layout()
            comparison_path = self.output_dir / "visual_comparison.png"
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"   💾 Visual comparison saved: {comparison_path}")
            return True

        except Exception as e:
            print(f"   ⚠️  Could not generate visual comparison: {e}")
            return False

    def run_comprehensive_test(self):
        """Führe umfassenden visuellen Test aus"""
        print("🔍 COMPREHENSIVE VISUAL QUALITY TEST")
        print("=" * 60)

        # Pfade prüfen
        original_path = self.input_dir / self.test_file

        if not original_path.exists():
            print(f"❌ Original file not found: {original_path}")
            return False

        if not self.processed_path.exists():
            print(f"❌ Processed file not found: {self.processed_path}")
            return False

        print(f"📁 Original: {self.test_file}")
        print(f"📁 Processed: {self.processed_file}")

        # Lade Bilder
        original_gif = Image.open(original_path)
        processed_gif = Image.open(self.processed_path)

        # Erstes Frame für Analyse
        original_frame = original_gif.convert('RGBA')
        processed_frame = processed_gif.convert('RGBA')

        # Führe alle Analysen durch
        all_results = {}

        # 1. Transparenz-Analyse
        transparency_results = self.analyze_transparency_quality(
            original_frame, processed_frame)
        all_results.update(transparency_results)

        # 2. Farb-Analyse
        color_results = self.analyze_color_quality(
            original_frame, processed_frame)
        all_results.update(color_results)

        # 3. Pixelisierungs-Analyse
        pixel_results = self.analyze_pixelization_quality(
            original_frame, processed_frame)
        all_results.update(pixel_results)

        # 4. Animations-Analyse
        animation_results = self.analyze_animation_quality(
            original_path, self.processed_path)
        all_results.update(animation_results)

        # 5. Datei-Analyse
        file_results = self.analyze_file_quality(
            original_path, self.processed_path)
        all_results.update(file_results)

        # 6. Visuelle Vergleiche generieren
        self.generate_visual_comparison(original_frame, processed_frame)

        # Gesamtbewertung
        print("\n" + "=" * 60)
        print("🎯 OVERALL QUALITY ASSESSMENT")
        print("=" * 60)

        quality_scores = []

        # Transparenz-Score
        if all_results['transparency']['preservation_good']:
            transparency_score = 100
        else:
            transparency_score = max(
                0, 100 - (all_results['transparency']['difference'] * 1000))
        quality_scores.append(('Transparency', transparency_score))

        # Farb-Score
        color_score = 0
        if all_results['colors']['color_preservation_good']:
            color_score += 50
        if all_results['colors']['range_preservation_good']:
            color_score += 50
        quality_scores.append(('Colors', color_score))

        # Pixelisierungs-Score
        pixel_score = 0
        if all_results['pixelization']['pixelization_effective']:
            pixel_score += 50
        if all_results['pixelization']['structure_preserved']:
            pixel_score += 50
        quality_scores.append(('Pixelization', pixel_score))

        # Animations-Score
        animation_score = 0
        if all_results['animation']['frame_preservation']:
            animation_score += 33
        if all_results['animation']['timing_preservation']:
            animation_score += 33
        if all_results['animation']['motion_preserved']:
            animation_score += 34
        quality_scores.append(('Animation', animation_score))

        # Datei-Score
        file_score = 100 if all_results['file']['size_efficient'] else 50
        quality_scores.append(('File Quality', file_score))

        # Zeige Scores
        print("📊 QUALITY SCORES:")
        total_score = 0
        for category, score in quality_scores:
            status = "✅ EXCELLENT" if score >= 90 else "🟡 GOOD" if score >= 70 else "🟠 FAIR" if score >= 50 else "❌ POOR"
            print(f"   • {category:15}: {score:3.0f}% {status}")
            total_score += score

        overall_score = total_score / len(quality_scores)
        overall_status = "✅ EXCELLENT" if overall_score >= 90 else "🟡 GOOD" if overall_score >= 70 else "🟠 FAIR" if overall_score >= 50 else "❌ POOR"

        print(f"\n🏆 OVERALL SCORE: {overall_score:.0f}% {overall_status}")

        # Empfehlung
        if overall_score >= 80:
            print(f"\n🎉 RECOMMENDATION: PROCEED WITH BATCH PROCESSING")
            print(f"✅ Quality is sufficient for all 249 files")
            return True
        elif overall_score >= 60:
            print(f"\n⚠️  RECOMMENDATION: MINOR ADJUSTMENTS NEEDED")
            print(f"🔧 Consider tweaking parameters before full batch")
            return False
        else:
            print(f"\n❌ RECOMMENDATION: MAJOR IMPROVEMENTS NEEDED")
            print(f"🛠️  Significant changes required before batch processing")
            return False


if __name__ == "__main__":
    tester = ComprehensiveVisualTest()
    success = tester.run_comprehensive_test()

    if success:
        print(f"\n🚀 READY FOR BATCH PROCESSING!")
    else:
        print(f"\n🔧 IMPROVEMENTS NEEDED FIRST!")
