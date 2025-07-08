#!/usr/bin/env python3
"""
Quick Pokemon Card Verifier - No Unicode Issues
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import imagehash
from scipy.spatial.distance import cosine


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


class QuickCardVerifier:
    def __init__(self):
        self.setup_directories()

    def setup_directories(self):
        """Create output directories if they don't exist"""
        dirs = ["output", "output/reports", "output/visualizations", "logs"]
        for directory in dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def verify_magikarp_cards(self, card1_path, card2_path):
        """Verify the two Magikarp cards against each other"""
        print(f"Loading cards:")
        print(f"  Card 1: {card1_path}")
        print(f"  Card 2: {card2_path}")

        # Load images
        img1 = Image.open(card1_path).convert('RGB')
        img2 = Image.open(card2_path).convert('RGB')

        print(f"Card 1 size: {img1.size}")
        print(f"Card 2 size: {img2.size}")

        # Convert to numpy arrays and resize
        gen_np = cv2.resize(np.array(img1), (512, 512))
        ref_np = cv2.resize(np.array(img2), (512, 512))

        print("Analyzing similarity...")

        # Calculate metrics
        results = {}

        # 1. Structural Similarity (SSIM)
        gen_gray = cv2.cvtColor(gen_np, cv2.COLOR_RGB2GRAY)
        ref_gray = cv2.cvtColor(ref_np, cv2.COLOR_RGB2GRAY)
        ssim_score = ssim(gen_gray, ref_gray)
        results['ssim_score'] = float(ssim_score)
        print(f"SSIM Score: {ssim_score:.3f}")

        # 2. Perceptual hash comparison
        gen_hash = imagehash.phash(Image.fromarray(gen_np))
        ref_hash = imagehash.phash(Image.fromarray(ref_np))
        hash_diff = gen_hash - ref_hash
        hash_similarity = 1.0 - hash_diff / 64.0
        results['hash_similarity'] = float(max(0.0, hash_similarity))
        print(f"Hash Similarity: {hash_similarity:.3f}")

        # 3. Color analysis
        color_score = self.analyze_colors(gen_np, ref_np)
        results['color_score'] = float(color_score)
        print(f"Color Score: {color_score:.3f}")

        # 4. Histogram comparison
        hist_score = self.compare_histograms(gen_np, ref_np)
        results['histogram_score'] = float(hist_score)
        print(f"Histogram Score: {hist_score:.3f}")

        # 5. Pixel difference
        pixel_diff = np.mean(np.abs(gen_np.astype(
            float) - ref_np.astype(float))) / 255.0
        pixel_similarity = 1.0 - pixel_diff
        results['pixel_similarity'] = float(pixel_similarity)
        print(f"Pixel Similarity: {pixel_similarity:.3f}")

        # Calculate final score
        weights = {
            'ssim': 0.30,
            'hash': 0.25,
            'color': 0.20,
            'histogram': 0.15,
            'pixel': 0.10
        }

        final_score = (
            weights['ssim'] * results['ssim_score'] +
            weights['hash'] * results['hash_similarity'] +
            weights['color'] * results['color_score'] +
            weights['histogram'] * results['histogram_score'] +
            weights['pixel'] * results['pixel_similarity']
        )

        results['final_score'] = float(final_score)

        # Assessment
        if final_score > 0.85:
            assessment = "HIGH SIMILARITY"
        elif final_score > 0.70:
            assessment = "MODERATE SIMILARITY"
        else:
            assessment = "LOW SIMILARITY"

        results['assessment'] = assessment

        print(f"\nFinal Score: {final_score:.3f}")
        print(f"Assessment: {assessment}")

        # Create visualization
        self.create_visualization(img1, img2, results)

        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"output/reports/magikarp_comparison_{timestamp}.json"

        # Convert numpy types for JSON serialization
        results_json = convert_numpy_types(results)
        results_json['card1_path'] = str(card1_path)
        results_json['card2_path'] = str(card2_path)
        results_json['timestamp'] = timestamp

        with open(report_path, 'w') as f:
            json.dump(results_json, f, indent=2)

        print(f"Report saved: {report_path}")

        return results

    def analyze_colors(self, img1, img2):
        """Analyze color distribution similarity"""
        # Convert to LAB color space
        lab1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
        lab2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)

        # Calculate histograms
        hist1 = cv2.calcHist([lab1], [0, 1, 2], None, [
                             50, 50, 50], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([lab2], [0, 1, 2], None, [
                             50, 50, 50], [0, 256, 0, 256, 0, 256])

        # Normalize
        hist1 = hist1.flatten() / np.sum(hist1)
        hist2 = hist2.flatten() / np.sum(hist2)

        # Calculate similarity
        color_similarity = 1.0 - cosine(hist1, hist2)
        return max(0.0, color_similarity)

    def compare_histograms(self, img1, img2):
        """Compare RGB histograms"""
        scores = []
        for channel in range(3):
            hist1 = cv2.calcHist([img1], [channel], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [channel], None, [256], [0, 256])

            # Normalize
            hist1 = hist1.flatten() / np.sum(hist1)
            hist2 = hist2.flatten() / np.sum(hist2)

            # Correlation
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            scores.append(correlation)

        return np.mean(scores)

    def create_visualization(self, img1, img2, results):
        """Create visualization of verification results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"output/visualizations/magikarp_comparison_{timestamp}.png"

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f"Magikarp Card Comparison\nSimilarity Score: {results['final_score']:.3f}", fontsize=16)

        # Images
        axes[0, 0].imshow(img1)
        axes[0, 0].set_title("Card 1 (migakarp.png)")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(img2)
        axes[0, 1].set_title("Card 2 (Migakarp.png)")
        axes[0, 1].axis('off')

        # Difference
        img1_resized = cv2.resize(np.array(img1), (512, 512))
        img2_resized = cv2.resize(np.array(img2), (512, 512))
        diff = np.abs(img1_resized.astype(float) - img2_resized.astype(float))
        axes[0, 2].imshow(diff.astype(np.uint8))
        axes[0, 2].set_title("Pixel Difference")
        axes[0, 2].axis('off')

        # Scores
        score_names = ['SSIM', 'Hash', 'Color', 'Histogram', 'Pixel']
        score_values = [
            results['ssim_score'],
            results['hash_similarity'],
            results['color_score'],
            results['histogram_score'],
            results['pixel_similarity']
        ]

        axes[1, 0].barh(score_names, score_values, color='lightblue')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_title('Component Scores')
        axes[1, 0].set_xlim(0, 1)

        # Histograms
        axes[1, 1].hist(img1_resized[:, :, 0].flatten(), bins=50,
                        alpha=0.7, label='Card 1 R', color='red')
        axes[1, 1].hist(img2_resized[:, :, 0].flatten(), bins=50,
                        alpha=0.7, label='Card 2 R', color='darkred')
        axes[1, 1].set_title('Red Channel Histogram')
        axes[1, 1].legend()

        # Assessment
        assessment = results['assessment']
        if "HIGH" in assessment:
            color = 'green'
        elif "MODERATE" in assessment:
            color = 'orange'
        else:
            color = 'red'

        axes[1, 2].text(0.5, 0.5, f"{assessment}\n{results['final_score']:.3f}",
                        ha='center', va='center', fontsize=16, color=color, weight='bold')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved: {output_path}")


def main():
    verifier = QuickCardVerifier()

    # Verify the two Magikarp cards
    card1 = "input_cards/to_verify/migakarp.png"
    card2 = "input_cards/Migakarp.png"

    print("=== MAGIKARP CARD COMPARISON ===")

    try:
        results = verifier.verify_magikarp_cards(card1, card2)

        print("\n=== RESULTS SUMMARY ===")
        print(f"Final Similarity Score: {results['final_score']:.3f}")
        print(f"Assessment: {results['assessment']}")
        print("\nComponent Breakdown:")
        print(f"  SSIM (Structural): {results['ssim_score']:.3f}")
        print(f"  Hash (Perceptual): {results['hash_similarity']:.3f}")
        print(f"  Color Distribution: {results['color_score']:.3f}")
        print(f"  Histogram Match: {results['histogram_score']:.3f}")
        print(f"  Pixel Similarity: {results['pixel_similarity']:.3f}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
