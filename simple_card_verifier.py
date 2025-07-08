#!/usr/bin/env python3
"""
Simplified Pokemon Card Authenticity Verification
Works with basic dependencies for quick testing
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


class SimpleCardVerifier:
    def __init__(self):
        """Initialize the simple card verification system"""
        self.setup_logging()
        self.setup_directories()

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = f"logs/simple_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """Create output directories if they don't exist"""
        dirs = ["output", "output/reports", "output/visualizations", "logs"]
        for directory in dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def parse_filename(self, filename):
        """Parse filename to extract card information"""
        name = Path(filename).stem.lower()

        # For simple names like "migakarp" or "magikarp"
        if "magikarp" in name or "migakarp" in name:
            return {
                'set_name': 'custom',
                'card_number': '129',
                'card_name': 'magikarp',
                'type': 'generated' if 'generated' in name else 'unknown'
            }

        # Try standard format parsing
        parts = name.split('_')
        if len(parts) >= 4:
            return {
                'set_name': parts[0],
                'card_number': parts[1],
                'card_name': '_'.join(parts[2:-1]),
                'type': parts[-1]
            }

        # Fallback for non-standard names
        return {
            'set_name': 'custom',
            'card_number': '000',
            'card_name': name,
            'type': 'unknown'
        }

    def find_reference_card(self, generated_card_info):
        """Find matching reference card for a generated card"""
        # First try to find exact match in reference_cards
        possible_paths = [
            f"reference_cards/custom/custom_129_magikarp_reference.png",
            f"reference_cards/base_set/base_set_129_magikarp_reference.png",
            f"input_cards/Migakarp.png",  # Use the other provided file as reference
        ]

        for path in possible_paths:
            if Path(path).exists():
                return Path(path)

        return None

    def analyze_layout_components(self, generated_img, reference_img):
        """Analyze specific layout components using traditional CV methods"""
        # Convert to numpy arrays and resize
        gen_np = cv2.resize(np.array(generated_img), (512, 512))
        ref_np = cv2.resize(np.array(reference_img), (512, 512))

        analysis = {}

        # 1. Structural Similarity (SSIM)
        gen_gray = cv2.cvtColor(gen_np, cv2.COLOR_RGB2GRAY)
        ref_gray = cv2.cvtColor(ref_np, cv2.COLOR_RGB2GRAY)
        ssim_score, ssim_diff = ssim(gen_gray, ref_gray, full=True)
        analysis['ssim_score'] = ssim_score

        # 2. Perceptual hash comparison
        gen_hash = imagehash.phash(Image.fromarray(gen_np))
        ref_hash = imagehash.phash(Image.fromarray(ref_np))
        hash_similarity = 1.0 - (gen_hash - ref_hash) / 64.0
        analysis['hash_similarity'] = max(0.0, hash_similarity)

        # 3. Color analysis
        color_score = self.analyze_colors(gen_np, ref_np)
        analysis['color_score'] = color_score

        # 4. Border analysis
        border_score = self.analyze_borders(gen_np, ref_np)
        analysis['border_score'] = border_score

        # 5. Histogram comparison
        hist_score = self.compare_histograms(gen_np, ref_np)
        analysis['histogram_score'] = hist_score

        return analysis

    def analyze_colors(self, gen_img, ref_img):
        """Analyze color distribution similarity"""
        # Convert to LAB color space
        gen_lab = cv2.cvtColor(gen_img, cv2.COLOR_RGB2LAB)
        ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_RGB2LAB)

        # Calculate histograms
        gen_hist = cv2.calcHist([gen_lab], [0, 1, 2], None, [
                                50, 50, 50], [0, 256, 0, 256, 0, 256])
        ref_hist = cv2.calcHist([ref_lab], [0, 1, 2], None, [
                                50, 50, 50], [0, 256, 0, 256, 0, 256])

        # Normalize
        gen_hist = gen_hist.flatten() / np.sum(gen_hist)
        ref_hist = ref_hist.flatten() / np.sum(ref_hist)

        # Calculate similarity
        color_similarity = 1.0 - cosine(gen_hist, ref_hist)
        return max(0.0, color_similarity)

    def analyze_borders(self, gen_img, ref_img):
        """Analyze border consistency"""
        border_thickness = 20

        # Extract borders
        gen_border = self.extract_border_region(gen_img, border_thickness)
        ref_border = self.extract_border_region(ref_img, border_thickness)

        # Compare
        border_diff = np.mean(
            np.abs(gen_border.astype(float) - ref_border.astype(float)))
        border_score = max(0.0, 1.0 - border_diff / 255.0)

        return border_score

    def extract_border_region(self, img, thickness):
        """Extract border region from image"""
        h, w = img.shape[:2]
        border = np.zeros_like(img)

        # Extract borders
        border[:thickness, :] = img[:thickness, :]
        border[-thickness:, :] = img[-thickness:, :]
        border[:, :thickness] = img[:, :thickness]
        border[:, -thickness:] = img[:, -thickness:]

        return border

    def compare_histograms(self, gen_img, ref_img):
        """Compare RGB histograms"""
        scores = []
        for channel in range(3):
            gen_hist = cv2.calcHist(
                [gen_img], [channel], None, [256], [0, 256])
            ref_hist = cv2.calcHist(
                [ref_img], [channel], None, [256], [0, 256])

            # Normalize
            gen_hist = gen_hist.flatten() / np.sum(gen_hist)
            ref_hist = ref_hist.flatten() / np.sum(ref_hist)

            # Correlation
            correlation = cv2.compareHist(
                gen_hist, ref_hist, cv2.HISTCMP_CORREL)
            scores.append(correlation)

        return np.mean(scores)

    def calculate_authenticity_score(self, layout_analysis):
        """Calculate final authenticity score"""
        weights = {
            'ssim': 0.30,
            'hash': 0.25,
            'color': 0.20,
            'border': 0.15,
            'histogram': 0.10
        }

        final_score = (
            weights['ssim'] * layout_analysis['ssim_score'] +
            weights['hash'] * layout_analysis['hash_similarity'] +
            weights['color'] * layout_analysis['color_score'] +
            weights['border'] * layout_analysis['border_score'] +
            weights['histogram'] * layout_analysis['histogram_score']
        )

        return {
            'final_score': final_score,
            'component_scores': layout_analysis
        }

    def create_visualization(self, generated_img, reference_img, results, output_path):
        """Create visualization of verification results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f"Simple Card Verification\nFinal Score: {results['final_score']:.3f}", fontsize=16)

        # Images
        axes[0, 0].imshow(generated_img)
        axes[0, 0].set_title("Generated Card")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(reference_img)
        axes[0, 1].set_title("Reference Card")
        axes[0, 1].axis('off')

        # Difference
        gen_resized = cv2.resize(np.array(generated_img), (512, 512))
        ref_resized = cv2.resize(np.array(reference_img), (512, 512))
        diff = np.abs(gen_resized.astype(float) - ref_resized.astype(float))
        axes[0, 2].imshow(diff.astype(np.uint8))
        axes[0, 2].set_title("Pixel Difference")
        axes[0, 2].axis('off')

        # Scores
        scores = results['component_scores']
        score_names = list(scores.keys())
        score_values = list(scores.values())

        axes[1, 0].barh(score_names, score_values, color='lightblue')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_title('Component Scores')
        axes[1, 0].set_xlim(0, 1)

        # Histograms
        axes[1, 1].hist(gen_resized[:, :, 0].flatten(), bins=50,
                        alpha=0.7, label='Generated R', color='red')
        axes[1, 1].hist(ref_resized[:, :, 0].flatten(), bins=50,
                        alpha=0.7, label='Reference R', color='darkred')
        axes[1, 1].set_title('Red Channel Histogram')
        axes[1, 1].legend()

        # Assessment
        if results['final_score'] > 0.85:
            assessment = "HIGH AUTHENTICITY"
            color = 'green'
        elif results['final_score'] > 0.70:
            assessment = "MODERATE AUTHENTICITY"
            color = 'orange'
        else:
            assessment = "LOW AUTHENTICITY"
            color = 'red'

        axes[1, 2].text(0.5, 0.5, f"{assessment}\n{results['final_score']:.3f}",
                        ha='center', va='center', fontsize=16, color=color, weight='bold')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def verify_card(self, generated_card_path, reference_card_path=None):
        """Verify a single card against its reference"""
        self.logger.info(f"🔍 Verifying card: {generated_card_path}")

        # Parse generated card info
        gen_info = self.parse_filename(generated_card_path)

        # Find reference card if not provided
        if reference_card_path is None:
            reference_card_path = self.find_reference_card(gen_info)
            if reference_card_path is None:
                # Create a synthetic reference using the other file
                self.logger.warning(
                    "No reference card found - using comparative analysis")
                # Use the other Magikarp file if available
                other_file = "input_cards/Migakarp.png" if "to_verify" in generated_card_path else "input_cards/to_verify/migakarp.png"
                if Path(other_file).exists():
                    reference_card_path = other_file
                else:
                    raise FileNotFoundError(
                        f"No reference card found for {generated_card_path}")

        self.logger.info(f"📋 Using reference: {reference_card_path}")

        # Load images
        generated_img = Image.open(generated_card_path).convert('RGB')
        reference_img = Image.open(reference_card_path).convert('RGB')

        # Analyze layout components
        self.logger.info("📐 Analyzing layout components...")
        layout_analysis = self.analyze_layout_components(
            generated_img, reference_img)

        # Calculate authenticity score
        results = self.calculate_authenticity_score(layout_analysis)

        # Create output paths
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        card_id = f"{gen_info['set_name']}_{gen_info['card_number']}_{gen_info['card_name']}"

        # Generate report
        report = {
            'timestamp': timestamp,
            'generated_card': str(generated_card_path),
            'reference_card': str(reference_card_path),
            'card_info': gen_info,
            'authenticity_score': results['final_score'],
            'component_scores': results['component_scores'],
            'assessment': 'AUTHENTIC' if results['final_score'] > 0.85 else
            'SUSPICIOUS' if results['final_score'] > 0.70 else 'LIKELY_FAKE'
        }

        # Save report
        report_path = f"output/reports/simple_verification_{card_id}_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Create visualization
        viz_path = f"output/visualizations/simple_verification_{card_id}_{timestamp}.png"
        self.create_visualization(
            generated_img, reference_img, results, viz_path)

        self.logger.info(
            f"✅ Verification complete. Score: {results['final_score']:.3f}")
        self.logger.info(f"📊 Report saved: {report_path}")
        self.logger.info(f"🖼️  Visualization saved: {viz_path}")

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Simple Pokemon Card Authenticity Verification")
    parser.add_argument("--single", "-s", required=True,
                        help="Path to card to verify")
    parser.add_argument("--reference", "-r",
                        help="Reference card path (optional)")

    args = parser.parse_args()

    # Initialize verifier
    verifier = SimpleCardVerifier()

    # Verify card
    try:
        report = verifier.verify_card(args.single, args.reference)
        print(f"\n🎯 Verification Results:")
        print(f"Authenticity Score: {report['authenticity_score']:.3f}")
        print(f"Assessment: {report['assessment']}")
        print(f"\nComponent Scores:")
        for component, score in report['component_scores'].items():
            print(f"  {component}: {score:.3f}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
