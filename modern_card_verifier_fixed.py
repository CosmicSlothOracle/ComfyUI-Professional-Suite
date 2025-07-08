#!/usr/bin/env python3
"""
Modern Pokemon Card Authenticity Verification System - Production Ready
Fixed version with proper dependency handling
"""

import imagehash
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cosine
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import timm
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import os
import json
import argparse
import configparser
from pathlib import Path
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')


# Try to import transformers, fallback to basic models if unavailable
try:
    from transformers import ViTImageProcessor, ViTForImageClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available, using fallback models")


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


class ModernCardVerifier:
    def __init__(self, config_path="config/verification_config.ini"):
        """Initialize the modern card verification system"""

        # Set up default config if file doesn't exist
        self.setup_default_config()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")

        self.setup_logging()
        self.load_models()
        self.setup_transforms()

    def setup_default_config(self):
        """Setup default configuration"""
        self.config = {
            'image_size': 512,
            'confidence_threshold': 0.85,
            'weights': {
                'vit': 0.30 if TRANSFORMERS_AVAILABLE else 0.0,
                'dino': 0.25 if TRANSFORMERS_AVAILABLE else 0.0,
                'resnet': 0.15,
                'ssim': 0.30,
                'border': 0.10,
                'color': 0.08,
                'hash': 0.07
            }
        }

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = f"logs/modern_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_models(self):
        """Load all required models with fallback handling"""
        self.logger.info("Loading AI models...")

        # Always load ResNet (most reliable)
        try:
            self.resnet_model = timm.create_model(
                'resnet50', pretrained=True, num_classes=0)
            self.resnet_model.to(self.device)
            self.resnet_model.eval()
            self.logger.info("✅ ResNet-50 loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Error loading ResNet: {str(e)}")
            self.resnet_model = None

        # Try to load Vision Transformer models
        if TRANSFORMERS_AVAILABLE:
            try:
                self.vit_processor = ViTImageProcessor.from_pretrained(
                    'google/vit-base-patch16-224')
                self.vit_model = ViTForImageClassification.from_pretrained(
                    'google/vit-base-patch16-224')
                self.vit_model.to(self.device)
                self.vit_model.eval()
                self.logger.info("✅ ViT model loaded successfully")
            except Exception as e:
                self.logger.error(f"❌ Error loading ViT: {str(e)}")
                self.vit_model = None
                self.vit_processor = None
        else:
            self.vit_model = None
            self.vit_processor = None

        self.logger.info("🤖 Model loading complete")

    def setup_transforms(self):
        """Setup image transformation pipelines"""
        self.transform = transforms.Compose([
            transforms.Resize(
                (self.config['image_size'], self.config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        if TRANSFORMERS_AVAILABLE and self.vit_processor:
            self.transform_vit = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ])

    def parse_filename(self, filename):
        """Parse filename to extract card information"""
        name = Path(filename).stem.lower()

        # Handle special cases like Magikarp
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
        set_name = generated_card_info['set_name']
        card_number = generated_card_info['card_number']
        card_name = generated_card_info['card_name']

        # Search patterns
        search_patterns = [
            f"reference_cards/{set_name}/{set_name}_{card_number}_{card_name}_reference.*",
            f"reference_cards/custom/{set_name}_{card_number}_{card_name}_reference.*",
            f"reference_cards/{set_name}/*{card_name}*reference.*",
            f"reference_cards/custom/*{card_name}*reference.*"
        ]

        for pattern in search_patterns:
            matches = list(Path(".").glob(pattern))
            if matches:
                return matches[0]

        return None

    def extract_features(self, image):
        """Extract features using available models"""
        features = {}

        # Preprocess image
        pil_image = image if isinstance(
            image, Image.Image) else Image.fromarray(image)

        # ResNet features (most reliable)
        if self.resnet_model:
            try:
                image_tensor = self.transform(
                    pil_image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    resnet_features = self.resnet_model(
                        image_tensor).cpu().numpy()
                    features['resnet'] = resnet_features.flatten()
            except Exception as e:
                self.logger.warning(
                    f"ResNet feature extraction failed: {str(e)}")

        # ViT features (if available)
        if TRANSFORMERS_AVAILABLE and self.vit_model and self.vit_processor:
            try:
                inputs = self.vit_processor(
                    images=pil_image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    vit_outputs = self.vit_model(
                        **inputs, output_hidden_states=True)
                    vit_features = vit_outputs.hidden_states[-1][:,
                                                                 0, :].cpu().numpy()
                    features['vit'] = vit_features.flatten()
            except Exception as e:
                self.logger.warning(f"ViT feature extraction failed: {str(e)}")

        return features

    def analyze_layout_components(self, generated_img, reference_img):
        """Analyze layout components using traditional CV methods"""
        # Convert to numpy arrays and resize
        gen_np = cv2.resize(np.array(
            generated_img), (self.config['image_size'], self.config['image_size']))
        ref_np = cv2.resize(np.array(
            reference_img), (self.config['image_size'], self.config['image_size']))

        analysis = {}

        # 1. Structural Similarity (SSIM)
        gen_gray = cv2.cvtColor(gen_np, cv2.COLOR_RGB2GRAY)
        ref_gray = cv2.cvtColor(ref_np, cv2.COLOR_RGB2GRAY)
        ssim_score = ssim(gen_gray, ref_gray)
        analysis['ssim_score'] = float(ssim_score)

        # 2. Perceptual hash comparison
        gen_hash = imagehash.phash(Image.fromarray(gen_np))
        ref_hash = imagehash.phash(Image.fromarray(ref_np))
        hash_similarity = 1.0 - (gen_hash - ref_hash) / 64.0
        analysis['hash_similarity'] = float(max(0.0, hash_similarity))

        # 3. Color analysis
        color_score = self.analyze_colors(gen_np, ref_np)
        analysis['color_score'] = float(color_score)

        # 4. Border analysis
        border_score = self.analyze_borders(gen_np, ref_np)
        analysis['border_score'] = float(border_score)

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

    def calculate_authenticity_score(self, features_gen, features_ref, layout_analysis):
        """Calculate final authenticity score using available features"""

        weights = self.config['weights']
        total_weight = 0
        score = 0

        # Deep learning features (if available)
        if 'vit' in features_gen and 'vit' in features_ref:
            vit_similarity = 1.0 - \
                cosine(features_gen['vit'], features_ref['vit'])
            score += weights['vit'] * vit_similarity
            total_weight += weights['vit']

        if 'resnet' in features_gen and 'resnet' in features_ref:
            resnet_similarity = 1.0 - \
                cosine(features_gen['resnet'], features_ref['resnet'])
            score += weights['resnet'] * resnet_similarity
            total_weight += weights['resnet']

        # Traditional CV features (always available)
        score += weights['ssim'] * layout_analysis['ssim_score']
        score += weights['border'] * layout_analysis['border_score']
        score += weights['color'] * layout_analysis['color_score']
        score += weights['hash'] * layout_analysis['hash_similarity']

        total_weight += weights['ssim'] + weights['border'] + \
            weights['color'] + weights['hash']

        # Normalize by available weights
        final_score = score / total_weight if total_weight > 0 else 0

        return {
            'final_score': float(final_score),
            'vit_similarity': 1.0 - cosine(features_gen['vit'], features_ref['vit']) if 'vit' in features_gen else None,
            'resnet_similarity': 1.0 - cosine(features_gen['resnet'], features_ref['resnet']) if 'resnet' in features_gen else None,
            'layout_scores': layout_analysis
        }

    def create_visualization(self, generated_img, reference_img, results, output_path):
        """Create visualization of verification results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f"Modern Card Verification\nAuthenticity Score: {results['final_score']:.3f}", fontsize=16)

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
        scores = []
        if results['vit_similarity'] is not None:
            scores.append(('ViT Similarity', results['vit_similarity']))
        if results['resnet_similarity'] is not None:
            scores.append(('ResNet Similarity', results['resnet_similarity']))

        scores.extend([
            ('SSIM Score', results['layout_scores']['ssim_score']),
            ('Border Score', results['layout_scores']['border_score']),
            ('Color Score', results['layout_scores']['color_score']),
            ('Hash Similarity', results['layout_scores']['hash_similarity'])
        ])

        score_names = [s[0] for s in scores]
        score_values = [s[1] for s in scores]

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
        self.logger.info(f"Verifying card: {generated_card_path}")

        # Parse generated card info
        gen_info = self.parse_filename(generated_card_path)

        # Find reference card if not provided
        if reference_card_path is None:
            reference_card_path = self.find_reference_card(gen_info)
            if reference_card_path is None:
                # Use the other file as reference for comparison
                other_files = [
                    "input_cards/Migakarp.png",
                    "input_cards/to_verify/migakarp.png"
                ]
                for other_file in other_files:
                    if Path(other_file).exists() and str(other_file) != str(generated_card_path):
                        reference_card_path = other_file
                        self.logger.warning(
                            f"Using comparative reference: {reference_card_path}")
                        break

                if reference_card_path is None:
                    raise FileNotFoundError(
                        f"No reference card found for {generated_card_path}")

        self.logger.info(f"Using reference: {reference_card_path}")

        # Load images
        generated_img = Image.open(generated_card_path).convert('RGB')
        reference_img = Image.open(reference_card_path).convert('RGB')

        # Extract features
        self.logger.info("Extracting AI features...")
        features_gen = self.extract_features(generated_img)
        features_ref = self.extract_features(reference_img)

        # Analyze layout components
        self.logger.info("Analyzing layout components...")
        layout_analysis = self.analyze_layout_components(
            generated_img, reference_img)

        # Calculate authenticity score
        results = self.calculate_authenticity_score(
            features_gen, features_ref, layout_analysis)

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
            'component_scores': {
                'vit_similarity': results['vit_similarity'],
                'resnet_similarity': results['resnet_similarity'],
                'ssim_score': results['layout_scores']['ssim_score'],
                'border_score': results['layout_scores']['border_score'],
                'color_score': results['layout_scores']['color_score'],
                'hash_similarity': results['layout_scores']['hash_similarity']
            },
            'assessment': 'AUTHENTIC' if results['final_score'] > 0.85 else
            'SUSPICIOUS' if results['final_score'] > 0.70 else 'LIKELY_FAKE',
            'models_used': {
                'vit_available': results['vit_similarity'] is not None,
                'resnet_available': results['resnet_similarity'] is not None,
                'transformers_available': TRANSFORMERS_AVAILABLE
            }
        }

        # Convert numpy types for JSON serialization
        report = convert_numpy_types(report)

        # Save report
        Path("output/reports").mkdir(parents=True, exist_ok=True)
        report_path = f"output/reports/modern_verification_{card_id}_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Create visualization
        Path("output/visualizations").mkdir(parents=True, exist_ok=True)
        viz_path = f"output/visualizations/modern_verification_{card_id}_{timestamp}.png"
        self.create_visualization(
            generated_img, reference_img, results, viz_path)

        self.logger.info(
            f"Verification complete. Score: {results['final_score']:.3f}")
        self.logger.info(f"Report saved: {report_path}")
        self.logger.info(f"Visualization saved: {viz_path}")

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Modern Pokemon Card Authenticity Verification")
    parser.add_argument("--single", "-s",
                        help="Path to card to verify")
    parser.add_argument("--reference", "-r",
                        help="Reference card path (optional)")

    args = parser.parse_args()

    # Initialize verifier
    verifier = ModernCardVerifier()

    if args.single:
        # Single card verification
        try:
            report = verifier.verify_card(args.single, args.reference)
            print(f"\n🎯 Modern Verification Results:")
            print(f"Authenticity Score: {report['authenticity_score']:.3f}")
            print(f"Assessment: {report['assessment']}")
            print(f"\nModels Used:")
            for model, available in report['models_used'].items():
                status = "✅" if available else "❌"
                print(f"  {status} {model}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    else:
        # Default: verify Magikarp files
        card1 = "input_cards/to_verify/migakarp.png"
        if Path(card1).exists():
            try:
                report = verifier.verify_card(card1)
                print(f"\n🎯 Modern Verification Results:")
                print(
                    f"Authenticity Score: {report['authenticity_score']:.3f}")
                print(f"Assessment: {report['assessment']}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        else:
            print("No default cards found. Use --single to specify a card to verify.")


if __name__ == "__main__":
    main()
