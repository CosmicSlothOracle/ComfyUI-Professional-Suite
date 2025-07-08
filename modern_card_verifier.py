#!/usr/bin/env python3
"""
Modern Pokemon Card Authenticity Verification System
Single-card processing with Vision Transformer analysis
"""

import os
import json
import argparse
import configparser
from pathlib import Path
from datetime import datetime
import logging

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import AutoImageProcessor, AutoModel  # Fixed import for DINOv2
import timm
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as ssim
import imagehash


class ModernCardVerifier:
    def __init__(self, config_path="config/verification_config.ini"):
        """Initialize the modern card verification system"""
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")

        self.setup_logging()
        self.load_models()
        self.setup_transforms()

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = f"logs/verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_models(self):
        """Load all required models"""
        self.logger.info("🤖 Loading Vision Transformer models...")

        try:
            # Load ViT-Large
            self.vit_processor = ViTImageProcessor.from_pretrained(
                'models/vision_transformer/vit-large')
            self.vit_model = ViTForImageClassification.from_pretrained(
                'models/vision_transformer/vit-large')
            self.vit_model.to(self.device)
            self.vit_model.eval()

            # Load DINOv2
            self.dino_model = AutoModel.from_pretrained(
                'facebook/dinov2-large')
            self.dino_model.to(self.device)
            self.dino_model.eval()

            # Load ResNet50
            self.resnet_model = timm.create_model(
                'resnet50', pretrained=True, num_classes=0)  # Feature extractor
            self.resnet_model.to(self.device)
            self.resnet_model.eval()

            self.logger.info("✅ All models loaded successfully")

        except Exception as e:
            self.logger.error(f"❌ Error loading models: {str(e)}")
            # Fallback to online download
            self.logger.info("📥 Downloading models from Hugging Face...")
            self.download_models_fallback()

    def download_models_fallback(self):
        """Fallback model download if local models not found"""
        self.vit_processor = ViTImageProcessor.from_pretrained(
            'google/vit-large-patch16-224')
        self.vit_model = ViTForImageClassification.from_pretrained(
            'google/vit-large-patch16-224')
        self.vit_model.to(self.device)
        self.vit_model.eval()

        self.dino_model = AutoModel.from_pretrained(
            'facebook/dinov2-large')
        self.dino_model.to(self.device)
        self.dino_model.eval()

        self.resnet_model = timm.create_model(
            'resnet50', pretrained=True, num_classes=0)
        self.resnet_model.to(self.device)
        self.resnet_model.eval()

    def setup_transforms(self):
        """Setup image transformation pipelines"""
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        self.transform_vit = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    def parse_filename(self, filename):
        """Parse filename to extract card information"""
        # Remove extension
        name = Path(filename).stem

        # Expected format: {set_name}_{card_number}_{card_name}_{type}
        parts = name.split('_')
        if len(parts) >= 4:
            set_name = parts[0]
            card_number = parts[1]
            card_name = '_'.join(parts[2:-1])  # Handle multi-word names
            card_type = parts[-1]  # 'reference' or 'generated'

            return {
                'set_name': set_name,
                'card_number': card_number,
                'card_name': card_name,
                'type': card_type
            }
        return None

    def find_reference_card(self, generated_card_info):
        """Find matching reference card for a generated card"""
        set_name = generated_card_info['set_name']
        card_number = generated_card_info['card_number']
        card_name = generated_card_info['card_name']

        # Search in the specific set directory
        set_dir = Path(f"reference_cards/{set_name}")
        if not set_dir.exists():
            # Try custom directory
            set_dir = Path("reference_cards/custom")

        # Look for exact match
        reference_pattern = f"{set_name}_{card_number}_{card_name}_reference.*"
        for ref_file in set_dir.glob(reference_pattern):
            return ref_file

        # If no exact match, look for same card number in set
        reference_pattern = f"{set_name}_{card_number}_*_reference.*"
        for ref_file in set_dir.glob(reference_pattern):
            return ref_file

        return None

    def extract_features(self, image):
        """Extract features using ensemble of models"""
        features = {}

        # Preprocess image
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # ViT features
        inputs = self.vit_processor(
            images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vit_outputs = self.vit_model(**inputs, output_hidden_states=True)
            # Use last hidden state as features
            # CLS token
            vit_features = vit_outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            features['vit'] = vit_features.flatten()

        # DINOv2 features
        with torch.no_grad():
            dino_outputs = self.dino_model(**inputs)
            # For DINOv2, use the pooler output or last hidden state
            if hasattr(dino_outputs, 'pooler_output') and dino_outputs.pooler_output is not None:
                dino_features = dino_outputs.pooler_output.cpu().numpy()
            else:
                # CLS token
                dino_features = dino_outputs.last_hidden_state[:, 0, :].cpu(
                ).numpy()
            features['dino'] = dino_features.flatten()

        # ResNet features
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            resnet_features = self.resnet_model(image_tensor).cpu().numpy()
            features['resnet'] = resnet_features.flatten()

        return features

    def analyze_layout_components(self, generated_img, reference_img):
        """Analyze specific layout components"""
        # Convert to numpy arrays
        gen_np = np.array(generated_img)
        ref_np = np.array(reference_img)

        # Resize to standard size
        gen_resized = cv2.resize(gen_np, (512, 512))
        ref_resized = cv2.resize(ref_np, (512, 512))

        analysis = {}

        # 1. Border analysis
        border_score = self.analyze_borders(gen_resized, ref_resized)
        analysis['border_score'] = border_score

        # 2. Color analysis
        color_score = self.analyze_colors(gen_resized, ref_resized)
        analysis['color_score'] = color_score

        # 3. Structural similarity
        # Convert to grayscale for SSIM
        gen_gray = cv2.cvtColor(gen_resized, cv2.COLOR_RGB2GRAY)
        ref_gray = cv2.cvtColor(ref_resized, cv2.COLOR_RGB2GRAY)
        ssim_score = ssim(gen_gray, ref_gray, full=True)[0]
        analysis['ssim_score'] = ssim_score

        # 4. Perceptual hash comparison
        gen_hash = imagehash.phash(Image.fromarray(gen_resized))
        ref_hash = imagehash.phash(Image.fromarray(ref_resized))
        hash_similarity = 1.0 - (gen_hash - ref_hash) / 64.0
        analysis['hash_similarity'] = hash_similarity

        return analysis

    def analyze_borders(self, gen_img, ref_img):
        """Analyze border consistency"""
        border_thickness = 20  # pixels

        # Extract borders
        gen_border = self.extract_border_region(gen_img, border_thickness)
        ref_border = self.extract_border_region(ref_img, border_thickness)

        # Compare border similarity
        border_diff = np.mean(
            np.abs(gen_border.astype(float) - ref_border.astype(float)))
        border_score = max(0.0, 1.0 - border_diff / 255.0)

        return border_score

    def extract_border_region(self, img, thickness):
        """Extract border region from image"""
        h, w = img.shape[:2]
        border = np.zeros_like(img)

        # Top and bottom borders
        border[:thickness, :] = img[:thickness, :]
        border[-thickness:, :] = img[-thickness:, :]

        # Left and right borders
        border[:, :thickness] = img[:, :thickness]
        border[:, -thickness:] = img[:, -thickness:]

        return border

    def analyze_colors(self, gen_img, ref_img):
        """Analyze color distribution similarity"""
        # Convert to LAB color space for perceptual comparison
        gen_lab = cv2.cvtColor(gen_img, cv2.COLOR_RGB2LAB)
        ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_RGB2LAB)

        # Calculate histograms
        gen_hist = cv2.calcHist([gen_lab], [0, 1, 2], None, [
                                50, 50, 50], [0, 256, 0, 256, 0, 256])
        ref_hist = cv2.calcHist([ref_lab], [0, 1, 2], None, [
                                50, 50, 50], [0, 256, 0, 256, 0, 256])

        # Normalize histograms
        gen_hist = gen_hist.flatten() / np.sum(gen_hist)
        ref_hist = ref_hist.flatten() / np.sum(ref_hist)

        # Calculate similarity using cosine similarity
        color_similarity = 1.0 - cosine(gen_hist, ref_hist)

        return max(0.0, color_similarity)

    def calculate_authenticity_score(self, features_gen, features_ref, layout_analysis):
        """Calculate final authenticity score using ensemble approach"""

        # Feature similarity scores
        vit_similarity = 1.0 - cosine(features_gen['vit'], features_ref['vit'])
        dino_similarity = 1.0 - \
            cosine(features_gen['dino'], features_ref['dino'])
        resnet_similarity = 1.0 - \
            cosine(features_gen['resnet'], features_ref['resnet'])

        # Weighted combination of all scores
        weights = {
            'vit': 0.30,
            'dino': 0.25,
            'resnet': 0.15,
            'border': 0.10,
            'color': 0.08,
            'ssim': 0.07,
            'hash': 0.05
        }

        final_score = (
            weights['vit'] * vit_similarity +
            weights['dino'] * dino_similarity +
            weights['resnet'] * resnet_similarity +
            weights['border'] * layout_analysis['border_score'] +
            weights['color'] * layout_analysis['color_score'] +
            weights['ssim'] * layout_analysis['ssim_score'] +
            weights['hash'] * layout_analysis['hash_similarity']
        )

        return {
            'final_score': final_score,
            'vit_similarity': vit_similarity,
            'dino_similarity': dino_similarity,
            'resnet_similarity': resnet_similarity,
            'layout_scores': layout_analysis
        }

    def create_visualization(self, generated_img, reference_img, results, output_path):
        """Create visualization of the verification results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f"Card Authenticity Verification\nFinal Score: {results['final_score']:.3f}", fontsize=16)

        # Original images
        axes[0, 0].imshow(generated_img)
        axes[0, 0].set_title("Generated Card")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(reference_img)
        axes[0, 1].set_title("Reference Card")
        axes[0, 1].axis('off')

        # Difference visualization
        gen_resized = cv2.resize(np.array(generated_img), (512, 512))
        ref_resized = cv2.resize(np.array(reference_img), (512, 512))
        diff = np.abs(gen_resized.astype(float) - ref_resized.astype(float))
        axes[0, 2].imshow(diff.astype(np.uint8))
        axes[0, 2].set_title("Pixel Difference")
        axes[0, 2].axis('off')

        # Score breakdown
        scores = [
            ('ViT Similarity', results['vit_similarity']),
            ('DINO Similarity', results['dino_similarity']),
            ('ResNet Similarity', results['resnet_similarity']),
            ('Border Score', results['layout_scores']['border_score']),
            ('Color Score', results['layout_scores']['color_score']),
            ('SSIM Score', results['layout_scores']['ssim_score'])
        ]

        score_names = [s[0] for s in scores]
        score_values = [s[1] for s in scores]

        axes[1, 0].barh(score_names, score_values, color='skyblue')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_title('Component Scores')
        axes[1, 0].set_xlim(0, 1)

        # Color distribution comparison
        gen_colors = gen_resized.reshape(-1, 3)
        ref_colors = ref_resized.reshape(-1, 3)

        axes[1, 1].hist(gen_colors[:, 0], bins=50, alpha=0.7,
                        label='Generated R', color='red')
        axes[1, 1].hist(ref_colors[:, 0], bins=50, alpha=0.7,
                        label='Reference R', color='darkred')
        axes[1, 1].set_title('Red Channel Distribution')
        axes[1, 1].legend()

        # Overall assessment
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
        if not gen_info:
            raise ValueError(f"Invalid filename format: {generated_card_path}")

        # Find reference card if not provided
        if reference_card_path is None:
            reference_card_path = self.find_reference_card(gen_info)
            if reference_card_path is None:
                raise FileNotFoundError(
                    f"No reference card found for {generated_card_path}")

        self.logger.info(f"📋 Using reference: {reference_card_path}")

        # Load images
        generated_img = Image.open(generated_card_path).convert('RGB')
        reference_img = Image.open(reference_card_path).convert('RGB')

        # Extract features
        self.logger.info("🔬 Extracting features...")
        features_gen = self.extract_features(generated_img)
        features_ref = self.extract_features(reference_img)

        # Analyze layout components
        self.logger.info("📐 Analyzing layout components...")
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
                'dino_similarity': results['dino_similarity'],
                'resnet_similarity': results['resnet_similarity'],
                'border_score': results['layout_scores']['border_score'],
                'color_score': results['layout_scores']['color_score'],
                'ssim_score': results['layout_scores']['ssim_score'],
                'hash_similarity': results['layout_scores']['hash_similarity']
            },
            'assessment': 'AUTHENTIC' if results['final_score'] > 0.85 else
            'SUSPICIOUS' if results['final_score'] > 0.70 else 'LIKELY_FAKE'
        }

        # Save report
        report_path = f"output/reports/verification_{card_id}_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Create visualization
        viz_path = f"output/visualizations/verification_{card_id}_{timestamp}.png"
        self.create_visualization(
            generated_img, reference_img, results, viz_path)

        self.logger.info(
            f"✅ Verification complete. Score: {results['final_score']:.3f}")
        self.logger.info(f"📊 Report saved: {report_path}")
        self.logger.info(f"🖼️  Visualization saved: {viz_path}")

        return report

    def verify_batch(self, input_dir):
        """Verify all cards in input directory"""
        input_path = Path(input_dir)
        processed_count = 0

        for card_file in input_path.glob("*_generated.*"):
            try:
                report = self.verify_card(card_file)

                # Move processed card
                processed_dir = Path("input_cards/processed")
                processed_dir.mkdir(exist_ok=True)
                card_file.rename(processed_dir / card_file.name)

                processed_count += 1

            except Exception as e:
                self.logger.error(f"❌ Error processing {card_file}: {str(e)}")

        self.logger.info(
            f"🎉 Batch verification complete. Processed {processed_count} cards.")


def main():
    parser = argparse.ArgumentParser(
        description="Modern Pokemon Card Authenticity Verification")
    parser.add_argument("--input", "-i", default="input_cards/to_verify",
                        help="Input directory containing cards to verify")
    parser.add_argument("--output", "-o", default="output",
                        help="Output directory for results")
    parser.add_argument("--single", "-s",
                        help="Verify a single card (provide path to generated card)")
    parser.add_argument("--reference", "-r",
                        help="Reference card path (for single card verification)")

    args = parser.parse_args()

    # Initialize verifier
    verifier = ModernCardVerifier()

    if args.single:
        # Single card verification
        try:
            report = verifier.verify_card(args.single, args.reference)
            print(f"\n🎯 Verification Results:")
            print(f"Authenticity Score: {report['authenticity_score']:.3f}")
            print(f"Assessment: {report['assessment']}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    else:
        # Batch verification
        verifier.verify_batch(args.input)


if __name__ == "__main__":
    main()
