#!/usr/bin/env python3
"""
🔧 Setup Script for Trading Card Authenticity Analyzer
Installs and configures all required AI models and dependencies
"""

import subprocess
import sys
import os
import logging
from pathlib import Path


def setup_logging():
    """Setup logging"""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


def install_package(package_name, logger):
    """Install a Python package via pip"""
    try:
        logger.info(f"Installing {package_name}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name])
        logger.info(f"✓ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to install {package_name}: {e}")
        return False


def verify_installation(package_name, logger):
    """Verify package installation"""
    try:
        __import__(package_name)
        logger.info(f"✓ {package_name} verified")
        return True
    except ImportError:
        logger.error(f"✗ {package_name} not available")
        return False


def main():
    """Main setup function"""
    logger = setup_logging()

    print("=" * 60)
    print("TRADING CARD AUTHENTICITY ANALYZER - SETUP")
    print("=" * 60)
    print("Installing AI models and dependencies...")
    print()

    # Required packages with AI model support
    packages = [
        # Core AI frameworks
        "torch>=2.0.0",
        "torchvision",
        "transformers>=4.21.0",

        # Hugging Face models
        "sentence-transformers",
        "datasets",

        # Computer Vision
        "opencv-python",
        "Pillow>=9.0.0",
        "easyocr",

        # Scientific computing
        "numpy>=1.21.0",
        "scikit-learn",

        # Utilities
        "tqdm",
        "requests"
    ]

    success_count = 0
    total_packages = len(packages)

    # Install packages
    for package in packages:
        if install_package(package, logger):
            success_count += 1
        print()

    # Summary
    print("=" * 60)
    print(
        f"INSTALLATION SUMMARY: {success_count}/{total_packages} packages installed")
    print("=" * 60)

    if success_count == total_packages:
        print("✓ All dependencies installed successfully!")

        # Test AI model loading
        print("\nTesting AI model access...")
        test_model_access(logger)

    else:
        print(f"✗ {total_packages - success_count} packages failed to install")
        print("Please check the error messages above and install manually if needed")


def test_model_access(logger):
    """Test if AI models can be accessed"""

    tests = [
        ("transformers", "Hugging Face Transformers"),
        ("sentence_transformers", "SentenceTransformers"),
        ("cv2", "OpenCV"),
        ("torch", "PyTorch"),
        ("easyocr", "EasyOCR")
    ]

    print("\nModel Access Tests:")
    print("-" * 30)

    for module, name in tests:
        if verify_installation(module, logger):
            print(f"✓ {name} - Ready")
        else:
            print(f"✗ {name} - Not available")

    # Test specific model loading
    print("\nAdvanced Model Tests:")
    print("-" * 30)

    try:
        from transformers import TrOCRProcessor
        processor = TrOCRProcessor.from_pretrained(
            'microsoft/trocr-base-printed')
        print("✓ TrOCR Model - Loaded successfully")
    except Exception as e:
        print(f"✗ TrOCR Model - Failed: {str(e)[:50]}...")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ SentenceTransformer - Loaded successfully")
    except Exception as e:
        print(f"✗ SentenceTransformer - Failed: {str(e)[:50]}...")

    try:
        from transformers import pipeline
        captioner = pipeline(
            "image-to-text", model="Salesforce/blip-image-captioning-base")
        print("✓ BLIP Captioning - Loaded successfully")
    except Exception as e:
        print(f"✗ BLIP Captioning - Failed: {str(e)[:50]}...")

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("You can now run: python trading_card_authenticity_analyzer.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
