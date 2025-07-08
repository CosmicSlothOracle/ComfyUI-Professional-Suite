#!/usr/bin/env python3
"""
Pokemon Card Authenticity Enhancement - Easy Runner
==================================================

Simple script to enhance your custom Pokemon cards for better authenticity scores.
Uses research-backed 2024-2025 AI enhancement techniques.

Usage:
    python run_enhancement_workflow.py --generated your_card.png --reference official_card.png
"""

import argparse
import sys
import os
from pathlib import Path
import cv2
import numpy as np
from authenticity_enhancement_workflow import AuthenticityEnhancementWorkflow


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'opencv-python', 'numpy', 'pillow', 'torch', 'torchvision',
        'scikit-learn', 'scikit-image'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'pillow':
                import PIL
            elif package == 'scikit-learn':
                import sklearn
            elif package == 'scikit-image':
                import skimage
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   {package}")
        print("\nInstall them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    print("✓ All dependencies are installed!")
    return True


def validate_input_files(generated_path: str, reference_path: str):
    """Validate that input files exist and are valid images"""

    # Check if files exist
    if not os.path.exists(generated_path):
        raise FileNotFoundError(f"Generated card not found: {generated_path}")

    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference card not found: {reference_path}")

    # Check if files are valid images
    try:
        generated_img = cv2.imread(generated_path)
        if generated_img is None:
            raise ValueError(
                f"Cannot read generated card as image: {generated_path}")

        reference_img = cv2.imread(reference_path)
        if reference_img is None:
            raise ValueError(
                f"Cannot read reference card as image: {reference_path}")

        print(
            f"✓ Generated card: {generated_img.shape[1]}x{generated_img.shape[0]} pixels")
        print(
            f"✓ Reference card: {reference_img.shape[1]}x{reference_img.shape[0]} pixels")

    except Exception as e:
        raise ValueError(f"Error validating images: {e}")


def print_enhancement_summary(report: dict):
    """Print a user-friendly summary of the enhancement process"""

    print("\n" + "="*70)
    print("🎨 AUTHENTICITY ENHANCEMENT RESULTS")
    print("="*70)

    # Analysis summary
    analysis = report['analysis']
    print("\n📊 AUTHENTICITY ANALYSIS:")

    structural = analysis['structural_issues']
    print(f"   Structure Quality: {structural['ssim_score']:.3f} (SSIM)")
    print(f"   Edge Similarity: {structural['edge_similarity']:.3f}")
    print(
        f"   Structure Fix Needed: {'Yes' if structural['needs_structure_fix'] else 'No'}")

    color = analysis['color_issues']
    print(f"   Color Similarity: {color['color_similarity']:.3f}")
    print(
        f"   Color Fix Needed: {'Yes' if color['needs_color_fix'] else 'No'}")

    texture = analysis['texture_issues']
    print(f"   Texture Sharpness: {texture['texture_sharpness']:.1f}")
    print(
        f"   Texture Fix Needed: {'Yes' if texture['needs_texture_fix'] else 'No'}")

    layout = analysis['layout_issues']
    print(f"   Layout Similarity: {layout['layout_similarity']:.3f}")
    print(
        f"   Layout Fix Needed: {'Yes' if layout['needs_layout_fix'] else 'No'}")

    # Enhancements applied
    applied = report['enhancements_applied']
    print("\n🔧 ENHANCEMENTS APPLIED:")

    enhancement_names = {
        'progressive_denoising': 'Progressive Denoising (Artifact Removal)',
        'color_palette_transfer': 'Color Palette Transfer',
        'layout_structure_enhancement': 'Layout Structure Enhancement',
        'texture_refinement': 'Texture Detail Refinement',
        'quality_refinement': 'Final Quality Polish'
    }

    for key, applied_status in applied.items():
        name = enhancement_names.get(key, key.replace('_', ' ').title())
        status = "✅ Applied" if applied_status else "⏭️  Skipped (not needed)"
        print(f"   {name}: {status}")

    print(f"\n💾 Enhanced card saved to: {report['output_file']}")
    print(
        f"📄 Full report saved to: {report['output_file'].replace('.png', '_report.json')}")

    print("\n" + "="*70)
    print("🎯 NEXT STEPS:")
    print("1. Run the enhanced card through your verification system")
    print("2. Compare the new authenticity scores")
    print("3. If needed, adjust enhancement parameters and re-run")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Enhance Pokemon card authenticity using AI-powered techniques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enhance a single card
  python run_enhancement_workflow.py --generated my_card.png --reference official_card.png

  # Use specific output directory
  python run_enhancement_workflow.py -g my_card.png -r official_card.png --output enhanced_cards/

  # Batch process multiple cards
  python run_enhancement_workflow.py --batch generated_cards/ --references reference_cards/
        """
    )

    parser.add_argument(
        '-g', '--generated',
        required=False,
        help="Path to the generated/custom card image"
    )

    parser.add_argument(
        '-r', '--reference',
        required=False,
        help="Path to the official reference card image"
    )

    parser.add_argument(
        '--output',
        default="output/enhanced_cards",
        help="Output directory for enhanced cards (default: output/enhanced_cards)"
    )

    parser.add_argument(
        '--check-deps',
        action='store_true',
        help="Check if all dependencies are installed"
    )

    args = parser.parse_args()

    # Check dependencies if requested
    if args.check_deps:
        if check_dependencies():
            print("✅ All dependencies satisfied!")
        else:
            sys.exit(1)
        return

    # Validate required arguments for main processing
    if not args.generated or not args.reference:
        parser.error("Both --generated and --reference are required unless using --check-deps")

    print("🎮 Pokemon Card Authenticity Enhancement Workflow")
    print("📚 Using research-backed 2024-2025 AI techniques")
    print("-" * 50)

    try:
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)

        # Validate input files
        print("\n🔍 Validating input files...")
        validate_input_files(args.generated, args.reference)

        # Initialize and run workflow
        print("\n🚀 Starting enhancement workflow...")
        workflow = AuthenticityEnhancementWorkflow(output_dir=args.output)

        result = workflow.process_card(
            generated_card_path=args.generated,
            reference_card_path=args.reference
        )

        # Display results
        print_enhancement_summary(result)

        # Success message
        print(f"\n🎉 SUCCESS! Your card has been enhanced for better authenticity!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nFor help, run: python run_enhancement_workflow.py --help")
        sys.exit(1)


if __name__ == "__main__":
    main()
