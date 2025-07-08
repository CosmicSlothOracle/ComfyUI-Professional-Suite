#!/usr/bin/env python3
"""
Verify Transparency - Quick check if transparency is preserved
"""

from pathlib import Path
from PIL import Image
import numpy as np


def check_transparency(image_path):
    """Prüfe ob Bild Transparenz hat"""
    try:
        img = Image.open(image_path)

        # Konvertiere zu RGBA
        if img.mode == 'P' and 'transparency' in img.info:
            img = img.convert('RGBA')
        elif img.mode != 'RGBA':
            return False, 0.0

        # Analysiere Alpha-Kanal
        alpha_channel = np.array(img)[:, :, 3]
        transparent_pixels = np.sum(alpha_channel < 128)
        total_pixels = alpha_channel.size
        transparency_ratio = transparent_pixels / total_pixels

        return transparency_ratio > 0.01, transparency_ratio

    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return False, 0.0


def main():
    base_dir = Path("C:/Users/Public/ComfyUI-master")

    # Original file
    original_path = base_dir / "input" / \
        "0ea53a3cdbfdcf14caf1c8cccdb60143_fast_transparent_converted.gif"

    # Result file
    result_path = base_dir / "output" / "transparency_test_fixed" / \
        "transparency_fixed_0ea53a3cdbfdcf14caf1c8cccdb60143_fast_transparent_converted.gif"

    print("🔍 TRANSPARENCY VERIFICATION")
    print("=" * 40)

    # Check original
    if original_path.exists():
        has_orig_trans, orig_ratio = check_transparency(original_path)
        print(f"📁 Original: {original_path.name}")
        print(f"   • Has transparency: {has_orig_trans}")
        print(f"   • Transparency ratio: {orig_ratio:.1%}")
    else:
        print("❌ Original file not found")
        return

    # Check result
    if result_path.exists():
        has_result_trans, result_ratio = check_transparency(result_path)
        print(f"\n📁 Result: {result_path.name}")
        print(f"   • Has transparency: {has_result_trans}")
        print(f"   • Transparency ratio: {result_ratio:.1%}")

        # Compare
        print(f"\n📈 COMPARISON:")
        print(f"   • Original: {orig_ratio:.1%} transparent")
        print(f"   • Result:   {result_ratio:.1%} transparent")

        if has_result_trans and result_ratio > 0.01:
            print(f"\n✅ TRANSPARENCY PRESERVED!")
            print(f"🎉 Ready for full batch processing!")
            return True
        else:
            print(f"\n❌ TRANSPARENCY LOST!")
            return False
    else:
        print("❌ Result file not found")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🚀 START BATCH PROCESSING? The transparency preservation works!")
    else:
        print(f"\n⚠️  Fix needed before batch processing")
