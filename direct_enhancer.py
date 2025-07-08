#!/usr/bin/env python3
"""
DIRECT CARD ENHANCER
====================
Simple, effective Pokemon card authenticity improvement.
No complex workflows - just results.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import json
from datetime import datetime


class DirectCardEnhancer:
    def __init__(self):
        self.output_dir = "output/enhanced"
        os.makedirs(self.output_dir, exist_ok=True)

    def enhance_card(self, generated_path, reference_path):
        """Direct enhancement - no bullshit, just results"""
        print(f"🎯 ENHANCING: {os.path.basename(generated_path)}")

        # Load images
        generated = cv2.imread(generated_path)
        reference = cv2.imread(reference_path)

        if generated is None or reference is None:
            raise ValueError("Cannot load images")

        # Resize for processing
        target_size = (512, 768)
        generated_resized = cv2.resize(generated, target_size)
        reference_resized = cv2.resize(reference, target_size)

        # STEP 1: COLOR MATCHING
        print("🔄 Step 1: Color matching...")
        enhanced = self._match_colors(generated_resized, reference_resized)

        # STEP 2: STRUCTURE ENHANCEMENT
        print("🔄 Step 2: Structure enhancement...")
        enhanced = self._enhance_structure(enhanced, reference_resized)

        # STEP 3: TEXTURE REFINEMENT
        print("🔄 Step 3: Texture refinement...")
        enhanced = self._refine_texture(enhanced)

        # STEP 4: FINAL POLISH
        print("🔄 Step 4: Final polish...")
        enhanced = self._final_polish(enhanced)

        # Save result
        output_path = os.path.join(
            self.output_dir, f"enhanced_{os.path.basename(generated_path)}")
        cv2.imwrite(output_path, enhanced)

        # Generate report
        report = {
            "original_file": generated_path,
            "reference_file": reference_path,
            "enhanced_file": output_path,
            "timestamp": datetime.now().isoformat(),
            "enhancements": ["color_matching", "structure_enhancement", "texture_refinement", "final_polish"]
        }

        report_path = output_path.replace('.png', '_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ ENHANCED: {output_path}")
        return output_path

    def _match_colors(self, generated, reference):
        """Match color palette from reference to generated"""
        # Convert to LAB color space
        generated_lab = cv2.cvtColor(generated, cv2.COLOR_BGR2LAB)
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)

        # Calculate color statistics
        gen_mean, gen_std = cv2.meanStdDev(generated_lab)
        ref_mean, ref_std = cv2.meanStdDev(reference_lab)

        # Apply color transfer (70% reference, 30% original)
        transfer_strength = 0.7
        enhanced_lab = generated_lab.copy().astype(np.float32)

        for i in range(3):
            enhanced_lab[:, :, i] = (enhanced_lab[:, :, i] - gen_mean[i]) * (ref_std[i] / gen_std[i]) * \
                transfer_strength + \
                ref_mean[i] * transfer_strength + \
                gen_mean[i] * (1 - transfer_strength)

        enhanced_lab = np.clip(enhanced_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def _enhance_structure(self, generated, reference):
        """Enhance structural details"""
        # Edge detection for structure preservation
        gray_gen = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

        # Detect edges
        edges_gen = cv2.Canny(gray_gen, 50, 150)
        edges_ref = cv2.Canny(gray_ref, 50, 150)

        # Create edge mask
        edge_mask = cv2.bitwise_and(edges_gen, edges_ref)
        edge_mask = cv2.dilate(edge_mask, None, iterations=1)

        # Apply structure enhancement
        enhanced = generated.copy().astype(np.float32)

        # Sharpen in edge areas
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(generated, -1, kernel)

        # Blend based on edge mask
        edge_mask_3d = np.stack([edge_mask/255.0] * 3, axis=2)
        enhanced = enhanced + \
            (sharpened.astype(np.float32) - enhanced) * edge_mask_3d * 0.3

        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def _refine_texture(self, image):
        """Refine texture details"""
        # Bilateral filter for edge-preserving smoothing
        smoothed = cv2.bilateralFilter(image, 9, 75, 75)

        # Unsharp mask for detail enhancement
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

        # Blend original, smoothed, and sharpened
        enhanced = cv2.addWeighted(image, 0.6, smoothed, 0.2, 0)
        enhanced = cv2.addWeighted(enhanced, 0.8, unsharp_mask, 0.2, 0)

        return enhanced

    def _final_polish(self, image):
        """Final quality polish"""
        # Convert to PIL for final adjustments
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Subtle contrast enhancement
        contrast_enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = contrast_enhancer.enhance(1.1)

        # Subtle saturation boost
        saturation_enhancer = ImageEnhance.Color(enhanced)
        enhanced = saturation_enhancer.enhance(1.05)

        # Convert back to OpenCV
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)


def main():
    """Main execution"""
    enhancer = DirectCardEnhancer()

    # Test with your Magikarp cards
    try:
        enhanced_path = enhancer.enhance_card(
            "input_cards/to_verify/migakarp.png",
            "input_cards/Migakarp.png"
        )

        print("\n" + "="*50)
        print("🎉 ENHANCEMENT COMPLETE!")
        print("="*50)
        print(f"Enhanced card: {enhanced_path}")
        print(f"Report: {enhanced_path.replace('.png', '_report.json')}")
        print("\nNow test it with your verification system!")

    except Exception as e:
        print(f"❌ ERROR: {e}")


if __name__ == "__main__":
    main()
