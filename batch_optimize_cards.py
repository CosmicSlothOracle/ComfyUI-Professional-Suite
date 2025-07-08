#!/usr/bin/env python3
"""
🎮 Batch Trading Card Optimizer
Praktisches Script für die Optimierung aller Trading Cards im Input-Verzeichnis
"""

import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import sys

# Image Processing
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch


def setup_logging():
    """Logging konfigurieren"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                log_dir / f"batch_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class TradingCardOptimizer:
    def __init__(self):
        # Configure logging with proper encoding for Windows
        self.setup_logging()

        # Setup output directories
        self.setup_directories()

        # TCG-specific enhancement parameters based on reference cards
        self.tcg_params = {
            # Authentic Pokemon yellow
            'pokemon_yellow_border': [255, 232, 89],
            'text_contrast_boost': 1.4,
            'border_sharpness': 2.0,
            'artwork_saturation': 1.15,
            'shadow_enhancement': 0.3,
            'card_dimensions': (625, 875)  # Standard TCG proportions
        }

        self.enhancement_params = {
            'upscale_factor': 2,
            'contrast_factor': 1.3,  # Increased for better text clarity
            'sharpness_factor': 1.8,  # Enhanced for crisp linework
            'color_factor': 1.2,     # Improved color vibrancy
            'unsharp_radius': 2,
            'unsharp_percent': 150,
            'unsharp_threshold': 3
        }

    def setup_logging(self):
        """Setup logging with Windows-compatible encoding"""
        # Create logs directory
        os.makedirs("logs", exist_ok=True)

        # Setup file handler with UTF-8 encoding
        log_filename = f"logs/tcg_optimizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()  # Console output without emojis
            ]
        )
        self.logger = logging.getLogger(__name__)

    def detect_tcg_elements(self, image):
        """Detect and analyze TCG-specific elements"""
        height, width = image.shape[:2]

        # Define regions based on standard TCG layout
        regions = {
            'border': {'top': 0, 'bottom': height, 'left': 0, 'right': width},
            'artwork': {'top': int(height*0.15), 'bottom': int(height*0.58),
                        'left': int(width*0.08), 'right': int(width*0.92)},
            'text_area': {'top': int(height*0.60), 'bottom': int(height*0.85),
                          'left': int(width*0.08), 'right': int(width*0.92)},
            'stats': {'top': int(height*0.85), 'bottom': height,
                      'left': int(width*0.08), 'right': int(width*0.92)}
        }

        # Analyze color distribution
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Detect dominant colors in border area
        border_region = image[0:50, 0:width]  # Top border sample
        border_colors = self.get_dominant_colors(border_region, 3)

        return regions, border_colors

    def get_dominant_colors(self, image_region, k=3):
        """Extract dominant colors from image region"""
        data = image_region.reshape((-1, 3))
        data = np.float32(data)

        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        return centers.astype(int)

    def enhance_tcg_specific(self, image):
        """Apply TCG-specific enhancements"""
        # Convert to PIL for advanced operations
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Detect card elements
        regions, border_colors = self.detect_tcg_elements(image)

        # Step 1: Enhance border definition
        pil_image = self.enhance_border_clarity(pil_image)

        # Step 2: Boost artwork colors while preserving authenticity
        pil_image = self.enhance_artwork_region(pil_image, regions['artwork'])

        # Step 3: Improve text legibility
        pil_image = self.enhance_text_clarity(pil_image, regions['text_area'])

        # Step 4: Sharpen linework
        pil_image = self.enhance_linework(pil_image)

        # Step 5: Preserve authentic TCG color palette
        pil_image = self.preserve_tcg_colors(pil_image, border_colors)

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def enhance_border_clarity(self, pil_image):
        """Enhance border definition for authentic TCG look"""
        # Apply edge enhancement
        edge_filter = ImageFilter.UnsharpMask(
            radius=1.5,
            percent=200,
            threshold=2
        )
        return pil_image.filter(edge_filter)

    def enhance_artwork_region(self, pil_image, artwork_region):
        """Enhance artwork while preserving style"""
        # Selective color enhancement
        enhancer = ImageEnhance.Color(pil_image)
        enhanced = enhancer.enhance(self.tcg_params['artwork_saturation'])

        # Improve contrast in artwork area
        contrast_enhancer = ImageEnhance.Contrast(enhanced)
        return contrast_enhancer.enhance(1.25)

    def enhance_text_clarity(self, pil_image, text_region):
        """Improve text readability"""
        # Boost contrast for better text visibility
        contrast_enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = contrast_enhancer.enhance(
            self.tcg_params['text_contrast_boost'])

        # Sharpen text areas
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
        return sharpness_enhancer.enhance(1.6)

    def enhance_linework(self, pil_image):
        """Enhance linework and details"""
        # Apply specialized sharpening for line art
        detail_filter = ImageFilter.UnsharpMask(
            radius=self.enhancement_params['unsharp_radius'],
            percent=self.enhancement_params['unsharp_percent'],
            threshold=self.enhancement_params['unsharp_threshold']
        )
        return pil_image.filter(detail_filter)

    def preserve_tcg_colors(self, pil_image, dominant_colors):
        """Preserve authentic TCG color schemes"""
        # Convert to numpy for color processing
        img_array = np.array(pil_image)

        # Enhance yellow tones (Pokemon card borders)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # Boost yellow saturation for authentic Pokemon look
        yellow_mask = cv2.inRange(hsv, np.array(
            [20, 100, 100]), np.array([30, 255, 255]))
        hsv[yellow_mask > 0, 1] = np.minimum(
            hsv[yellow_mask > 0, 1] * 1.15, 255)

        enhanced_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(enhanced_rgb)

    def enhance_image_basic(self, image_path):
        """Enhanced processing with TCG-specific optimizations"""
        try:
            self.logger.info(f"Processing: {os.path.basename(image_path)}")

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            original_quality = self.calculate_quality_score(image)

            # Apply TCG-specific enhancements
            enhanced_image = self.enhance_tcg_specific(image)

            # Apply additional traditional enhancements
            enhanced_image = self.apply_traditional_enhancements(
                enhanced_image)

            # Calculate quality improvement
            final_quality = self.calculate_quality_score(enhanced_image)

            # Save enhanced image
            output_filename = f"enhanced_{os.path.splitext(os.path.basename(image_path))[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            output_path = os.path.join("output/enhanced", output_filename)

            cv2.imwrite(output_path, enhanced_image)

            self.logger.info(
                f"[SUCCESS] Enhanced {os.path.basename(image_path)} -> {output_filename} (Quality: {final_quality:.2f})")

            return {
                'success': True,
                'input_file': os.path.basename(image_path),
                'output_file': output_filename,
                'original_quality': original_quality,
                'enhanced_quality': final_quality,
                'improvement': final_quality - original_quality
            }

        except Exception as e:
            self.logger.error(
                f"[ERROR] Failed to process {os.path.basename(image_path)}: {str(e)}")
            return {
                'success': False,
                'input_file': os.path.basename(image_path),
                'error': str(e)
            }

    def process_batch(self, input_directory):
        """Process all images in directory with improved progress tracking"""

        # Get all image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(input_directory)
                       if f.lower().endswith(image_extensions)]

        if not image_files:
            self.logger.warning("No image files found in input directory")
            return None

        self.logger.info(
            f"Starting batch processing of {len(image_files)} images")
        self.logger.info("=" * 60)

        results = []
        successful = 0
        failed = 0
        start_time = time.time()

        for i, filename in enumerate(image_files, 1):
            self.logger.info(f"\n--- Processing {i}/{len(image_files)} ---")

            image_path = os.path.join(input_directory, filename)
            result = self.enhance_image_basic(image_path)

            results.append(result)

            if result['success']:
                successful += 1
            else:
                failed += 1

            # Progress update
            progress = (i / len(image_files)) * 100
            self.logger.info(
                f"Progress: {progress:.1f}% ({successful} success, {failed} failed)")

        # Final summary with ASCII characters
        total_time = time.time() - start_time

        self.logger.info("\n" + "=" * 60)
        self.logger.info("BATCH PROCESSING COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Images: {len(image_files)}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Total Time: {total_time:.1f}s")

        # Calculate average quality
        successful_results = [r for r in results if r['success']]
        if successful_results:
            avg_quality = sum(r['enhanced_quality']
                              for r in successful_results) / len(successful_results)
            self.logger.info(f"Avg Quality Score: {avg_quality:.2f}")

        # Save detailed report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(image_files),
            'successful': successful,
            'failed': failed,
            'processing_time_seconds': total_time,
            'average_quality': avg_quality if successful_results else 0,
            'results': results
        }

        report_path = os.path.join(
            "output/reports", f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Report saved: {report_path}")
        self.logger.info(f"Enhanced images: output/enhanced/")
        self.logger.info("=" * 60)

        return report_data

    def apply_traditional_enhancements(self, image):
        """Apply traditional image enhancement techniques"""
        # Convert to PIL for enhancement operations
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Upscaling
        original_size = pil_image.size
        new_width = int(original_size[0] *
                        self.enhancement_params['upscale_factor'])
        new_height = int(
            original_size[1] * self.enhancement_params['upscale_factor'])
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

        # Contrast enhancement
        contrast_enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = contrast_enhancer.enhance(
            self.enhancement_params['contrast_factor'])

        # Sharpness enhancement
        sharpness_enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = sharpness_enhancer.enhance(
            self.enhancement_params['sharpness_factor'])

        # Color enhancement
        color_enhancer = ImageEnhance.Color(pil_image)
        pil_image = color_enhancer.enhance(
            self.enhancement_params['color_factor'])

        # Final detail enhancement
        unsharp_filter = ImageFilter.UnsharpMask(
            radius=self.enhancement_params['unsharp_radius'],
            percent=self.enhancement_params['unsharp_percent'],
            threshold=self.enhancement_params['unsharp_threshold']
        )
        pil_image = pil_image.filter(unsharp_filter)

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def calculate_quality_score(self, image):
        """Calculate quality score based on edge sharpness and contrast"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calculate edge sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Normalize edge score (typical range 0-2000, normalize to 0-1)
            edge_score = min(laplacian_var / 1000.0, 1.0)

            # Calculate contrast using standard deviation
            contrast_score = min(np.std(gray) / 80.0, 1.0)

            # Calculate brightness distribution
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist.ravel() / hist.sum()
            brightness_score = 1.0 - abs(0.5 - np.mean(gray) / 255.0)

            # Weighted quality score
            quality_score = (
                edge_score * 0.4 +
                contrast_score * 0.4 +
                brightness_score * 0.2
            )

            return min(quality_score, 1.0)

        except Exception as e:
            self.logger.warning(f"Quality calculation failed: {e}")
            return 0.5

    def setup_directories(self):
        """Create necessary output directories"""
        directories = [
            "output/enhanced",
            "output/analysis",
            "output/reports",
            "logs"
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        self.logger.info("Output directories created")


def advanced_enhance_with_opencv(image_path: Path) -> Path:
    """Erweiterte OpenCV-basierte Verbesserung"""
    logger = logging.getLogger(__name__)

    try:
        # Bild laden
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("Could not load image")

        # 1. Noise Reduction
        denoised = cv2.bilateralFilter(img, 9, 75, 75)

        # 2. Sharpening Kernel
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # 3. Kontrast mit CLAHE
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # 4. Upscaling mit OpenCV
        height, width = enhanced.shape[:2]
        enhanced = cv2.resize(enhanced, (width*2, height*2),
                              interpolation=cv2.INTER_CUBIC)

        # Speichern
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"opencv_enhanced_{image_path.stem}_{timestamp}.png"
        output_path = Path("output/enhanced") / output_filename

        cv2.imwrite(str(output_path), enhanced)
        logger.info(f"OpenCV enhanced: {output_filename}")

        return output_path

    except Exception as e:
        logger.error(f"OpenCV enhancement failed for {image_path.name}: {e}")
        return image_path


def main():
    """Hauptfunktion"""
    print("TRADING CARD BATCH OPTIMIZER")
    print("=" * 50)

    # Logging setup
    logger = setup_logging()

    # GPU Check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(
            0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.info("Using CPU processing")

    # System info
    import psutil
    logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info(f"CPU: {psutil.cpu_count()} cores")

    # Optimizer initialisieren
    optimizer = TradingCardOptimizer()

    # Batch-Processing starten
    try:
        batch_report = optimizer.process_batch("input")

        if "error" not in batch_report:
            print(
                f"\nSUCCESS! Processed {batch_report['successful']}/{batch_report['total_images']} images")
            print(
                f"Average Quality Score: {batch_report['average_quality']:.2f}")
            print(
                f"Total Time: {batch_report['processing_time_seconds']:.1f} seconds")
            print(f"Results in: output/enhanced/")
        else:
            print(f"Batch processing failed: {batch_report['error']}")

    except Exception as e:
        logger.error(f"Critical error: {e}")
        print(f"Processing failed: {e}")


if __name__ == "__main__":
    main()
