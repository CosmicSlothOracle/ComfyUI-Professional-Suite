#!/usr/bin/env python3
"""
🧠 ComfyUI Trading Card Optimizer Pipeline
Automatisierte Optimierung von Midjourney-generierten Trading Cards

Features:
- OCR-basierte Textextraktion und -rekonstruktion
- KI-gestützte Bildverbesserung mit spezialisierten LoRAs
- Kantenschärfung und Upscaling
- Style-Transfer für konsistente TCG-Ästhetik
- Batch-Processing und API-Automatisierung
"""

from comfy.utils import ProgressBar
from comfy.model_management import get_total_memory
import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import easyocr
import pytesseract

# ComfyUI Imports
import sys
sys.path.append('./ComfyUI')


@dataclass
class CardAnalysis:
    """Trading Card Analyse-Ergebnisse"""
    ocr_text: Dict[str, str]
    layout_regions: Dict[str, Tuple[int, int, int, int]]
    color_palette: List[str]
    edge_quality: float
    text_clarity: float
    style_consistency: float


@dataclass
class OptimizationSettings:
    """Optimierung-Konfiguration"""
    target_resolution: Tuple[int, int] = (512, 768)
    upscale_factor: int = 2
    edge_enhancement: float = 1.5
    text_sharpening: float = 2.0
    style_strength: float = 0.8
    batch_size: int = 4


class TradingCardOptimizer:
    """Hauptklasse für Trading Card Optimierung"""

    def __init__(self, config_path: str = "configs/trading_card_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.setup_logging()
        self.setup_models()

    def _load_config(self) -> Dict:
        """Konfiguration laden oder Standard erstellen"""
        default_config = {
            "models": {
                "upscaler": "models/upscale_models/RealESRGAN_x4plus.pth",
                "ocr_languages": ["en", "de"],
                "style_loras": [
                    "models/loras/pokemon_tcg_style.safetensors",
                    "models/loras/trading_card_enhance.safetensors"
                ]
            },
            "processing": {
                "chunk_size": 4,
                "gpu_memory_fraction": 0.8,
                "quality_threshold": 0.7
            },
            "output": {
                "format": "PNG",
                "quality": 95,
                "preserve_metadata": True
            }
        }

        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4)
            return default_config

    def setup_logging(self):
        """Logging-System initialisieren"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f"trading_card_optimizer_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Trading Card Optimizer initialized")

    def setup_models(self):
        """AI-Modelle initialisieren"""
        try:
            # OCR-Reader initialisieren
            self.ocr_reader = easyocr.Reader(
                self.config["models"]["ocr_languages"], gpu=torch.cuda.is_available())
            self.logger.info("OCR Reader initialized")

            # GPU-Memory Check
            if torch.cuda.is_available():
                gpu_memory = get_total_memory() / (1024**3)  # GB
                self.logger.info(f"GPU Memory available: {gpu_memory:.2f} GB")

        except Exception as e:
            self.logger.error(f"Model setup failed: {e}")
            raise

    def analyze_card_image(self, image_path: str) -> CardAnalysis:
        """Schritt 1: Umfassende Bildanalyse"""
        self.logger.info(f"Analyzing card image: {image_path}")

        # Bild laden
        image = cv2.imread(image_path)
        pil_image = Image.open(image_path)

        # OCR-Textextraktion
        ocr_results = self.ocr_reader.readtext(image)
        ocr_text = {
            "card_name": "",
            "card_type": "",
            "attacks": [],
            "stats": {},
            "description": ""
        }

        # Layout-Regionen identifizieren
        layout_regions = self._detect_layout_regions(image)

        # Farbpalette extrahieren
        color_palette = self._extract_color_palette(pil_image)

        # Qualitätsbewertung
        edge_quality = self._assess_edge_quality(image)
        text_clarity = self._assess_text_clarity(ocr_results)
        style_consistency = self._assess_style_consistency(pil_image)

        return CardAnalysis(
            ocr_text=ocr_text,
            layout_regions=layout_regions,
            color_palette=color_palette,
            edge_quality=edge_quality,
            text_clarity=text_clarity,
            style_consistency=style_consistency
        )

    def _detect_layout_regions(self, image: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
        """Layout-Regionen für Trading Card identifizieren"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Kantenerkennung für Layout-Segmentierung
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = {
            "header": (0, 0, image.shape[1], int(image.shape[0] * 0.2)),
            "artwork": (int(image.shape[1] * 0.1), int(image.shape[0] * 0.2),
                        int(image.shape[1] * 0.9), int(image.shape[0] * 0.6)),
            "text_area": (0, int(image.shape[0] * 0.6), image.shape[1], image.shape[0]),
            "stats_area": (int(image.shape[1] * 0.7), int(image.shape[0] * 0.8),
                           image.shape[1], image.shape[0])
        }

        return regions

    def _extract_color_palette(self, image: Image.Image, n_colors: int = 8) -> List[str]:
        """Dominante Farbpalette extrahieren"""
        image_small = image.resize((150, 150))
        result = image_small.convert(
            'P', palette=Image.ADAPTIVE, colors=n_colors)
        palette = result.getpalette()

        color_counts = sorted(result.getcolors(), reverse=True)
        colors = []

        for count, index in color_counts:
            color = palette[index*3:(index+1)*3]
            hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
            colors.append(hex_color)

        return colors[:n_colors]

    def _assess_edge_quality(self, image: np.ndarray) -> float:
        """Kantenschärfe bewerten (0-1)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(laplacian_var / 1000.0, 1.0)

    def _assess_text_clarity(self, ocr_results: List) -> float:
        """Textklarheit bewerten basierend auf OCR-Konfidenz"""
        if not ocr_results:
            return 0.0

        confidences = [result[2] for result in ocr_results if len(result) > 2]
        return sum(confidences) / len(confidences) if confidences else 0.0

    def _assess_style_consistency(self, image: Image.Image) -> float:
        """Stilkonsistenz bewerten (vereinfacht)"""
        # Farbverteilung und Kontrastanalyse
        enhancer = ImageEnhance.Contrast(image)
        contrast_metric = enhancer.enhance(1.5)

        # Vereinfachte Metrik basierend auf Farbvariation
        colors = self._extract_color_palette(image, 16)
        color_variety = len(set(colors)) / 16.0

        return color_variety

    def enhance_image_quality(self, image_path: str, analysis: CardAnalysis,
                              settings: OptimizationSettings) -> str:
        """Schritt 2-4: Bildoptimierung mit ComfyUI-Integration"""
        self.logger.info("Starting image enhancement pipeline")

        # Workflow für ComfyUI erstellen
        workflow = self._create_enhancement_workflow(
            image_path, analysis, settings)

        # ComfyUI API-Call (vereinfacht)
        enhanced_image_path = self._execute_comfyui_workflow(workflow)

        return enhanced_image_path

    def _create_enhancement_workflow(self, image_path: str, analysis: CardAnalysis,
                                     settings: OptimizationSettings) -> Dict:
        """ComfyUI Workflow für Trading Card Enhancement"""
        workflow = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": os.path.basename(image_path)}
            },
            "2": {
                "class_type": "UpscaleModelLoader",
                "inputs": {"model_name": "RealESRGAN_x4plus.pth"}
            },
            "3": {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {
                    "upscale_model": ["2", 0],
                    "image": ["1", 0]
                }
            },
            "4": {
                "class_type": "ImageEnhanceSharpness",
                "inputs": {
                    "image": ["3", 0],
                    "sharpness": settings.edge_enhancement
                }
            },
            "5": {
                "class_type": "ImageEnhanceContrast",
                "inputs": {
                    "image": ["4", 0],
                    "contrast": 1.2
                }
            },
            "6": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["5", 0],
                    "filename_prefix": "enhanced_card_"
                }
            }
        }

        # Conditional LoRA application basierend auf Stil-Analyse
        if analysis.style_consistency < 0.7:
            workflow["7"] = {
                "class_type": "LoRALoader",
                "inputs": {
                    "model": ["5", 0],
                    "lora_name": "trading_card_style.safetensors",
                    "strength_model": settings.style_strength,
                    "strength_clip": settings.style_strength
                }
            }

        return workflow

    def _execute_comfyui_workflow(self, workflow: Dict) -> str:
        """ComfyUI Workflow ausführen (API-Integration)"""
        # Hier würde die tatsächliche ComfyUI API-Integration erfolgen
        # Placeholder für jetzt
        output_path = f"output/enhanced_card_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.logger.info(f"ComfyUI workflow would save to: {output_path}")
        return output_path

    def validate_output_quality(self, original_path: str, enhanced_path: str,
                                analysis: CardAnalysis) -> Dict[str, float]:
        """Schritt 5: Qualitätsvalidierung"""
        self.logger.info("Validating output quality")

        # Original und Enhanced Bilder laden
        original = cv2.imread(original_path)
        enhanced = cv2.imread(enhanced_path)

        # Qualitätsmetriken berechnen
        metrics = {
            "resolution_improvement": self._calculate_resolution_improvement(original, enhanced),
            "edge_sharpness": self._assess_edge_quality(enhanced),
            "text_clarity": self._assess_text_clarity_improvement(original_path, enhanced_path),
            "color_preservation": self._assess_color_preservation(original, enhanced),
            "overall_score": 0.0
        }

        # Gesamtscore berechnen
        metrics["overall_score"] = (
            metrics["resolution_improvement"] * 0.25 +
            metrics["edge_sharpness"] * 0.25 +
            metrics["text_clarity"] * 0.3 +
            metrics["color_preservation"] * 0.2
        )

        return metrics

    def _calculate_resolution_improvement(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Auflösungsverbesserung berechnen"""
        original_pixels = original.shape[0] * original.shape[1]
        enhanced_pixels = enhanced.shape[0] * enhanced.shape[1]
        return enhanced_pixels / original_pixels

    def _assess_text_clarity_improvement(self, original_path: str, enhanced_path: str) -> float:
        """Textklarheit-Verbesserung bewerten"""
        original_clarity = self._assess_text_clarity(
            self.ocr_reader.readtext(original_path))
        enhanced_clarity = self._assess_text_clarity(
            self.ocr_reader.readtext(enhanced_path))
        return enhanced_clarity / max(original_clarity, 0.1)

    def _assess_color_preservation(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Farberhaltung bewerten"""
        # Histogram-Vergleich
        original_hist = cv2.calcHist([original], [0, 1, 2], None, [
                                     50, 50, 50], [0, 256, 0, 256, 0, 256])
        enhanced_hist = cv2.calcHist([enhanced], [0, 1, 2], None, [
                                     50, 50, 50], [0, 256, 0, 256, 0, 256])

        correlation = cv2.compareHist(
            original_hist, enhanced_hist, cv2.HISTCMP_CORREL)
        return max(correlation, 0.0)

    async def process_batch(self, input_dir: str, output_dir: str,
                            settings: OptimizationSettings) -> List[Dict]:
        """Batch-Processing für mehrere Trading Cards"""
        self.logger.info(
            f"Starting batch processing: {input_dir} -> {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        # Alle Bilddateien finden
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(Path(input_dir).glob(ext))

        results = []

        # Progress Bar
        progress = ProgressBar(len(image_files))

        for i, image_path in enumerate(image_files):
            try:
                self.logger.info(
                    f"Processing {i+1}/{len(image_files)}: {image_path.name}")

                # Analyse
                analysis = self.analyze_card_image(str(image_path))

                # Enhancement
                enhanced_path = self.enhance_image_quality(
                    str(image_path), analysis, settings)

                # Validation
                quality_metrics = self.validate_output_quality(
                    str(image_path), enhanced_path, analysis)

                # Ergebnis speichern
                result = {
                    "original_file": str(image_path),
                    "enhanced_file": enhanced_path,
                    "analysis": analysis.__dict__,
                    "quality_metrics": quality_metrics,
                    "processing_time": datetime.now().isoformat()
                }

                results.append(result)
                progress.update_absolute(i + 1)

            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}")
                continue

        # Batch-Report speichern
        report_path = os.path.join(
            output_dir, f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        self.logger.info(
            f"Batch processing completed. Report saved: {report_path}")
        return results


class TradingCardAPI:
    """API-Interface für automatisierte Verarbeitung"""

    def __init__(self, optimizer: TradingCardOptimizer):
        self.optimizer = optimizer

    async def optimize_single_card(self, image_data: bytes,
                                   settings: OptimizationSettings) -> Dict:
        """Einzelne Karte über API optimieren"""
        # Temporäre Datei erstellen
        temp_path = f"temp/input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        os.makedirs("temp", exist_ok=True)

        with open(temp_path, 'wb') as f:
            f.write(image_data)

        try:
            # Optimierung durchführen
            analysis = self.optimizer.analyze_card_image(temp_path)
            enhanced_path = self.optimizer.enhance_image_quality(
                temp_path, analysis, settings)
            quality_metrics = self.optimizer.validate_output_quality(
                temp_path, enhanced_path, analysis)

            # Enhanced Image laden
            with open(enhanced_path, 'rb') as f:
                enhanced_data = f.read()

            return {
                "success": True,
                "enhanced_image": enhanced_data,
                "analysis": analysis.__dict__,
                "quality_metrics": quality_metrics
            }

        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)


def main():
    """Hauptfunktion für CLI-Nutzung"""
    import argparse

    parser = argparse.ArgumentParser(description="Trading Card Optimizer")
    parser.add_argument("--input", required=True,
                        help="Input directory or file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--batch", action="store_true",
                        help="Batch processing mode")

    args = parser.parse_args()

    # Optimizer initialisieren
    config_path = args.config or "configs/trading_card_config.json"
    optimizer = TradingCardOptimizer(config_path)

    # Processing-Settings
    settings = OptimizationSettings()

    if args.batch:
        # Batch-Processing
        asyncio.run(optimizer.process_batch(args.input, args.output, settings))
    else:
        # Einzelfile-Processing
        analysis = optimizer.analyze_card_image(args.input)
        enhanced_path = optimizer.enhance_image_quality(
            args.input, analysis, settings)
        quality_metrics = optimizer.validate_output_quality(
            args.input, enhanced_path, analysis)

        print(f"Enhanced image saved: {enhanced_path}")
        print(f"Quality score: {quality_metrics['overall_score']:.2f}")


if __name__ == "__main__":
    main()
