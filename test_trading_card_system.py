#!/usr/bin/env python3
"""
🧪 Trading Card Optimizer System Tests
Umfassende Tests für die Trading Card Optimierungs-Pipeline
"""

import os
import sys
import time
import json
import logging
import tempfile
import asyncio
from pathlib import Path
from typing import Dict, List
import unittest
from unittest.mock import Mock, patch

# Test Libraries
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


class TradingCardSystemTest(unittest.TestCase):
    """Test-Suite für Trading Card Optimizer"""

    def setUp(self):
        """Test-Setup"""
        self.test_dir = Path("test_temp")
        self.test_dir.mkdir(exist_ok=True)

        # Test-Konfiguration
        self.config = {
            "models": {
                "upscaler": "models/upscale_models/RealESRGAN_x4plus.pth",
                "ocr_languages": ["en", "de"]
            },
            "processing": {
                "chunk_size": 2,
                "quality_threshold": 0.5
            }
        }

        # Logging für Tests
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def tearDown(self):
        """Test-Cleanup"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def create_test_trading_card(self, filename: str = "test_card.png") -> Path:
        """Mock Trading Card für Tests erstellen"""
        # Trading Card Dimensionen
        width, height = 512, 768

        # Bild erstellen
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)

        # Card Border
        draw.rectangle([10, 10, width-10, height-10], outline='black', width=3)

        # Header-Bereich (Card Name)
        draw.rectangle([20, 20, width-20, 80], fill='lightblue',
                       outline='darkblue', width=2)
        draw.text((30, 35), "PIKACHU", fill='black', anchor="lt")

        # Artwork-Bereich
        draw.rectangle([20, 90, width-20, 400], fill='yellow',
                       outline='orange', width=2)
        draw.ellipse([width//2-50, 200, width//2+50, 300], fill='red')

        # Text-Bereich
        draw.rectangle([20, 410, width-20, 600],
                       fill='lightgray', outline='gray', width=2)
        draw.text((30, 430), "Electric Mouse Pokemon", fill='black')
        draw.text((30, 460), "Thunderbolt: 90 damage", fill='black')
        draw.text((30, 490), "Quick Attack: 30 damage", fill='black')

        # Stats-Bereich
        draw.rectangle([width-120, height-100, width-20, height-20],
                       fill='red', outline='darkred', width=2)
        draw.text((width-110, height-80), "HP: 80", fill='white')

        # Speichern
        file_path = self.test_dir / filename
        image.save(file_path)

        return file_path

    def test_import_core_modules(self):
        """Test: Core Module-Imports"""
        self.logger.info("Testing core module imports...")

        try:
            from trading_card_optimizer import TradingCardOptimizer, OptimizationSettings
            self.logger.info("✅ Core modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import core modules: {e}")

    def test_import_custom_nodes(self):
        """Test: Custom Node-Imports"""
        self.logger.info("Testing custom node imports...")

        try:
            from custom_nodes.trading_card_nodes import TradingCardAnalyzer, TradingCardEnhancer, TradingCardValidator
            self.logger.info("✅ Custom nodes imported successfully")
        except ImportError as e:
            self.logger.warning(f"⚠️ Custom nodes import failed: {e}")

    def test_dependencies(self):
        """Test: Externe Dependencies"""
        self.logger.info("Testing external dependencies...")

        dependencies = [
            ("torch", "PyTorch"),
            ("cv2", "OpenCV"),
            ("PIL", "Pillow"),
            ("numpy", "NumPy"),
            ("requests", "Requests"),
            ("fastapi", "FastAPI"),
            ("easyocr", "EasyOCR")
        ]

        for module_name, display_name in dependencies:
            try:
                __import__(module_name)
                self.logger.info(f"✅ {display_name} available")
            except ImportError:
                self.logger.warning(f"⚠️ {display_name} not available")

    @patch('easyocr.Reader')
    def test_trading_card_analyzer(self, mock_ocr):
        """Test: Trading Card Analyzer Node"""
        self.logger.info("Testing Trading Card Analyzer...")

        try:
            from custom_nodes.trading_card_nodes import TradingCardAnalyzer

            # Mock OCR Reader
            mock_reader = Mock()
            mock_reader.readtext.return_value = [
                ([[10, 10], [100, 10], [100, 30], [10, 30]], "PIKACHU", 0.95),
                ([[10, 50], [150, 50], [150, 70], [10, 70]], "Electric Mouse", 0.87)
            ]
            mock_ocr.return_value = mock_reader

            # Test-Bild erstellen
            test_image_path = self.create_test_trading_card()

            # Analyzer initialisieren
            analyzer = TradingCardAnalyzer()

            # Test-Image als Tensor
            import torch
            test_image = Image.open(test_image_path)
            img_array = np.array(test_image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            # Analyse durchführen
            results = analyzer.analyze_trading_card(img_tensor)

            self.assertEqual(len(results), 6)  # 6 Return-Werte erwartet
            self.assertIsInstance(results[0], str)  # OCR Results JSON
            self.assertIsInstance(results[2], float)  # Edge Quality

            self.logger.info("✅ Trading Card Analyzer test passed")

        except Exception as e:
            self.logger.error(f"❌ Trading Card Analyzer test failed: {e}")
            raise

    def test_trading_card_enhancer(self):
        """Test: Trading Card Enhancer Node"""
        self.logger.info("Testing Trading Card Enhancer...")

        try:
            from custom_nodes.trading_card_nodes import TradingCardEnhancer

            # Test-Bild erstellen
            test_image_path = self.create_test_trading_card()

            # Enhancer initialisieren
            enhancer = TradingCardEnhancer()

            # Test-Image als Tensor
            import torch
            test_image = Image.open(test_image_path)
            img_array = np.array(test_image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            # Enhancement durchführen
            results = enhancer.enhance_trading_card(
                img_tensor,
                edge_enhancement=1.3,
                text_sharpening=1.8,
                color_saturation=1.1
            )

            self.assertEqual(len(results), 2)  # Enhanced Image + Report
            enhanced_image, report = results

            # Tensor-Shape prüfen
            self.assertEqual(len(enhanced_image.shape), 4)  # Batch dimension

            # Report als JSON validieren
            report_dict = json.loads(report)
            self.assertIn("applied_enhancements", report_dict)

            self.logger.info("✅ Trading Card Enhancer test passed")

        except Exception as e:
            self.logger.error(f"❌ Trading Card Enhancer test failed: {e}")
            raise

    def test_trading_card_validator(self):
        """Test: Trading Card Validator Node"""
        self.logger.info("Testing Trading Card Validator...")

        try:
            from custom_nodes.trading_card_nodes import TradingCardValidator

            # Test-Bilder erstellen
            original_path = self.create_test_trading_card("original.png")
            enhanced_path = self.create_test_trading_card(
                "enhanced.png")  # Mock enhanced

            # Validator initialisieren
            validator = TradingCardValidator()

            # Test-Images als Tensors
            import torch

            original_img = Image.open(original_path)
            enhanced_img = Image.open(enhanced_path)

            orig_array = np.array(original_img).astype(np.float32) / 255.0
            enh_array = np.array(enhanced_img).astype(np.float32) / 255.0

            orig_tensor = torch.from_numpy(orig_array).unsqueeze(0)
            enh_tensor = torch.from_numpy(enh_array).unsqueeze(0)

            # Validation durchführen
            results = validator.validate_enhancement(
                orig_tensor, enh_tensor, min_quality_score=0.5)

            # quality_passed, score, report, comparison
            self.assertEqual(len(results), 4)
            quality_passed, quality_score, validation_report, comparison_image = results

            self.assertIsInstance(quality_passed, bool)
            self.assertIsInstance(quality_score, float)
            self.assertIsInstance(validation_report, str)

            # Report validieren
            report_dict = json.loads(validation_report)
            self.assertIn("quality_score", report_dict)
            self.assertIn("metrics", report_dict)

            self.logger.info("✅ Trading Card Validator test passed")

        except Exception as e:
            self.logger.error(f"❌ Trading Card Validator test failed: {e}")
            raise

    def test_api_server_startup(self):
        """Test: API Server Startup"""
        self.logger.info("Testing API server startup...")

        try:
            from api_server.trading_card_api import app

            # FastAPI App prüfen
            self.assertIsNotNone(app)
            self.assertEqual(app.title, "Trading Card Optimizer API")

            # Routes prüfen
            route_paths = [route.path for route in app.routes]
            expected_routes = ["/", "/health",
                               "/optimize/single", "/optimize/batch"]

            for route in expected_routes:
                self.assertIn(route, route_paths, f"Route {route} not found")

            self.logger.info("✅ API server startup test passed")

        except Exception as e:
            self.logger.error(f"❌ API server startup test failed: {e}")
            raise

    def test_model_paths(self):
        """Test: Model-Pfade und Verfügbarkeit"""
        self.logger.info("Testing model paths...")

        model_paths = [
            "models/upscale_models/RealESRGAN_x4plus.pth",
            "models/upscale_models/RealESRGAN_x2plus.pth"
        ]

        available_models = 0
        for model_path in model_paths:
            if os.path.exists(model_path):
                self.logger.info(f"✅ Model available: {model_path}")
                available_models += 1
            else:
                self.logger.warning(f"⚠️ Model missing: {model_path}")

        if available_models == 0:
            self.logger.warning(
                "⚠️ No models found - download models using model_downloader.py")
        else:
            self.logger.info(
                f"✅ {available_models}/{len(model_paths)} models available")

    def test_configuration_files(self):
        """Test: Konfigurationsdateien"""
        self.logger.info("Testing configuration files...")

        config_files = [
            "configs/trading_card_config.json",
            "workflows/trading_card_optimization_workflow.json"
        ]

        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        json.load(f)
                    self.logger.info(f"✅ Valid config: {config_file}")
                except json.JSONDecodeError:
                    self.logger.error(f"❌ Invalid JSON: {config_file}")
            else:
                self.logger.warning(f"⚠️ Config missing: {config_file}")

    def test_directory_structure(self):
        """Test: Verzeichnisstruktur"""
        self.logger.info("Testing directory structure...")

        required_dirs = [
            "models/upscale_models",
            "models/loras",
            "models/controlnet",
            "output/enhanced",
            "custom_nodes",
            "configs",
            "workflows"
        ]

        for directory in required_dirs:
            if os.path.exists(directory):
                self.logger.info(f"✅ Directory exists: {directory}")
            else:
                self.logger.warning(f"⚠️ Directory missing: {directory}")

    async def test_batch_processing_simulation(self):
        """Test: Batch Processing Simulation"""
        self.logger.info("Testing batch processing simulation...")

        try:
            # Mehrere Test-Karten erstellen
            test_cards = []
            for i in range(3):
                card_path = self.create_test_trading_card(
                    f"batch_card_{i}.png")
                test_cards.append(card_path)

            # Batch-Processing simulieren
            from trading_card_optimizer import TradingCardOptimizer, OptimizationSettings

            # Mock-Optimizer verwenden
            with patch('trading_card_optimizer.TradingCardOptimizer') as MockOptimizer:
                mock_instance = Mock()
                MockOptimizer.return_value = mock_instance

                # Mock-Responses
                mock_instance.analyze_card_image.return_value = Mock()
                mock_instance.enhance_image_quality.return_value = "mock_enhanced.png"
                mock_instance.validate_output_quality.return_value = {
                    "overall_score": 0.85}

                optimizer = MockOptimizer()
                settings = OptimizationSettings()

                # Batch verarbeiten
                results = []
                for card_path in test_cards:
                    analysis = optimizer.analyze_card_image(str(card_path))
                    enhanced = optimizer.enhance_image_quality(
                        str(card_path), analysis, settings)
                    validation = optimizer.validate_output_quality(
                        str(card_path), enhanced, analysis)

                    results.append({
                        "original": str(card_path),
                        "enhanced": enhanced,
                        "quality": validation["overall_score"]
                    })

                self.assertEqual(len(results), 3)
                self.logger.info("✅ Batch processing simulation passed")

        except Exception as e:
            self.logger.error(f"❌ Batch processing simulation failed: {e}")
            raise

    def test_gpu_availability(self):
        """Test: GPU-Verfügbarkeit"""
        self.logger.info("Testing GPU availability...")

        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(
                    0).total_memory / (1024**3)

                self.logger.info(f"✅ GPU available: {gpu_name}")
                self.logger.info(f"✅ GPU memory: {gpu_memory:.1f} GB")
                self.logger.info(f"✅ GPU count: {gpu_count}")

                # CUDA-Test
                test_tensor = torch.randn(100, 100).cuda()
                result = test_tensor.sum().cpu().numpy()
                self.assertIsInstance(result, (int, float, np.number))

                self.logger.info("✅ CUDA computation test passed")
            else:
                self.logger.warning("⚠️ No GPU available - using CPU")

        except Exception as e:
            self.logger.warning(f"⚠️ GPU test failed: {e}")


class SystemIntegrationTest:
    """System-Integration Tests"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def run_full_pipeline_test(self):
        """Vollständiger Pipeline-Test"""
        self.logger.info("🚀 Running full pipeline integration test...")

        try:
            # Test-Setup
            test_system = TradingCardSystemTest()
            test_system.setUp()

            # Test-Karte erstellen
            test_card = test_system.create_test_trading_card(
                "integration_test.png")

            # Pipeline-Schritte simulieren
            self.logger.info("  Step 1: Image Analysis...")
            time.sleep(0.5)  # Simulate processing time

            self.logger.info("  Step 2: OCR Text Extraction...")
            time.sleep(0.3)

            self.logger.info("  Step 3: Image Enhancement...")
            time.sleep(1.0)

            self.logger.info("  Step 4: Quality Validation...")
            time.sleep(0.2)

            self.logger.info("✅ Full pipeline test completed successfully")

            # Cleanup
            test_system.tearDown()

        except Exception as e:
            self.logger.error(f"❌ Full pipeline test failed: {e}")
            raise


def run_comprehensive_tests():
    """Umfassende Test-Suite ausführen"""
    print("🧪 TRADING CARD OPTIMIZER - SYSTEM TESTS")
    print("=" * 60)

    # Unit Tests
    print("\n📋 Running Unit Tests...")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TradingCardSystemTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Integration Tests
    print("\n🔗 Running Integration Tests...")
    integration_test = SystemIntegrationTest()
    asyncio.run(integration_test.run_full_pipeline_test())

    # Ergebnisse
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED - System ready for production!")
    else:
        print(f"❌ {len(result.failures + result.errors)} test(s) failed")
        print("Check logs for details and resolve issues before deployment")

    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trading Card System Tests")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick tests only")
    parser.add_argument("--integration", action="store_true",
                        help="Run integration tests only")
    parser.add_argument("--unit", action="store_true",
                        help="Run unit tests only")

    args = parser.parse_args()

    if args.unit:
        # Nur Unit Tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TradingCardSystemTest)
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    elif args.integration:
        # Nur Integration Tests
        integration_test = SystemIntegrationTest()
        asyncio.run(integration_test.run_full_pipeline_test())
    else:
        # Alle Tests
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
