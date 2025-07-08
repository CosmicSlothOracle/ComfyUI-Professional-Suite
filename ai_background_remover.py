#!/usr/bin/env python3
"""
AI-POWERED BACKGROUND REMOVER
State-of-the-art KI-basierte Hintergrundentfernung für Spritesheets
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from pathlib import Path
import requests
import os
from typing import Optional, Tuple, List
import logging

# Moderne KI-Modelle für Background Removal
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("⚠️  rembg not available. Install with: pip install rembg")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not available. Install with: pip install transformers")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  onnxruntime not available. Install with: pip install onnxruntime")


class AIBackgroundRemover:
    """
    KI-basierte Hintergrundentfernung mit mehreren State-of-the-Art Modellen
    """

    def __init__(self):
        self.models = {}
        self.fallback_methods = []
        self.quality_threshold = 0.15  # Minimum transparency ratio

        self._initialize_models()

    def _initialize_models(self):
        """Initialisiert verfügbare KI-Modelle"""
        print("🤖 Initializing AI Background Removal Models...")

        # 1. REMBG - Robust Background Removal
        if REMBG_AVAILABLE:
            try:
                # Verschiedene Modelle für verschiedene Use Cases
                self.models['u2net'] = new_session('u2net')  # General purpose
                self.models['u2net_human_seg'] = new_session(
                    'u2net_human_seg')  # Human figures
                self.models['u2netp'] = new_session('u2netp')  # Lightweight
                # High quality
                self.models['isnet-general-use'] = new_session(
                    'isnet-general-use')
                print("✅ REMBG models loaded")
            except Exception as e:
                print(f"⚠️  REMBG initialization failed: {e}")

        # 2. Transformers-based models
        if TRANSFORMERS_AVAILABLE:
            try:
                # Segment Anything Model (SAM) für präzise Segmentierung
                # self.sam_model = pipeline("image-segmentation", model="facebook/sam-vit-base")
                print("✅ Transformers models ready")
            except Exception as e:
                print(f"⚠️  Transformers initialization failed: {e}")

        # 3. Fallback: Verbesserter traditioneller Ansatz
        self.fallback_methods = [
            self._advanced_color_based_removal,
            self._grabcut_removal,
            self._watershed_removal
        ]

        print(
            f"🎯 {len(self.models)} AI models + {len(self.fallback_methods)} fallback methods ready")

    def remove_background_ai(self, image: np.ndarray, method: str = 'auto') -> Tuple[np.ndarray, float]:
        """
        KI-basierte Hintergrundentfernung

        Args:
            image: Input image (BGR format)
            method: 'auto', 'u2net', 'u2net_human_seg', 'isnet-general-use', etc.

        Returns:
            tuple: (RGBA image with transparent background, quality_score)
        """

        if method == 'auto':
            method = self._select_optimal_model(image)

        # Konvertiere zu RGB für KI-Modelle
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        best_result = None
        best_quality = 0.0

        # Versuche KI-Modelle
        if method in self.models and REMBG_AVAILABLE:
            try:
                print(f"🤖 Using AI model: {method}")

                # REMBG Background Removal
                output_pil = remove(image_pil, session=self.models[method])
                result_rgba = np.array(output_pil)

                # Qualitätsbewertung
                quality = self._assess_quality(result_rgba)

                if quality > best_quality:
                    best_result = result_rgba
                    best_quality = quality

                print(f"✅ AI model quality: {quality:.2f}")

            except Exception as e:
                print(f"⚠️  AI model {method} failed: {e}")

        # Fallback zu traditionellen Methoden wenn KI nicht gut genug
        if best_quality < self.quality_threshold:
            print("🔄 AI quality insufficient, trying fallback methods...")

            for fallback_method in self.fallback_methods:
                try:
                    result_rgba, quality = fallback_method(image)

                    if quality > best_quality:
                        best_result = result_rgba
                        best_quality = quality
                        print(f"✅ Fallback method quality: {quality:.2f}")

                        if quality > self.quality_threshold:
                            break

                except Exception as e:
                    print(f"⚠️  Fallback method failed: {e}")
                    continue

        # Post-Processing für bessere Qualität
        if best_result is not None:
            best_result = self._post_process_result(best_result)
            final_quality = self._assess_quality(best_result)
            return best_result, final_quality

        # Letzter Fallback: Original mit minimalem Processing
        print("⚠️  All methods failed, returning original with basic processing")
        rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        return rgba_image, 0.1

    def _select_optimal_model(self, image: np.ndarray) -> str:
        """Wählt das optimale Modell basierend auf Bildcharakteristika"""
        h, w = image.shape[:2]

        # Analysiere Bildinhalt
        is_large = w > 1024 or h > 1024

        # Erkenne ob Menschen/Charaktere im Bild
        has_human_features = self._detect_human_features(image)

        if has_human_features and 'u2net_human_seg' in self.models:
            return 'u2net_human_seg'
        elif is_large and 'u2netp' in self.models:
            return 'u2netp'  # Lightweight für große Bilder
        elif 'isnet-general-use' in self.models:
            return 'isnet-general-use'  # Beste Qualität
        elif 'u2net' in self.models:
            return 'u2net'  # Standard
        else:
            return 'auto'

    def _detect_human_features(self, image: np.ndarray) -> bool:
        """Vereinfachte Erkennung von menschlichen Charakteristika"""
        # Einfache Heuristik basierend auf Hautfarbtönen
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Hautfarb-Bereiche (vereinfacht)
        skin_lower = np.array([0, 30, 60])
        skin_upper = np.array([20, 150, 255])
        skin_mask1 = cv2.inRange(hsv, skin_lower, skin_upper)

        skin_lower2 = np.array([160, 30, 60])
        skin_upper2 = np.array([180, 150, 255])
        skin_mask2 = cv2.inRange(hsv, skin_lower2, skin_upper2)

        skin_mask = skin_mask1 + skin_mask2
        skin_ratio = np.sum(skin_mask > 0) / (image.shape[0] * image.shape[1])

        return skin_ratio > 0.05  # Mehr als 5% Hautfarbe

    def _assess_quality(self, rgba_image: np.ndarray) -> float:
        """Bewertet die Qualität der Hintergrundentfernung"""
        if len(rgba_image.shape) != 3 or rgba_image.shape[2] != 4:
            return 0.0

        alpha_channel = rgba_image[:, :, 3]
        total_pixels = alpha_channel.size

        # Transparenz-Ratio
        transparent_pixels = np.sum(alpha_channel == 0)
        transparency_ratio = transparent_pixels / total_pixels

        # Semi-transparente Pixel (gute Kanten)
        semi_transparent = np.sum((alpha_channel > 0) & (alpha_channel < 255))
        edge_quality = semi_transparent / total_pixels

        # Opaque Pixel (Objekt-Bereich)
        opaque_pixels = np.sum(alpha_channel == 255)
        object_ratio = opaque_pixels / total_pixels

        # Quality Score basierend auf ausgewogenen Verhältnissen
        if transparency_ratio < 0.05:  # Zu wenig Hintergrund entfernt
            return transparency_ratio * 2
        elif transparency_ratio > 0.95:  # Zu viel entfernt
            return (1 - transparency_ratio) * 2
        else:
            # Gute Balance zwischen Transparenz und Objekt
            balance_score = min(transparency_ratio, object_ratio) * 2
            edge_bonus = min(edge_quality * 10, 0.2)  # Bonus für gute Kanten
            return min(balance_score + edge_bonus, 1.0)

    def _post_process_result(self, rgba_image: np.ndarray) -> np.ndarray:
        """Nachbearbeitung für bessere Qualität"""
        if len(rgba_image.shape) != 3 or rgba_image.shape[2] != 4:
            return rgba_image

        alpha = rgba_image[:, :, 3]

        # Morphologische Bereinigung der Alpha-Maske
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Schließe kleine Löcher
        alpha_cleaned = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

        # Entferne kleine Fragmente
        alpha_cleaned = cv2.morphologyEx(alpha_cleaned, cv2.MORPH_OPEN, kernel)

        # Weiche Kanten für bessere Qualität
        if np.any(alpha_cleaned > 0):
            alpha_blurred = cv2.GaussianBlur(alpha_cleaned, (3, 3), 0.5)
            alpha_final = np.where(
                alpha_cleaned > 0, alpha_blurred, 0).astype(np.uint8)
        else:
            alpha_final = alpha_cleaned

        # Aktualisiere Alpha-Kanal
        result = rgba_image.copy()
        result[:, :, 3] = alpha_final

        return result

    def _advanced_color_based_removal(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Verbesserter farb-basierter Ansatz"""
        h, w = image.shape[:2]

        # Multi-Zone Sampling statt nur Ecken
        zones = self._sample_background_zones(image)
        bg_colors = self._analyze_background_colors(zones)

        # Adaptive Toleranz
        tolerance = self._calculate_adaptive_tolerance(zones)

        # Erstelle Maske für jeden Hintergrundfarb-Kandidaten
        combined_mask = np.zeros((h, w), dtype=bool)

        for bg_color in bg_colors:
            diff = np.abs(image.astype(int) - bg_color.astype(int))
            mask = np.all(diff <= tolerance, axis=2)
            combined_mask |= mask

        # Konvertiere zu RGBA
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgba_image = np.dstack(
            [image_rgb, np.full((h, w), 255, dtype=np.uint8)])

        # Setze Hintergrund transparent
        rgba_image[combined_mask, 3] = 0

        quality = self._assess_quality(rgba_image)
        return rgba_image, quality

    def _sample_background_zones(self, image: np.ndarray) -> List[np.ndarray]:
        """Samples multiple zones for better background detection"""
        h, w = image.shape[:2]
        zone_size = min(h, w) // 15  # Größere Sampling-Zone

        zones = []

        # Ecken
        zones.extend([
            image[:zone_size, :zone_size],
            image[:zone_size, -zone_size:],
            image[-zone_size:, :zone_size],
            image[-zone_size:, -zone_size:]
        ])

        # Kanten-Mitten
        zones.extend([
            image[:zone_size, w//2-zone_size //
                  2:w//2+zone_size//2],  # Top center
            image[-zone_size:, w//2-zone_size//2:w //
                  2+zone_size//2],  # Bottom center
            image[h//2-zone_size//2:h//2+zone_size //
                  2, :zone_size],  # Left center
            image[h//2-zone_size//2:h//2+zone_size //
                  2, -zone_size:]  # Right center
        ])

        return zones

    def _analyze_background_colors(self, zones: List[np.ndarray]) -> List[np.ndarray]:
        """Analysiert Hintergrundfarben aus mehreren Zonen"""
        all_colors = []

        for zone in zones:
            if zone.size > 0:
                # Dominante Farben per K-Means
                pixels = zone.reshape(-1, 3).astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS +
                            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, labels, centers = cv2.kmeans(
                    pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                all_colors.extend(centers)

        if not all_colors:
            return [np.array([255, 255, 255], dtype=np.uint8)]  # Default white

        # Clustere alle gefundenen Farben
        all_colors = np.array(all_colors, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, _, final_centers = cv2.kmeans(all_colors, min(
            3, len(all_colors)), None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        return [center.astype(np.uint8) for center in final_centers]

    def _calculate_adaptive_tolerance(self, zones: List[np.ndarray]) -> int:
        """Berechnet adaptive Toleranz basierend auf Farbvarianz"""
        variances = []

        for zone in zones:
            if zone.size > 0:
                variance = np.var(zone.reshape(-1, 3), axis=0)
                variances.append(np.mean(variance))

        if not variances:
            return 25  # Default

        avg_variance = np.mean(variances)

        # Adaptive Toleranz: Höhere Varianz = Höhere Toleranz
        if avg_variance < 100:
            return 15  # Niedrige Toleranz für einheitliche Hintergründe
        elif avg_variance < 500:
            return 25  # Standard Toleranz
        else:
            return 40  # Hohe Toleranz für variable Hintergründe

    def _grabcut_removal(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """GrabCut-basierte Hintergrundentfernung"""
        # Erstelle initiale Maske
        h, w = image.shape[:2]
        mask = np.zeros((h, w), np.uint8)

        # Definiere Rechteck um das wahrscheinliche Objekt
        border = min(h, w) // 10
        rect = (border, border, w - 2*border, h - 2*border)

        # GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(image, mask, rect, bgd_model,
                        fgd_model, 5, cv2.GC_INIT_WITH_RECT)

            # Erstelle finale Maske
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            # Konvertiere zu RGBA
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgba_image = np.dstack([image_rgb, mask2 * 255])

            quality = self._assess_quality(rgba_image)
            return rgba_image, quality

        except Exception as e:
            print(f"GrabCut failed: {e}")
            return self._advanced_color_based_removal(image)

    def _watershed_removal(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Watershed-basierte Segmentierung"""
        # Konvertiere zu Graustufen
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Find sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(
            dist_transform, 0.7*dist_transform.max(), 255, 0)

        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Apply watershed
        markers = cv2.watershed(image, markers)

        # Create mask
        mask = np.where(markers > 1, 255, 0).astype(np.uint8)

        # Konvertiere zu RGBA
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgba_image = np.dstack([image_rgb, mask])

        quality = self._assess_quality(rgba_image)
        return rgba_image, quality


def install_dependencies():
    """Installiert erforderliche KI-Abhängigkeiten"""
    dependencies = [
        "rembg[gpu]",  # Für GPU-Unterstützung
        "transformers",
        "torch",
        "torchvision",
        "onnxruntime",
        "Pillow",
        "opencv-python"
    ]

    print("🤖 Installing AI dependencies...")
    import subprocess
    import sys

    for dep in dependencies:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} installed")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to install {dep}: {e}")


if __name__ == "__main__":
    # Test der KI-basierten Hintergrundentfernung
    print("🤖 AI Background Remover Test")

    # Installiere Dependencies falls nötig
    if not REMBG_AVAILABLE:
        install_dependencies()

    # Erstelle AI Remover
    ai_remover = AIBackgroundRemover()

    # Test mit Beispielbild
    test_image_path = Path("input/Mann_steigt_aus_Limousine_aus.png")
    if test_image_path.exists():
        print(f"🧪 Testing with: {test_image_path}")

        image = cv2.imread(str(test_image_path))
        result_rgba, quality = ai_remover.remove_background_ai(image)

        print(f"🎯 Quality Score: {quality:.2f}")

        # Speichere Ergebnis
        output_path = Path("output/ai_background_test.png")
        output_path.parent.mkdir(exist_ok=True)

        result_pil = Image.fromarray(result_rgba, 'RGBA')
        result_pil.save(output_path)

        print(f"💾 Result saved to: {output_path}")
    else:
        print("⚠️  Test image not found")
