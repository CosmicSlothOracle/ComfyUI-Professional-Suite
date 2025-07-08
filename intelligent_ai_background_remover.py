#!/usr/bin/env python3
"""
INTELLIGENT AI BACKGROUND REMOVER
State-of-the-art KI-basierte Hintergrundentfernung für Spritesheets
"""

import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import requests
import os
from typing import Optional, Tuple, List
import logging

# KI-Modelle für Background Removal
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class IntelligentAIBackgroundRemover:
    """
    Intelligente KI-basierte Hintergrundentfernung mit mehreren Modellen
    """

    def __init__(self):
        self.models = {}
        self.quality_threshold = 0.15
        self._initialize_models()

    def _initialize_models(self):
        """Initialisiert verfügbare KI-Modelle"""
        print("🤖 Initializing AI Background Removal Models...")

        if REMBG_AVAILABLE:
            try:
                # Verschiedene spezialisierte Modelle
                self.models['u2net'] = new_session('u2net')  # Universal
                self.models['u2net_human_seg'] = new_session(
                    'u2net_human_seg')  # Menschen
                # Hohe Qualität
                self.models['isnet-general-use'] = new_session(
                    'isnet-general-use')
                self.models['silueta'] = new_session('silueta')  # Für Objekte
                print("✅ REMBG AI models loaded successfully")
            except Exception as e:
                print(f"⚠️  REMBG initialization failed: {e}")
                REMBG_AVAILABLE = False

    def remove_background_intelligent(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Intelligente KI-basierte Hintergrundentfernung

        Returns:
            tuple: (RGBA image with transparent background, quality_score)
        """

        # Wähle bestes Modell für dieses Bild
        optimal_model = self._select_optimal_model(image)

        # Konvertiere zu RGB für KI-Modelle
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        best_result = None
        best_quality = 0.0

        # Versuche KI-Modelle in Reihenfolge der Qualität
        model_priority = [optimal_model,
                          'isnet-general-use', 'u2net', 'silueta']

        for model_name in model_priority:
            if model_name in self.models and REMBG_AVAILABLE:
                try:
                    print(f"🤖 Trying AI model: {model_name}")

                    # KI-basierte Background Removal
                    output_pil = remove(
                        image_pil, session=self.models[model_name])
                    result_rgba = np.array(output_pil)

                    # Qualitätsbewertung
                    quality = self._assess_quality_advanced(result_rgba, image)
                    print(f"   Quality: {quality:.3f}")

                    if quality > best_quality:
                        best_result = result_rgba
                        best_quality = quality

                    # Wenn Qualität gut genug, nutze dieses Ergebnis
                    if quality > 0.6:
                        break

                except Exception as e:
                    print(f"⚠️  AI model {model_name} failed: {e}")
                    continue

        # Post-Processing für noch bessere Qualität
        if best_result is not None:
            best_result = self._post_process_intelligent(best_result, image)
            final_quality = self._assess_quality_advanced(best_result, image)

            print(f"✅ Final AI result quality: {final_quality:.3f}")
            return best_result, final_quality

        # Fallback zu verbesserter traditioneller Methode
        print("🔄 AI failed, using advanced fallback...")
        return self._advanced_fallback_removal(image)

    def _select_optimal_model(self, image: np.ndarray) -> str:
        """Wählt das optimale KI-Modell basierend auf Bildanalyse"""
        h, w = image.shape[:2]

        # Analysiere Bildinhalt
        has_human = self._detect_human_content(image)
        is_complex = self._analyze_complexity(image)
        image_size = w * h

        # Intelligente Modellauswahl
        if has_human and 'u2net_human_seg' in self.models:
            return 'u2net_human_seg'
        elif is_complex and 'isnet-general-use' in self.models:
            return 'isnet-general-use'
        elif image_size > 1024*1024 and 'silueta' in self.models:
            return 'silueta'  # Schneller für große Bilder
        else:
            return 'u2net'  # Standard hohe Qualität

    def _detect_human_content(self, image: np.ndarray) -> bool:
        """Erkennt menschliche Inhalte im Bild"""
        # Konvertiere zu HSV für bessere Hautfarberkennung
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Erweiterte Hautfarb-Bereiche
        skin_ranges = [
            ([0, 30, 60], [20, 150, 255]),      # Heller Hautton
            ([160, 30, 60], [180, 150, 255]),   # Rötlicher Hautton
            ([10, 50, 20], [25, 255, 255])      # Dunklerer Hautton
        ]

        total_skin = 0
        for lower, upper in skin_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            total_skin += np.sum(mask > 0)

        skin_ratio = total_skin / (image.shape[0] * image.shape[1])

        # Zusätzliche Heuristiken
        has_face_like_structures = self._detect_face_structures(image)

        return skin_ratio > 0.03 or has_face_like_structures

    def _detect_face_structures(self, image: np.ndarray) -> bool:
        """Einfache Gesichtserkennung über Haar-Cascade"""
        try:
            # Nutze OpenCV's Haar-Cascade für Gesichtserkennung
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            return len(faces) > 0
        except:
            return False

    def _analyze_complexity(self, image: np.ndarray) -> bool:
        """Analysiert die Komplexität des Bildes"""
        # Kantenerkennung für Komplexitätsanalyse
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])

        # Farb-Histogramm-Analyse
        hist = cv2.calcHist([image], [0, 1, 2], None, [
                            8, 8, 8], [0, 256, 0, 256, 0, 256])
        color_diversity = np.sum(hist > 0) / (8 * 8 * 8)

        # Komplex wenn viele Kanten oder viele verschiedene Farben
        return edge_density > 0.1 or color_diversity > 0.3

    def _assess_quality_advanced(self, rgba_image: np.ndarray, original_image: np.ndarray) -> float:
        """Erweiterte Qualitätsbewertung der Hintergrundentfernung"""
        if len(rgba_image.shape) != 3 or rgba_image.shape[2] != 4:
            return 0.0

        alpha = rgba_image[:, :, 3]
        h, w = alpha.shape
        total_pixels = h * w

        # 1. Transparenz-Analyse
        transparent = np.sum(alpha == 0)
        semi_transparent = np.sum((alpha > 0) & (alpha < 255))
        opaque = np.sum(alpha == 255)

        transparency_ratio = transparent / total_pixels
        edge_ratio = semi_transparent / total_pixels
        object_ratio = opaque / total_pixels

        # 2. Kantenschärfe-Analyse
        edge_quality = self._analyze_edge_quality(alpha)

        # 3. Objekt-Kohärenz (zusammenhängende Bereiche)
        coherence_score = self._analyze_object_coherence(alpha)

        # 4. Hintergrund-Entfernung Vollständigkeit
        background_removal_score = self._analyze_background_removal(
            rgba_image, original_image)

        # Gewichtete Gesamtbewertung
        if transparency_ratio < 0.05:  # Kaum Hintergrund entfernt
            return transparency_ratio * 0.5
        elif transparency_ratio > 0.95:  # Zu viel entfernt
            return (1 - transparency_ratio) * 0.5
        else:
            # Balancierte Bewertung
            balance_score = min(transparency_ratio * 2, object_ratio * 2) * 0.4
            edge_score = edge_quality * 0.3
            coherence_score_weighted = coherence_score * 0.2
            background_score = background_removal_score * 0.1

            total_score = balance_score + edge_score + \
                coherence_score_weighted + background_score
            return min(total_score, 1.0)

    def _analyze_edge_quality(self, alpha: np.ndarray) -> float:
        """Analysiert die Qualität der Kanten"""
        # Gradient der Alpha-Maske
        grad_x = cv2.Sobel(alpha, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(alpha, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Scharfe Kanten haben hohe Gradienten
        sharp_edges = np.sum(gradient_magnitude > 50)
        total_edge_pixels = np.sum(gradient_magnitude > 10)

        if total_edge_pixels == 0:
            return 0.0

        return min(sharp_edges / total_edge_pixels, 1.0)

    def _analyze_object_coherence(self, alpha: np.ndarray) -> float:
        """Analysiert die Kohärenz der extrahierten Objekte"""
        # Connected Components auf Alpha-Kanal
        binary_alpha = (alpha > 128).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_alpha)

        if num_labels <= 1:  # Nur Hintergrund
            return 0.0

        # Größe des größten Objekts
        areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
        if len(areas) == 0:
            return 0.0

        largest_area = np.max(areas)
        total_object_area = np.sum(areas)

        # Hohe Kohärenz wenn ein großes Hauptobjekt existiert
        coherence = largest_area / total_object_area if total_object_area > 0 else 0

        # Bonus für weniger fragmentierte Objekte
        fragment_penalty = max(0, 1 - (len(areas) - 1) * 0.1)

        return min(coherence * fragment_penalty, 1.0)

    def _analyze_background_removal(self, rgba_image: np.ndarray, original: np.ndarray) -> float:
        """Analysiert wie gut der Hintergrund entfernt wurde"""
        # Vergleiche originale Ecken mit extrahiertem Ergebnis
        h, w = original.shape[:2]
        corner_size = min(h, w) // 20

        # Originalecken
        corners_orig = [
            original[:corner_size, :corner_size],
            original[:corner_size, -corner_size:],
            original[-corner_size:, :corner_size],
            original[-corner_size:, -corner_size:]
        ]

        # Extrahierte Ecken (Alpha-Kanal)
        alpha = rgba_image[:, :, 3]
        corners_alpha = [
            alpha[:corner_size, :corner_size],
            alpha[:corner_size, -corner_size:],
            alpha[-corner_size:, :corner_size],
            alpha[-corner_size:, -corner_size:]
        ]

        # Bewerte wie viel Hintergrund in den Ecken entfernt wurde
        removal_scores = []
        for orig_corner, alpha_corner in zip(corners_orig, corners_alpha):
            transparent_ratio = np.sum(alpha_corner == 0) / alpha_corner.size
            removal_scores.append(transparent_ratio)

        return np.mean(removal_scores)

    def _post_process_intelligent(self, rgba_image: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Intelligente Nachbearbeitung"""
        if len(rgba_image.shape) != 3 or rgba_image.shape[2] != 4:
            return rgba_image

        alpha = rgba_image[:, :, 3]

        # 1. Adaptive morphologische Bereinigung
        kernel_size = max(3, min(alpha.shape) // 200)  # Adaptive Kernel-Größe
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Schließe kleine Löcher
        alpha_closed = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

        # Entferne kleine Fragmente
        alpha_opened = cv2.morphologyEx(alpha_closed, cv2.MORPH_OPEN, kernel)

        # 2. Intelligente Kantenglättung
        # Nur glätten wo es sinnvoll ist (nicht alle Kanten)
        blur_mask = self._create_blur_mask(alpha_opened)
        alpha_blurred = cv2.GaussianBlur(alpha_opened, (5, 5), 1.0)
        alpha_final = np.where(blur_mask, alpha_blurred, alpha_opened)

        # 3. Aktualisiere das Ergebnis
        result = rgba_image.copy()
        result[:, :, 3] = alpha_final.astype(np.uint8)

        return result

    def _create_blur_mask(self, alpha: np.ndarray) -> np.ndarray:
        """Erstellt Maske für selektive Kantenglättung"""
        # Erkenne scharfe Kanten
        edges = cv2.Canny(alpha, 50, 150)

        # Erweitere Kanten-Bereich
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        # Blur nur an Kanten, nicht im Objektinneren
        return edges_dilated > 0

    def _advanced_fallback_removal(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Erweiterte Fallback-Methode wenn KI versagt"""
        print("🔄 Using advanced fallback background removal...")

        # Multi-Methoden-Ansatz
        methods = [
            self._grabcut_advanced,
            self._watershed_advanced,
            self._color_clustering_advanced
        ]

        best_result = None
        best_quality = 0.0

        for method in methods:
            try:
                result, quality = method(image)
                if quality > best_quality:
                    best_result = result
                    best_quality = quality

                if quality > 0.4:  # Ausreichende Qualität erreicht
                    break

            except Exception as e:
                print(f"⚠️  Fallback method failed: {e}")
                continue

        if best_result is not None:
            return best_result, best_quality

        # Absoluter Fallback: Minimal-Processing
        print("⚠️  All methods failed, returning minimal processing")
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        return rgba, 0.05

    def _grabcut_advanced(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Verbesserter GrabCut-Algorithmus"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), np.uint8)

        # Intelligentere Rechteck-Definition
        border_h = h // 8
        border_w = w // 8
        rect = (border_w, border_h, w - 2*border_w, h - 2*border_h)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Mehr Iterationen für bessere Qualität
        cv2.grabCut(image, mask, rect, bgd_model,
                    fgd_model, 8, cv2.GC_INIT_WITH_RECT)

        # Verfeinerte Maske
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Morphologische Verbesserung
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

        # Konvertiere zu RGBA
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgba_image = np.dstack([image_rgb, mask2 * 255])

        quality = self._assess_quality_advanced(rgba_image, image)
        return rgba_image, quality

    def _watershed_advanced(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Verbesserter Watershed-Algorithmus"""
        # Preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Noise Reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(
            denoised, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Sure foreground
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(
            dist_transform, 0.6*dist_transform.max(), 255, 0)

        # Unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Apply watershed
        markers = cv2.watershed(image, markers)

        # Create refined mask
        mask = np.where(markers > 1, 255, 0).astype(np.uint8)

        # Post-processing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Convert to RGBA
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgba_image = np.dstack([image_rgb, mask])

        quality = self._assess_quality_advanced(rgba_image, image)
        return rgba_image, quality

    def _color_clustering_advanced(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Erweiterte farb-basierte Segmentierung mit K-Means"""
        h, w = image.shape[:2]

        # Reshape für K-Means
        pixels = image.reshape(-1, 3).astype(np.float32)

        # K-Means Clustering für Farbsegmentierung
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = min(8, len(np.unique(pixels.reshape(-1))))  # Adaptive K

        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Identifiziere Hintergrundfarben (Ecken-Analyse)
        corner_labels = self._get_corner_labels(labels.reshape(h, w))
        background_clusters = set(corner_labels)

        # Erstelle Maske
        segmented = centers[labels.flatten()].reshape(h, w, 3).astype(np.uint8)

        # Background mask
        bg_mask = np.zeros((h, w), dtype=bool)
        for bg_cluster in background_clusters:
            cluster_mask = (labels.reshape(h, w) == bg_cluster)
            bg_mask |= cluster_mask

        # Foreground mask
        fg_mask = ~bg_mask

        # Morphological refinement
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask.astype(
            np.uint8), cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Convert to RGBA
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgba_image = np.dstack([image_rgb, fg_mask * 255])

        quality = self._assess_quality_advanced(rgba_image, image)
        return rgba_image, quality

    def _get_corner_labels(self, label_image: np.ndarray) -> List[int]:
        """Extrahiert Labels aus den Bildecken"""
        h, w = label_image.shape
        corner_size = min(h, w) // 20

        corners = [
            label_image[:corner_size, :corner_size],
            label_image[:corner_size, -corner_size:],
            label_image[-corner_size:, :corner_size],
            label_image[-corner_size:, -corner_size:]
        ]

        corner_labels = []
        for corner in corners:
            unique_labels, counts = np.unique(corner, return_counts=True)
            most_common = unique_labels[np.argmax(counts)]
            corner_labels.append(most_common)

        return corner_labels


def install_ai_dependencies():
    """Installiert KI-Abhängigkeiten für Background Removal"""
    dependencies = [
        "rembg[gpu]",
        "torch",
        "torchvision",
        "transformers",
        "onnxruntime-gpu",  # Für GPU-Beschleunigung
        "Pillow>=8.0.0",
        "opencv-python>=4.5.0"
    ]

    print("🤖 Installing AI dependencies for background removal...")
    import subprocess
    import sys

    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", dep, "--upgrade"])
            print(f"✅ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to install {dep}: {e}")

            # Fallback ohne GPU
            if 'gpu' in dep:
                fallback_dep = dep.replace('[gpu]', '').replace('-gpu', '')
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", fallback_dep])
                    print(f"✅ {fallback_dep} (CPU version) installed")
                except:
                    print(f"❌ Failed to install {fallback_dep}")


if __name__ == "__main__":
    # Test und Installation
    print("🤖 Intelligent AI Background Remover")

    # Prüfe ob Dependencies installiert sind
    if not REMBG_AVAILABLE:
        print("⚠️  AI dependencies missing. Installing...")
        install_ai_dependencies()
        print("🔄 Please restart the script after installation.")
    else:
        # Test mit Beispielbild
        ai_remover = IntelligentAIBackgroundRemover()

        test_path = Path("input/Mann_steigt_aus_Limousine_aus.png")
        if test_path.exists():
            print(f"🧪 Testing with: {test_path}")

            image = cv2.imread(str(test_path))
            result, quality = ai_remover.remove_background_intelligent(image)

            print(f"🎯 AI Quality Score: {quality:.3f}")

            # Speichere Ergebnis
            output_path = Path("output/ai_test_result.png")
            output_path.parent.mkdir(exist_ok=True)

            Image.fromarray(result, 'RGBA').save(output_path)
            print(f"💾 AI result saved to: {output_path}")
        else:
            print("⚠️  Test image not found, but AI system is ready!")
