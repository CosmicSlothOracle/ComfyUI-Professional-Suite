#!/usr/bin/env python3
"""
AI-BASED SPRITESHEET PROCESSOR
Verwendet modernste Machine Learning Algorithmen für intelligente Sprite-Erkennung
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import time
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from skimage.segmentation import watershed, felzenszwalb, slic
from skimage.feature import peak_local_maxima
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects
from skimage.filters import rank, gaussian
from skimage.util import img_as_ubyte
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


class AISpritesheetProcessor:
    """AI-Based Spritesheet Processor mit modernsten ML-Algorithmen"""

    def __init__(self):
        self.input_dir = Path("input")
        self.output_dir = Path("output/ai_sprites_processed")
        self.processed_count = 0
        self.failed_count = 0

    def intelligent_background_detection(self, image: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """AI-basierte Multi-Algorithmus Background Detection"""
        print("🧠 AI Background Detection...")

        # Sammle Features aus verschiedenen Zonen
        features = self._extract_background_features(image)

        # Führe verschiedene Clustering-Algorithmen aus
        clustering_results = self._apply_clustering_algorithms(features)

        # Bewerte die Algorithmen
        best_algorithm = self._evaluate_clustering_results(
            clustering_results, image)

        # Erstelle finale Maske
        background_mask = self._create_intelligent_mask(image, best_algorithm)

        return background_mask, best_algorithm['confidence'], best_algorithm['metadata']

    def _extract_background_features(self, image: np.ndarray) -> np.ndarray:
        """Extrahiert AI-Features für Background Detection"""
        h, w = image.shape[:2]
        features = []

        # Strategische Sampling-Zonen
        zones = [
            # Corners (verschiedene Größen)
            image[:max(10, h//20), :max(10, w//20)],  # top-left
            image[:max(10, h//20), -max(10, w//20):],  # top-right
            image[-max(10, h//20):, :max(10, w//20)],  # bottom-left
            image[-max(10, h//20):, -max(10, w//20):],  # bottom-right

            # Edges (adaptive)
            image[:max(5, h//40), :],  # top edge
            image[-max(5, h//40):, :],  # bottom edge
            image[:, :max(5, w//40)],  # left edge
            image[:, -max(5, w//40):],  # right edge

            # Quadrants borders
            image[:h//4, :w//4],  # top-left quadrant
            image[:h//4, 3*w//4:],  # top-right quadrant
            image[3*h//4:, :w//4],  # bottom-left quadrant
            image[3*h//4:, 3*w//4:],  # bottom-right quadrant
        ]

        # Extrahiere Features aus jeder Zone
        for zone in zones:
            if zone.size > 0:
                # RGB Features
                rgb_features = zone.reshape(-1, 3)
                features.append(rgb_features)

                # HSV Features
                hsv_zone = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
                hsv_features = hsv_zone.reshape(-1, 3)
                features.append(hsv_features)

                # LAB Features
                lab_zone = cv2.cvtColor(zone, cv2.COLOR_BGR2LAB)
                lab_features = lab_zone.reshape(-1, 3)
                features.append(lab_features)

        if features:
            return np.vstack(features)
        else:
            return np.array([[255, 255, 255]])  # Fallback

    def _apply_clustering_algorithms(self, features: np.ndarray) -> Dict[str, Dict]:
        """Wendet verschiedene AI-Clustering-Algorithmen an"""
        results = {}

        # Standardisiere Features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        print("  🔬 Applying K-Means...")
        # 1. K-Means Clustering (verschiedene K-Werte)
        best_kmeans = None
        best_kmeans_score = -1

        for k in range(2, min(8, len(features) // 10 + 1)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)

            if len(set(labels)) > 1:
                score = silhouette_score(features_scaled, labels)
                if score > best_kmeans_score:
                    best_kmeans_score = score
                    best_kmeans = {
                        'algorithm': kmeans,
                        'labels': labels,
                        'centers': scaler.inverse_transform(kmeans.cluster_centers_),
                        'score': score,
                        'n_clusters': k
                    }

        if best_kmeans:
            results['kmeans'] = best_kmeans

        print("  🔬 Applying DBSCAN...")
        # 2. DBSCAN Clustering (adaptive parameters)
        distances = pdist(features_scaled)
        eps_values = np.percentile(distances, [10, 25, 50])

        best_dbscan = None
        best_dbscan_score = -1

        for eps in eps_values:
            for min_samples in [3, 5, 10]:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(features_scaled)

                if len(set(labels)) > 1 and -1 not in labels:
                    score = silhouette_score(features_scaled, labels)
                    if score > best_dbscan_score:
                        best_dbscan_score = score
                        best_dbscan = {
                            'algorithm': dbscan,
                            'labels': labels,
                            'score': score,
                            'eps': eps,
                            'min_samples': min_samples
                        }

        if best_dbscan:
            results['dbscan'] = best_dbscan

        print("  🔬 Applying Mean Shift...")
        # 3. Mean Shift Clustering
        try:
            bandwidth = estimate_bandwidth(features_scaled, quantile=0.3)
            if bandwidth > 0:
                ms = MeanShift(bandwidth=bandwidth)
                labels = ms.fit_predict(features_scaled)

                if len(set(labels)) > 1:
                    score = silhouette_score(features_scaled, labels)
                    results['meanshift'] = {
                        'algorithm': ms,
                        'labels': labels,
                        'centers': scaler.inverse_transform(ms.cluster_centers_),
                        'score': score,
                        'bandwidth': bandwidth
                    }
        except:
            pass

        print("  🔬 Applying Gaussian Mixture...")
        # 4. Gaussian Mixture Models
        best_gmm = None
        best_gmm_score = -1

        for n_components in range(2, min(6, len(features) // 20 + 1)):
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            labels = gmm.fit_predict(features_scaled)

            if len(set(labels)) > 1:
                score = silhouette_score(features_scaled, labels)
                if score > best_gmm_score:
                    best_gmm_score = score
                    best_gmm = {
                        'algorithm': gmm,
                        'labels': labels,
                        'centers': scaler.inverse_transform(gmm.means_),
                        'score': score,
                        'n_components': n_components
                    }

        if best_gmm:
            results['gmm'] = best_gmm

        return results

    def _evaluate_clustering_results(self, clustering_results: Dict, image: np.ndarray) -> Dict:
        """Bewertet die Clustering-Ergebnisse und wählt den besten aus"""
        if not clustering_results:
            return {'algorithm': 'fallback', 'confidence': 0.0, 'metadata': {}}

        best_result = None
        best_score = -1

        for algo_name, result in clustering_results.items():
            # Kombiniere Silhouette Score mit Domain-spezifischen Metriken
            silhouette = result['score']

            # Bonus für weniger Cluster (Background soll einfach sein)
            n_clusters = len(set(result['labels']))
            cluster_penalty = max(0, (n_clusters - 3) * 0.1)

            # Bonus für konsistente Cluster-Größen
            unique_labels, counts = np.unique(
                result['labels'], return_counts=True)
            if len(counts) > 1:
                cluster_balance = 1.0 - (np.std(counts) / np.mean(counts))
            else:
                cluster_balance = 1.0

            # Gesamtscore
            total_score = silhouette + cluster_balance * 0.3 - cluster_penalty

            if total_score > best_score:
                best_score = total_score
                best_result = {
                    'algorithm': algo_name,
                    'data': result,
                    'confidence': min(1.0, max(0.0, total_score)),
                    'metadata': {
                        'silhouette_score': silhouette,
                        'n_clusters': n_clusters,
                        'cluster_balance': cluster_balance,
                        'total_score': total_score
                    }
                }

        return best_result or {'algorithm': 'fallback', 'confidence': 0.0, 'metadata': {}}

    def _create_intelligent_mask(self, image: np.ndarray, clustering_result: Dict) -> np.ndarray:
        """Erstellt intelligente Maske basierend auf Clustering-Ergebnis"""
        if clustering_result['algorithm'] == 'fallback':
            return self._fallback_mask(image)

        # Identifiziere Background-Cluster
        bg_clusters = self._identify_background_clusters(
            clustering_result, image)

        # Erstelle Pixel-Level Maske
        mask = self._create_pixel_mask(image, clustering_result, bg_clusters)

        # Intelligente Post-Processing
        mask = self._intelligent_morphology(mask, image)

        return mask

    def _identify_background_clusters(self, clustering_result: Dict, image: np.ndarray) -> List[int]:
        """Identifiziert Background-Cluster durch Corner-Analyse"""
        h, w = image.shape[:2]
        corner_pixels = [
            image[0, 0], image[0, w-1], image[h-1, 0], image[h-1, w-1]
        ]

        # Finde Cluster-Zugehörigkeit der Ecken
        data = clustering_result['data']
        if 'centers' in data:
            centers = data['centers']

            # Bestimme nächstgelegene Cluster für jede Ecke
            bg_clusters = set()
            for corner in corner_pixels:
                distances = [np.linalg.norm(corner - center)
                             for center in centers]
                closest_cluster = np.argmin(distances)
                bg_clusters.add(closest_cluster)

            return list(bg_clusters)
        else:
            return [0]  # Fallback

    def _create_pixel_mask(self, image: np.ndarray, clustering_result: Dict, bg_clusters: List[int]) -> np.ndarray:
        """Erstellt Pixel-Level Maske"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        data = clustering_result['data']
        if 'centers' in data:
            centers = data['centers']

            # Klassifiziere jeden Pixel
            for y in range(h):
                for x in range(w):
                    pixel = image[y, x]
                    distances = [np.linalg.norm(pixel - center)
                                 for center in centers]
                    closest_cluster = np.argmin(distances)

                    if closest_cluster not in bg_clusters:
                        mask[y, x] = 255

        return mask

    def _intelligent_morphology(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Intelligente morphologische Operationen"""
        h, w = image.shape[:2]

        # Adaptive Kernel-Größe
        kernel_size = max(3, min(h, w) // 100)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Closing (Löcher füllen)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Opening (Noise entfernen)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Edge-guided Refinement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Verwende Edges für Masken-Verfeinerung
        dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        edge_mask = cv2.morphologyEx(
            mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        edge_refinement = dilated_edges & edge_mask

        mask[edge_refinement > 0] = 255

        return mask

    def _fallback_mask(self, image: np.ndarray) -> np.ndarray:
        """Fallback Maske bei Clustering-Fehlern"""
        # Einfache Corner-basierte Methode
        h, w = image.shape[:2]
        corners = [image[0, 0], image[0, w-1], image[h-1, 0], image[h-1, w-1]]
        bg_color = np.mean(corners, axis=0)

        diff = np.abs(image.astype(float) - bg_color.astype(float))
        distance = np.sqrt(np.sum(diff**2, axis=2))

        return (distance > 30).astype(np.uint8) * 255

    def advanced_frame_extraction(self, image: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
        """Advanced AI-basierte Frame-Extraktion"""
        print("🎯 AI Frame Extraction...")

        # Watershed Segmentation für bessere Trennung
        segmented_mask = self._watershed_segmentation(image, mask)

        # Intelligente Connected Components
        components = self._intelligent_connected_components(segmented_mask)

        # AI-basierte Filterung
        valid_components = self._ai_component_filtering(components, image)

        # Frame-Extraktion mit Optimierung
        frames = self._extract_optimized_frames(image, valid_components)

        return frames

    def _watershed_segmentation(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Watershed Segmentation für bessere Objekttrennung"""
        # Distanz-Transform
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        # Lokale Maxima finden
        local_maxima = peak_local_maxima(
            dist_transform, min_distance=20, threshold_abs=0.3 * dist_transform.max())

        # Marker erstellen
        markers = np.zeros(mask.shape, dtype=np.int32)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1

        # Watershed anwenden
        labels = watershed(-dist_transform, markers, mask=mask)

        return labels

    def _intelligent_connected_components(self, segmented_mask: np.ndarray) -> List[Dict]:
        """Intelligente Connected Components Analysis"""
        # Verschiedene Connectivity-Levels testen
        components_4 = self._analyze_connectivity(segmented_mask, 4)
        components_8 = self._analyze_connectivity(segmented_mask, 8)

        # Wähle beste Connectivity basierend auf Ergebnissen
        if len(components_8) > 0 and len(components_4) > 0:
            # Bevorzuge 8-Connectivity wenn es mehr sinnvolle Components gibt
            ratio = len(components_8) / len(components_4)
            if 0.7 <= ratio <= 1.5:  # Ähnliche Anzahl
                return components_8
            elif ratio > 1.5:  # Zu viele in 8-connectivity
                return components_4
            else:
                return components_8
        elif len(components_8) > 0:
            return components_8
        else:
            return components_4

    def _analyze_connectivity(self, mask: np.ndarray, connectivity: int) -> List[Dict]:
        """Analysiert Connected Components mit gegebener Connectivity"""
        if isinstance(mask, np.ndarray) and mask.dtype == np.int32:
            # Watershed Labels
            unique_labels = np.unique(mask)
            components = []

            for label_val in unique_labels:
                if label_val == 0:  # Background
                    continue

                component_mask = (mask == label_val).astype(np.uint8)
                props = regionprops(label(component_mask))

                if props:
                    prop = props[0]
                    components.append({
                        'area': prop.area,
                        'bbox': prop.bbox,
                        'centroid': prop.centroid,
                        'mask': component_mask
                    })
        else:
            # Standard Connected Components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=connectivity
            )

            components = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                    stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

                components.append({
                    'area': area,
                    'bbox': (y, x, y+h, x+w),
                    'centroid': centroids[i][::-1],  # y, x
                    'mask': (labels == i).astype(np.uint8)
                })

        return components

    def _ai_component_filtering(self, components: List[Dict], image: np.ndarray) -> List[Dict]:
        """AI-basierte Filterung von Components"""
        if not components:
            return []

        h, w = image.shape[:2]
        total_pixels = h * w

        # Extrahiere Features für ML-basierte Filterung
        features = []
        for comp in components:
            area = comp['area']
            bbox = comp['bbox']
            y1, x1, y2, x2 = bbox
            width = x2 - x1
            height = y2 - y1

            # Geometrische Features
            aspect_ratio = width / height if height > 0 else 0
            extent = area / (width * height) if width > 0 and height > 0 else 0

            # Relative Features
            relative_area = area / total_pixels
            relative_width = width / w
            relative_height = height / h

            features.append([
                area, width, height, aspect_ratio, extent,
                relative_area, relative_width, relative_height
            ])

        features = np.array(features)

        # Clustering für Outlier-Erkennung
        if len(features) > 1:
            # Isolation Forest würde hier ideal sein, aber DBSCAN als Alternative
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Konservative DBSCAN Parameter für Outlier Detection
            dbscan = DBSCAN(eps=0.5, min_samples=1)
            labels = dbscan.fit_predict(features_scaled)

            # Behalte Components die nicht als Outlier markiert wurden
            valid_components = []
            for i, (comp, label) in enumerate(zip(components, labels)):
                if label != -1:  # Nicht Outlier
                    # Zusätzliche heuristische Filter
                    area = comp['area']
                    bbox = comp['bbox']
                    y1, x1, y2, x2 = bbox
                    width = x2 - x1
                    height = y2 - y1
                    aspect_ratio = width / height if height > 0 else 0

                    if (area > 500 and  # Mindestgröße
                        area < total_pixels * 0.8 and  # Nicht zu groß
                        0.1 < aspect_ratio < 10 and  # Vernünftige Proportionen
                            width > 10 and height > 10):  # Mindest-Dimensionen
                        valid_components.append(comp)

            return valid_components
        else:
            return components

    def _extract_optimized_frames(self, image: np.ndarray, components: List[Dict]) -> List[np.ndarray]:
        """Extrahiert optimierte Frames"""
        frames = []

        for comp in components:
            bbox = comp['bbox']
            y1, x1, y2, x2 = bbox

            # Intelligentes Padding
            padding = max(5, min((x2-x1), (y2-y1)) // 10)

            # Erweiterte Bounding Box
            y1_pad = max(0, y1 - padding)
            x1_pad = max(0, x1 - padding)
            y2_pad = min(image.shape[0], y2 + padding)
            x2_pad = min(image.shape[1], x2 + padding)

            # Frame extrahieren
            frame_rgb = image[y1_pad:y2_pad, x1_pad:x2_pad]

            # Maske für Transparenz
            mask_region = comp['mask'][y1_pad:y2_pad, x1_pad:x2_pad]

            # RGBA Frame erstellen
            frame_rgba = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGBA)
            frame_rgba[:, :, 3] = mask_region * 255

            # Frame-Optimierung
            optimized_frame = self._optimize_frame(frame_rgba)

            frames.append(optimized_frame)

        return frames

    def _optimize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Optimiert einzelnen Frame"""
        # Zu PIL konvertieren für Verbesserungen
        pil_image = Image.fromarray(frame, 'RGBA')

        # Kontrast leicht erhöhen
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.1)

        # Schärfe leicht erhöhen
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.1)

        # Farben leicht verstärken
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(1.05)

        return np.array(pil_image)

    def upscale_frame_ai(self, frame: np.ndarray, scale_factor: int = 2) -> np.ndarray:
        """AI-basierte Frame-Hochskalierung"""
        # Für Pixel-Art: INTER_NEAREST
        # Für detaillierte Sprites: INTER_CUBIC

        # Bestimme Skalierungsalgorithmus basierend auf Frame-Charakteristiken
        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        if edge_density < 0.1:  # Wenige Edges = Pixel Art
            interpolation = cv2.INTER_NEAREST
        else:  # Viele Details = Smooth Scaling
            interpolation = cv2.INTER_CUBIC

        h, w = frame.shape[:2]
        new_h, new_w = h * scale_factor, w * scale_factor

        return cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    def process_single_spritesheet(self, image_path: Path) -> Dict[str, Any]:
        """Verarbeitet ein einzelnes Spritesheet mit AI"""
        print(f"\n🤖 AI Processing: {image_path.name}")

        try:
            # Lade Bild
            image = cv2.imread(str(image_path))
            if image is None:
                return {"success": False, "error": "Could not load image"}

            print(f"📊 Image: {image.shape[1]}x{image.shape[0]} pixels")

            # AI Background Detection
            mask, confidence, metadata = self.intelligent_background_detection(
                image)
            print(f"🎯 Background Detection Confidence: {confidence:.2f}")

            # AI Frame Extraction
            frames = self.advanced_frame_extraction(image, mask)
            print(f"🎮 Extracted {len(frames)} frames")

            if not frames:
                return {"success": False, "error": "No frames extracted"}

            # Speichere Ergebnisse
            sprite_name = image_path.stem
            sprite_output_dir = self.output_dir / sprite_name
            sprite_output_dir.mkdir(parents=True, exist_ok=True)

            saved_frames = []
            for i, frame in enumerate(frames):
                # AI Upscaling
                upscaled_frame = self.upscale_frame_ai(frame, scale_factor=2)

                # Speichere Frame
                frame_filename = f"ai_frame_{i:03d}.png"
                frame_path = sprite_output_dir / frame_filename

                pil_frame = Image.fromarray(upscaled_frame, 'RGBA')
                pil_frame.save(frame_path, 'PNG')
                saved_frames.append(str(frame_path))

            # Erstelle AI-Report
            ai_report = {
                "algorithm_used": metadata.get('algorithm', 'unknown'),
                "confidence": confidence,
                "frames_extracted": len(frames),
                "clustering_metadata": metadata
            }

            report_path = sprite_output_dir / "ai_analysis_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(ai_report, f, indent=2, ensure_ascii=False)

            # Erstelle animiertes GIF
            gif_path = sprite_output_dir / f"{sprite_name}_ai_animated.gif"
            gif_frames = [Image.fromarray(frame, 'RGBA') for frame in frames]

            if gif_frames:
                gif_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=gif_frames[1:],
                    duration=400,
                    loop=0,
                    disposal=2
                )

            print(f"✅ AI Processing Complete: {len(frames)} frames saved")
            return {
                "success": True,
                "frames_count": len(frames),
                "confidence": confidence,
                "algorithm": metadata.get('algorithm', 'unknown'),
                "output_dir": str(sprite_output_dir),
                "saved_frames": saved_frames,
                "gif_path": str(gif_path),
                "ai_report": ai_report
            }

        except Exception as e:
            print(f"❌ AI Processing Error: {e}")
            return {"success": False, "error": str(e)}

    def get_spritesheet_files(self) -> List[Path]:
        """Findet alle Spritesheet-Dateien"""
        sprite_files = []

        # Hauptverzeichnis
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            sprite_files.extend(self.input_dir.glob(ext))

        # Sprite-Sheets Unterverzeichnis
        sprite_sheets_dir = self.input_dir / "sprite_sheets"
        if sprite_sheets_dir.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                sprite_files.extend(sprite_sheets_dir.glob(ext))

        # Filtere nach Größe
        valid_files = []
        for file in sprite_files:
            try:
                if file.stat().st_size > 100_000:  # > 100KB
                    valid_files.append(file)
                    print(
                        f"🎯 AI Target: {file.name} ({file.stat().st_size // 1024}KB)")
            except:
                continue

        return valid_files

    def process_all_spritesheets_ai(self):
        """Verarbeitet alle Spritesheets mit AI"""
        print("🤖 AI BATCH SPRITESHEET PROCESSING")
        print("=" * 70)
        print("🧠 Using Advanced ML Algorithms:")
        print("  • K-Means Clustering with Silhouette Analysis")
        print("  • DBSCAN for Outlier Detection")
        print("  • Mean Shift Clustering")
        print("  • Gaussian Mixture Models")
        print("  • Watershed Segmentation")
        print("  • Intelligent Connected Components")
        print("  • AI-based Frame Filtering")
        print("=" * 70)

        # Erstelle Output-Verzeichnis
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Finde alle Spritesheet-Dateien
        sprite_files = self.get_spritesheet_files()

        if not sprite_files:
            print("❌ No spritesheet files found!")
            return

        print(f"\n🎯 Found {len(sprite_files)} files for AI processing")

        # Verarbeite jede Datei
        results = []
        for i, sprite_file in enumerate(sprite_files, 1):
            print(f"\n[{i}/{len(sprite_files)}] " + "="*50)

            start_time = time.time()
            result = self.process_single_spritesheet(sprite_file)
            end_time = time.time()

            result["filename"] = sprite_file.name
            result["processing_time"] = end_time - start_time
            results.append(result)

            if result["success"]:
                self.processed_count += 1
                print(f"✅ AI Success in {result['processing_time']:.2f}s")
                print(f"🎯 Algorithm: {result.get('algorithm', 'unknown')}")
                print(f"📊 Confidence: {result.get('confidence', 0):.2f}")
            else:
                self.failed_count += 1
                print(f"❌ AI Failed: {result.get('error', 'Unknown error')}")

        # Speichere Comprehensive AI Report
        comprehensive_report = {
            "processing_summary": {
                "total_files": len(sprite_files),
                "successful": self.processed_count,
                "failed": self.failed_count,
                "success_rate": self.processed_count / len(sprite_files) if sprite_files else 0
            },
            "detailed_results": results,
            "ai_algorithms_used": {
                "background_detection": ["K-Means", "DBSCAN", "Mean Shift", "Gaussian Mixture"],
                "segmentation": ["Watershed", "Connected Components"],
                "filtering": ["AI-based Component Analysis", "Outlier Detection"],
                "optimization": ["Intelligent Upscaling", "Frame Enhancement"]
            }
        }

        report_path = self.output_dir / "comprehensive_ai_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)

        # Finale Statistiken
        print("\n" + "="*70)
        print("🎉 AI BATCH PROCESSING COMPLETE!")
        print(f"✅ Successfully processed: {self.processed_count}")
        print(f"❌ Failed: {self.failed_count}")
        print(f"📊 Total files: {len(sprite_files)}")
        print(
            f"💯 AI Success rate: {(self.processed_count / len(sprite_files) * 100):.1f}%")
        print(f"📋 Comprehensive report: {report_path}")
        print("🧠 AI Algorithms provided intelligent sprite analysis!")


def main():
    """Hauptausführung"""
    processor = AISpritesheetProcessor()
    processor.process_all_spritesheets_ai()


if __name__ == "__main__":
    print("🚀 AI-BASED SPRITESHEET PROCESSOR")
    print("Powered by Machine Learning & Computer Vision")

    main()
