#!/usr/bin/env python3
"""
INTELLIGENT SPRITESHEET PROCESSOR
Automatische Frame-Erkennung durch Connected Component Analysis
Löst das Problem unregelmäßiger Spritesheet-Layouts vollständig!
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
from pathlib import Path
import json
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


class IntelligentSpritesheetProcessor:
    def __init__(self, output_dir="output/intelligent_sprites"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_dir / "extracted_frames"
        self.analysis_dir = self.output_dir / "analysis"
        self.aligned_dir = self.output_dir / "aligned_frames"

        for dir_path in [self.frames_dir, self.analysis_dir, self.aligned_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def remove_background_advanced(self, image: np.ndarray, method="adaptive") -> np.ndarray:
        """Erweiterte Hintergrundentfernung mit mehreren Methoden"""

        if len(image.shape) == 3:
            # Zu RGBA konvertieren
            if image.shape[2] == 3:
                image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            else:
                image_rgba = image.copy()
        else:
            image_rgba = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)

        if method == "corner_detection":
            # Hintergrundfarbe aus Ecken ermitteln
            h, w = image.shape[:2]
            corner_size = min(h, w) // 20

            corners = [
                image[0:corner_size, 0:corner_size],
                image[0:corner_size, w-corner_size:w],
                image[h-corner_size:h, 0:corner_size],
                image[h-corner_size:h, w-corner_size:w]
            ]

            # Häufigste Eckfarbe
            corner_pixels = []
            for corner in corners:
                if len(corner.shape) == 3:
                    pixels = corner.reshape(-1, 3)
                    for pixel in pixels:
                        corner_pixels.append(tuple(pixel))

            from collections import Counter
            bg_color = Counter(corner_pixels).most_common(1)[0][0]

            # Maske erstellen
            tolerance = 30
            if len(image.shape) == 3:
                diff = np.abs(image.astype(int) -
                              np.array(bg_color, dtype=int))
                mask = np.all(diff <= tolerance, axis=2)
            else:
                mask = np.abs(image.astype(int) - int(bg_color)) <= tolerance

            # Alpha-Kanal setzen
            image_rgba[mask, 3] = 0  # Transparent
            image_rgba[~mask, 3] = 255  # Opaque

        elif method == "adaptive":
            # Adaptiver Threshold für komplexere Hintergründe
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(
                image.shape) == 3 else image

            # Otsu's threshold für automatische Schwellenwertfindung
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphological operations um Noise zu entfernen
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # Alpha-Kanal basierend auf binary mask
            image_rgba[:, :, 3] = binary

        return image_rgba

    def find_connected_components(self, image_rgba: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Findet zusammenhängende Komponenten (Frames) nach Background Removal"""

        # Alpha-Kanal als Maske verwenden
        alpha = image_rgba[:, :, 3]

        # Connected Components Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            alpha, connectivity=8
        )

        components = []

        # Durch alle Components iterieren (außer Background = Label 0)
        for i in range(1, num_labels):
            # Statistiken der Component
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]

            # Filtere zu kleine Components (Noise)
            min_area = 500  # Mindestgröße für einen Frame
            if area < min_area:
                continue

            # Filtere zu dünne/breite Components
            aspect_ratio = w / h if h > 0 else float('inf')
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                continue

            component_info = {
                'id': i,
                'bbox': (x, y, w, h),
                'area': area,
                'centroid': (cx, cy),
                'aspect_ratio': aspect_ratio
            }

            components.append(component_info)

        print(f"🔍 Gefunden: {len(components)} potentielle Frames")
        return components, labels

    def extract_frame_from_component(self, image_rgba: np.ndarray, component: Dict) -> np.ndarray:
        """Extrahiert einen Frame basierend auf Connected Component"""

        x, y, w, h = component['bbox']

        # Frame mit etwas Padding extrahieren
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image_rgba.shape[1], x + w + padding)
        y_end = min(image_rgba.shape[0], y + h + padding)

        frame = image_rgba[y_start:y_end, x_start:x_end]

        return frame

    def cluster_frames_by_size(self, components: List[Dict]) -> Dict[str, List[Dict]]:
        """Clustert Frames nach ähnlicher Größe für bessere Organisation"""

        if not components:
            return {}

        # Features für Clustering: [width, height, area]
        features = []
        for comp in components:
            x, y, w, h = comp['bbox']
            features.append([w, h, comp['area']])

        features = np.array(features)

        # DBSCAN Clustering für ähnliche Größen
        clustering = DBSCAN(eps=50, min_samples=1).fit(features)
        labels = clustering.labels_

        # Gruppiere Components nach Cluster
        clusters = {}
        for i, label in enumerate(labels):
            cluster_name = f"size_group_{label}"
            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append(components[i])

        # Sortiere Clusters nach durchschnittlicher Größe
        sorted_clusters = {}
        for name, cluster_components in clusters.items():
            avg_area = np.mean([c['area'] for c in cluster_components])
            sorted_name = f"group_{len(sorted_clusters)+1}_avg_area_{int(avg_area)}"
            sorted_clusters[sorted_name] = cluster_components

        return sorted_clusters

    def align_frames_in_grid(self, frames: List[np.ndarray], target_size: Tuple[int, int] = None) -> List[np.ndarray]:
        """Richtet Frames in einem einheitlichen Grid aus"""

        if not frames:
            return frames

        # Bestimme Zielgröße falls nicht angegeben
        if target_size is None:
            # Verwende die häufigste Größe als Referenz
            sizes = [frame.shape[:2] for frame in frames]
            from collections import Counter
            target_size = Counter(sizes).most_common(1)[0][0]

        target_h, target_w = target_size
        aligned_frames = []

        for i, frame in enumerate(frames):
            # Resize Frame auf Zielgröße mit Aspect Ratio Preservation
            h, w = frame.shape[:2]

            # Berechne Skalierungsfaktor
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)

            # Resize
            resized = cv2.resize(frame, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)

            # Erstelle Canvas mit Zielgröße
            canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)

            # Zentriere resized frame auf canvas
            start_y = (target_h - new_h) // 2
            start_x = (target_w - new_w) // 2
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized

            aligned_frames.append(canvas)

        return aligned_frames

    def create_analysis_visualization(self, original: np.ndarray, labels: np.ndarray,
                                      components: List[Dict], output_path: Path):
        """Erstellt Visualisierung der Analyse für Debugging"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Original
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Spritesheet")
        axes[0, 0].axis('off')

        # Nach Background Removal
        alpha_mask = labels > 0
        masked = original.copy()
        masked[~alpha_mask] = [255, 0, 255]  # Magenta für removed background
        axes[0, 1].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("Nach Background Removal")
        axes[0, 1].axis('off')

        # Connected Components
        colored_labels = cv2.applyColorMap(
            (labels * 255 / labels.max()).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        axes[1, 0].imshow(colored_labels)
        axes[1, 0].set_title("Connected Components")
        axes[1, 0].axis('off')

        # Detected Frames with Bounding Boxes
        result = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        for i, comp in enumerate(components):
            x, y, w, h = comp['bbox']
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result, f"Frame {i+1}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        axes[1, 1].imshow(result)
        axes[1, 1].set_title(f"Erkannte Frames: {len(components)}")
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def process_spritesheet(self, image_path: str, bg_removal_method="adaptive") -> Dict:
        """Hauptfunktion: Verarbeitet ein Spritesheet vollständig"""

        print(f"\n🎮 Verarbeite Spritesheet: {Path(image_path).name}")

        # Lade Bild
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Konnte Bild nicht laden: {image_path}")

        # Konvertiere zu RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        print(f"📐 Bildgröße: {image.shape}")

        # 1. Background Removal
        print("🎯 Entferne Hintergrund...")
        image_no_bg = self.remove_background_advanced(
            image_rgb, method=bg_removal_method)

        # 2. Connected Components Analysis
        print("🔍 Analysiere Connected Components...")
        components, labels = self.find_connected_components(image_no_bg)

        if not components:
            print("❌ Keine Frames gefunden!")
            return {'error': 'Keine Frames erkannt'}

        # 3. Extrahiere Frames
        print("📦 Extrahiere Frames...")
        extracted_frames = []
        frame_info = []

        for i, comp in enumerate(components):
            frame = self.extract_frame_from_component(image_no_bg, comp)
            extracted_frames.append(frame)

            # Speichere Frame
            frame_filename = f"frame_{i+1:03d}.png"
            frame_path = self.frames_dir / frame_filename

            # Konvertiere zu PIL und speichere
            pil_frame = Image.fromarray(frame, 'RGBA')
            pil_frame.save(frame_path)

            frame_info.append({
                'id': i+1,
                'filename': frame_filename,
                'bbox': comp['bbox'],
                'area': comp['area'],
                'centroid': comp['centroid']
            })

        # 4. Clustere nach Größe
        print("📊 Clustere Frames nach Größe...")
        clusters = self.cluster_frames_by_size(components)

        # 5. Frame Alignment
        print("⚙️ Richte Frames aus...")
        aligned_frames = self.align_frames_in_grid(extracted_frames)

        # Speichere aligned frames
        for i, aligned_frame in enumerate(aligned_frames):
            aligned_filename = f"aligned_frame_{i+1:03d}.png"
            aligned_path = self.aligned_dir / aligned_filename

            pil_aligned = Image.fromarray(aligned_frame, 'RGBA')
            pil_aligned.save(aligned_path)

        # 6. Erstelle Analyse-Visualisierung
        print("📈 Erstelle Analyse-Visualisierung...")
        analysis_path = self.analysis_dir / \
            f"analysis_{Path(image_path).stem}.png"
        self.create_analysis_visualization(
            image, labels, components, analysis_path)

        # 7. Erstelle JSON Report
        report = {
            'input_file': str(image_path),
            'processing_method': bg_removal_method,
            'total_frames': len(components),
            'image_size': image.shape[:2],
            'frames': frame_info,
            'size_clusters': {name: len(cluster) for name, cluster in clusters.items()},
            'output_dirs': {
                'frames': str(self.frames_dir),
                'aligned': str(self.aligned_dir),
                'analysis': str(self.analysis_dir)
            }
        }

        report_path = self.analysis_dir / \
            f"report_{Path(image_path).stem}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✅ Erfolgreich verarbeitet!")
        print(f"   📦 {len(components)} Frames extrahiert")
        print(f"   📁 Gespeichert in: {self.output_dir}")
        print(f"   📊 Analyse: {analysis_path}")

        return report

    def create_gif_from_frames(self, frame_dir: Path, output_path: Path, duration=500):
        """Erstellt GIF aus extrahierten Frames"""

        frame_files = sorted(frame_dir.glob("*.png"))
        if not frame_files:
            print("❌ Keine Frames für GIF gefunden")
            return

        frames = []
        for frame_file in frame_files:
            frame = Image.open(frame_file)
            frames.append(frame)

        # Speichere als GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            transparency=0,
            disposal=2
        )

        print(f"🎬 GIF erstellt: {output_path}")


def main():
    """Test der intelligenten Spritesheet-Verarbeitung"""

    processor = IntelligentSpritesheetProcessor()

    # Teste mit vorhandenen Bildern
    test_images = [
        "input/Mann_steigt_aus_Limousine_aus.png",
        "input/2D_Sprites_des_Mannes_im_Anzug.png",
        "input/Kampfer_im_Anzug_mit_Waffe.png"
    ]

    for image_path in test_images:
        if os.path.exists(image_path):
            try:
                # Verarbeite mit verschiedenen Methoden
                for method in ["corner_detection", "adaptive"]:
                    print(f"\n{'='*60}")
                    print(f"Testing {method} method on {image_path}")
                    print(f"{'='*60}")

                    result = processor.process_spritesheet(
                        image_path, bg_removal_method=method)

                    if 'error' not in result:
                        # Erstelle GIF aus aligned frames
                        gif_path = processor.output_dir / \
                            f"animation_{Path(image_path).stem}_{method}.gif"
                        processor.create_gif_from_frames(
                            processor.aligned_dir, gif_path)

            except Exception as e:
                print(f"❌ Fehler bei {image_path}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
