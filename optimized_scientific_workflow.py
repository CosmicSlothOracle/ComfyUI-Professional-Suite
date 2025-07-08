#!/usr/bin/env python3
"""
OPTIMIERTER WISSENSCHAFTLICHER SPRITESHEET-WORKFLOW
Basierend auf kritischer Selbstanalyse implementiert:
- Systematische Multi-Algorithmus Evaluation
- Quantitative Metriken und Validierung
- Parallele Verarbeitung für Effizienz
- Wissenschaftlich fundierte Methodologie
"""

import cv2
import numpy as np
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from sklearn.cluster import KMeans
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class EvaluationResult:
    """Struktur für Evaluationsergebnisse"""
    image: str
    algorithm: str
    frame_count: int
    processing_time: float
    foreground_ratio: float
    consistency_score: float
    success: bool
    error: str = ""


class OptimizedSegmentationFramework:
    """Optimiertes Framework für systematische Spritesheet-Segmentierung"""

    def __init__(self):
        self.algorithms = {}
        self.register_algorithms()

    def register_algorithms(self):
        """Registriert alle Test-Algorithmen"""

        def four_corner_bg_detection(image):
            """Original 4-Ecken Methode"""
            h, w = image.shape[:2]
            corners = [image[0, 0], image[0, w-1],
                       image[h-1, 0], image[h-1, w-1]]
            bg_color = np.mean(corners, axis=0)
            diff = np.abs(image.astype(float) - bg_color.astype(float))
            distance = np.sqrt(np.sum(diff**2, axis=2))
            return (distance > 25).astype(np.uint8) * 255

        def twelve_point_bg_detection(image):
            """12-Punkt Methode mit K-Means"""
            h, w = image.shape[:2]
            points = []

            # Sample 12 border points
            for i in range(3):
                x = int(w * (i + 1) / 4)
                points.extend([image[0, x], image[h-1, x]])
            for i in range(3):
                y = int(h * (i + 1) / 4)
                points.extend([image[y, 0], image[y, w-1]])

            points = np.array(points)
            if len(points) >= 2:
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                labels = kmeans.fit_predict(points)
                cluster_sizes = [np.sum(labels == 0), np.sum(labels == 1)]
                bg_cluster = 0 if cluster_sizes[0] > cluster_sizes[1] else 1
                bg_color = kmeans.cluster_centers_[bg_cluster]
            else:
                bg_color = np.mean(points, axis=0)

            # Adaptive tolerance
            image_variance = np.var(image.reshape(-1, 3), axis=0)
            tolerance = max(25, np.mean(image_variance) * 0.5)

            diff = np.abs(image.astype(float) - bg_color.astype(float))
            distance = np.sqrt(np.sum(diff**2, axis=2))
            return (distance > tolerance).astype(np.uint8) * 255

        def sixteen_point_bg_detection(image):
            """16-Punkt Methode (Enhanced)"""
            h, w = image.shape[:2]
            points = []

            # Sample 16 border points systematisch
            for i in range(4):
                x = int(w * (i + 1) / 5)
                points.extend([image[0, x], image[h-1, x]])
            for i in range(4):
                y = int(h * (i + 1) / 5)
                points.extend([image[y, 0], image[y, w-1]])

            points = np.array(points)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(points)
            cluster_sizes = [np.sum(labels == 0), np.sum(labels == 1)]
            bg_cluster = 0 if cluster_sizes[0] > cluster_sizes[1] else 1
            bg_color = kmeans.cluster_centers_[bg_cluster]

            # Enhanced adaptive tolerance
            image_std = np.std(image.reshape(-1, 3), axis=0)
            tolerance = max(30, np.mean(image_std) * 0.8)

            diff = np.abs(image.astype(float) - bg_color.astype(float))
            distance = np.sqrt(np.sum(diff**2, axis=2))
            return (distance > tolerance).astype(np.uint8) * 255

        def twenty_point_bg_detection(image):
            """20-Punkt Methode für erhöhte Robustheit"""
            h, w = image.shape[:2]
            points = []

            # Sample 20 border points
            for i in range(5):
                x = int(w * (i + 1) / 6)
                points.extend([image[0, x], image[h-1, x]])
            for i in range(5):
                y = int(h * (i + 1) / 6)
                points.extend([image[y, 0], image[y, w-1]])

            points = np.array(points)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(points)
            cluster_sizes = [np.sum(labels == 0), np.sum(labels == 1)]
            bg_cluster = 0 if cluster_sizes[0] > cluster_sizes[1] else 1
            bg_color = kmeans.cluster_centers_[bg_cluster]

            # Ultra-adaptive tolerance
            image_stats = np.std(image.reshape(-1, 3), axis=0)
            tolerance = max(35, np.mean(image_stats) * 1.0)

            diff = np.abs(image.astype(float) - bg_color.astype(float))
            distance = np.sqrt(np.sum(diff**2, axis=2))
            return (distance > tolerance).astype(np.uint8) * 255

        def edge_detection_bg(image):
            """Edge-Detection basierte Segmentierung"""
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    cv2.fillPoly(mask, [contour], 255)
            return mask

        def watershed_bg_detection(image):
            """Watershed-basierte Segmentierung"""
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Noise removal
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(
                gray, cv2.MORPH_OPEN, kernel, iterations=2)

            # Sure background
            sure_bg = cv2.dilate(opening, kernel, iterations=3)

            # Sure foreground
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(
                dist_transform, 0.7*dist_transform.max(), 255, 0)

            # Unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)

            # Marker labelling
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0

            # Apply watershed
            markers = cv2.watershed(image, markers)

            # Create foreground mask
            foreground_mask = (markers > 1).astype(np.uint8) * 255
            return foreground_mask

        self.algorithms = {
            '4-Corner': four_corner_bg_detection,
            '12-Point': twelve_point_bg_detection,
            '16-Point': sixteen_point_bg_detection,
            '20-Point': twenty_point_bg_detection,
            'Edge-Detection': edge_detection_bg,
            'Watershed': watershed_bg_detection
        }

    def extract_frames(self, foreground_mask):
        """Extrahiert Frames mit Connected Components"""
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8)

        frames = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            # Verbesserte Filterkriterien
            if area > 100 and 0.1 < w/h < 10 and w > 10 and h > 10:
                frames.append((x, y, w, h))
        return frames

    def calculate_consistency_score(self, frames):
        """Berechnet Konsistenz-Score basierend auf Frame-Größen"""
        if len(frames) < 2:
            return 1.0

        areas = [w * h for x, y, w, h in frames]
        mean_area = np.mean(areas)
        std_area = np.std(areas)

        # Coefficient of Variation (niedrigere Werte = konsistenter)
        cv = std_area / mean_area if mean_area > 0 else float('inf')

        # Normalisiert auf 0-1 Skala (1 = perfekt konsistent)
        consistency = 1 / (1 + cv)
        return consistency

    def evaluate_single_image(self, image_path, algorithm_name, bg_detector):
        """Evaluiert einen Algorithmus auf einem Bild"""
        try:
            start_time = time.time()
            image = cv2.imread(image_path)
            if image is None:
                return EvaluationResult(
                    image=os.path.basename(image_path),
                    algorithm=algorithm_name,
                    frame_count=0,
                    processing_time=0,
                    foreground_ratio=0,
                    consistency_score=0,
                    success=False,
                    error="Could not load image"
                )

            # Background detection
            foreground_mask = bg_detector(image)

            # Frame extraction
            frames = self.extract_frames(foreground_mask)

            processing_time = time.time() - start_time

            # Metrics
            total_pixels = image.shape[0] * image.shape[1]
            foreground_pixels = np.sum(foreground_mask > 0)
            foreground_ratio = foreground_pixels / total_pixels

            # Consistency score
            consistency_score = self.calculate_consistency_score(frames)

            return EvaluationResult(
                image=os.path.basename(image_path),
                algorithm=algorithm_name,
                frame_count=len(frames),
                processing_time=processing_time,
                foreground_ratio=foreground_ratio,
                consistency_score=consistency_score,
                success=True
            )

        except Exception as e:
            return EvaluationResult(
                image=os.path.basename(image_path),
                algorithm=algorithm_name,
                frame_count=0,
                processing_time=0,
                foreground_ratio=0,
                consistency_score=0,
                success=False,
                error=str(e)
            )

    def run_systematic_evaluation(self, test_images, max_workers=4):
        """Führt systematische Evaluation aller Algorithmen durch"""
        print(
            f"🔬 Evaluating {len(self.algorithms)} algorithms on {len(test_images)} images...")

        tasks = []
        for img_path in test_images:
            for algo_name, bg_detector in self.algorithms.items():
                tasks.append((img_path, algo_name, bg_detector))

        print(f"📊 Total evaluations to perform: {len(tasks)}")

        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(
                self.evaluate_single_image, *task) for task in tasks]

            results = []
            completed = 0
            for future in futures:
                result = future.result()
                results.append(result)
                completed += 1

                if completed % 20 == 0:
                    print(
                        f"✅ Completed {completed}/{len(futures)} evaluations...")

        print(f"🎯 Evaluation complete: {len(results)} results generated")
        return results

    def analyze_results(self, results):
        """Analysiert Ergebnisse und generiert wissenschaftlichen Report"""
        # Convert to DataFrame
        data = []
        for result in results:
            data.append({
                'image': result.image,
                'algorithm': result.algorithm,
                'frame_count': result.frame_count,
                'processing_time': result.processing_time,
                'foreground_ratio': result.foreground_ratio,
                'consistency_score': result.consistency_score,
                'success': result.success,
                'error': result.error
            })

        df = pd.DataFrame(data)

        # Filter successful runs
        successful_df = df[df['success'] == True]

        if successful_df.empty:
            return {
                'error': 'No successful evaluations',
                'total_evaluations': len(df),
                'success_rate': 0
            }

        # Algorithm performance statistics
        algo_stats = successful_df.groupby('algorithm').agg({
            'frame_count': ['mean', 'std', 'min', 'max', 'median'],
            'processing_time': ['mean', 'std'],
            'foreground_ratio': ['mean', 'std'],
            'consistency_score': ['mean', 'std']
        }).round(4)

        # Success rates
        success_rates = df.groupby('algorithm')['success'].mean()

        # Consistency analysis (Coefficient of Variation for frame counts)
        consistency_analysis = successful_df.groupby('algorithm')['frame_count'].apply(
            lambda x: {
                'cv': x.std() / x.mean() if x.mean() > 0 else float('inf'),
                'range': x.max() - x.min(),
                'iqr': x.quantile(0.75) - x.quantile(0.25)
            }
        )

        # Performance ranking
        performance_scores = {}
        for algo in successful_df['algorithm'].unique():
            algo_data = successful_df[successful_df['algorithm'] == algo]

            # Composite score: niedrige Varianz + realistische Frame-Zahlen + schnelle Verarbeitung
            mean_frames = algo_data['frame_count'].mean()
            cv_frames = algo_data['frame_count'].std(
            ) / mean_frames if mean_frames > 0 else float('inf')
            mean_time = algo_data['processing_time'].mean()
            mean_consistency = algo_data['consistency_score'].mean()

            # Scoring (niedrigere CV und Zeit = besser, höhere Konsistenz = besser)
            score = (mean_consistency * 100) / \
                (cv_frames + 0.01) / (mean_time + 0.01)
            performance_scores[algo] = score

        # Sort by performance
        ranked_algorithms = sorted(
            performance_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            'algorithm_statistics': algo_stats,
            'success_rates': success_rates.to_dict(),
            'consistency_analysis': {k: v for k, v in consistency_analysis.items()},
            'performance_ranking': ranked_algorithms,
            'total_evaluations': len(df),
            'successful_evaluations': len(successful_df),
            'overall_success_rate': len(successful_df) / len(df),
            'raw_data': df
        }

    def generate_scientific_report(self, analysis):
        """Generiert wissenschaftlichen Evaluations-Report"""
        report = f"""
=== WISSENSCHAFTLICHER EVALUATIONS-REPORT ===
SYSTEMATISCHE SPRITESHEET-SEGMENTIERUNG ANALYSE

METHODOLOGIE:
- {len(self.algorithms)} Algorithmen getestet
- {analysis['total_evaluations']} Evaluationen durchgeführt
- Parallelverarbeitung für Effizienz
- Quantitative Metriken: Frame Count, Processing Time, Consistency Score

ERFOLGSRATEN:
"""
        for algo, rate in analysis['success_rates'].items():
            report += f"  {algo}: {rate:.1%}\n"

        report += f"\nGESAMT-ERFOLGSRATE: {analysis['overall_success_rate']:.1%}\n"

        report += "\nPERFORMANCE RANKING:\n"
        for i, (algo, score) in enumerate(analysis['performance_ranking'], 1):
            report += f"  {i}. {algo}: {score:.2f}\n"

        if 'algorithm_statistics' in analysis:
            report += "\nFRAME COUNT STATISTIKEN:\n"
            for algo in analysis['algorithm_statistics'].index:
                stats = analysis['algorithm_statistics'].loc[algo]
                mean_frames = stats[('frame_count', 'mean')]
                std_frames = stats[('frame_count', 'std')]
                mean_time = stats[('processing_time', 'mean')]
                consistency = stats[('consistency_score', 'mean')]

                cv = analysis['consistency_analysis'][algo]['cv']

                report += f"  {algo}:\n"
                report += f"    Frames: {mean_frames:.1f} ± {std_frames:.1f}\n"
                report += f"    Variationskoeffizient: {cv:.3f}\n"
                report += f"    Konsistenz-Score: {consistency:.3f}\n"
                report += f"    Verarbeitungszeit: {mean_time:.3f}s\n\n"

        report += "=== EMPFEHLUNGEN ===\n"

        if analysis['performance_ranking']:
            best_algo = analysis['performance_ranking'][0][0]
            report += f"EMPFOHLENER ALGORITHMUS: {best_algo}\n"

            best_stats = analysis['algorithm_statistics'].loc[best_algo]
            best_cv = analysis['consistency_analysis'][best_algo]['cv']

            report += f"Begründung:\n"
            report += f"- Konsistenteste Ergebnisse (CV: {best_cv:.3f})\n"
            report += f"- Durchschnittlich {best_stats[('frame_count', 'mean')]:.1f} Frames detektiert\n"
            report += f"- Verarbeitungszeit: {best_stats[('processing_time', 'mean')]:.3f}s\n"

        return report

    def save_comprehensive_results(self, analysis, results, output_dir="scientific_evaluation_results"):
        """Speichert umfassende Evaluations-Ergebnisse"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())

        # CSV Export
        csv_path = os.path.join(output_dir, f"evaluation_data_{timestamp}.csv")
        analysis['raw_data'].to_csv(csv_path, index=False)

        # JSON Report
        json_data = {
            'metadata': {
                'timestamp': timestamp,
                'algorithms_tested': list(self.algorithms.keys()),
                'total_evaluations': analysis['total_evaluations'],
                'methodology': 'Systematic multi-algorithm evaluation with quantitative metrics'
            },
            'results': {
                'success_rates': analysis['success_rates'],
                'performance_ranking': analysis['performance_ranking'],
                'consistency_analysis': analysis['consistency_analysis']
            },
            'raw_results': [r.__dict__ for r in results]
        }

        json_path = os.path.join(
            output_dir, f"scientific_report_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        # Text Report
        report_text = self.generate_scientific_report(analysis)
        report_path = os.path.join(
            output_dir, f"scientific_report_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)

        return {
            'csv_path': csv_path,
            'json_path': json_path,
            'report_path': report_path,
            'output_dir': output_dir
        }


def main():
    """Hauptfunktion für optimierten wissenschaftlichen Workflow"""
    print("🔬 === OPTIMIERTER WISSENSCHAFTLICHER SPRITESHEET-WORKFLOW ===")
    print("Implementiert systematische Multi-Algorithmus Evaluation\n")

    # Initialize framework
    framework = OptimizedSegmentationFramework()

    # Get test images
    input_dir = Path('input')
    test_images = list(input_dir.glob('*.png'))

    if not test_images:
        print("❌ No test images found in input directory!")
        return

    # Limit for demonstration (remove in production)
    max_images = 30
    if len(test_images) > max_images:
        test_images = test_images[:max_images]
        print(f"📝 Limited to first {max_images} images for demonstration")

    print(f"📂 Found {len(test_images)} test images")
    print(f"🧪 Testing {len(framework.algorithms)} algorithms:")
    for algo in framework.algorithms.keys():
        print(f"   - {algo}")

    # Run systematic evaluation
    start_time = time.time()
    results = framework.run_systematic_evaluation(
        [str(img) for img in test_images])
    evaluation_time = time.time() - start_time

    # Analyze results
    analysis = framework.analyze_results(results)

    if 'error' in analysis:
        print(f"❌ Error in analysis: {analysis['error']}")
        return

    # Display results
    print(f"\n🎯 === EVALUATION RESULTS ({evaluation_time:.1f}s) ===")
    print(f"Total evaluations: {analysis['total_evaluations']}")
    print(f"Successful: {analysis['successful_evaluations']}")
    print(f"Overall success rate: {analysis['overall_success_rate']:.1%}")

    print("\n🏆 PERFORMANCE RANKING:")
    for i, (algo, score) in enumerate(analysis['performance_ranking'], 1):
        success_rate = analysis['success_rates'][algo]
        print(f"  {i}. {algo}: {score:.2f} (Success: {success_rate:.1%})")

    # Generate and display scientific report
    report = framework.generate_scientific_report(analysis)
    print("\n" + report)

    # Save comprehensive results
    file_paths = framework.save_comprehensive_results(analysis, results)

    print(f"💾 Results saved to: {file_paths['output_dir']}/")
    print(f"   - CSV data: {os.path.basename(file_paths['csv_path'])}")
    print(f"   - JSON report: {os.path.basename(file_paths['json_path'])}")
    print(f"   - Text report: {os.path.basename(file_paths['report_path'])}")

    print("\n✅ === WISSENSCHAFTLICHE EVALUATION ABGESCHLOSSEN ===")

    # Recommendation
    if analysis['performance_ranking']:
        best_algo = analysis['performance_ranking'][0][0]
        print(f"\n🎯 EMPFEHLUNG: Verwende {best_algo} für optimale Ergebnisse!")


if __name__ == "__main__":
    main()
