#!/usr/bin/env python3
"""
Wissenschaftliches Validierungs-Framework für Spritesheet-Segmentierung
Implementiert systematische Evaluation mit quantitativen Metriken
"""

import cv2
import numpy as np
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


@dataclass
class SegmentationResult:
    """Ergebnis einer Segmentierung"""
    algorithm_name: str
    filename: str
    frame_count: int
    processing_time: float
    foreground_ratio: float
    components_found: int
    frame_sizes: List[Tuple[int, int]]


@dataclass
class ValidationMetrics:
    """Validierungs-Metriken"""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    frame_count_error: float
    processing_time: float


class BackgroundDetector:
    """Basis-Klasse für Background Detection Algorithmen"""

    def __init__(self, name: str):
        self.name = name

    def detect_background(self, image: np.ndarray) -> np.ndarray:
        """Detektiert Background und returns Foreground-Maske"""
        raise NotImplementedError


class FourCornerDetector(BackgroundDetector):
    """Original 4-Ecken Background Detection"""

    def __init__(self):
        super().__init__("4-Corner")

    def detect_background(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]

        # Sample 4 corners
        corners = [
            image[0, 0],           # top-left
            image[0, w-1],         # top-right
            image[h-1, 0],         # bottom-left
            image[h-1, w-1]        # bottom-right
        ]

        # Average corner color
        bg_color = np.mean(corners, axis=0)

        # Create mask
        tolerance = 25
        diff = np.abs(image.astype(float) - bg_color.astype(float))
        distance = np.sqrt(np.sum(diff**2, axis=2))
        foreground_mask = distance > tolerance

        return foreground_mask.astype(np.uint8) * 255


class MultiPointDetector(BackgroundDetector):
    """16-Punkt Multi-Zone Background Detection"""

    def __init__(self, num_points: int = 16):
        super().__init__(f"{num_points}-Point")
        self.num_points = num_points

    def detect_background(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]

        # Generate sampling points around border
        points = []

        # Top and bottom edges
        for i in range(self.num_points // 4):
            x = int(w * (i + 1) / (self.num_points // 4 + 1))
            points.append(image[0, x])      # top
            points.append(image[h-1, x])    # bottom

        # Left and right edges
        for i in range(self.num_points // 4):
            y = int(h * (i + 1) / (self.num_points // 4 + 1))
            points.append(image[y, 0])      # left
            points.append(image[y, w-1])    # right

        # K-means clustering for background color
        points = np.array(points)
        if len(points) >= 2:
            kmeans = KMeans(n_clusters=2, random_state=42)
            labels = kmeans.fit_predict(points)

            # Larger cluster is likely background
            cluster_sizes = [np.sum(labels == 0), np.sum(labels == 1)]
            bg_cluster = 0 if cluster_sizes[0] > cluster_sizes[1] else 1
            bg_color = kmeans.cluster_centers_[bg_cluster]
        else:
            bg_color = np.mean(points, axis=0)

        # Create adaptive tolerance mask
        image_variance = np.var(image.reshape(-1, 3), axis=0)
        adaptive_tolerance = max(25, np.mean(image_variance) * 0.5)

        diff = np.abs(image.astype(float) - bg_color.astype(float))
        distance = np.sqrt(np.sum(diff**2, axis=2))
        foreground_mask = distance > adaptive_tolerance

        return foreground_mask.astype(np.uint8) * 255


class WatershedDetector(BackgroundDetector):
    """Watershed-basierte Segmentierung"""

    def __init__(self):
        super().__init__("Watershed")

    def detect_background(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Sure foreground area
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

        # Create foreground mask
        foreground_mask = (markers > 1).astype(np.uint8) * 255

        return foreground_mask


class EdgeDetector(BackgroundDetector):
    """Edge-Detection basierte Segmentierung"""

    def __init__(self):
        super().__init__("Edge-Detection")

    def detect_background(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges to create regions
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Fill contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros(gray.shape, dtype=np.uint8)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                cv2.fillPoly(mask, [contour], 255)

        return mask


class ScientificValidator:
    """Wissenschaftlicher Validator für Spritesheet-Segmentierung"""

    def __init__(self, ground_truth_file: str = "spritesheet_ground_truth.json"):
        self.ground_truth_file = ground_truth_file
        self.ground_truth = self.load_ground_truth()

        # Initialize algorithms
        self.algorithms = [
            FourCornerDetector(),
            MultiPointDetector(12),
            MultiPointDetector(16),
            MultiPointDetector(20),
            WatershedDetector(),
            EdgeDetector()
        ]

    def load_ground_truth(self) -> Dict:
        """Lädt Ground Truth Daten"""
        if os.path.exists(self.ground_truth_file):
            with open(self.ground_truth_file, 'r') as f:
                return json.load(f)
        else:
            print(
                f"Warning: Ground truth file {self.ground_truth_file} not found!")
            return {}

    def extract_frames(self, foreground_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Extrahiert Frames mit Connected Components"""
        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8)

        frames = []
        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, w, h, area = stats[i]

            # Filter by area and aspect ratio
            if area > 100 and 0.1 < w/h < 10:
                frames.append((x, y, w, h))

        return frames

    def evaluate_algorithm(self, algorithm: BackgroundDetector, image_path: str) -> SegmentationResult:
        """Evaluiert einen Algorithmus auf einem Bild"""
        start_time = time.time()

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Detect background
        foreground_mask = algorithm.detect_background(image)

        # Extract frames
        frames = self.extract_frames(foreground_mask)

        processing_time = time.time() - start_time

        # Calculate metrics
        total_pixels = image.shape[0] * image.shape[1]
        foreground_pixels = np.sum(foreground_mask > 0)
        foreground_ratio = foreground_pixels / total_pixels

        frame_sizes = [(w, h) for x, y, w, h in frames]

        return SegmentationResult(
            algorithm_name=algorithm.name,
            filename=image_path,
            frame_count=len(frames),
            processing_time=processing_time,
            foreground_ratio=foreground_ratio,
            components_found=len(frames),
            frame_sizes=frame_sizes
        )

    def calculate_validation_metrics(self, result: SegmentationResult, ground_truth_count: int) -> ValidationMetrics:
        """Berechnet Validierungs-Metriken"""
        predicted_count = result.frame_count

        # Frame count accuracy
        frame_count_error = abs(
            predicted_count - ground_truth_count) / ground_truth_count

        # Binary classification metrics (simplified)
        # True if frame count is within 20% of ground truth
        tolerance = 0.2
        correct_prediction = frame_count_error <= tolerance

        accuracy = 1.0 if correct_prediction else 0.0
        precision = accuracy  # Simplified for frame count prediction
        recall = accuracy
        f1 = accuracy

        return ValidationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            frame_count_error=frame_count_error,
            processing_time=result.processing_time
        )

    def run_comprehensive_evaluation(self, test_images: List[str] = None) -> pd.DataFrame:
        """Führt umfassende Evaluation aller Algorithmen durch"""
        if test_images is None:
            # Use all images with ground truth
            test_images = list(self.ground_truth.keys())

        if not test_images:
            print("No test images available!")
            return pd.DataFrame()

        print(
            f"Running evaluation on {len(test_images)} images with {len(self.algorithms)} algorithms...")

        results = []

        for img_path in test_images:
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue

            ground_truth_count = self.ground_truth.get(
                img_path, {}).get('frame_count', 0)
            if ground_truth_count == 0:
                print(f"Warning: No ground truth for {img_path}")
                continue

            print(
                f"\nEvaluating: {os.path.basename(img_path)} (GT: {ground_truth_count} frames)")

            for algorithm in self.algorithms:
                try:
                    # Run algorithm
                    seg_result = self.evaluate_algorithm(algorithm, img_path)

                    # Calculate metrics
                    val_metrics = self.calculate_validation_metrics(
                        seg_result, ground_truth_count)

                    # Store result
                    results.append({
                        'image': os.path.basename(img_path),
                        'algorithm': algorithm.name,
                        'predicted_frames': seg_result.frame_count,
                        'ground_truth_frames': ground_truth_count,
                        'frame_count_error': val_metrics.frame_count_error,
                        'accuracy': val_metrics.accuracy,
                        'precision': val_metrics.precision,
                        'recall': val_metrics.recall,
                        'f1_score': val_metrics.f1_score,
                        'processing_time': val_metrics.processing_time,
                        'foreground_ratio': seg_result.foreground_ratio
                    })

                    print(f"  {algorithm.name}: {seg_result.frame_count} frames, "
                          f"error: {val_metrics.frame_count_error:.2%}, "
                          f"time: {val_metrics.processing_time:.3f}s")

                except Exception as e:
                    print(f"  {algorithm.name}: ERROR - {e}")

        return pd.DataFrame(results)

    def generate_performance_report(self, df: pd.DataFrame) -> Dict:
        """Generiert Performance-Report"""
        if df.empty:
            return {}

        # Algorithm performance summary
        algo_stats = df.groupby('algorithm').agg({
            'frame_count_error': ['mean', 'std', 'min', 'max'],
            'accuracy': 'mean',
            'processing_time': 'mean',
            'foreground_ratio': 'mean'
        }).round(4)

        # Best algorithm per metric
        best_algorithms = {
            'lowest_error': df.loc[df['frame_count_error'].idxmin(), 'algorithm'],
            'highest_accuracy': df.loc[df['accuracy'].idxmax(), 'algorithm'],
            'fastest': df.loc[df['processing_time'].idxmin(), 'algorithm']
        }

        # Overall ranking (lower error + higher accuracy + faster time)
        df['composite_score'] = (
            1 - df['frame_count_error']) * df['accuracy'] / (df['processing_time'] + 0.001)
        overall_ranking = df.groupby('algorithm')[
            'composite_score'].mean().sort_values(ascending=False)

        report = {
            'algorithm_statistics': algo_stats.to_dict(),
            'best_algorithms': best_algorithms,
            'overall_ranking': overall_ranking.to_dict(),
            'total_evaluations': len(df),
            'unique_images': df['image'].nunique(),
            'algorithms_tested': df['algorithm'].nunique()
        }

        return report

    def save_results(self, df: pd.DataFrame, report: Dict, output_dir: str = "validation_results"):
        """Speichert Evaluations-Ergebnisse"""
        os.makedirs(output_dir, exist_ok=True)

        # Save CSV
        csv_path = os.path.join(output_dir, "evaluation_results.csv")
        df.to_csv(csv_path, index=False)

        # Save JSON report
        json_path = os.path.join(output_dir, "performance_report.json")
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate plots
        self.create_visualizations(df, output_dir)

        print(f"\nResults saved to {output_dir}/")
        print(f"- evaluation_results.csv")
        print(f"- performance_report.json")
        print(f"- visualization plots")

    def create_visualizations(self, df: pd.DataFrame, output_dir: str):
        """Erstellt Visualisierungen"""
        # Algorithm comparison plot
        plt.figure(figsize=(12, 8))

        # Error rates by algorithm
        plt.subplot(2, 2, 1)
        error_stats = df.groupby('algorithm')[
            'frame_count_error'].mean().sort_values()
        error_stats.plot(kind='bar')
        plt.title('Average Frame Count Error by Algorithm')
        plt.ylabel('Error Rate')
        plt.xticks(rotation=45)

        # Processing time comparison
        plt.subplot(2, 2, 2)
        time_stats = df.groupby('algorithm')[
            'processing_time'].mean().sort_values()
        time_stats.plot(kind='bar', color='orange')
        plt.title('Average Processing Time by Algorithm')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)

        # Accuracy comparison
        plt.subplot(2, 2, 3)
        acc_stats = df.groupby('algorithm')[
            'accuracy'].mean().sort_values(ascending=False)
        acc_stats.plot(kind='bar', color='green')
        plt.title('Average Accuracy by Algorithm')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)

        # Error distribution
        plt.subplot(2, 2, 4)
        df.boxplot(column='frame_count_error', by='algorithm', ax=plt.gca())
        plt.title('Frame Count Error Distribution')
        plt.suptitle('')  # Remove default title
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Hauptfunktion für systematische Evaluation"""
    print("=== SCIENTIFIC SPRITESHEET SEGMENTATION VALIDATOR ===\n")

    # Initialize validator
    validator = ScientificValidator()

    if not validator.ground_truth:
        print("ERROR: No ground truth data found!")
        print("Please run ground_truth_creator.py first to annotate test images.")
        return

    print(f"Loaded ground truth for {len(validator.ground_truth)} images")

    # Run comprehensive evaluation
    results_df = validator.run_comprehensive_evaluation()

    if results_df.empty:
        print("No evaluation results generated!")
        return

    # Generate performance report
    performance_report = validator.generate_performance_report(results_df)

    # Display key findings
    print("\n=== KEY FINDINGS ===")
    print(f"Total evaluations: {performance_report['total_evaluations']}")
    print(f"Images tested: {performance_report['unique_images']}")
    print(f"Algorithms tested: {performance_report['algorithms_tested']}")

    print("\nBest algorithms:")
    for metric, algorithm in performance_report['best_algorithms'].items():
        print(f"  {metric}: {algorithm}")

    print("\nOverall ranking:")
    for i, (algorithm, score) in enumerate(performance_report['overall_ranking'].items(), 1):
        print(f"  {i}. {algorithm}: {score:.4f}")

    # Save results
    validator.save_results(results_df, performance_report)

    print("\n=== EVALUATION COMPLETE ===")
    print("Results saved to validation_results/ directory")


if __name__ == "__main__":
    main()
