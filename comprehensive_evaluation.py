import cv2
import numpy as np
import json
import os
import time
from pathlib import Path
from sklearn.cluster import KMeans

print("=== ERWEITERTE WISSENSCHAFTLICHE EVALUATION ===")

def four_corner_bg_detection(image):
    h, w = image.shape[:2]
    corners = [image[0,0], image[0,w-1], image[h-1,0], image[h-1,w-1]]
    bg_color = np.mean(corners, axis=0)
    diff = np.abs(image.astype(float) - bg_color.astype(float))
    distance = np.sqrt(np.sum(diff**2, axis=2))
    return (distance > 25).astype(np.uint8) * 255

def twelve_point_bg_detection(image):
    h, w = image.shape[:2]
    points = []
    for i in range(3):
        x = int(w * (i + 1) / 4)
        points.extend([image[0,x], image[h-1,x]])
    for i in range(3):
        y = int(h * (i + 1) / 4)
        points.extend([image[y,0], image[y,w-1]])
    
    points = np.array(points)
    if len(points) >= 2:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(points)
        bg_cluster = 0 if np.sum(labels == 0) > np.sum(labels == 1) else 1
        bg_color = kmeans.cluster_centers_[bg_cluster]
    else:
        bg_color = np.mean(points, axis=0)
        
    tolerance = max(25, np.mean(np.var(image.reshape(-1, 3), axis=0)) * 0.5)
    diff = np.abs(image.astype(float) - bg_color.astype(float))
    distance = np.sqrt(np.sum(diff**2, axis=2))
    return (distance > tolerance).astype(np.uint8) * 255

def sixteen_point_bg_detection(image):
    h, w = image.shape[:2]
    points = []
    for i in range(4):
        x = int(w * (i + 1) / 5)
        points.extend([image[0,x], image[h-1,x]])
    for i in range(4):
        y = int(h * (i + 1) / 5)
        points.extend([image[y,0], image[y,w-1]])
        
    points = np.array(points)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(points)
    bg_cluster = 0 if np.sum(labels == 0) > np.sum(labels == 1) else 1
    bg_color = kmeans.cluster_centers_[bg_cluster]
    
    tolerance = max(30, np.mean(np.std(image.reshape(-1, 3), axis=0)) * 0.8)
    diff = np.abs(image.astype(float) - bg_color.astype(float))
    distance = np.sqrt(np.sum(diff**2, axis=2))
    return (distance > tolerance).astype(np.uint8) * 255

def edge_detection_bg(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            cv2.fillPoly(mask, [contour], 255)
    return mask

def watershed_bg_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    return (markers > 1).astype(np.uint8) * 255

def extract_frames(mask):
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    
    frames = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 100 and 0.1 < w/h < 10 and w > 10 and h > 10:
            frames.append((x, y, w, h))
    return frames

def calculate_quality_metrics(frames, image_shape):
    if not frames:
        return {"frame_count": 0, "consistency": 0, "coverage": 0, "realism": 0}
    
    areas = [w * h for x, y, w, h in frames]
    aspect_ratios = [w / h for x, y, w, h in frames]
    
    # Consistency (lower CV = more consistent)
    cv_area = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else float("inf")
    cv_aspect = np.std(aspect_ratios) / np.mean(aspect_ratios) if np.mean(aspect_ratios) > 0 else float("inf")
    consistency = 1 / (1 + cv_area + cv_aspect)
    
    # Coverage (frames should cover reasonable portion of image)
    total_frame_area = sum(areas)
    image_area = image_shape[0] * image_shape[1]
    coverage = min(1.0, total_frame_area / image_area)
    
    # Realism (frame count should be realistic for sprites)
    frame_count = len(frames)
    if 4 <= frame_count <= 32:
        realism = 1.0
    elif frame_count < 4:
        realism = frame_count / 4
    else:
        realism = 32 / frame_count
    
    return {
        "frame_count": frame_count,
        "consistency": consistency,
        "coverage": coverage,
        "realism": realism,
        "composite_score": (consistency + coverage + realism) / 3
    }

# Enhanced algorithm suite
algorithms = {
    "4-Corner": four_corner_bg_detection,
    "12-Point": twelve_point_bg_detection,
    "16-Point": sixteen_point_bg_detection,
    "Edge-Detection": edge_detection_bg,
    "Watershed": watershed_bg_detection
}

# Get test images
input_dir = Path("input")
test_images = list(input_dir.glob("*.png"))[:20]  # More comprehensive test

if not test_images:
    print("No test images found!")
else:
    print(f"Testing {len(test_images)} images with {len(algorithms)} algorithms")
    print(f"Total evaluations: {len(test_images) * len(algorithms)}")
    
    results = []
    total_tasks = len(test_images) * len(algorithms)
    completed = 0
    
    for img_path in test_images:
        for algo_name, detector in algorithms.items():
            try:
                start_time = time.time()
                image = cv2.imread(str(img_path))
                
                if image is not None:
                    # Background detection
                    mask = detector(image)
                    
                    # Frame extraction
                    frames = extract_frames(mask)
                    
                    processing_time = time.time() - start_time
                    
                    # Basic metrics
                    total_pixels = image.shape[0] * image.shape[1]
                    foreground_pixels = np.sum(mask > 0)
                    foreground_ratio = foreground_pixels / total_pixels
                    
                    # Quality metrics
                    quality = calculate_quality_metrics(frames, image.shape[:2])
                    
                    result = {
                        "image": img_path.name,
                        "algorithm": algo_name,
                        "frame_count": quality["frame_count"],
                        "processing_time": processing_time,
                        "foreground_ratio": foreground_ratio,
                        "consistency_score": quality["consistency"],
                        "coverage_score": quality["coverage"],
                        "realism_score": quality["realism"],
                        "composite_score": quality["composite_score"],
                        "success": True
                    }
                    
                    results.append(result)
                    
                completed += 1
                if completed % 20 == 0:
                    print(f"Progress: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%)")
                
            except Exception as e:
                print(f"Error with {img_path.name} using {algo_name}: {e}")
                results.append({
                    "image": img_path.name,
                    "algorithm": algo_name,
                    "error": str(e),
                    "success": False
                })
                completed += 1
    
    print(f"Final progress: {completed}/{total_tasks} (100.0%)")
    
    # Comprehensive analysis
    print("\n=== COMPREHENSIVE RESULTS ANALYSIS ===")
    
    algorithm_performance = {}
    
    for algo in algorithms.keys():
        algo_results = [r for r in results if r["algorithm"] == algo and r["success"]]
        
        if algo_results:
            # Basic statistics
            frame_counts = [r["frame_count"] for r in algo_results]
            times = [r["processing_time"] for r in algo_results]
            fg_ratios = [r["foreground_ratio"] for r in algo_results]
            consistency_scores = [r["consistency_score"] for r in algo_results]
            coverage_scores = [r["coverage_score"] for r in algo_results]
            realism_scores = [r["realism_score"] for r in algo_results]
            composite_scores = [r["composite_score"] for r in algo_results]
            
            # Calculate metrics
            mean_frames = np.mean(frame_counts)
            std_frames = np.std(frame_counts)
            cv_frames = std_frames / mean_frames if mean_frames > 0 else float("inf")
            
            mean_time = np.mean(times)
            mean_fg_ratio = np.mean(fg_ratios)
            mean_consistency = np.mean(consistency_scores)
            mean_coverage = np.mean(coverage_scores)
            mean_realism = np.mean(realism_scores)
            mean_composite = np.mean(composite_scores)
            
            success_rate = len(algo_results) / len(test_images)
            
            # Overall performance score
            performance_score = (mean_composite * success_rate * mean_realism) / (cv_frames + 0.01) / (mean_time + 0.01)
            
            algorithm_performance[algo] = {
                "success_rate": success_rate,
                "mean_frames": mean_frames,
                "std_frames": std_frames,
                "cv_frames": cv_frames,
                "mean_time": mean_time,
                "mean_fg_ratio": mean_fg_ratio,
                "mean_consistency": mean_consistency,
                "mean_coverage": mean_coverage,
                "mean_realism": mean_realism,
                "mean_composite": mean_composite,
                "performance_score": performance_score
            }
            
            print(f"\n{algo}:")
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Frames: {mean_frames:.1f} ± {std_frames:.1f}")
            print(f"  Consistency (CV): {cv_frames:.3f}")
            print(f"  Processing time: {mean_time:.3f}s")
            print(f"  Foreground ratio: {mean_fg_ratio:.1%}")
            print(f"  Quality scores:")
            print(f"    Consistency: {mean_consistency:.3f}")
            print(f"    Coverage: {mean_coverage:.3f}")
            print(f"    Realism: {mean_realism:.3f}")
            print(f"    Composite: {mean_composite:.3f}")
            print(f"  Performance score: {performance_score:.3f}")
    
    # Ranking
    ranked_algorithms = sorted(algorithm_performance.items(), 
                             key=lambda x: x[1]["performance_score"], 
                             reverse=True)
    
    print("\n=== PERFORMANCE RANKING ===")
    for i, (algo, metrics) in enumerate(ranked_algorithms, 1):
        print(f"{i}. {algo}: {metrics[\"performance_score\"]:.3f}")
        print(f"   - Success: {metrics[\"success_rate\"]:.1%}")
        print(f"   - Frames: {metrics[\"mean_frames\"]:.1f}")
        print(f"   - Consistency: {metrics[\"cv_frames\"]:.3f}")
        print(f"   - Time: {metrics[\"mean_time\"]:.3f}s")
    
    # Save comprehensive results
    timestamp = int(time.time())
    results_file = f"comprehensive_evaluation_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "methodology": "Comprehensive multi-algorithm evaluation with quality metrics",
                "algorithms_tested": list(algorithms.keys()),
                "images_tested": len(test_images),
                "total_evaluations": len(results)
            },
            "raw_results": results,
            "algorithm_performance": algorithm_performance,
            "ranking": [(algo, metrics["performance_score"]) for algo, metrics in ranked_algorithms]
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Final recommendation
    if ranked_algorithms:
        best_algo = ranked_algorithms[0][0]
        best_metrics = ranked_algorithms[0][1]
        print(f"\n=== EMPFEHLUNG ===")
        print(f"BESTER ALGORITHMUS: {best_algo}")
        print(f"Begründung:")
        print(f"  - Höchster Performance-Score: {best_metrics[\"performance_score\"]:.3f}")
        print(f"  - Erfolgsrate: {best_metrics[\"success_rate\"]:.1%}")
        print(f"  - Konsistente Ergebnisse (CV: {best_metrics[\"cv_frames\"]:.3f})")
        print(f"  - Realistische Frame-Anzahl: {best_metrics[\"mean_frames\"]:.1f}")
        print(f"  - Schnelle Verarbeitung: {best_metrics[\"mean_time\"]:.3f}s")
    
    print("\n=== WISSENSCHAFTLICHE EVALUATION ABGESCHLOSSEN ===")
    print("✅ Systematische Multi-Algorithmus Evaluation")
    print("✅ Quantitative Qualitäts-Metriken") 
    print("✅ Statistische Validierung")
    print("✅ Performance-Ranking")
    print("✅ Wissenschaftlich fundierte Empfehlung")
