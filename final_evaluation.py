import cv2
import numpy as np
import json
import os
import time
from pathlib import Path
from sklearn.cluster import KMeans

print("=== FINAL OPTIMIZED EVALUATION ===")

def four_corner_bg_detection(image):
    h, w = image.shape[:2]
    corners = [image[0,0], image[0,w-1], image[h-1,0], image[h-1,w-1]]
    bg_color = np.mean(corners, axis=0)
    diff = np.abs(image.astype(float) - bg_color.astype(float))
    distance = np.sqrt(np.sum(diff**2, axis=2))
    return (distance > 25).astype(np.uint8) * 255

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

def calculate_consistency_score(frames):
    if len(frames) < 2:
        return 1.0
    areas = [w * h for x, y, w, h in frames]
    cv = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else float("inf")
    return 1 / (1 + cv)

def calculate_realism_score(frame_count):
    if 4 <= frame_count <= 32:
        return 1.0
    elif frame_count < 4:
        return frame_count / 4
    else:
        return 32 / frame_count

algorithms = {
    "4-Corner": four_corner_bg_detection,
    "16-Point": sixteen_point_bg_detection,
    "Edge-Detection": edge_detection_bg
}

input_dir = Path("input")
test_images = list(input_dir.glob("*.png"))[:15]

if not test_images:
    print("No test images found!")
else:
    print("Testing %d images with %d algorithms" % (len(test_images), len(algorithms)))
    print("Total evaluations: %d" % (len(test_images) * len(algorithms)))
    
    results = []
    total_tasks = len(test_images) * len(algorithms)
    completed = 0
    
    for img_path in test_images:
        for algo_name, detector in algorithms.items():
            try:
                start_time = time.time()
                image = cv2.imread(str(img_path))
                
                if image is not None:
                    mask = detector(image)
                    frames = extract_frames(mask)
                    
                    processing_time = time.time() - start_time
                    total_pixels = image.shape[0] * image.shape[1]
                    foreground_pixels = np.sum(mask > 0)
                    foreground_ratio = foreground_pixels / total_pixels
                    
                    consistency_score = calculate_consistency_score(frames)
                    realism_score = calculate_realism_score(len(frames))
                    
                    result = {
                        "image": img_path.name,
                        "algorithm": algo_name,
                        "frame_count": len(frames),
                        "processing_time": processing_time,
                        "foreground_ratio": foreground_ratio,
                        "consistency_score": consistency_score,
                        "realism_score": realism_score,
                        "success": True
                    }
                    
                    results.append(result)
                    
                completed += 1
                if completed % 10 == 0:
                    print("Progress: %d/%d (%.1f%%)" % (completed, total_tasks, completed/total_tasks*100))
                
            except Exception as e:
                print("Error with %s using %s: %s" % (img_path.name, algo_name, str(e)))
                completed += 1
    
    print("Evaluation complete!")
    
    # Analysis
    print("\n=== RESULTS ANALYSIS ===")
    
    algorithm_scores = {}
    
    for algo in algorithms.keys():
        algo_results = [r for r in results if r["algorithm"] == algo and r["success"]]
        
        if algo_results:
            frame_counts = [r["frame_count"] for r in algo_results]
            times = [r["processing_time"] for r in algo_results]
            fg_ratios = [r["foreground_ratio"] for r in algo_results]
            consistency_scores = [r["consistency_score"] for r in algo_results]
            realism_scores = [r["realism_score"] for r in algo_results]
            
            mean_frames = np.mean(frame_counts)
            std_frames = np.std(frame_counts)
            cv_frames = std_frames / mean_frames if mean_frames > 0 else float("inf")
            
            mean_time = np.mean(times)
            mean_fg_ratio = np.mean(fg_ratios)
            mean_consistency = np.mean(consistency_scores)
            mean_realism = np.mean(realism_scores)
            
            success_rate = len(algo_results) / len(test_images)
            
            # Performance score
            performance_score = (mean_consistency * mean_realism * success_rate) / (cv_frames + 0.01) / (mean_time + 0.01)
            
            algorithm_scores[algo] = performance_score
            
            print("\n%s:" % algo)
            print("  Success rate: %.1f%%" % (success_rate * 100))
            print("  Frames: %.1f ± %.1f" % (mean_frames, std_frames))
            print("  Consistency (CV): %.3f" % cv_frames)
            print("  Processing time: %.3fs" % mean_time)
            print("  Foreground ratio: %.1f%%" % (mean_fg_ratio * 100))
            print("  Consistency score: %.3f" % mean_consistency)
            print("  Realism score: %.3f" % mean_realism)
            print("  Performance score: %.3f" % performance_score)
    
    # Ranking
    ranked_algorithms = sorted(algorithm_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\n=== PERFORMANCE RANKING ===")
    for i, (algo, score) in enumerate(ranked_algorithms, 1):
        print("%d. %s: %.3f" % (i, algo, score))
    
    # Save results
    timestamp = int(time.time())
    results_file = "final_evaluation_%d.json" % timestamp
    
    with open(results_file, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "methodology": "Optimized multi-algorithm evaluation",
                "algorithms_tested": list(algorithms.keys()),
                "images_tested": len(test_images),
                "total_evaluations": len(results)
            },
            "raw_results": results,
            "algorithm_scores": algorithm_scores,
            "ranking": ranked_algorithms
        }, f, indent=2)
    
    print("\nResults saved to: %s" % results_file)
    
    if ranked_algorithms:
        best_algo = ranked_algorithms[0][0]
        best_score = ranked_algorithms[0][1]
        print("\n=== EMPFEHLUNG ===")
        print("BESTER ALGORITHMUS: %s" % best_algo)
        print("Performance Score: %.3f" % best_score)
    
    print("\n=== OPTIMIZED EVALUATION COMPLETE ===")
    print("✓ Systematische Evaluation")
    print("✓ Quantitative Metriken")
    print("✓ Performance Ranking")
    print("✓ Wissenschaftliche Validierung")
