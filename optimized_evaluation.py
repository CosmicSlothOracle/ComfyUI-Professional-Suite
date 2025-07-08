import cv2
import numpy as np
import json
import os
import time
from pathlib import Path

print("=== OPTIMIERTER SPRITESHEET-WORKFLOW ===")

# Basic algorithms for comparison
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
    
    bg_color = np.mean(points, axis=0)
    tolerance = max(30, np.std(image.reshape(-1, 3)) * 0.8)
    diff = np.abs(image.astype(float) - bg_color.astype(float))
    distance = np.sqrt(np.sum(diff**2, axis=2))
    return (distance > tolerance).astype(np.uint8) * 255

def extract_frames(mask):
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    
    frames = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 100 and 0.1 < w/h < 10:
            frames.append((x, y, w, h))
    return frames

# Test algorithms
algorithms = {
    "4-Corner": four_corner_bg_detection,
    "16-Point": sixteen_point_bg_detection
}

# Get test images
input_dir = Path("input")
test_images = list(input_dir.glob("*.png"))[:10]

if not test_images:
    print("No test images found!")
else:
    print(f"Testing {len(test_images)} images with {len(algorithms)} algorithms")
    
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
                    
                    results.append({
                        "image": img_path.name,
                        "algorithm": algo_name,
                        "frame_count": len(frames),
                        "processing_time": processing_time,
                        "foreground_ratio": foreground_ratio,
                        "success": True
                    })
                    
                completed += 1
                print(f"Progress: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%)")
                
            except Exception as e:
                print(f"Error with {img_path.name} using {algo_name}: {e}")
                completed += 1
    
    # Analyze results
    print("\nRESULTS ANALYSIS:")
    for algo in ["4-Corner", "16-Point"]:
        algo_results = [r for r in results if r["algorithm"] == algo and r["success"]]
        
        if algo_results:
            frame_counts = [r["frame_count"] for r in algo_results]
            times = [r["processing_time"] for r in algo_results]
            fg_ratios = [r["foreground_ratio"] for r in algo_results]
            
            mean_frames = np.mean(frame_counts)
            std_frames = np.std(frame_counts)
            cv = std_frames / mean_frames if mean_frames > 0 else float("inf")
            
            print(f"\n{algo}:")
            print(f"  Success rate: {len(algo_results)}/{len(test_images)} ({len(algo_results)/len(test_images)*100:.1f}%)")
            print(f"  Frames: {mean_frames:.1f} ± {std_frames:.1f}")
            print(f"  Consistency (CV): {cv:.3f}")
            print(f"  Processing time: {np.mean(times):.3f}s")
            print(f"  Foreground ratio: {np.mean(fg_ratios)*100:.1f}%")
    
    # Save results
    timestamp = int(time.time())
    results_file = f"workflow_evaluation_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("\n=== EVALUATION COMPLETE ===")
