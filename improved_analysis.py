import cv2
import numpy as np
from pathlib import Path
import statistics
from collections import Counter

def improved_background_detection(image):
    """Verbesserte Hintergrund-Erkennung mit mehr Sampling-Punkten"""
    h, w = image.shape[:2]
    
    # 16-Punkt Sampling (statt nur 4 Ecken)
    sample_points = [
        # Ecken
        (0, 0, 40, 40),
        (w-40, 0, 40, 40),
        (0, h-40, 40, 40),
        (w-40, h-40, 40, 40),
        # Mitten der Kanten
        (w//2-20, 0, 40, 20),
        (w//2-20, h-20, 40, 20),
        (0, h//2-20, 20, 40),
        (w-20, h//2-20, 20, 40),
        # Zusätzliche Punkte
        (w//4-20, h//4-20, 40, 40),
        (3*w//4-20, h//4-20, 40, 40),
        (w//4-20, 3*h//4-20, 40, 40),
        (3*w//4-20, 3*h//4-20, 40, 40),
        # Zentrale Randpunkte
        (w//8-10, h//8-10, 20, 20),
        (7*w//8-10, h//8-10, 20, 20),
        (w//8-10, 7*h//8-10, 20, 20),
        (7*w//8-10, 7*h//8-10, 20, 20)
    ]
    
    colors = []
    for x, y, sw, sh in sample_points:
        x = max(0, min(x, w-sw))
        y = max(0, min(y, h-sh))
        sample = image[y:y+sh, x:x+sw]
        if sample.size > 0:
            colors.append(sample.mean(axis=(0, 1)))
    
    if not colors:
        return None, None
    
    # Finde den häufigsten Farbbereich
    # Clustere ähnliche Farben
    color_clusters = []
    for color in colors:
        found_cluster = False
        for cluster in color_clusters:
            if np.linalg.norm(color - cluster[0]) < 30:  # Ähnlichkeitsthreshold
                cluster.append(color)
                found_cluster = True
                break
        if not found_cluster:
            color_clusters.append([color])
    
    # Wähle größten Cluster als Hintergrund
    largest_cluster = max(color_clusters, key=len)
    bg_color = np.mean(largest_cluster, axis=0)
    
    # Adaptive Toleranz basierend auf Cluster-Varianz
    cluster_std = np.std(largest_cluster, axis=0).mean()
    tolerance = max(15, min(50, int(cluster_std * 3)))
    
    return bg_color, tolerance

def intelligent_component_filtering(components, target_count=12):
    """Intelligente Komponenten-Filterung"""
    if not components:
        return []
    
    areas = [c["area"] for c in components]
    area_mean = statistics.mean(areas)
    area_median = statistics.median(areas)
    area_std = statistics.stdev(areas) if len(areas) > 1 else 0
    
    # Identifiziere problematische Komponenten
    too_small = [c for c in components if c["area"] < area_median * 0.3]
    too_large = [c for c in components if c["area"] > area_median * 3]
    normal = [c for c in components if c not in too_small and c not in too_large]
    
    print(f"Component classification:")
    print(f"  Too small: {len(too_small)}")
    print(f"  Too large: {len(too_large)}")
    print(f"  Normal: {len(normal)}")
    
    # Wenn wir zu viele Komponenten haben, filtere aggressiver
    if len(components) > target_count * 2:
        # Behalte nur Komponenten im idealen Größenbereich
        ideal_min = area_median * 0.5
        ideal_max = area_median * 2.0
        filtered = [c for c in components if ideal_min <= c["area"] <= ideal_max]
        if len(filtered) >= target_count // 2:
            return sorted(filtered, key=lambda x: x["area"], reverse=True)[:target_count]
    
    # Standard-Filterung
    result = normal.copy()
    
    # Füge moderate große Komponenten hinzu falls nötig
    if len(result) < target_count // 2:
        moderate_large = [c for c in too_large if c["area"] < area_mean * 2]
        result.extend(moderate_large)
    
    return sorted(result, key=lambda x: x["area"], reverse=True)[:target_count]

print("Testing improved algorithms on: Der_Zauber_des_alten_Mannes")

# Lade Bild
image_path = Path("input/Der_Zauber_des_alten_Mannes.png")
image = cv2.imread(str(image_path))
upscaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

print(f"Image size: {upscaled.shape[:2]}")

# Teste verbesserte Hintergrund-Erkennung
bg_color, tolerance = improved_background_detection(upscaled)

if bg_color is not None:
    print(f"Improved BG color: {bg_color}")
    print(f"Adaptive tolerance: {tolerance}")
    
    # Erstelle Maske
    mask = np.all(np.abs(upscaled - bg_color) <= tolerance, axis=-1)
    foreground_mask = ~mask
    
    foreground_ratio = foreground_mask.sum() / foreground_mask.size
    print(f"Foreground ratio: {foreground_ratio:.3f}")
    
    if 0.05 <= foreground_ratio <= 0.8:  # Sinnvoller Bereich
        # Connected Components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            foreground_mask.astype(np.uint8), connectivity=8)
        
        components = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > 1000:  # Etwas höhere Mindestfläche
                components.append({
                    "id": i,
                    "x": x, "y": y,
                    "width": w, "height": h,
                    "area": area
                })
        
        print(f"Components found: {len(components)}")
        
        if components:
            # Intelligente Filterung
            filtered_components = intelligent_component_filtering(components, target_count=12)
            
            print(f"After intelligent filtering: {len(filtered_components)}")
            
            print("Final components:")
            for i, comp in enumerate(filtered_components):
                w = comp["width"]
                h = comp["height"]
                area = comp["area"]
                print(f"  {i+1}: {w}x{h} (area: {area})")
        else:
            print("No valid components found")
    else:
        print(f"Invalid foreground ratio: {foreground_ratio:.3f}")
else:
    print("Background detection failed")

