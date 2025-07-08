import cv2
import numpy as np
from pathlib import Path
import json
import statistics

print("Detailed analysis of: Der_Zauber_des_alten_Mannes")

# Lade das ursprüngliche Bild
image_path = Path("input/Der_Zauber_des_alten_Mannes.png")
if not image_path.exists():
    print("Image not found!")
    exit()

image = cv2.imread(str(image_path))
original_size = image.shape[:2]
print(f"Original size: {original_size}")

# Simuliere Original Workflow
upscaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
print(f"Upscaled size: {upscaled.shape[:2]}")

# Original Background Detection (4 Ecken)
h, w = upscaled.shape[:2]
corners = [
    upscaled[0:20, 0:20].mean(axis=(0, 1)),
    upscaled[0:20, w-20:w].mean(axis=(0, 1)),
    upscaled[h-20:h, 0:20].mean(axis=(0, 1)),
    upscaled[h-20:h, w-20:w].mean(axis=(0, 1))
]

bg_color = np.mean(corners, axis=0)
tolerance = 25
print(f"Background color: {bg_color}")
print(f"Tolerance: {tolerance}")

mask = np.all(np.abs(upscaled - bg_color) <= tolerance, axis=-1)
foreground_mask = ~mask

foreground_ratio = foreground_mask.sum() / foreground_mask.size
print(f"Foreground ratio: {foreground_ratio:.3f}")

# Connected Components Analysis
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    foreground_mask.astype(np.uint8), connectivity=8)

print(f"Total components found: {num_labels - 1}")

# Analysiere Komponenten
components = []
for i in range(1, num_labels):  # Skip background
    x, y, w_comp, h_comp, area = stats[i]
    if area > 500:  # Original Mindestfläche
        components.append({
            "id": i,
            "x": x, "y": y,
            "width": w_comp, "height": h_comp,
            "area": area,
            "aspect_ratio": w_comp / h_comp if h_comp > 0 else 0
        })

print(f"Components after filtering (>500px): {len(components)}")

if components:
    areas = [c["area"] for c in components]
    widths = [c["width"] for c in components]
    heights = [c["height"] for c in components]
    
    print(f"Area range: {min(areas)} - {max(areas)}")
    print(f"Size range: {min(widths)}x{min(heights)} - {max(widths)}x{max(heights)}")
    print(f"Area mean: {statistics.mean(areas):.0f}")
    print(f"Area std: {statistics.stdev(areas) if len(areas) > 1 else 0:.0f}")
    
    print("First 12 components:")
    for i, comp in enumerate(components[:12]):
        print(f"  Component {i+1}: {comp[\"width\"]}x{comp[\"height\"]} (area: {comp[\"area\"]})")

