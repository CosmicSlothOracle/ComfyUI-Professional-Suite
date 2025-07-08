import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import json
from datetime import datetime
import statistics

class OriginalWorkflowFixed:
    """Original Workflow mit intelligenten Fixes für Connected Components-Probleme"""
    
    def __init__(self):
        self.session_name = f"fixed_session_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}"
        self.base_dir = Path("output/original_fixed")
        self.session_dir = self.base_dir / self.session_name
        
        (self.session_dir / "animations").mkdir(parents=True, exist_ok=True)
        (self.session_dir / "individual_sprites").mkdir(parents=True, exist_ok=True)
    
    def improved_background_detection(self, image):
        """FIXED: 16-Punkt Sampling statt 4 Ecken"""
        h, w = image.shape[:2]
        
        sample_points = [
            (0, 0, 40, 40), (w-40, 0, 40, 40), (0, h-40, 40, 40), (w-40, h-40, 40, 40),
            (w//2-20, 0, 40, 20), (w//2-20, h-20, 40, 20), (0, h//2-20, 20, 40), (w-20, h//2-20, 20, 40),
            (w//4-20, h//4-20, 40, 40), (3*w//4-20, h//4-20, 40, 40), (w//4-20, 3*h//4-20, 40, 40), (3*w//4-20, 3*h//4-20, 40, 40)
        ]
        
        colors = []
        for x, y, sw, sh in sample_points:
            x = max(0, min(x, w-sw))
            y = max(0, min(y, h-sh))
            sample = image[y:y+sh, x:x+sw]
            if sample.size > 0:
                colors.append(sample.mean(axis=(0, 1)))
        
        # Farb-Clustering
        color_clusters = []
        for color in colors:
            found = False
            for cluster in color_clusters:
                if np.linalg.norm(color - cluster[0]) < 30:
                    cluster.append(color)
                    found = True
                    break
            if not found:
                color_clusters.append([color])
        
        largest_cluster = max(color_clusters, key=len)
        bg_color = np.mean(largest_cluster, axis=0)
        
        # Adaptive Toleranz
        cluster_std = np.std(largest_cluster, axis=0).mean()
        tolerance = max(15, min(50, int(cluster_std * 3)))
        
        return bg_color, tolerance
    
    def intelligent_filtering(self, components, target_count=12):
        """FIXED: Intelligente Komponenten-Filterung"""
        if not components:
            return []
        
        areas = [c[\"area\"] for c in components]
        area_median = statistics.median(areas)
        
        # Klassifikation
        too_small = [c for c in components if c[\"area\"] < area_median * 0.3]
        too_large = [c for c in components if c[\"area\"] > area_median * 3]
        normal = [c for c in components if c not in too_small and c not in too_large]
        
        # Aggressive Filterung bei Übersegmentierung
        if len(components) > target_count * 2:
            ideal_min = area_median * 0.5
            ideal_max = area_median * 2.0
            filtered = [c for c in components if ideal_min <= c[\"area\"] <= ideal_max]
            if len(filtered) >= target_count // 2:
                return sorted(filtered, key=lambda x: x[\"area\"], reverse=True)[:target_count]
        
        result = normal.copy()
        if len(result) < target_count // 2:
            area_mean = statistics.mean(areas)
            moderate_large = [c for c in too_large if c[\"area\"] < area_mean * 2]
            result.extend(moderate_large)
        
        return sorted(result, key=lambda x: x[\"area\"], reverse=True)[:target_count]
    
    def process_spritesheet_fixed(self, image_path):
        """Original Workflow mit intelligenten Fixes"""
        print(f\"Processing: {image_path.name}\")
        
        # Lade Bild
        image = cv2.imread(str(image_path))
        if image is None:
            return {\"error\": \"Could not load image\"}
        
        # 2x Upscaling + Enhancement (Original beibehalten)
        upscaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        pil_image = Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
        enhanced = pil_image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.2)
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.05)
        enhanced = ImageEnhance.Color(enhanced).enhance(1.1)
        processed_image = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
        
        # FIXED: Verbesserte Hintergrund-Erkennung
        bg_color, tolerance = self.improved_background_detection(processed_image)
        
        mask = np.all(np.abs(processed_image - bg_color) <= tolerance, axis=-1)
        foreground_mask = ~mask
        foreground_ratio = foreground_mask.sum() / foreground_mask.size
        
        # Validierung
        if not (0.05 <= foreground_ratio <= 0.8):
            return {\"error\": f\"Invalid foreground ratio: {foreground_ratio:.3f}\"}
        
        # Connected Components (Original beibehalten)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            foreground_mask.astype(np.uint8), connectivity=8)
        
        components = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > 500:  # Original Mindestfläche beibehalten
                components.append({
                    \"id\": i, \"x\": x, \"y\": y, \"width\": w, \"height\": h, \"area\": area
                })
        
        # FIXED: Intelligente Filterung
        filtered_components = self.intelligent_filtering(components)
        
        # Frame-Extraktion mit 30px Padding (Original beibehalten)
        frames = []
        h, w = processed_image.shape[:2]
        padding = 30
        
        for comp in filtered_components:
            x, y, cw, ch = comp[\"x\"], comp[\"y\"], comp[\"width\"], comp[\"height\"]
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + cw + padding)
            y2 = min(h, y + ch + padding)
            
            frame = processed_image[y1:y2, x1:x2]
            if frame.size > 0:
                frames.append(frame)
        
        # Vaporwave Filter + Speichern (Original beibehalten)
        sprite_dir = self.session_dir / \"individual_sprites\" / image_path.stem
        sprite_dir.mkdir(exist_ok=True)
        
        frame_info = []
        for i, frame in enumerate(frames, 1):
            # Vaporwave Filter (Original 37.5%)
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            np_frame = np.array(pil_frame, dtype=np.float32)
            np_frame[:, :, 0] = np.clip(np_frame[:, :, 0] * 1.15, 0, 255)
            np_frame[:, :, 2] = np.clip(np_frame[:, :, 2] * 1.225, 0, 255)
            
            vaporwave_frame = Image.fromarray(np_frame.astype(np.uint8))
            vaporwave_frame = ImageEnhance.Color(vaporwave_frame).enhance(1.3)
            vaporwave_frame = ImageEnhance.Contrast(vaporwave_frame).enhance(1.1125)
            
            final_frame = cv2.cvtColor(np.array(vaporwave_frame), cv2.COLOR_RGB2BGR)
            
            # Speichern
            frame_filename = f\"fixed_frame_{i:03d}.png\"
            frame_path = sprite_dir / frame_filename
            cv2.imwrite(str(frame_path), final_frame)
            
            frame_info.append({
                \"id\": i,
                \"filename\": frame_filename,
                \"size\": f\"{frame.shape[1]}x{frame.shape[0]}\",
                \"area\": frame.shape[0] * frame.shape[1]
            })
        
        return {
            \"input_file\": str(image_path),
            \"timestamp\": datetime.now().isoformat(),
            \"workflow\": \"Original Fixed\",
            \"background_color\": bg_color.tolist(),
            \"tolerance\": int(tolerance),
            \"foreground_ratio\": float(foreground_ratio),
            \"original_components\": len(components),
            \"filtered_components\": len(filtered_components),
            \"total_frames\": len(frames),
            \"frames\": frame_info
        }

# Test mit dem problematischen Fall
processor = OriginalWorkflowFixed()

test_files = [
    \"Der_Zauber_des_alten_Mannes.png\",
    \"2D_Sprites_des_Mannes_im_Anzug.png\",
    \"Mann_steigt_aus_Limousine_aus.png\"
]

print(\"🔧 ORIGINAL WORKFLOW FIXED TEST\")
print(\"=\" * 50)

for filename in test_files:
    test_path = Path(f\"input/{filename}\")
    if test_path.exists():
        result = processor.process_spritesheet_fixed(test_path)
        
        if \"error\" not in result:
            print(f\"✅ {filename}:\")
            print(f\"   Frames: {result[\"total_frames\"]}\")
            print(f\"   Components: {result[\"original_components\"]} -> {result[\"filtered_components\"]}\")
            print(f\"   Foreground: {result[\"foreground_ratio\"]:.3f}\")
        else:
            print(f\"❌ {filename}: {result[\"error\"]}\")
    else:
        print(f\"⚠️ {filename}: Not found\")

print(f\"\\nFixed results saved to: {processor.session_dir}\")

