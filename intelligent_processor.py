import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict
import statistics
from collections import Counter

class IntelligentSpritesheetProcessor:
    """Intelligente Spritesheet-Verarbeitung mit verbesserter Frame-Extraktion"""
    
    def __init__(self):
        self.session_name = f"intelligent_session_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}"
        self.base_dir = Path("output/intelligent_processing")
        self.session_dir = self.base_dir / self.session_name
        
        (self.session_dir / "animations").mkdir(parents=True, exist_ok=True)
        (self.session_dir / "individual_sprites").mkdir(parents=True, exist_ok=True)
        (self.session_dir / "debug").mkdir(parents=True, exist_ok=True)
    
    def improved_background_detection(self, image):
        """Verbesserte Hintergrund-Erkennung mit 16-Punkt Sampling"""
        h, w = image.shape[:2]
        
        sample_points = [
            # Ecken (größere Bereiche)
            (0, 0, 40, 40),
            (w-40, 0, 40, 40),
            (0, h-40, 40, 40),
            (w-40, h-40, 40, 40),
            # Mitten der Kanten
            (w//2-20, 0, 40, 20),
            (w//2-20, h-20, 40, 20),
            (0, h//2-20, 20, 40),
            (w-20, h//2-20, 20, 40),
            # Viertel-Punkte
            (w//4-20, h//4-20, 40, 40),
            (3*w//4-20, h//4-20, 40, 40),
            (w//4-20, 3*h//4-20, 40, 40),
            (3*w//4-20, 3*h//4-20, 40, 40),
            # Zusätzliche Randpunkte
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
        
        # Clustere ähnliche Farben
        color_clusters = []
        for color in colors:
            found_cluster = False
            for cluster in color_clusters:
                if np.linalg.norm(color - cluster[0]) < 30:
                    cluster.append(color)
                    found_cluster = True
                    break
            if not found_cluster:
                color_clusters.append([color])
        
        # Größter Cluster = Hintergrund
        largest_cluster = max(color_clusters, key=len)
        bg_color = np.mean(largest_cluster, axis=0)
        
        # Adaptive Toleranz
        cluster_std = np.std(largest_cluster, axis=0).mean()
        tolerance = max(15, min(50, int(cluster_std * 3)))
        
        return bg_color, tolerance
    
    def intelligent_component_filtering(self, components, target_count=12):
        """Intelligente Komponenten-Filterung"""
        if not components:
            return []
        
        areas = [c["area"] for c in components]
        area_median = statistics.median(areas)
        
        # Klassifiziere Komponenten
        too_small = [c for c in components if c["area"] < area_median * 0.3]
        too_large = [c for c in components if c["area"] > area_median * 3]
        normal = [c for c in components if c not in too_small and c not in too_large]
        
        # Aggressive Filterung bei zu vielen Komponenten
        if len(components) > target_count * 2:
            ideal_min = area_median * 0.5
            ideal_max = area_median * 2.0
            filtered = [c for c in components if ideal_min <= c["area"] <= ideal_max]
            if len(filtered) >= target_count // 2:
                return sorted(filtered, key=lambda x: x["area"], reverse=True)[:target_count]
        
        # Standard-Filterung
        result = normal.copy()
        
        # Füge moderate große Komponenten hinzu falls nötig
        if len(result) < target_count // 2:
            area_mean = statistics.mean(areas)
            moderate_large = [c for c in too_large if c["area"] < area_mean * 2]
            result.extend(moderate_large)
        
        return sorted(result, key=lambda x: x["area"], reverse=True)[:target_count]
    
    def extract_intelligent_frames(self, image, padding=30):
        """Intelligente Frame-Extraktion"""
        # Verbesserte Hintergrund-Erkennung
        bg_color, tolerance = self.improved_background_detection(image)
        
        # Erstelle Maske
        mask = np.all(np.abs(image - bg_color) <= tolerance, axis=-1)
        foreground_mask = ~mask
        
        foreground_ratio = foreground_mask.sum() / foreground_mask.size
        
        if not (0.05 <= foreground_ratio <= 0.8):
            return [], {"error": f"Invalid foreground ratio: {foreground_ratio:.3f}"}
        
        # Connected Components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            foreground_mask.astype(np.uint8), connectivity=8)
        
        # Sammle Komponenten
        components = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > 1000:  # Höhere Mindestfläche
                components.append({
                    "id": i,
                    "x": x, "y": y,
                    "width": w, "height": h,
                    "area": area
                })
        
        # Intelligente Filterung
        filtered_components = self.intelligent_component_filtering(components)
        
        # Extrahiere Frames mit Padding
        frames = []
        h, w = image.shape[:2]
        
        for comp in filtered_components:
            x, y, cw, ch = comp["x"], comp["y"], comp["width"], comp["height"]
            
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + cw + padding)
            y2 = min(h, y + ch + padding)
            
            frame = image[y1:y2, x1:x2]
            if frame.size > 0:
                frames.append(frame)
        
        analysis = {
            "background_color": bg_color.tolist(),
            "tolerance": int(tolerance),
            "foreground_ratio": float(foreground_ratio),
            "total_components": len(components),
            "filtered_components": len(filtered_components),
            "extracted_frames": len(frames)
        }
        
        return frames, analysis
    
    def apply_vaporwave_filter(self, frame, intensity=0.375):
        """Verstärkter Vaporwave-Filter"""
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Farbverschiebung
        np_frame = np.array(pil_frame, dtype=np.float32)
        np_frame[:, :, 0] = np.clip(np_frame[:, :, 0] * (1.0 + intensity * 0.4), 0, 255)
        np_frame[:, :, 2] = np.clip(np_frame[:, :, 2] * (1.0 + intensity * 0.6), 0, 255)
        
        vaporwave_frame = Image.fromarray(np_frame.astype(np.uint8))
        
        # Verstärkungen
        vaporwave_frame = ImageEnhance.Color(vaporwave_frame).enhance(1.0 + intensity * 0.8)
        vaporwave_frame = ImageEnhance.Contrast(vaporwave_frame).enhance(1.0 + intensity * 0.3)
        
        return cv2.cvtColor(np.array(vaporwave_frame), cv2.COLOR_RGB2BGR)
    
    def create_gif(self, frames, output_path):
        """Erstelle GIF Animation"""
        if not frames:
            return
        
        pil_frames = []
        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb_frame))
        
        if pil_frames:
            pil_frames[0].save(
                str(output_path),
                save_all=True,
                append_images=pil_frames[1:],
                duration=200,
                loop=0,
                optimize=True
            )
    
    def process_spritesheet(self, image_path):
        """Verarbeite ein Spritesheet mit intelligenten Algorithmen"""
        print(f"\\n🧠 INTELLIGENT PROCESSING: {image_path.name}")
        
        # Lade und skaliere Bild
        image = cv2.imread(str(image_path))
        if image is None:
            return {"error": f"Could not load {image_path}"}
        
        # 2x Upscaling + Enhancement
        print("   📈 2x Upscaling + Enhancement...")
        upscaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        pil_image = Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
        enhanced = pil_image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.2)
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.05)
        enhanced = ImageEnhance.Color(enhanced).enhance(1.1)
        processed_image = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
        
        # Intelligente Frame-Extraktion
        print("   🧠 Intelligente Frame-Extraktion...")
        frames, analysis = self.extract_intelligent_frames(processed_image)
        
        if not frames:
            return {"error": "No frames extracted", "analysis": analysis}
        
        print(f"   ✅ {len(frames)} Frames intelligent extrahiert")
        
        # Verarbeite Frames
        sprite_dir = self.session_dir / "individual_sprites" / image_path.stem
        sprite_dir.mkdir(exist_ok=True)
        
        processed_frames = []
        frame_info = []
        
        for i, frame in enumerate(frames, 1):
            # Vaporwave-Filter
            vaporwave_frame = self.apply_vaporwave_filter(frame, 0.375)
            
            # Speichern
            frame_filename = f"intelligent_frame_{i:03d}.png"
            frame_path = sprite_dir / frame_filename
            
            frame_rgba = cv2.cvtColor(vaporwave_frame, cv2.COLOR_BGR2BGRA)
            cv2.imwrite(str(frame_path), frame_rgba)
            
            frame_info.append({
                "id": i,
                "filename": frame_filename,
                "size": f"{frame.shape[1]}x{frame.shape[0]}",
                "area": frame.shape[0] * frame.shape[1]
            })
            
            processed_frames.append(vaporwave_frame)
        
        # GIF erstellen
        gif_path = self.session_dir / "animations" / f"{image_path.stem}_intelligent_vaporwave.gif"
        self.create_gif(processed_frames, gif_path)
        
        return {
            "input_file": str(image_path),
            "timestamp": datetime.now().isoformat(),
            "workflow": "Intelligent Processing",
            "analysis": analysis,
            "total_frames": len(frames),
            "frames": frame_info,
            "gif_animation": str(gif_path)
        }

# Test mit dem problematischen Fall
processor = IntelligentSpritesheetProcessor()

test_file = Path("input/Der_Zauber_des_alten_Mannes.png")
if test_file.exists():
    print("🧠 INTELLIGENT SPRITESHEET PROCESSING TEST")
    print("=" * 60)
    
    result = processor.process_spritesheet(test_file)
    
    if "error" not in result:
        print(f"\\n📊 ERGEBNIS:")
        print(f"   Frames extrahiert: {result.get(\"total_frames\", \"N/A\")}")
        print(f"   Analysis: {result.get(\"analysis\", {})}")
        print(f"   GIF erstellt: {result.get(\"gif_animation\", \"N/A\")}")
        
        # Speichere Report
        report_path = processor.session_dir / f"{test_file.stem}_intelligent_report.json"
        with open(report_path, \"w\", encoding=\"utf-8\") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"   Report: {report_path}")
    else:
        print(f"\\n❌ FEHLER: {result[\"error\"]}")
else:
    print("❌ Test-Datei nicht gefunden")

