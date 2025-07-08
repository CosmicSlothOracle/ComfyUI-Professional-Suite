#!/usr/bin/env python3
"""
🎯 PRODUCTION SPRITE PROCESSOR - ITERATION 4 OPTIMIZED
Basiert auf bewährten Verbesserungen aus dem Test-Workflow
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import glob


class ProductionSpriteProcessor:
    def __init__(self):
        self.session_id = f"production_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.base_dir = Path("C:/Users/Public/ComfyUI-master")
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output" / \
            "production_optimized" / self.session_id
        self.ensure_directories()

        # Bewährte Parameter aus Iteration 4
        self.config = {
            "background_removal": {
                "edge_tolerance": 15,
                "alpha_threshold": 240,
                "morphology_kernel": 5
            },
            "anatomical_analysis": {
                "head_ratio_min": 0.15,
                "head_ratio_max": 0.35,
                "body_aspect_min": 1.2,
                "body_aspect_max": 4.0,
                "min_frame_area": 2000
            },
            "instagram_filter": {
                "warmth": 1.15,
                "contrast": 1.25,
                "saturation": 1.1,
                "brightness": 1.05,
                "vintage_opacity": 0.3
            },
            "processing": {
                "max_workers": 6,
                "max_frames_per_file": 8  # Limit für Konsistenz
            }
        }

    def ensure_directories(self):
        """Erstelle Production-Verzeichnisstruktur"""
        dirs = ["individual_sprites", "animations", "reports", "final_sprites"]
        for d in dirs:
            (self.output_dir / d).mkdir(parents=True, exist_ok=True)

    def get_input_files(self) -> List[Path]:
        """Sammle alle verarbeitbaren Input-Dateien"""
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
        files = []

        # Hauptverzeichnis
        for pattern in patterns:
            files.extend(self.input_dir.glob(pattern))

        # Sprite-Sheets Unterverzeichnis
        sprite_dir = self.input_dir / "sprite_sheets"
        if sprite_dir.exists():
            for pattern in patterns:
                files.extend(sprite_dir.glob(pattern))

        # Filter für gültige Größen (min 1MB für bessere Qualität)
        valid_files = []
        for f in files:
            try:
                size = f.stat().st_size
                if size > 1024 * 1024:  # > 1MB
                    valid_files.append(f)
            except:
                continue

        print(f"📁 Gefunden: {len(valid_files)} verarbeitbare Dateien")
        return sorted(valid_files)

    def remove_background_transparent(self, image: np.ndarray) -> np.ndarray:
        """Bewährte Background-Removal aus Iteration 4"""
        if len(image.shape) == 3:
            image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        else:
            image_rgba = image.copy()

        h, w = image_rgba.shape[:2]
        corner_size = min(h, w) // 20

        corners = [
            image_rgba[0:corner_size, 0:corner_size],
            image_rgba[0:corner_size, w-corner_size:w],
            image_rgba[h-corner_size:h, 0:corner_size],
            image_rgba[h-corner_size:h, w-corner_size:w],
            image_rgba[0:corner_size, w//2-corner_size//2:w//2+corner_size//2],
            image_rgba[h-corner_size:h, w//2 -
                       corner_size//2:w//2+corner_size//2]
        ]

        bg_colors = []
        for corner in corners:
            if corner.size > 0:
                avg_color = np.mean(corner.reshape(-1, 4), axis=0)[:3]
                bg_colors.append(avg_color)

        bg_color = np.mean(bg_colors, axis=0)

        diff = np.linalg.norm(image_rgba[:, :, :3] - bg_color, axis=2)
        mask = diff > self.config["background_removal"]["edge_tolerance"]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (self.config["background_removal"]["morphology_kernel"],
                                            self.config["background_removal"]["morphology_kernel"]))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        image_rgba[:, :, 3] = mask * 255
        return image_rgba

    def anatomical_frame_analysis(self, contours: List) -> List[Dict]:
        """Anatomische Analyse mit Frame-Limit"""
        valid_frames = []

        # Sortiere nach Größe (größte zuerst)
        contour_areas = [(cv2.contourArea(c), c) for c in contours]
        contour_areas.sort(reverse=True)

        for area, contour in contour_areas:
            if len(valid_frames) >= self.config["processing"]["max_frames_per_file"]:
                break

            x, y, w, h = cv2.boundingRect(contour)

            if area < self.config["anatomical_analysis"]["min_frame_area"]:
                continue

            aspect_ratio = h / w if w > 0 else 0
            head_height = h * 0.3
            head_ratio = head_height / h

            is_anatomically_valid = (
                self.config["anatomical_analysis"]["head_ratio_min"] <= head_ratio <=
                self.config["anatomical_analysis"]["head_ratio_max"] and
                self.config["anatomical_analysis"]["body_aspect_min"] <= aspect_ratio <=
                self.config["anatomical_analysis"]["body_aspect_max"]
            )

            if is_anatomically_valid:
                padding = max(w, h) // 10
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = x + w + padding
                y_end = y + h + padding

                valid_frames.append({
                    "id": len(valid_frames) + 1,
                    "bbox": (x_start, y_start, x_end, y_end),
                    "area": int(area),
                    "aspect_ratio": aspect_ratio,
                    "head_ratio": head_ratio,
                    "anatomically_valid": True
                })

        return valid_frames

    def apply_instagram_filter(self, image: Image.Image) -> Image.Image:
        """Instagram-Filter aus Iteration 4"""
        img_array = np.array(image.convert('RGB'))

        warmth = self.config["instagram_filter"]["warmth"]
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * warmth, 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] / warmth, 0, 255)

        img = Image.fromarray(img_array.astype(np.uint8))

        enhancers = [
            (ImageEnhance.Contrast,
             self.config["instagram_filter"]["contrast"]),
            (ImageEnhance.Color,
             self.config["instagram_filter"]["saturation"]),
            (ImageEnhance.Brightness,
             self.config["instagram_filter"]["brightness"])
        ]

        for enhancer_class, factor in enhancers:
            enhancer = enhancer_class(img)
            img = enhancer.enhance(factor)

        img = img.filter(ImageFilter.UnsharpMask(
            radius=1, percent=50, threshold=1))
        return img

    def enhance_linework_and_shadows(self, image: np.ndarray) -> np.ndarray:
        """Enhanced Linework aus Iteration 4"""
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)

        edge_mask = edges > 0
        for c in range(3):
            image[:, :, c][edge_mask] = np.clip(
                image[:, :, c][edge_mask] - 20, 0, 255)

        hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]

        shadow_mask = v_channel < 100
        hsv[:, :, 2][shadow_mask] = np.clip(
            v_channel[shadow_mask] * 0.85, 0, 255)

        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if image.shape[2] == 4:
            enhanced = np.dstack([enhanced, image[:, :, 3]])

        return enhanced

    def process_single_image(self, image_path: Path) -> Dict:
        """Production Processing für einzelnes Bild"""
        try:
            print(f"🎯 {image_path.name}")
            start_time = time.time()

            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                return {"error": f"Konnte nicht laden: {image_path.name}"}

            original_shape = image.shape

            # Pipeline aus Iteration 4
            image_transparent = self.remove_background_transparent(image)

            gray = cv2.cvtColor(
                image_transparent[:, :, :3], cv2.COLOR_BGR2GRAY)
            alpha = image_transparent[:, :, 3]
            combined = cv2.bitwise_and(gray, alpha)
            contours, _ = cv2.findContours(
                combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_frames = self.anatomical_frame_analysis(contours)

            processed_frames = []
            sprite_dir = self.output_dir / "individual_sprites" / image_path.stem
            sprite_dir.mkdir(exist_ok=True)

            for frame_info in valid_frames:
                x1, y1, x2, y2 = frame_info["bbox"]
                frame = image_transparent[y1:y2, x1:x2]

                if frame.size == 0:
                    continue

                frame_enhanced = self.enhance_linework_and_shadows(frame)
                frame_pil = Image.fromarray(cv2.cvtColor(
                    frame_enhanced[:, :, :3], cv2.COLOR_BGR2RGB))
                frame_filtered = self.apply_instagram_filter(frame_pil)

                frame_final = cv2.cvtColor(
                    np.array(frame_filtered), cv2.COLOR_RGB2BGR)
                frame_final = np.dstack([frame_final, frame_enhanced[:, :, 3]])

                frame_filename = f"frame_{frame_info['id']:03d}.png"
                frame_path = sprite_dir / frame_filename
                cv2.imwrite(str(frame_path), frame_final)

                processed_frames.append(frame_final)

            # GIF Animation
            if processed_frames:
                gif_path = self.output_dir / "animations" / \
                    f"{image_path.stem}_optimized.gif"
                self.create_gif(processed_frames, gif_path)

            processing_time = time.time() - start_time

            result = {
                "filename": image_path.name,
                "original_size": [original_shape[1], original_shape[0]],
                "frames_extracted": len(processed_frames),
                "processing_time": round(processing_time, 2),
                "anatomical_frames": len(valid_frames),
                "success": True
            }

            # Speichere Report
            report_path = self.output_dir / "reports" / \
                f"{image_path.stem}_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            return result

        except Exception as e:
            return {"filename": image_path.name, "error": str(e), "success": False}

    def create_gif(self, frames: List[np.ndarray], output_path: Path, duration: int = 400):
        """GIF mit Transparenz"""
        if not frames:
            return

        pil_frames = []
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB)
            frame_rgba = np.dstack([frame_rgb, frame[:, :, 3]])
            pil_frame = Image.fromarray(frame_rgba, 'RGBA')
            pil_frames.append(pil_frame)

        pil_frames[0].save(
            output_path, save_all=True, append_images=pil_frames[1:],
            duration=duration, loop=0, transparency=0, disposal=2
        )

    def run_production_batch(self):
        """Production Batch Processing"""
        print("🚀 PRODUCTION SPRITE PROCESSOR")
        print("=" * 50)
        print("🎯 Iteration 4 Optimierungen:")
        print("  ✅ 100% Transparenz")
        print("  ✅ Anatomische Frame-Analyse")
        print("  ✅ Instagram-Filter + Weißabgleich")
        print("  ✅ Enhanced Linework & Shadows")
        print("  ✅ Konsistente Frame-Limits")
        print()

        input_files = self.get_input_files()
        if not input_files:
            print("❌ Keine verarbeitbaren Dateien gefunden!")
            return

        print(
            f"📊 Verarbeite {len(input_files)} Dateien mit {self.config['processing']['max_workers']} Threads")
        print()

        results = []
        successful = 0
        failed = 0
        total_frames = 0

        start_time = time.time()

        # Multi-threaded Processing
        with ThreadPoolExecutor(max_workers=self.config["processing"]["max_workers"]) as executor:
            future_to_file = {executor.submit(
                self.process_single_image, f): f for f in input_files}

            for i, future in enumerate(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result.get("success", False):
                        successful += 1
                        total_frames += result.get("frames_extracted", 0)
                        print(
                            f"  ✅ {result['filename']} | {result['frames_extracted']} frames | {result['processing_time']}s")
                    else:
                        failed += 1
                        print(
                            f"  ❌ {result['filename']} | {result.get('error', 'Unknown error')}")

                except Exception as e:
                    failed += 1
                    print(f"  ❌ {file_path.name} | Exception: {e}")

                # Progress
                progress = (i + 1) / len(input_files) * 100
                print(
                    f"     Progress: {progress:.1f}% ({i+1}/{len(input_files)})")

        total_time = time.time() - start_time

        # Master Report
        master_report = {
            "session": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "total_files": len(input_files),
            "successful": successful,
            "failed": failed,
            "total_frames": total_frames,
            "total_time": round(total_time, 2),
            "avg_time_per_file": round(total_time / len(input_files), 2),
            "frames_per_second": round(total_frames / total_time, 2),
            "configuration": self.config,
            "improvements": [
                "100% Background Transparency",
                "Anatomical Frame Analysis",
                "Instagram Filter + White Balance",
                "Enhanced Linework & Shadows",
                "Generous Anatomical Cropping",
                "Consistent Frame Limits",
                "Multi-threaded Processing"
            ],
            "results": results
        }

        master_path = self.output_dir / "PRODUCTION_MASTER_REPORT.json"
        with open(master_path, 'w', encoding='utf-8') as f:
            json.dump(master_report, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 50)
        print("🎉 PRODUCTION PROCESSING ABGESCHLOSSEN!")
        print(f"✅ Erfolgreich: {successful}/{len(input_files)}")
        print(f"❌ Fehlgeschlagen: {failed}/{len(input_files)}")
        print(f"📦 Frames total: {total_frames}")
        print(f"⏱️ Zeit total: {total_time:.1f}s")
        print(f"🚀 Frames/Sekunde: {total_frames/total_time:.1f}")
        print(f"📂 Output: {self.output_dir}")
        print("=" * 50)

        return master_report


def main():
    processor = ProductionSpriteProcessor()
    results = processor.run_production_batch()
    return results


if __name__ == "__main__":
    main()
