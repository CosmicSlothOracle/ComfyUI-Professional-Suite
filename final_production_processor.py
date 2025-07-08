#!/usr/bin/env python3
"""
🎯 FINAL PRODUCTION PROCESSOR - OPTIMIZED
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import json
import time
from pathlib import Path
from datetime import datetime


class FinalProductionProcessor:
    def __init__(self):
        self.session_id = f"final_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.base_dir = Path("C:/Users/Public/ComfyUI-master")
        self.output_dir = self.base_dir / "output" / "final_optimized" / self.session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_final_processing(self):
        """Führe finale Verarbeitung durch"""
        print("🚀 FINAL PRODUCTION PROCESSING")
        print("=" * 50)

        input_files = list(self.base_dir.glob("input/*.png"))
        input_files.extend(
            list(self.base_dir.glob("input/sprite_sheets/*.png")))

        # Filter für > 1MB
        valid_files = [f for f in input_files if f.stat().st_size >
                       1024 * 1024]

        print(f"📁 Verarbeite {len(valid_files)} Dateien")

        results = []
        for i, file_path in enumerate(valid_files[:10]):  # Limit für Test
            print(f"🎯 [{i+1}] {file_path.name}")

            try:
                image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    continue

                # Simple transparent background removal
                if len(image.shape) == 3:
                    image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
                else:
                    image_rgba = image.copy()

                # Basic edge-based mask
                gray = cv2.cvtColor(image_rgba[:, :, :3], cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)

                # Create mask
                mask = cv2.morphologyEx(
                    edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

                # Apply transparency
                image_rgba[:, :, 3] = mask

                # Save result
                output_path = self.output_dir / f"processed_{file_path.name}"
                cv2.imwrite(str(output_path), image_rgba)

                results.append({
                    "file": file_path.name,
                    "output": str(output_path),
                    "success": True
                })

                print(f"  ✅ Gespeichert: {output_path.name}")

            except Exception as e:
                results.append(
                    {"file": file_path.name, "error": str(e), "success": False})
                print(f"  ❌ Fehler: {e}")

        # Report
        report = {
            "session": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "processed": len([r for r in results if r.get("success", False)]),
            "total": len(results),
            "results": results
        }

        report_path = self.output_dir / "FINAL_REPORT.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(
            f"\n📊 FERTIG: {report['processed']}/{report['total']} erfolgreich")
        print(f"📂 Output: {self.output_dir}")

        return report


def main():
    processor = FinalProductionProcessor()
    return processor.run_final_processing()


if __name__ == "__main__":
    main()
