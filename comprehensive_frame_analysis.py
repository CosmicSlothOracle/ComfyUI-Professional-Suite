#!/usr/bin/env python3
"""
COMPREHENSIVE FRAME ANALYSIS
Systematische Analyse der Connected Components-Probleme bei der Spritesheet-Verarbeitung

ZIEL:
1. Quantifizierung der Häufigkeit von Frame-Extraktionsproblemen
2. Identifikation der Hauptursachen
3. Statistische Analyse der Frame-Größenverteilungen
4. Entwicklung einer robusten Lösung
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from datetime import datetime


class ComprehensiveFrameAnalysis:
    """Umfassende Analyse der Frame-Extraktions-Probleme"""

    def __init__(self):
        self.session_name = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.analysis_dir = Path("output/frame_analysis") / self.session_name
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        # Sammle Daten
        self.all_results = []
        self.problem_cases = []
        self.size_distributions = []
        self.frame_count_distribution = Counter()

    def load_existing_results(self) -> Dict:
        """Lade alle existierenden Verarbeitungs-Ergebnisse"""
        results = {}

        # Suche in allen Session-Verzeichnissen
        base_dir = Path("output/original_optimized")
        if base_dir.exists():
            for session_dir in base_dir.iterdir():
                if session_dir.is_dir():
                    reports_dir = session_dir / "reports"
                    if reports_dir.exists():
                        for report_file in reports_dir.glob("*_report.json"):
                            try:
                                with open(report_file, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    filename = report_file.stem.replace(
                                        '_report', '')
                                    results[filename] = data
                            except Exception as e:
                                print(
                                    f"   ⚠️ Fehler beim Laden {report_file}: {e}")

        return results

    def analyze_frame_size_problems(self, results: Dict) -> Dict:
        """Analysiere Frame-Größenprobleme"""
        print("📊 Analysiere Frame-Größenprobleme...")

        analysis = {
            "total_files": len(results),
            "problematic_files": 0,
            "oversegmentation": [],  # Zu viele kleine Frames
            "undersegmentation": [],  # Zu wenige große Frames
            "inconsistent_sizes": [],  # Stark variierende Größen
            "size_statistics": {}
        }

        all_frame_areas = []
        all_frame_counts = []

        for filename, data in results.items():
            if "frames" not in data or not data["frames"]:
                continue

            frames = data["frames"]
            frame_count = len(frames)
            all_frame_counts.append(frame_count)
            self.frame_count_distribution[frame_count] += 1

            # Berechne Frame-Größen
            areas = []
            sizes = []
            for frame in frames:
                if "area" in frame:
                    areas.append(frame["area"])
                    all_frame_areas.append(frame["area"])
                elif "size" in frame:
                    # Parse size string wie "1137x1207"
                    try:
                        w, h = map(int, frame["size"].split('x'))
                        area = w * h
                        areas.append(area)
                        all_frame_areas.append(area)
                        sizes.append((w, h))
                    except:
                        continue

            if not areas:
                continue

            # Statistiken für diese Datei
            area_mean = statistics.mean(areas)
            area_std = statistics.stdev(areas) if len(areas) > 1 else 0
            area_cv = area_std / area_mean if area_mean > 0 else 0  # Coefficient of Variation

            file_analysis = {
                "filename": filename,
                "frame_count": frame_count,
                "area_mean": area_mean,
                "area_std": area_std,
                "area_cv": area_cv,
                "area_min": min(areas),
                "area_max": max(areas),
                "areas": areas,
                "sizes": sizes
            }

            self.all_results.append(file_analysis)

            # Identifiziere Probleme
            is_problematic = False

            # 1. Übersegmentierung (zu viele Frames)
            if frame_count > 32:  # Typische Spritesheets haben 4-32 Frames
                analysis["oversegmentation"].append(filename)
                is_problematic = True

            # 2. Untersegmentierung (zu wenige Frames)
            elif frame_count < 2:
                analysis["undersegmentation"].append(filename)
                is_problematic = True

            # 3. Inkonsistente Größen (hohe Varianz)
            elif area_cv > 1.0:  # Coefficient of Variation > 100%
                analysis["inconsistent_sizes"].append(filename)
                is_problematic = True

            if is_problematic:
                analysis["problematic_files"] += 1
                self.problem_cases.append(file_analysis)

        # Globale Statistiken
        if all_frame_areas:
            analysis["size_statistics"] = {
                "total_frames": len(all_frame_areas),
                "mean_area": statistics.mean(all_frame_areas),
                "median_area": statistics.median(all_frame_areas),
                "std_area": statistics.stdev(all_frame_areas),
                "min_area": min(all_frame_areas),
                "max_area": max(all_frame_areas)
            }

        if all_frame_counts:
            analysis["frame_count_stats"] = {
                "mean_count": statistics.mean(all_frame_counts),
                "median_count": statistics.median(all_frame_counts),
                "min_count": min(all_frame_counts),
                "max_count": max(all_frame_counts)
            }

        return analysis

    def analyze_specific_problem_case(self, filename: str) -> Dict:
        """Detailanalyse eines spezifischen Problemfalls"""
        print(f"🔍 Detailanalyse: {filename}")

        # Lade ursprüngliches Bild
        image_path = Path("input") / f"{filename}.png"
        if not image_path.exists():
            return {"error": f"Image not found: {image_path}"}

        # Simuliere Original-Workflow
        image = cv2.imread(str(image_path))
        if image is None:
            return {"error": f"Could not load image: {image_path}"}

        original_size = image.shape[:2]

        # 2x Upscaling (wie im Original)
        upscaled = cv2.resize(image, None, fx=2, fy=2,
                              interpolation=cv2.INTER_CUBIC)

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

        mask = np.all(np.abs(upscaled - bg_color) <= tolerance, axis=-1)
        foreground_mask = ~mask

        # Connected Components Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            foreground_mask.astype(np.uint8), connectivity=8)

        # Analysiere gefundene Komponenten
        components = []
        for i in range(1, num_labels):  # Skip background
            x, y, w_comp, h_comp, area = stats[i]
            if area > 500:  # Original Mindestfläche
                components.append({
                    "id": i,
                    "x": x, "y": y,
                    "width": w_comp, "height": h_comp,
                    "area": area,
                    "aspect_ratio": w_comp / h_comp if h_comp > 0 else 0,
                    "centroid": centroids[i].tolist()
                })

        # Statistiken der Komponenten
        if components:
            areas = [c["area"] for c in components]
            widths = [c["width"] for c in components]
            heights = [c["height"] for c in components]

            component_stats = {
                "count": len(components),
                "area_mean": statistics.mean(areas),
                "area_median": statistics.median(areas),
                "area_std": statistics.stdev(areas) if len(areas) > 1 else 0,
                "area_min": min(areas),
                "area_max": max(areas),
                "size_range": f"{min(widths)}x{min(heights)} - {max(widths)}x{max(heights)}"
            }
        else:
            component_stats = {"count": 0}

        # Speichere Debug-Visualisierung
        debug_path = self.analysis_dir / f"{filename}_debug.png"
        self.save_debug_visualization(
            upscaled, foreground_mask, components, debug_path)

        return {
            "filename": filename,
            "original_size": original_size,
            "upscaled_size": upscaled.shape[:2],
            "background_color": bg_color.tolist(),
            "tolerance": tolerance,
            "foreground_ratio": float(foreground_mask.sum() / foreground_mask.size),
            "component_stats": component_stats,
            # Nur ersten 10 für Übersichtlichkeit
            "components": components[:10],
            "debug_visualization": str(debug_path)
        }

    def save_debug_visualization(self, image: np.ndarray, mask: np.ndarray,
                                 components: List[Dict], output_path: Path):
        """Speichere Debug-Visualisierung"""
        h, w = image.shape[:2]

        # 3-Panel Visualisierung
        debug_img = np.zeros((h, w*3, 3), dtype=np.uint8)

        # Panel 1: Original
        debug_img[:, :w] = image

        # Panel 2: Maske
        mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
        mask_colored[mask] = [0, 255, 0]  # Grün für Vordergrund
        debug_img[:, w:2*w] = mask_colored

        # Panel 3: Komponenten-Bounding Boxes
        components_img = image.copy()
        for i, comp in enumerate(components[:20]):  # Max 20 Components
            x, y, w_comp, h_comp = comp["x"], comp["y"], comp["width"], comp["height"]
            color = (0, 255, 255) if comp["area"] > 10000 else (
                255, 0, 0)  # Gelb für große, Rot für kleine
            cv2.rectangle(components_img, (x, y),
                          (x + w_comp, y + h_comp), color, 2)
            cv2.putText(
                components_img, f"{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        debug_img[:, 2*w:] = components_img

        cv2.imwrite(str(output_path), debug_img)

    def generate_comprehensive_report(self, size_analysis: Dict) -> str:
        """Generiere umfassenden Analysereport"""
        print("📋 Generiere umfassenden Report...")

        report = []
        report.append("# COMPREHENSIVE FRAME ANALYSIS REPORT")
        report.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)

        # Übersicht
        report.append(f"\n## ÜBERSICHT")
        report.append(f"Analysierte Dateien: {size_analysis['total_files']}")
        report.append(
            f"Problematische Dateien: {size_analysis['problematic_files']} ({size_analysis['problematic_files']/size_analysis['total_files']*100:.1f}%)")

        # Problem-Kategorien
        report.append(f"\n## PROBLEM-KATEGORIEN")
        report.append(
            f"Übersegmentierung (>32 Frames): {len(size_analysis['oversegmentation'])}")
        report.append(
            f"Untersegmentierung (<2 Frames): {len(size_analysis['undersegmentation'])}")
        report.append(
            f"Inkonsistente Größen (CV>100%): {len(size_analysis['inconsistent_sizes'])}")

        # Frame-Count Verteilung
        report.append(f"\n## FRAME-COUNT VERTEILUNG")
        for count in sorted(self.frame_count_distribution.keys())[:10]:
            freq = self.frame_count_distribution[count]
            report.append(f"{count:3d} Frames: {freq:3d} Dateien")
        if len(self.frame_count_distribution) > 10:
            report.append("...")

        # Größenstatistiken
        if "size_statistics" in size_analysis:
            stats = size_analysis["size_statistics"]
            report.append(f"\n## GLOBALE FRAME-GRÖSSENSTATISTIKEN")
            report.append(f"Total Frames: {stats['total_frames']}")
            report.append(f"Mittlere Fläche: {stats['mean_area']:.0f} Pixel²")
            report.append(f"Median Fläche: {stats['median_area']:.0f} Pixel²")
            report.append(
                f"Standardabweichung: {stats['std_area']:.0f} Pixel²")
            report.append(
                f"Bereich: {stats['min_area']:.0f} - {stats['max_area']:.0f} Pixel²")

        # Worst Cases
        report.append(f"\n## WORST CASES")

        # Sortiere nach Problemschwere
        worst_overseg = sorted([(f, self.frame_count_distribution.get(
            next((r['frame_count'] for r in self.all_results if r['filename'] == f), 0), 0))
            for f in size_analysis['oversegmentation']], key=lambda x: x[1], reverse=True)[:5]

        report.append(f"\nÜbersegmentierung (Top 5):")
        for filename, count in worst_overseg:
            report.append(f"  • {filename}: {count} Frames")

        worst_inconsistent = sorted([(r['filename'], r['area_cv'])
                                     for r in self.all_results if r['filename'] in size_analysis['inconsistent_sizes']],
                                    key=lambda x: x[1], reverse=True)[:5]

        report.append(f"\nInkonsistente Größen (Top 5):")
        for filename, cv in worst_inconsistent:
            report.append(f"  • {filename}: CV = {cv:.2f}")

        # Empfehlungen
        report.append(f"\n## EMPFEHLUNGEN")
        report.append("1. Implementierung adaptiver Größenfilterung")
        report.append("2. Statistische Validierung der Frame-Extraktion")
        report.append("3. Intelligente Zusammenfassung kleiner Fragmente")
        report.append("4. Grid-basierte Aufspaltung großer Komponenten")
        report.append("5. Qualitätskontrolle mit erwarteten Frame-Anzahlen")

        return "\n".join(report)

    def run_comprehensive_analysis(self):
        """Führe die komplette Analyse durch"""
        print("🔬 COMPREHENSIVE FRAME ANALYSIS")
        print("=" * 50)

        # 1. Lade existierende Ergebnisse
        print("📂 Lade existierende Verarbeitungs-Ergebnisse...")
        results = self.load_existing_results()
        print(f"   Gefunden: {len(results)} verarbeitete Dateien")

        if not results:
            print("❌ Keine Ergebnisse gefunden!")
            return

        # 2. Analysiere Frame-Größenprobleme
        size_analysis = self.analyze_frame_size_problems(results)

        # 3. Detailanalyse problematischer Fälle
        print("\n🔍 Detailanalyse problematischer Fälle...")
        detailed_analyses = {}

        # Analysiere Top 3 problematische Fälle
        top_problems = []

        # Worst oversegmentation case
        if size_analysis["oversegmentation"]:
            worst_overseg = max(size_analysis["oversegmentation"],
                                key=lambda f: next((r['frame_count'] for r in self.all_results if r['filename'] == f), 0))
            top_problems.append(worst_overseg)

        # Worst inconsistent sizes case
        if size_analysis["inconsistent_sizes"]:
            worst_inconsistent = max(size_analysis["inconsistent_sizes"],
                                     key=lambda f: next((r['area_cv'] for r in self.all_results if r['filename'] == f), 0))
            top_problems.append(worst_inconsistent)

        # Undersegmentation case
        if size_analysis["undersegmentation"]:
            top_problems.append(size_analysis["undersegmentation"][0])

        for filename in top_problems[:3]:
            detailed_analyses[filename] = self.analyze_specific_problem_case(
                filename)

        # 4. Generiere Report
        report_text = self.generate_comprehensive_report(size_analysis)

        # 5. Speichere Ergebnisse
        report_path = self.analysis_dir / "comprehensive_analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        # Speichere JSON-Daten
        json_data = {
            "analysis_timestamp": datetime.now().isoformat(),
            "summary": size_analysis,
            "detailed_cases": detailed_analyses,
            # Erste 50 für Übersichtlichkeit
            "all_results": self.all_results[:50]
        }

        json_path = self.analysis_dir / "analysis_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        # Ausgabe der Hauptergebnisse
        print(f"\n📊 ANALYSE ABGESCHLOSSEN")
        print(f"   Analysierte Dateien: {size_analysis['total_files']}")
        print(
            f"   Problematische Dateien: {size_analysis['problematic_files']} ({size_analysis['problematic_files']/size_analysis['total_files']*100:.1f}%)")
        print(
            f"   Übersegmentierung: {len(size_analysis['oversegmentation'])}")
        print(
            f"   Untersegmentierung: {len(size_analysis['undersegmentation'])}")
        print(
            f"   Inkonsistente Größen: {len(size_analysis['inconsistent_sizes'])}")
        print(f"\n📋 Report gespeichert: {report_path}")
        print(f"📊 Daten gespeichert: {json_path}")

        return {
            "report_path": report_path,
            "json_path": json_path,
            "analysis_summary": size_analysis
        }


if __name__ == "__main__":
    analyzer = ComprehensiveFrameAnalysis()
    analyzer.run_comprehensive_analysis()
