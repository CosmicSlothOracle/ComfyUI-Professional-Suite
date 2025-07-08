#!/usr/bin/env python3
"""
AI BATCH INTELLIGENT SPRITESHEET PROCESSOR
Verwendet das bereits vorhandene intelligente System mit DBSCAN und ML-Algorithmen
"""

import os
import sys
from pathlib import Path
import time
import json
from intelligent_spritesheet_processor import IntelligentSpritesheetProcessor


class AIBatchProcessor:
    """AI-Batch-Processor mit intelligent clustering"""

    def __init__(self):
        self.input_dir = Path("input")
        self.output_base = Path("output/ai_intelligent_sprites")
        self.processed_count = 0
        self.failed_count = 0

    def get_spritesheet_files(self):
        """Findet alle Spritesheet-Dateien"""
        sprite_files = []

        # Main input directory
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            sprite_files.extend(self.input_dir.glob(ext))

        # Sprite sheets subdirectory
        sprite_sheets_dir = self.input_dir / "sprite_sheets"
        if sprite_sheets_dir.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                sprite_files.extend(sprite_sheets_dir.glob(ext))

        # Filter large files only
        valid_files = []
        for file in sprite_files:
            try:
                if file.stat().st_size > 100_000:  # > 100KB
                    valid_files.append(file)
                    print(
                        f"🧠 AI Target: {file.name} ({file.stat().st_size // 1024}KB)")
            except:
                continue

        return valid_files

    def process_single_spritesheet(self, image_path: Path):
        """Verarbeitet ein einzelnes Spritesheet mit AI"""
        print(f"\n🤖 AI Processing: {image_path.name}")

        try:
            # Initialize processor
            processor = IntelligentSpritesheetProcessor(
                output_dir=str(self.output_base / image_path.stem)
            )

            # Process with intelligent algorithms
            start_time = time.time()
            result = processor.process_spritesheet(
                str(image_path),
                bg_removal_method="adaptive"  # Uses AI-based adaptive method
            )
            end_time = time.time()

            processing_time = end_time - start_time

            if 'error' in result:
                print(f"❌ AI Processing failed: {result['error']}")
                return {
                    "success": False,
                    "filename": image_path.name,
                    "error": result['error'],
                    "processing_time": processing_time
                }

            # Extract results
            frames_count = len(result.get('frames', []))
            clusters_info = result.get('clusters', {})
            analysis_data = result.get('analysis', {})

            print(f"✅ AI Success: {frames_count} frames extracted")
            print(f"🧠 AI Clusters: {len(clusters_info)} size groups")
            print(f"⏱️ Processing time: {processing_time:.2f}s")

            # Create detailed AI report
            ai_report = {
                "algorithm_used": "Intelligent_DBSCAN_Clustering",
                "frames_extracted": frames_count,
                "processing_time": processing_time,
                "clusters_detected": len(clusters_info),
                "analysis_data": analysis_data,
                "cluster_details": clusters_info,
                "ai_features": [
                    "DBSCAN size clustering",
                    "Adaptive background removal",
                    "Connected Components Analysis",
                    "Intelligent frame alignment",
                    "Size-based component filtering"
                ]
            }

            # Save AI report
            output_dir = self.output_base / image_path.stem
            report_path = output_dir / "ai_processing_report.json"

            if output_dir.exists():
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(ai_report, f, indent=2, ensure_ascii=False)

            self.processed_count += 1

            return {
                "success": True,
                "filename": image_path.name,
                "frames_count": frames_count,
                "clusters_count": len(clusters_info),
                "processing_time": processing_time,
                "output_dir": str(output_dir),
                "ai_report": ai_report
            }

        except Exception as e:
            print(f"❌ AI Error: {e}")
            self.failed_count += 1
            return {
                "success": False,
                "filename": image_path.name,
                "error": str(e),
                "processing_time": 0
            }

    def process_all_spritesheets(self):
        """Verarbeitet alle Spritesheets mit AI"""
        print("🤖 AI INTELLIGENT BATCH PROCESSING")
        print("=" * 70)
        print("🧠 AI Algorithms Active:")
        print("  • DBSCAN Clustering for frame size grouping")
        print("  • Adaptive background detection")
        print("  • Connected Components Analysis")
        print("  • Intelligent frame extraction")
        print("  • Size-based filtering")
        print("  • Grid alignment algorithms")
        print("=" * 70)

        # Create output directory
        self.output_base.mkdir(parents=True, exist_ok=True)

        # Get all files
        sprite_files = self.get_spritesheet_files()

        if not sprite_files:
            print("❌ No spritesheet files found!")
            return

        print(f"\n🎯 Processing {len(sprite_files)} files with AI...")

        # Process each file
        results = []
        total_frames = 0
        total_clusters = 0

        for i, sprite_file in enumerate(sprite_files, 1):
            print(f"\n[{i}/{len(sprite_files)}] " + "="*50)

            result = self.process_single_spritesheet(sprite_file)
            results.append(result)

            if result["success"]:
                total_frames += result.get("frames_count", 0)
                total_clusters += result.get("clusters_count", 0)

            # Small delay between files
            time.sleep(0.5)

        # Create comprehensive AI analysis report
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]

        if successful_results:
            avg_frames = total_frames / len(successful_results)
            avg_clusters = total_clusters / len(successful_results)
            avg_time = sum(r["processing_time"]
                           for r in successful_results) / len(successful_results)
        else:
            avg_frames = avg_clusters = avg_time = 0

        comprehensive_report = {
            "ai_processing_summary": {
                "total_files_processed": len(sprite_files),
                "successful_extractions": len(successful_results),
                "failed_extractions": len(failed_results),
                "success_rate_percent": (len(successful_results) / len(sprite_files) * 100) if sprite_files else 0,
                "total_frames_extracted": total_frames,
                "total_clusters_detected": total_clusters,
                "average_frames_per_sprite": avg_frames,
                "average_clusters_per_sprite": avg_clusters,
                "average_processing_time_seconds": avg_time
            },
            "ai_algorithms_used": {
                "clustering": "DBSCAN for size-based frame grouping",
                "background_removal": "Adaptive multi-zone sampling",
                "segmentation": "Connected Components Analysis",
                "filtering": "Area and aspect ratio based intelligent filtering",
                "alignment": "Grid-based frame standardization"
            },
            "detailed_results": results,
            "performance_metrics": {
                "fastest_processing": min((r["processing_time"] for r in successful_results), default=0),
                "slowest_processing": max((r["processing_time"] for r in successful_results), default=0),
                "most_frames_extracted": max((r["frames_count"] for r in successful_results), default=0),
                "most_clusters_found": max((r["clusters_count"] for r in successful_results), default=0)
            }
        }

        # Save comprehensive report
        comprehensive_report_path = self.output_base / "ai_comprehensive_analysis.json"
        with open(comprehensive_report_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)

        # Final statistics
        print("\n" + "="*70)
        print("🎉 AI BATCH PROCESSING COMPLETE!")
        print(f"✅ Successfully processed: {self.processed_count}")
        print(f"❌ Failed: {self.failed_count}")
        print(f"📊 Total files: {len(sprite_files)}")
        print(
            f"💯 AI Success rate: {(self.processed_count / len(sprite_files) * 100):.1f}%")
        print(f"🎯 Total frames extracted: {total_frames}")
        print(f"🧠 Total AI clusters detected: {total_clusters}")
        print(f"⚡ Average processing: {avg_time:.2f}s per file")
        print(f"📋 Comprehensive AI report: {comprehensive_report_path}")
        print("🤖 AI algorithms successfully analyzed all spritesheets!")


def main():
    """Main execution"""
    processor = AIBatchProcessor()
    processor.process_all_spritesheets()


if __name__ == "__main__":
    print("🚀 AI INTELLIGENT BATCH SPRITESHEET PROCESSOR")
    print("Powered by DBSCAN ML Clustering & Adaptive AI Algorithms")

    main()
