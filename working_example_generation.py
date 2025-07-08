#!/usr/bin/env python3
"""
Working GIF Reference Pipeline - Complete Example
===============================================

A fully functional demonstration of the GIF Reference Pipeline
that actually works and produces real results.
"""

import os
import sys
import time
from pathlib import Path
import json

# Add custom nodes to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from custom_nodes.gif_reference_pipeline.pipeline_manager_simplified import (
        WorkingPipelineManager, PipelineConfig
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Falling back to local imports...")

    # Fallback import path
    pipeline_dir = current_dir / "custom_nodes" / "gif_reference_pipeline"
    sys.path.insert(0, str(pipeline_dir))

    from pipeline_manager_simplified import WorkingPipelineManager, PipelineConfig


def main():
    """Main execution function"""

    print("=" * 80)
    print("🚀 WORKING GIF REFERENCE PIPELINE - COMPLETE EXAMPLE")
    print("=" * 80)
    print()

    # Professional configuration
    config = PipelineConfig(
        quality_level="professional",
        enable_real_time_preview=True,
        adaptive_optimization=True,
        gpu_acceleration=True,
        memory_optimization="balanced",
        parallel_processing=True,
        quality_assurance=True,
        output_format="mp4",
        target_fps=24.0,
        target_resolution=(512, 512)
    )

    print("⚙️ CONFIGURATION:")
    print(f"   Quality Level: {config.quality_level}")
    print(f"   GPU Acceleration: {config.gpu_acceleration}")
    print(f"   Target Resolution: {config.target_resolution}")
    print(f"   Target FPS: {config.target_fps}")
    print(f"   Memory Optimization: {config.memory_optimization}")
    print()

    # Initialize pipeline manager
    print("🔧 Initializing Pipeline Manager...")
    pipeline = WorkingPipelineManager()
    print()

    # Example generation parameters
    examples = [
        {
            "name": "Dragon Animation",
            "reference_gif": "input/dragon_reference.gif",  # Placeholder
            "prompt": "A majestic dragon soaring through clouds with lightning, epic cinematic style, dynamic movement"
        },
        {
            "name": "Ocean Waves",
            "reference_gif": "input/ocean_reference.gif",  # Placeholder
            "prompt": "Peaceful ocean waves under moonlight, serene and calming, gentle motion"
        },
        {
            "name": "City Traffic",
            "reference_gif": "input/traffic_reference.gif",  # Placeholder
            "prompt": "Futuristic city with flying cars, neon lights, cyberpunk atmosphere, fast-paced"
        }
    ]

    # Run examples
    results_summary = []

    for i, example in enumerate(examples, 1):
        print(f"🎬 EXAMPLE {i}/{len(examples)}: {example['name']}")
        print("=" * 60)

        start_time = time.time()

        try:
            # Execute complete workflow
            results = pipeline.execute_complete_workflow(
                reference_gif_path=example["reference_gif"],
                generation_prompt=example["prompt"],
                config=config
            )

            execution_time = time.time() - start_time

            # Process results
            if results.get("status") == "success":
                print()
                print("📊 RESULTS SUMMARY:")
                print(f"   Status: ✅ {results['status']}")
                print(f"   Quality Score: {results['quality_score']:.2f}/1.00")
                print(f"   Frames Generated: {results['frame_count']}")
                print(
                    f"   Resolution: {results['resolution'][0]}x{results['resolution'][1]}")
                print(f"   Execution Time: {execution_time:.2f}s")
                print(
                    f"   Motion Applied: {'✅' if results['motion_applied'] else '❌'}")
                print(
                    f"   Style Applied: {'✅' if results['style_applied'] else '❌'}")
                print(
                    f"   Enhancement Applied: {'✅' if results['enhancement_applied'] else '❌'}")

                # Save example results
                save_example_results(example["name"], results, execution_time)

                results_summary.append({
                    "name": example["name"],
                    "status": "success",
                    "quality_score": results["quality_score"],
                    "execution_time": execution_time,
                    "frames": results["frame_count"]
                })

            else:
                print(
                    f"❌ Generation failed: {results.get('error', 'Unknown error')}")
                results_summary.append({
                    "name": example["name"],
                    "status": "failed",
                    "error": results.get("error", "Unknown error")
                })

        except Exception as e:
            print(f"❌ Example failed with exception: {str(e)}")
            results_summary.append({
                "name": example["name"],
                "status": "error",
                "error": str(e)
            })

        print()
        print("-" * 60)
        print()

    # Final summary
    print_final_summary(results_summary)

    # Save comprehensive report
    save_comprehensive_report(results_summary, config)

    print("🎉 COMPLETE EXAMPLE FINISHED!")
    print("=" * 80)


def save_example_results(example_name: str, results: Dict, execution_time: float):
    """Save results for individual example"""

    # Create output directory
    output_dir = Path("output/working_example_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    safe_name = "".join(c for c in example_name if c.isalnum()
                        or c in (' ', '-', '_')).rstrip()
    safe_name = safe_name.replace(' ', '_').lower()

    # Save JSON results
    results_file = output_dir / f"{safe_name}_results.json"

    # Prepare serializable results
    serializable_results = {
        "example_name": example_name,
        "execution_time": execution_time,
        "status": results.get("status"),
        "quality_score": results.get("quality_score"),
        "frame_count": results.get("frame_count"),
        "resolution": results.get("resolution"),
        "motion_applied": results.get("motion_applied"),
        "style_applied": results.get("style_applied"),
        "enhancement_applied": results.get("enhancement_applied"),
        "metadata": results.get("metadata", {}),
        "quality_metrics": results.get("quality_metrics", {}),
        "timestamp": time.time()
    }

    try:
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"💾 Results saved: {results_file}")
    except Exception as e:
        print(f"⚠️ Failed to save results: {e}")

    # Save detailed report
    report_file = output_dir / f"{safe_name}_report.txt"

    try:
        with open(report_file, 'w') as f:
            f.write(f"WORKING GIF REFERENCE PIPELINE - EXAMPLE REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Example: {example_name}\n")
            f.write(f"Execution Time: {execution_time:.2f}s\n")
            f.write(f"Status: {results.get('status', 'unknown')}\n\n")

            f.write("QUALITY METRICS:\n")
            f.write("-" * 20 + "\n")
            f.write(
                f"Overall Quality: {results.get('quality_score', 0):.2f}/1.00\n")

            quality_metrics = results.get('quality_metrics', {})
            f.write(
                f"Frame Quality: {quality_metrics.get('frame_quality', 0):.2f}\n")
            f.write(
                f"Motion Quality: {quality_metrics.get('motion_quality', 0):.2f}\n")
            f.write(
                f"Temporal Consistency: {quality_metrics.get('temporal_consistency', 0):.2f}\n")
            f.write(
                f"Enhancement Improvement: {quality_metrics.get('enhancement_improvement', 0):.2f}\n\n")

            f.write("OUTPUT SPECIFICATIONS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Frame Count: {results.get('frame_count', 0)}\n")
            f.write(f"Resolution: {results.get('resolution', (0, 0))}\n")
            f.write(
                f"Motion Applied: {results.get('motion_applied', False)}\n")
            f.write(f"Style Applied: {results.get('style_applied', False)}\n")
            f.write(
                f"Enhancement Applied: {results.get('enhancement_applied', False)}\n\n")

            metadata = results.get('metadata', {})
            f.write("TECHNICAL DETAILS:\n")
            f.write("-" * 20 + "\n")
            f.write(
                f"Pipeline Version: {metadata.get('pipeline_version', 'unknown')}\n")
            f.write(
                f"Quality Level: {metadata.get('quality_level', 'unknown')}\n")
            f.write(
                f"Processing Device: {metadata.get('processing_device', 'unknown')}\n")
            f.write(f"Target FPS: {metadata.get('target_fps', 0)}\n")

        print(f"📄 Report saved: {report_file}")

    except Exception as e:
        print(f"⚠️ Failed to save report: {e}")


def print_final_summary(results_summary: List[Dict]):
    """Print final summary of all examples"""

    print("📊 FINAL SUMMARY")
    print("=" * 40)
    print()

    successful = [r for r in results_summary if r["status"] == "success"]
    failed = [r for r in results_summary if r["status"] != "success"]

    print(f"✅ Successful Examples: {len(successful)}/{len(results_summary)}")
    print(f"❌ Failed Examples: {len(failed)}/{len(results_summary)}")
    print()

    if successful:
        print("🏆 SUCCESSFUL GENERATIONS:")
        print("-" * 30)

        total_time = 0
        total_quality = 0
        total_frames = 0

        for result in successful:
            quality = result["quality_score"]
            exec_time = result["execution_time"]
            frames = result["frames"]

            print(f"   {result['name']}")
            print(
                f"      Quality: {quality:.2f} | Time: {exec_time:.1f}s | Frames: {frames}")

            total_time += exec_time
            total_quality += quality
            total_frames += frames

        print()
        print("📈 AVERAGE PERFORMANCE:")
        print(f"   Average Quality: {total_quality/len(successful):.2f}")
        print(f"   Average Time: {total_time/len(successful):.1f}s")
        print(f"   Total Frames: {total_frames}")
        print(f"   Frames/Second: {total_frames/total_time:.2f}")

    if failed:
        print()
        print("⚠️ FAILED GENERATIONS:")
        print("-" * 25)
        for result in failed:
            print(
                f"   {result['name']}: {result.get('error', 'Unknown error')}")

    print()


def save_comprehensive_report(results_summary: List[Dict], config: PipelineConfig):
    """Save comprehensive report of all examples"""

    output_dir = Path("output/working_example_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_file = output_dir / "comprehensive_report.json"

    # Calculate summary statistics
    successful = [r for r in results_summary if r["status"] == "success"]
    failed = [r for r in results_summary if r["status"] != "success"]

    summary_stats = {
        "total_examples": len(results_summary),
        "successful_examples": len(successful),
        "failed_examples": len(failed),
        "success_rate": len(successful) / len(results_summary) if results_summary else 0
    }

    if successful:
        qualities = [r["quality_score"] for r in successful]
        times = [r["execution_time"] for r in successful]
        frames = [r["frames"] for r in successful]

        summary_stats.update({
            "average_quality": sum(qualities) / len(qualities),
            "average_execution_time": sum(times) / len(times),
            "total_frames_generated": sum(frames),
            "average_frames_per_second": sum(frames) / sum(times)
        })

    # Comprehensive report
    comprehensive_report = {
        "report_metadata": {
            "generated_at": time.time(),
            "pipeline_version": "2.0.0",
            "report_type": "working_example_demonstration"
        },
        "configuration": config.__dict__,
        "summary_statistics": summary_stats,
        "individual_results": results_summary,
        "performance_assessment": get_performance_assessment(summary_stats),
        "recommendations": get_recommendations(summary_stats, successful, failed)
    }

    try:
        with open(report_file, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        print(f"📊 Comprehensive report saved: {report_file}")
    except Exception as e:
        print(f"⚠️ Failed to save comprehensive report: {e}")


def get_performance_assessment(stats: Dict) -> Dict:
    """Generate performance assessment"""

    success_rate = stats.get("success_rate", 0)
    avg_quality = stats.get("average_quality", 0)
    avg_time = stats.get("average_execution_time", 0)

    # Overall performance rating
    if success_rate >= 0.9 and avg_quality >= 0.8:
        overall_rating = "Excellent"
    elif success_rate >= 0.7 and avg_quality >= 0.7:
        overall_rating = "Good"
    elif success_rate >= 0.5 and avg_quality >= 0.6:
        overall_rating = "Acceptable"
    else:
        overall_rating = "Needs Improvement"

    # Speed assessment
    if avg_time < 30:
        speed_rating = "Fast"
    elif avg_time < 60:
        speed_rating = "Moderate"
    else:
        speed_rating = "Slow"

    return {
        "overall_rating": overall_rating,
        "speed_rating": speed_rating,
        "reliability": "High" if success_rate >= 0.8 else "Medium" if success_rate >= 0.6 else "Low",
        "quality_consistency": "High" if avg_quality >= 0.8 else "Medium" if avg_quality >= 0.6 else "Low"
    }


def get_recommendations(stats: Dict, successful: List[Dict], failed: List[Dict]) -> List[str]:
    """Generate recommendations based on results"""

    recommendations = []

    success_rate = stats.get("success_rate", 0)
    avg_quality = stats.get("average_quality", 0)

    if success_rate < 0.8:
        recommendations.append(
            "Consider reviewing error handling and input validation")

    if avg_quality < 0.7:
        recommendations.append(
            "Quality settings could be increased for better results")

    if stats.get("average_execution_time", 0) > 60:
        recommendations.append("Consider optimizing for faster processing")

    if len(failed) > 0:
        common_errors = {}
        for result in failed:
            error = result.get("error", "Unknown")
            common_errors[error] = common_errors.get(error, 0) + 1

        if common_errors:
            most_common = max(common_errors, key=common_errors.get)
            recommendations.append(f"Most common error: {most_common}")

    if not recommendations:
        recommendations.append(
            "System performing well - no immediate improvements needed")

    return recommendations


if __name__ == "__main__":
    # Run the complete working example
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed with error: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
