#!/usr/bin/env python3
"""
Ultimate GIF Reference Pipeline - Example Generation
===================================================

Demonstrates the cutting-edge capabilities of the GIF Reference Pipeline
with a complete professional workflow example.

This example shows:
- Professional-grade analysis and motion extraction
- Cutting-edge AnimateDiff integration
- Real-time quality monitoring
- Adaptive AI optimization
- Broadcast-quality output generation
"""

from custom_nodes.gif_reference_pipeline.pipeline_manager import UltimatePipelineManager, PipelineConfig
import os
import sys
import time
from pathlib import Path

# Add the custom nodes to the path
current_dir = Path(__file__).parent
custom_nodes_path = current_dir / "custom_nodes" / "gif_reference_pipeline"
sys.path.insert(0, str(current_dir))


def run_ultimate_example_generation():
    """
    Run the ultimate example generation with professional quality
    """

    print("=" * 80)
    print("🚀 ULTIMATE GIF REFERENCE PIPELINE - EXAMPLE GENERATION")
    print("=" * 80)
    print()

    # Configuration for maximum quality
    config = PipelineConfig(
        quality_level="professional",           # professional quality
        enable_real_time_preview=True,          # real-time monitoring
        adaptive_optimization=True,             # AI-driven optimization
        gpu_acceleration=True,                  # GPU acceleration
        memory_optimization="balanced",         # balanced performance
        parallel_processing=True,               # parallel processing
        quality_assurance=True                  # quality control
    )

    # Initialize ultimate pipeline manager
    pipeline = UltimatePipelineManager()

    # Example parameters
    reference_gif = "input/example_animation.gif"  # Reference animation
    generation_prompt = "A majestic dragon soaring through clouds, cinematic lighting, epic fantasy style"

    print(f"📋 GENERATION CONFIGURATION")
    print(f"   Quality Level: {config.quality_level}")
    print(f"   Real-time Preview: {config.enable_real_time_preview}")
    print(f"   AI Optimization: {config.adaptive_optimization}")
    print(f"   GPU Acceleration: {config.gpu_acceleration}")
    print(f"   Quality Assurance: {config.quality_assurance}")
    print()

    print(f"🎯 GENERATION PARAMETERS")
    print(f"   Reference GIF: {reference_gif}")
    print(f"   Prompt: {generation_prompt}")
    print(f"   Expected Output: Professional 16-frame animation")
    print()

    # Execute the ultimate workflow
    start_time = time.time()

    print("🎬 STARTING ULTIMATE GENERATION WORKFLOW")
    print("-" * 50)

    try:
        # Run complete professional workflow
        results = pipeline.execute_complete_workflow(
            reference_gif_path=reference_gif,
            generation_prompt=generation_prompt,
            config=config
        )

        execution_time = time.time() - start_time

        # Display comprehensive results
        display_ultimate_results(results, execution_time)

        # Save professional outputs
        save_professional_outputs(results)

        # Generate quality report
        generate_quality_report(results)

        print()
        print("🎉 ULTIMATE GENERATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        return results

    except Exception as e:
        print(f"❌ ULTIMATE GENERATION FAILED: {str(e)}")
        print("=" * 80)
        return None


def display_ultimate_results(results, execution_time):
    """Display comprehensive results with professional formatting"""

    print()
    print("📊 ULTIMATE GENERATION RESULTS")
    print("=" * 50)

    if "error" in results:
        print(f"❌ Generation failed: {results['error']}")
        return

    # Quality Metrics
    print("🎯 QUALITY METRICS:")
    quality_score = results.get("quality_score", 0)
    pipeline_metrics = results.get("pipeline_metrics", {})

    print(
        f"   ┌─ Overall Quality Score: {quality_score:.2f}/1.00 {'🏆' if quality_score >= 0.9 else '⭐' if quality_score >= 0.8 else '🔧'}")
    print(
        f"   ├─ Analysis Quality: {pipeline_metrics.get('analysis_quality', 0):.2f}")
    print(
        f"   ├─ Generation Quality: {pipeline_metrics.get('generation_quality', 0):.2f}")
    print(
        f"   └─ Enhancement Quality: {pipeline_metrics.get('enhancement_quality', 0):.2f}")
    print()

    # Performance Statistics
    print("⚡ PERFORMANCE STATISTICS:")
    perf_stats = results.get("performance_stats", {})

    print(
        f"   ┌─ Total Processing Time: {perf_stats.get('total_processing_time', execution_time):.1f}s")
    print(
        f"   ├─ Frames Per Second: {perf_stats.get('frames_per_second', 0):.2f} fps")
    print(f"   ├─ Memory Usage: {perf_stats.get('memory_usage_mb', 0):.0f} MB")
    print(
        f"   └─ GPU Utilization: {perf_stats.get('gpu_utilization', 0)*100:.1f}%")
    print()

    # Output Information
    print("📽️ OUTPUT INFORMATION:")
    output_data = results.get("output_data", {})

    print(f"   ┌─ Frame Count: {output_data.get('frame_count', 0)} frames")
    print(
        f"   ├─ Resolution: {output_data.get('resolution', (0, 0))[0]}x{output_data.get('resolution', (0, 0))[1]}")
    print(f"   ├─ Frame Rate: {output_data.get('fps', 0):.1f} fps")
    print(f"   └─ Duration: {output_data.get('duration', 0):.2f}s")
    print()

    # Technical Details
    print("🔧 TECHNICAL DETAILS:")
    metadata = results.get("metadata", {})

    print(
        f"   ┌─ Pipeline Version: {metadata.get('pipeline_version', 'Unknown')}")
    print(f"   ├─ Quality Level: {metadata.get('quality_level', 'Unknown')}")
    print(f"   ├─ Processing Config: Professional")
    print(
        f"   └─ Generation Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata.get('timestamp', time.time())))}")


def save_professional_outputs(results):
    """Save professional-quality outputs"""

    print()
    print("💾 SAVING PROFESSIONAL OUTPUTS")
    print("-" * 30)

    output_dir = Path("output/ultimate_generation_example")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save generation results
        results_file = output_dir / "generation_results.json"

        # Prepare serializable results
        serializable_results = {
            "status": results.get("status", "unknown"),
            "quality_score": results.get("quality_score", 0),
            "pipeline_metrics": results.get("pipeline_metrics", {}),
            "performance_stats": results.get("performance_stats", {}),
            "metadata": results.get("metadata", {})
        }

        import json
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"✓ Results saved: {results_file}")

        # Save frame information (placeholder for actual frames)
        frame_info_file = output_dir / "frame_information.txt"
        with open(frame_info_file, 'w') as f:
            output_data = results.get("output_data", {})
            f.write(f"Frame Information\n")
            f.write(f"================\n\n")
            f.write(f"Frame Count: {output_data.get('frame_count', 0)}\n")
            f.write(f"Resolution: {output_data.get('resolution', (0, 0))}\n")
            f.write(f"FPS: {output_data.get('fps', 0)}\n")
            f.write(f"Duration: {output_data.get('duration', 0)}s\n")
            f.write(f"Quality Score: {results.get('quality_score', 0):.2f}\n")

        print(f"✓ Frame info saved: {frame_info_file}")

        # Create quality report
        quality_report_file = output_dir / "quality_report.txt"
        with open(quality_report_file, 'w') as f:
            f.write("ULTIMATE GIF REFERENCE PIPELINE - QUALITY REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(
                f"Overall Quality Score: {results.get('quality_score', 0):.2f}/1.00\n\n")

            pipeline_metrics = results.get("pipeline_metrics", {})
            f.write("Pipeline Quality Breakdown:\n")
            f.write(
                f"- Analysis Quality: {pipeline_metrics.get('analysis_quality', 0):.2f}\n")
            f.write(
                f"- Generation Quality: {pipeline_metrics.get('generation_quality', 0):.2f}\n")
            f.write(
                f"- Enhancement Quality: {pipeline_metrics.get('enhancement_quality', 0):.2f}\n\n")

            perf_stats = results.get("performance_stats", {})
            f.write("Performance Statistics:\n")
            f.write(
                f"- Processing Time: {perf_stats.get('total_processing_time', 0):.1f}s\n")
            f.write(
                f"- Frames Per Second: {perf_stats.get('frames_per_second', 0):.2f} fps\n")
            f.write(
                f"- Memory Usage: {perf_stats.get('memory_usage_mb', 0):.0f} MB\n")
            f.write(
                f"- GPU Utilization: {perf_stats.get('gpu_utilization', 0)*100:.1f}%\n")

        print(f"✓ Quality report saved: {quality_report_file}")

    except Exception as e:
        print(f"⚠️ Error saving outputs: {str(e)}")


def generate_quality_report(results):
    """Generate comprehensive quality assessment"""

    print()
    print("🔍 QUALITY ASSESSMENT REPORT")
    print("-" * 40)

    quality_score = results.get("quality_score", 0)
    pipeline_metrics = results.get("pipeline_metrics", {})

    # Overall Assessment
    if quality_score >= 0.95:
        assessment = "EXCEPTIONAL - Broadcast Quality ⭐⭐⭐⭐⭐"
    elif quality_score >= 0.90:
        assessment = "EXCELLENT - Professional Quality ⭐⭐⭐⭐"
    elif quality_score >= 0.80:
        assessment = "GOOD - Standard Quality ⭐⭐⭐"
    elif quality_score >= 0.70:
        assessment = "ACCEPTABLE - Draft Quality ⭐⭐"
    else:
        assessment = "NEEDS IMPROVEMENT ⭐"

    print(f"📊 Overall Assessment: {assessment}")
    print()

    # Detailed Breakdown
    print("📋 Detailed Quality Breakdown:")

    analysis_quality = pipeline_metrics.get("analysis_quality", 0)
    generation_quality = pipeline_metrics.get("generation_quality", 0)
    enhancement_quality = pipeline_metrics.get("enhancement_quality", 0)

    print(
        f"   ┌─ Motion Analysis: {'✓ Excellent' if analysis_quality >= 0.9 else '⚠ Good' if analysis_quality >= 0.8 else '❌ Needs Work'} ({analysis_quality:.2f})")
    print(
        f"   ├─ Generation Pipeline: {'✓ Excellent' if generation_quality >= 0.9 else '⚠ Good' if generation_quality >= 0.8 else '❌ Needs Work'} ({generation_quality:.2f})")
    print(
        f"   └─ Quality Enhancement: {'✓ Excellent' if enhancement_quality >= 0.9 else '⚠ Good' if enhancement_quality >= 0.8 else '❌ Needs Work'} ({enhancement_quality:.2f})")
    print()

    # Recommendations
    print("💡 Recommendations:")

    if quality_score >= 0.95:
        print("   ✨ Perfect! Ready for production use.")
    elif quality_score >= 0.90:
        print("   🎯 Excellent quality. Minor optimizations possible.")
    elif quality_score >= 0.80:
        print("   🔧 Good quality. Consider increasing quality settings.")
    else:
        print("   ⚙️ Quality improvements needed. Review configuration.")

    # Performance Assessment
    perf_stats = results.get("performance_stats", {})
    processing_time = perf_stats.get("total_processing_time", 0)

    print()
    print("⚡ Performance Assessment:")

    if processing_time < 30:
        perf_assessment = "EXCELLENT - Very Fast"
    elif processing_time < 60:
        perf_assessment = "GOOD - Acceptable Speed"
    elif processing_time < 120:
        perf_assessment = "MODERATE - Consider Optimization"
    else:
        perf_assessment = "SLOW - Optimization Needed"

    print(f"   Processing Speed: {perf_assessment} ({processing_time:.1f}s)")

    # Final Verdict
    print()
    print("🏆 FINAL VERDICT:")

    if quality_score >= 0.95 and processing_time < 60:
        verdict = "OUTSTANDING - Production Ready!"
    elif quality_score >= 0.90:
        verdict = "EXCELLENT - Professional Quality"
    elif quality_score >= 0.80:
        verdict = "GOOD - Suitable for Most Uses"
    else:
        verdict = "NEEDS IMPROVEMENT - Review Settings"

    print(f"   {verdict}")


if __name__ == "__main__":
    # Run the ultimate example generation
    results = run_ultimate_example_generation()

    if results:
        print()
        print("🎬 Example generation completed successfully!")
        print("Check the output directory for detailed results.")
    else:
        print()
        print("❌ Example generation failed.")
        print("Please check the error messages above.")
