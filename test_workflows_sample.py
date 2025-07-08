#!/usr/bin/env python3
"""
Test script for workflows 5-10 with a small sample of files
"""

import subprocess
import sys
from pathlib import Path
import time

# Small sample of input files for testing
sample_files = [
    "C:/Users/Public/ComfyUI-master/input/erdo.gif",
    "C:/Users/Public/ComfyUI-master/input/as.gif",
    "C:/Users/Public/ComfyUI-master/input/DwfOrtv.gif",
    "C:/Users/Public/ComfyUI-master/input/1.gif",
    "C:/Users/Public/ComfyUI-master/input/23.gif"
]

# Workflow scripts
workflows = [
    "workflow_05_multi_scale.py",
    "workflow_06_edge_preserving.py",
    "workflow_07_histogram_matching.py",
    "workflow_08_local_contrast.py",
    "workflow_09_gradient_enhancement.py",
    "workflow_10_soft_sharpening.py"
]


def run_workflow(workflow_script, input_file, output_dir):
    """Run a single workflow on a single input file"""
    try:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Warning: Input file does not exist: {input_file}")
            return False

        print(f"Running {workflow_script} on {input_path.name}")

        cmd = [
            sys.executable, workflow_script,
            "--input", str(input_path),
            "--output", output_dir
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")

        if result.returncode == 0:
            print(f"✓ Success: {workflow_script} -> {input_path.name}")
            return True
        else:
            print(f"✗ Error: {workflow_script} -> {input_path.name}")
            print(f"  Error output: {result.stderr}")
            return False

    except Exception as e:
        print(f"✗ Exception: {workflow_script} -> {input_file}: {e}")
        return False


def main():
    """Main execution function"""
    print("🧪 Testing workflows 5-10 with sample files")
    print(f"📁 Processing {len(sample_files)} sample files")
    print(f"🔧 Using {len(workflows)} workflows")

    output_dir = "C:/Users/Public/ComfyUI-master/output"

    # Statistics
    total_tasks = len(sample_files) * len(workflows)
    completed_tasks = 0
    failed_tasks = 0

    start_time = time.time()

    # Process each workflow
    for workflow_idx, workflow in enumerate(workflows, 1):
        print(f"\n{'='*50}")
        print(f"🔄 WORKFLOW {workflow_idx}/6: {workflow}")
        print(f"{'='*50}")

        workflow_start = time.time()
        workflow_success = 0
        workflow_failed = 0

        # Process each input file
        for file_idx, input_file in enumerate(sample_files, 1):
            print(f"\n[{file_idx}/{len(sample_files)}] ", end="")

            success = run_workflow(workflow, input_file, output_dir)

            if success:
                completed_tasks += 1
                workflow_success += 1
            else:
                failed_tasks += 1
                workflow_failed += 1

        workflow_time = time.time() - workflow_start
        print(f"\n📊 Workflow {workflow} completed:")
        print(f"   ✓ Success: {workflow_success}/{len(sample_files)}")
        print(f"   ✗ Failed: {workflow_failed}/{len(sample_files)}")
        print(f"   ⏱️ Time: {workflow_time:.1f}s")

    # Final statistics
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"🎉 SAMPLE TEST COMPLETE")
    print(f"{'='*50}")
    print(f"📈 Total tasks: {total_tasks}")
    print(f"✅ Completed: {completed_tasks}")
    print(f"❌ Failed: {failed_tasks}")
    print(f"📊 Success rate: {(completed_tasks/total_tasks)*100:.1f}%")
    print(f"⏱️ Total time: {total_time:.1f}s")
    print(f"📁 Output directory: {output_dir}")


if __name__ == "__main__":
    main()
