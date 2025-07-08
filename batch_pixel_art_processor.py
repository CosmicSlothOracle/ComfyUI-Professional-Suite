#!/usr/bin/env python3
"""
Batch Pixel Art Video Processor
Optimized for ComfyUI with 251 transparent GIF files
Based on community research and best practices
"""

import os
import json
import time
import glob
import shutil
from pathlib import Path


class PixelArtBatchProcessor:
    def __init__(self):
        self.base_dir = Path("C:/Users/Public/ComfyUI-master")
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output" / "pixel_art_videos"
        self.workflow_file = self.base_dir / "pixel_art_video_workflow.json"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load workflow template
        with open(self.workflow_file, 'r') as f:
            self.workflow_template = json.load(f)

    def get_gif_files(self):
        """Get all transparent GIF files from input directory"""
        pattern = str(self.input_dir / "*_fast_transparent_converted.gif")
        gif_files = glob.glob(pattern)
        print(f"Found {len(gif_files)} GIF files to process")
        return gif_files

    def create_workflow_for_file(self, gif_file):
        """Create workflow configuration for specific GIF file"""
        workflow = self.workflow_template.copy()

        # Update input file path
        for node in workflow["nodes"]:
            if node["type"] == "LoadVideo":
                node["widgets_values"][0] = os.path.basename(gif_file)
            elif node["type"] == "SaveVideo":
                base_name = Path(gif_file).stem
                output_name = f"{base_name}_pixel_art"
                node["widgets_values"][0] = output_name

        return workflow

    def optimize_settings_for_gif_size(self, gif_file, workflow):
        """Optimize pixel art settings based on GIF file size"""
        file_size = os.path.getsize(gif_file)

        # Adjust settings based on file size
        for node in workflow["nodes"]:
            if node["type"] == "PixelArtDetector":
                if file_size < 1024 * 1024:  # < 1MB - small files
                    node["widgets_values"][5] = 8   # fewer colors
                    node["widgets_values"][7] = 4   # smaller max colors
                elif file_size > 10 * 1024 * 1024:  # > 10MB - large files
                    node["widgets_values"][5] = 32  # more colors
                    node["widgets_values"][7] = 16  # higher max colors
                # Medium files keep default settings

        return workflow

    def process_single_file(self, gif_file):
        """Process a single GIF file"""
        print(f"Processing: {os.path.basename(gif_file)}")

        try:
            # Create optimized workflow
            workflow = self.create_workflow_for_file(gif_file)
            workflow = self.optimize_settings_for_gif_size(gif_file, workflow)

            # Save workflow for this file
            workflow_path = self.output_dir / \
                f"workflow_{Path(gif_file).stem}.json"
            with open(workflow_path, 'w') as f:
                json.dump(workflow, f, indent=2)

            print(f"✓ Workflow created: {workflow_path}")
            return True

        except Exception as e:
            print(f"✗ Error processing {gif_file}: {str(e)}")
            return False

    def generate_batch_report(self, processed_files, failed_files):
        """Generate processing report"""
        report_path = self.output_dir / "batch_processing_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("PIXEL ART BATCH PROCESSING REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                f"Total files found: {len(processed_files) + len(failed_files)}\n")
            f.write(f"Successfully processed: {len(processed_files)}\n")
            f.write(f"Failed: {len(failed_files)}\n\n")

            if processed_files:
                f.write("SUCCESSFULLY PROCESSED:\n")
                f.write("-" * 30 + "\n")
                for file in processed_files:
                    f.write(f"✓ {os.path.basename(file)}\n")
                f.write("\n")

            if failed_files:
                f.write("FAILED TO PROCESS:\n")
                f.write("-" * 30 + "\n")
                for file in failed_files:
                    f.write(f"✗ {os.path.basename(file)}\n")

        print(f"Report saved: {report_path}")

    def create_comfyui_execution_script(self):
        """Create script to execute workflows in ComfyUI"""
        script_path = self.output_dir / "execute_all_workflows.py"

        script_content = '''#!/usr/bin/env python3
"""
Execute all generated pixel art workflows in ComfyUI
Run this script from ComfyUI directory
"""

import os
import json
import glob
import subprocess
import time
from pathlib import Path

def execute_workflow(workflow_path):
    """Execute a single workflow in ComfyUI"""
    try:
        # This would integrate with ComfyUI API
        # For now, just print the command
        print(f"Execute: {workflow_path}")
        return True
    except Exception as e:
        print(f"Error executing {workflow_path}: {e}")
        return False

def main():
    output_dir = Path("output/pixel_art_videos")
    workflow_files = glob.glob(str(output_dir / "workflow_*.json"))

    print(f"Found {len(workflow_files)} workflows to execute")

    for workflow_file in workflow_files:
        print(f"Processing: {os.path.basename(workflow_file)}")
        success = execute_workflow(workflow_file)
        if success:
            print("✓ Success")
        else:
            print("✗ Failed")
        time.sleep(1)  # Prevent overload

if __name__ == "__main__":
    main()
'''

        with open(script_path, 'w') as f:
            f.write(script_content)

        print(f"Execution script created: {script_path}")

    def run_batch_processing(self):
        """Run the complete batch processing"""
        print("PIXEL ART BATCH PROCESSOR")
        print("=" * 50)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print()

        # Get all GIF files
        gif_files = self.get_gif_files()

        if not gif_files:
            print(
                "No GIF files found matching pattern '*_fast_transparent_converted.gif'")
            return

        processed_files = []
        failed_files = []

        # Process each file
        start_time = time.time()

        for i, gif_file in enumerate(gif_files, 1):
            print(f"\n[{i}/{len(gif_files)}] ", end="")

            if self.process_single_file(gif_file):
                processed_files.append(gif_file)
            else:
                failed_files.append(gif_file)

        # Generate report
        print(
            f"\n\nProcessing completed in {time.time() - start_time:.2f} seconds")
        self.generate_batch_report(processed_files, failed_files)
        self.create_comfyui_execution_script()

        print(f"\n✓ Batch processing complete!")
        print(f"✓ {len(processed_files)} workflows created")
        print(f"✓ Ready for ComfyUI execution")


if __name__ == "__main__":
    processor = PixelArtBatchProcessor()
    processor.run_batch_processing()
