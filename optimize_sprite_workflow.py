#!/usr/bin/env python3
"""
Sprite Workflow Optimizer
========================
Runs the sprite processing workflow 5 times with different parameters
to find the optimal configuration for frame detection and background removal.
"""

import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import cv2


class ComfyUIServer:
    def __init__(self, host="127.0.0.1", port=8188):
        self.base_url = f"http://{host}:{port}"

    def check_connection(self):
        """Check if ComfyUI server is running"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def wait_for_server(self, timeout=60):
        """Wait for ComfyUI server to become available"""
        print("Waiting for ComfyUI server...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.check_connection():
                print("✅ Connected to ComfyUI server")
                return True
            time.sleep(2)
            print(".", end="", flush=True)
        print("\n❌ Could not connect to ComfyUI server")
        return False

    def execute_workflow(self, workflow):
        """Execute a workflow and wait for completion"""
        try:
            # Queue the workflow
            response = requests.post(
                f"{self.base_url}/prompt",
                json=workflow,
                timeout=30
            )
            if response.status_code != 200:
                raise Exception(f"Failed to queue workflow: {response.text}")

            prompt_id = response.json()["prompt_id"]

            # Wait for completion
            while True:
                status = requests.get(
                    f"{self.base_url}/history/{prompt_id}",
                    timeout=30
                ).json()

                if status.get("completed", False):
                    return status
                elif status.get("error"):
                    raise Exception(f"Workflow error: {status['error']}")

                time.sleep(1)

        except requests.exceptions.RequestException as e:
            raise Exception(f"ComfyUI server error: {str(e)}")


class SpriteWorkflowOptimizer:
    def __init__(self):
        self.input_dir = Path("input/sprite_sheets")
        self.output_dir = Path("output/processed_sprites")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.server = ComfyUIServer()

        # Parameters to optimize
        self.params = {
            # Range for background removal threshold
            "background_tolerance": (15, 35),
            "min_frame_area": (600, 1000),    # Range for minimum frame area
            # Range for corner detection window
            "corner_detection_size": (20, 40),
            # Range for morphological operations
            "morphology_kernel_size": (2, 5),
        }

        # Quality metrics weights
        self.weights = {
            "frame_detection": 0.4,    # Weight for frame detection accuracy
            "background_removal": 0.3,  # Weight for background removal quality
            "transparency": 0.3,        # Weight for transparency consistency
        }

        self.results = []
        self.best_params = None
        self.best_score = 0

    def generate_params(self, iteration):
        """Generate parameters for each iteration using an optimization strategy"""
        if iteration == 0:
            # First iteration: use middle values
            return {
                "background_tolerance": np.mean(self.params["background_tolerance"]),
                "min_frame_area": np.mean(self.params["min_frame_area"]),
                "corner_detection_size": np.mean(self.params["corner_detection_size"]),
                "morphology_kernel_size": int(np.mean(self.params["morphology_kernel_size"]))
            }
        else:
            # Subsequent iterations: explore parameter space based on previous results
            if self.best_params is None:
                # If no best params yet, use random values within ranges
                return {
                    "background_tolerance": np.random.uniform(*self.params["background_tolerance"]),
                    "min_frame_area": np.random.uniform(*self.params["min_frame_area"]),
                    "corner_detection_size": np.random.uniform(*self.params["corner_detection_size"]),
                    "morphology_kernel_size": int(np.random.uniform(*self.params["morphology_kernel_size"]))
                }
            else:
                # Adjust parameters based on best results so far
                noise = 0.2  # 20% random adjustment
                return {
                    k: max(min(v * (1 + np.random.uniform(-noise, noise)),
                               self.params[k][1]), self.params[k][0])
                    for k, v in self.best_params.items()
                }

    def evaluate_frame_detection(self, output_path):
        """Evaluate frame detection quality"""
        try:
            frames = [f for f in output_path.glob("frame_*.png")]
            if not frames:
                return 0

            # Check frame consistency
            sizes = []
            for frame in frames:
                img = Image.open(frame)
                sizes.append(img.size)

            # All frames should have same size
            if len(set(sizes)) > 1:
                return 0.5

            # Score based on number of frames (expecting 8-12 frames for a typical animation)
            frame_count = len(frames)
            if 8 <= frame_count <= 12:
                return 1.0
            elif 6 <= frame_count <= 14:
                return 0.8
            else:
                return 0.6

        except Exception as e:
            print(f"Frame detection evaluation error: {e}")
            return 0

    def evaluate_background_removal(self, output_path):
        """Evaluate background removal quality"""
        try:
            frames = [f for f in output_path.glob("frame_*.png")]
            if not frames:
                return 0

            total_score = 0
            for frame in frames:
                img = cv2.imread(str(frame), cv2.IMREAD_UNCHANGED)
                if img is None or img.shape[-1] != 4:  # Should be RGBA
                    continue

                # Check alpha channel
                alpha = img[:, :, 3]

                # Calculate ratio of non-zero alpha pixels
                non_zero = np.count_nonzero(alpha)
                total = alpha.size
                ratio = non_zero / total

                # Score based on reasonable sprite size (expecting 10-40% of frame to be sprite)
                if 0.1 <= ratio <= 0.4:
                    total_score += 1
                elif 0.05 <= ratio <= 0.5:
                    total_score += 0.7
                else:
                    total_score += 0.3

            return total_score / len(frames)

        except Exception as e:
            print(f"Background removal evaluation error: {e}")
            return 0

    def evaluate_transparency(self, output_path):
        """Evaluate transparency consistency"""
        try:
            frames = [f for f in output_path.glob("frame_*.png")]
            if not frames:
                return 0

            alpha_stats = []
            for frame in frames:
                img = cv2.imread(str(frame), cv2.IMREAD_UNCHANGED)
                if img is None or img.shape[-1] != 4:
                    continue

                alpha = img[:, :, 3]
                # Get distribution of alpha values
                hist = cv2.calcHist([alpha], [0], None, [256], [0, 256])
                alpha_stats.append(hist)

            # Compare alpha distributions between frames
            total_score = 0
            for i in range(len(alpha_stats)):
                for j in range(i+1, len(alpha_stats)):
                    correlation = cv2.compareHist(
                        alpha_stats[i], alpha_stats[j], cv2.HISTCMP_CORREL)
                    # Only count positive correlations
                    total_score += max(0, correlation)

            max_comparisons = (len(frames) * (len(frames) - 1)) / 2
            return total_score / max_comparisons if max_comparisons > 0 else 0

        except Exception as e:
            print(f"Transparency evaluation error: {e}")
            return 0

    def calculate_total_score(self, frame_score, bg_score, trans_score):
        """Calculate weighted total score"""
        return (frame_score * self.weights["frame_detection"] +
                bg_score * self.weights["background_removal"] +
                trans_score * self.weights["transparency"])

    def process_sprite(self, sprite_path, params, iteration):
        """Process a single sprite with given parameters"""
        try:
            # Create output directory for this iteration
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / \
                f"iteration_{iteration}_{timestamp}"
            output_path.mkdir(parents=True, exist_ok=True)

            # Create workflow with current parameters
            workflow = {
                "1": {
                    "inputs": {
                        "image": str(sprite_path.name),
                        "upload": "image"
                    },
                    "class_type": "LoadImage"
                },
                "2": {
                    "inputs": {
                        "image": ["1", 0],
                        "background_tolerance": params["background_tolerance"],
                        "min_frame_area": params["min_frame_area"],
                        "corner_detection_size": params["corner_detection_size"],
                        "morphology_kernel_size": params["morphology_kernel_size"],
                        "output_gif": True,
                        "gif_duration": 500
                    },
                    "class_type": "IntelligentSpritesheetProcessor"
                },
                "3": {
                    "inputs": {
                        "images": ["2", 0],
                        "filename_prefix": str(output_path / "frame_")
                    },
                    "class_type": "SaveImage"
                }
            }

            # Save workflow
            workflow_path = output_path / "workflow.json"
            with open(workflow_path, 'w') as f:
                json.dump(workflow, f, indent=2)

            # Execute workflow
            print("Executing workflow...")
            result = self.server.execute_workflow(workflow)
            print("Workflow completed")

            # Evaluate results
            frame_score = self.evaluate_frame_detection(output_path)
            bg_score = self.evaluate_background_removal(output_path)
            trans_score = self.evaluate_transparency(output_path)
            total_score = self.calculate_total_score(
                frame_score, bg_score, trans_score)

            result = {
                "iteration": iteration,
                "params": params,
                "scores": {
                    "frame_detection": frame_score,
                    "background_removal": bg_score,
                    "transparency": trans_score,
                    "total": total_score
                },
                "output_path": str(output_path)
            }

            self.results.append(result)

            # Update best parameters if this is the best score
            if total_score > self.best_score:
                self.best_score = total_score
                self.best_params = params.copy()

            return result

        except Exception as e:
            print(f"Error processing sprite: {e}")
            return None

    def optimize(self, sprite_name):
        """Run optimization process over 5 iterations"""
        sprite_path = self.input_dir / sprite_name
        if not sprite_path.exists():
            raise FileNotFoundError(f"Sprite file not found: {sprite_path}")

        # Check ComfyUI server
        if not self.server.wait_for_server():
            print("Please start ComfyUI server and try again")
            return

        print(f"\n🎮 Starting optimization for: {sprite_name}")
        print("=" * 50)

        for i in range(5):
            print(f"\n📝 Iteration {i+1}/5")
            params = self.generate_params(i)

            # Round parameters for display
            display_params = {k: round(v, 2) if isinstance(v, float) else v
                              for k, v in params.items()}
            print(f"Parameters: {json.dumps(display_params, indent=2)}")

            result = self.process_sprite(sprite_path, params, i)
            if result:
                scores = result["scores"]
                print("\nScores:")
                print(f"  Frame Detection: {scores['frame_detection']:.2f}")
                print(
                    f"  Background Removal: {scores['background_removal']:.2f}")
                print(f"  Transparency: {scores['transparency']:.2f}")
                print(f"  Total Score: {scores['total']:.2f}")

                if scores['total'] > 0.8:
                    print("\n✨ Production-ready score achieved!")
                    break

            time.sleep(1)  # Small delay between iterations

        # Generate final report
        self.generate_report()

    def generate_report(self):
        """Generate optimization report"""
        if not self.results:
            return

        report_path = self.output_dir / "optimization_report.json"

        report = {
            "total_iterations": len(self.results),
            "best_score": self.best_score,
            "best_parameters": self.best_params,
            "all_results": self.results,
            "timestamp": datetime.now().isoformat()
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Optimization Report")
        print("=" * 50)
        print(f"Best Score: {self.best_score:.2f}")
        print("\nBest Parameters:")
        for k, v in self.best_params.items():
            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"\nDetailed report saved to: {report_path}")


def main():
    optimizer = SpriteWorkflowOptimizer()
    sprite_name = "Tanzende Anime-Figur im Sprite-Stil.png"

    try:
        optimizer.optimize(sprite_name)
    except Exception as e:
        print(f"Error during optimization: {e}")


if __name__ == "__main__":
    main()
