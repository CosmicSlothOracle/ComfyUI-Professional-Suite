#!/usr/bin/env python3
"""
🚀 ULTIMATE SPRITE PIPELINE INSTALLER
====================================
Automated installation and setup for the ultimate sprite processing pipeline.
Sets up dependencies, ComfyUI nodes, models, and directory structure.
"""

import os
import sys
import subprocess
import urllib.request
import json
import shutil
import platform
from pathlib import Path
from typing import List, Dict, Tuple
import zipfile
import tarfile


class UltimatePipelineInstaller:
    """Automated installer for Ultimate Sprite Processing Pipeline"""

    def __init__(self):
        self.system = platform.system()
        self.python_executable = sys.executable
        self.base_dir = Path.cwd()
        self.models_dir = self.base_dir / "models"
        self.comfyui_dir = self.base_dir / "ComfyUI"
        self.install_log = []

        print("🚀 ULTIMATE SPRITE PIPELINE INSTALLER")
        print("=" * 60)
        print(f"System: {self.system}")
        print(f"Python: {self.python_executable}")
        print(f"Base Directory: {self.base_dir}")
        print("=" * 60)

    def log(self, message: str, level: str = "INFO"):
        """Log installation progress"""
        formatted_msg = f"[{level}] {message}"
        print(formatted_msg)
        self.install_log.append(formatted_msg)

    def run_command(self, command: List[str], description: str) -> bool:
        """Run a system command with error handling"""
        try:
            self.log(f"Running: {description}")
            result = subprocess.run(
                command, check=True, capture_output=True, text=True)
            self.log(f"✅ {description} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"❌ {description} failed: {e.stderr}", "ERROR")
            return False
        except Exception as e:
            self.log(f"❌ {description} failed: {str(e)}", "ERROR")
            return False

    def download_file(self, url: str, destination: Path, description: str) -> bool:
        """Download a file with progress"""
        try:
            self.log(f"Downloading: {description}")
            destination.parent.mkdir(parents=True, exist_ok=True)

            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = (block_num * block_size / total_size) * 100
                    print(f"\r   Progress: {percent:.1f}%", end="", flush=True)

            urllib.request.urlretrieve(url, destination, progress_hook)
            print()  # New line after progress
            self.log(f"✅ Downloaded: {description}")
            return True
        except Exception as e:
            self.log(f"❌ Download failed for {description}: {str(e)}", "ERROR")
            return False

    def install_python_dependencies(self) -> bool:
        """Install Python dependencies"""
        self.log("🐍 Installing Python dependencies...")

        # First, upgrade pip
        if not self.run_command([
            self.python_executable, "-m", "pip", "install", "--upgrade", "pip"
        ], "Upgrading pip"):
            return False

        # Install PyTorch with CUDA support (if available)
        gpu_available = self.check_gpu_availability()
        if gpu_available:
            torch_command = [
                self.python_executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ]
        else:
            torch_command = [
                self.python_executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio"
            ]

        if not self.run_command(torch_command, "Installing PyTorch"):
            return False

        # Try to install requirements from file with Windows compatibility
        requirements_file = self.base_dir / "requirements_ultimate.txt"
        if requirements_file.exists():
            if not self.run_command([
                self.python_executable, "-m", "pip", "install", "-r", str(
                    requirements_file)
            ], "Installing requirements from file"):
                self.log(
                    "⚠️ Requirements file failed, installing packages individually...", "WARNING")
                return self._install_packages_individually()
        else:
            self.log(
                "⚠️ Requirements file not found, installing core packages...", "WARNING")
            return self._install_packages_individually()

        return True

    def _install_packages_individually(self) -> bool:
        """Install packages individually with Windows compatibility"""
        self.log("📦 Installing packages individually for Windows compatibility...")

        # Core packages (most likely to work on Windows)
        core_packages = [
            "numpy>=1.21.0",
            "opencv-python>=4.8.0",
            "Pillow>=9.0.0",
            "scipy>=1.7.0",
            "tqdm>=4.65.0",
            "rich>=13.0.0",
            "colorama>=0.4.6",
            "requests>=2.28.0",
            "websocket-client>=1.5.0"
        ]

        # AI packages (may fail on some systems)
        ai_packages = [
            "transformers>=4.30.0",
            "diffusers>=0.20.0",
            "accelerate>=0.20.0",
            "rembg>=2.0.50",
            "ultralytics>=8.0.0"  # YOLO alternative to MediaPipe
        ]

        # Optional packages (nice to have but not critical)
        optional_packages = [
            "pandas>=1.5.0",
            "matplotlib>=3.6.0",
            "seaborn>=0.11.0",
            "imageio>=2.25.0",
            "jupyter>=1.0.0",
            "psutil>=5.9.0"
        ]

        # Install core packages
        self.log("Installing core packages...")
        for package in core_packages:
            if not self.run_command([
                self.python_executable, "-m", "pip", "install", package
            ], f"Installing {package}"):
                self.log(
                    f"⚠️ Failed to install {package}, continuing...", "WARNING")

        # Install AI packages
        self.log("Installing AI packages...")
        for package in ai_packages:
            if not self.run_command([
                self.python_executable, "-m", "pip", "install", package
            ], f"Installing {package}"):
                self.log(
                    f"⚠️ Failed to install {package}, using fallback methods...", "WARNING")

        # Install optional packages
        self.log("Installing optional packages...")
        for package in optional_packages:
            if not self.run_command([
                self.python_executable, "-m", "pip", "install", package
            ], f"Installing optional {package}"):
                self.log(f"ℹ️ Optional package {package} skipped", "INFO")

        # Special handling for problematic Windows packages
        self.log("Checking for Windows-specific alternatives...")

        # Check if MediaPipe installed, if not, ensure YOLO is available
        try:
            import mediapipe
            self.log("   ✅ MediaPipe available")
        except ImportError:
            self.log("   ⚠️ MediaPipe not available, ensuring YOLO is installed...")
            self.run_command([
                self.python_executable, "-m", "pip", "install", "ultralytics"
            ], "Installing YOLO as MediaPipe alternative")

        # Check if scikit-image installed
        try:
            import skimage
            self.log("   ✅ Scikit-image available")
        except ImportError:
            self.log("   ⚠️ Scikit-image not available, using OpenCV alternatives")

        return True

    def check_gpu_availability(self) -> bool:
        """Check if NVIDIA GPU is available"""
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def setup_directory_structure(self) -> bool:
        """Create necessary directory structure"""
        self.log("📁 Setting up directory structure...")

        directories = [
            "input",
            "output/ultimate_sprites",
            "models/background_removal",
            "models/upscaling",
            "models/inpainting",
            "models/pose_estimation",
            "weights",
            "workflows",
            "temp",
            "logs"
        ]

        for directory in directories:
            dir_path = self.base_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.log(f"   ✓ Created: {directory}")

        return True

    def install_comfyui_integration(self) -> bool:
        """Install or update ComfyUI and required custom nodes"""
        self.log("🎨 Setting up ComfyUI integration...")

        # Check if ComfyUI exists
        if not self.comfyui_dir.exists():
            self.log("ComfyUI not found, attempting to clone...")
            if not self.run_command([
                "git", "clone", "https://github.com/comfyanonymous/ComfyUI.git"
            ], "Cloning ComfyUI"):
                self.log(
                    "⚠️ Failed to clone ComfyUI, continuing without it...", "WARNING")
                return True

        # Install ComfyUI dependencies
        comfyui_requirements = self.comfyui_dir / "requirements.txt"
        if comfyui_requirements.exists():
            if not self.run_command([
                self.python_executable, "-m", "pip", "install", "-r", str(
                    comfyui_requirements)
            ], "Installing ComfyUI requirements"):
                self.log("⚠️ Failed to install ComfyUI requirements", "WARNING")

        # Install essential custom nodes
        custom_nodes_dir = self.comfyui_dir / "custom_nodes"
        custom_nodes_dir.mkdir(exist_ok=True)

        essential_nodes = [
            {
                "name": "ComfyUI-BRIA_AI-RMBG",
                "url": "https://github.com/ZHO-ZHO-ZHO/ComfyUI-BRIA_AI-RMBG.git",
                "description": "BRIA AI background removal"
            },
            {
                "name": "ComfyUI_essentials",
                "url": "https://github.com/cubiq/ComfyUI_essentials.git",
                "description": "Essential nodes for image processing"
            },
            {
                "name": "ComfyUI-SUPIR",
                "url": "https://github.com/kijai/ComfyUI-SUPIR.git",
                "description": "SUPIR upscaling nodes"
            }
        ]

        for node in essential_nodes:
            node_path = custom_nodes_dir / node["name"]
            if not node_path.exists():
                if self.run_command([
                    "git", "clone", node["url"], str(node_path)
                ], f"Installing {node['description']}"):
                    # Install node-specific requirements if they exist
                    node_requirements = node_path / "requirements.txt"
                    if node_requirements.exists():
                        self.run_command([
                            self.python_executable, "-m", "pip", "install", "-r", str(
                                node_requirements)
                        ], f"Installing {node['name']} requirements")

        return True

    def download_essential_models(self) -> bool:
        """Download essential models for the pipeline"""
        self.log("📦 Downloading essential models...")

        # Background removal models (using REMBG's built-in download)
        try:
            import rembg
            self.log("Testing REMBG model downloads...")

            # This will automatically download u2net model on first use
            test_session = rembg.new_session('u2net')
            self.log("   ✓ U2Net model initialized")

            # Try BiRefNet if available
            try:
                birefnet_session = rembg.new_session('birefnet-general')
                self.log("   ✓ BiRefNet model initialized")
            except:
                self.log("   ⚠️ BiRefNet model not available", "WARNING")

        except ImportError:
            self.log("   ⚠️ REMBG not available for model download", "WARNING")

        # MediaPipe models (downloaded automatically)
        try:
            import mediapipe as mp
            # Initialize pose detection to trigger model download
            pose = mp.solutions.pose.Pose()
            self.log("   ✓ MediaPipe Pose model initialized")
            pose.close()
        except ImportError:
            self.log("   ⚠️ MediaPipe not available", "WARNING")

        # Create model configuration file
        model_config = {
            "background_removal": {
                "models": ["u2net", "birefnet-general", "isnet-general-use"],
                "default": "u2net"
            },
            "upscaling": {
                "models": ["realesrgan-x2plus", "opencv-cubic"],
                "default": "opencv-cubic"
            },
            "pose_estimation": {
                "models": ["mediapipe-pose"],
                "default": "mediapipe-pose"
            }
        }

        config_path = self.models_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)

        self.log(f"   ✓ Model configuration saved to {config_path}")
        return True

    def create_workflow_files(self) -> bool:
        """Create ComfyUI workflow files"""
        self.log("⚙️ Creating workflow files...")

        # Ultimate sprite processing workflow
        ultimate_workflow = {
            "workflow_name": "Ultimate Sprite Processing",
            "description": "State-of-the-art sprite processing with AI analysis",
            "version": "1.0.0",
            "nodes": {
                "1": {
                    "inputs": {"image": "INPUT_SPRITE.png", "upload": "image"},
                    "class_type": "LoadImage",
                    "_meta": {"title": "🎯 Load Sprite"}
                },
                "2": {
                    "inputs": {
                        "image": ["1", 0],
                        "model": "BiRefNet-HR",
                        "post_processing": True
                    },
                    "class_type": "BRIA_RMBG_BackgroundRemoval",
                    "_meta": {"title": "🎭 Perfect Background Removal"}
                },
                "3": {
                    "inputs": {
                        "image": ["2", 0],
                        "enable_pose_detection": True,
                        "enable_depth_analysis": True
                    },
                    "class_type": "AI_SpriteAnalyzer",
                    "_meta": {"title": "🧠 AI Analysis"}
                },
                "4": {
                    "inputs": {
                        "image": ["3", 0],
                        "mask": ["2", 1],
                        "intelligent_extraction": True
                    },
                    "class_type": "AdvancedFrameExtractor",
                    "_meta": {"title": "✂️ Frame Extraction"}
                },
                "5": {
                    "inputs": {
                        "frames": ["4", 0],
                        "ai_analysis": ["3", 1]
                    },
                    "class_type": "MotionAnalyzer",
                    "_meta": {"title": "🎬 Motion Analysis"}
                },
                "6": {
                    "inputs": {
                        "frames": ["4", 0],
                        "scale_factor": 2,
                        "model": "Real-ESRGAN"
                    },
                    "class_type": "StateOfTheArtUpscaler",
                    "_meta": {"title": "📈 2x Upscaling"}
                },
                "7": {
                    "inputs": {
                        "frames": ["4", 0],
                        "motion_analysis": ["5", 0],
                        "filename_prefix": "frame_1x_"
                    },
                    "class_type": "SaveFrames",
                    "_meta": {"title": "💾 Save Frames 1x"}
                },
                "8": {
                    "inputs": {
                        "frames": ["6", 0],
                        "motion_analysis": ["5", 0],
                        "filename_prefix": "frame_2x_"
                    },
                    "class_type": "SaveFrames",
                    "_meta": {"title": "💾 Save Frames 2x"}
                },
                "9": {
                    "inputs": {
                        "frames": ["4", 0],
                        "motion_analysis": ["5", 0],
                        "filename_prefix": "animated_1x"
                    },
                    "class_type": "SaveAnimatedGIF",
                    "_meta": {"title": "🎬 Save GIF 1x"}
                },
                "10": {
                    "inputs": {
                        "frames": ["6", 0],
                        "motion_analysis": ["5", 0],
                        "filename_prefix": "animated_2x"
                    },
                    "class_type": "SaveAnimatedGIF",
                    "_meta": {"title": "🎬 Save GIF 2x"}
                }
            }
        }

        workflow_path = self.base_dir / "workflows" / "ultimate_sprite_processing.json"
        with open(workflow_path, 'w') as f:
            json.dump(ultimate_workflow, f, indent=2)

        self.log(f"   ✓ Ultimate workflow saved to {workflow_path}")
        return True

    def create_example_files(self) -> bool:
        """Create example usage files"""
        self.log("📝 Creating example files...")

        # Example usage script
        example_script = '''#!/usr/bin/env python3
"""
Example usage of the Ultimate Sprite Processing Pipeline
"""

from ultimate_sprite_pipeline import process_sprite_files, create_processing_config

def main():
    print("🎮 Ultimate Sprite Processing Example")

    # Process with professional quality
    process_sprite_files(input_dir="input", quality_preset="professional")

    # Process with ultimate quality (slower but best results)
    # process_sprite_files(input_dir="input", quality_preset="ultimate")

if __name__ == "__main__":
    main()
'''

        example_path = self.base_dir / "example_usage.py"
        with open(example_path, 'w', encoding='utf-8') as f:
            f.write(example_script)

        # README file
        readme_content = '''# Ultimate Sprite Processing Pipeline

State-of-the-art AI-powered sprite processing with perfect background removal.

## Features
- BiRefNet-HR, BRIA RMBG 2.0, InSPyReNet background removal
- Hugging Face AI models for pose and depth analysis
- Stable Diffusion inpainting for pixel cluster filling
- SUPIR & Real-ESRGAN upscaling
- Motion analysis for intelligent GIF timing
- 4 output formats: frames 1x/2x, animated GIFs 1x/2x

## Quick Start
1. Place sprite images in the `input/` directory
2. Run: `python ultimate_sprite_pipeline.py`
3. Results will be in `output/ultimate_sprites/`

## Quality Presets
- `fast`: 30s per sprite
- `balanced`: 60s per sprite
- `professional`: 120s per sprite (recommended)
- `ultimate`: 300s per sprite (best quality)

## Requirements
- Python 3.8+
- CUDA GPU (recommended)
- 8GB+ RAM
- 4GB+ GPU memory (for ultimate quality)

## Installation
Run the installer: `python install_ultimate_pipeline.py`
'''

        readme_path = self.base_dir / "README_ULTIMATE.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        self.log("   ✓ Example files created")
        return True

    def run_installation_tests(self) -> bool:
        """Run tests to verify installation"""
        self.log("🧪 Running installation tests...")

        # Test Python imports
        test_imports = [
            "torch", "torchvision", "cv2", "numpy", "PIL",
            "transformers", "rembg", "mediapipe"
        ]

        for module in test_imports:
            try:
                __import__(module)
                self.log(f"   ✓ {module} import successful")
            except ImportError as e:
                self.log(f"   ❌ {module} import failed: {e}", "ERROR")
                return False

        # Test GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                self.log(
                    f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                self.log("   ⚠️ CUDA not available, will use CPU", "WARNING")
        except:
            self.log("   ❌ PyTorch test failed", "ERROR")
            return False

        # Test directory structure
        required_dirs = ["input", "output", "models", "workflows"]
        for directory in required_dirs:
            if (self.base_dir / directory).exists():
                self.log(f"   ✓ Directory exists: {directory}")
            else:
                self.log(f"   ❌ Missing directory: {directory}", "ERROR")
                return False

        return True

    def save_installation_log(self) -> bool:
        """Save installation log"""
        log_path = self.base_dir / "logs" / "installation.log"
        log_path.parent.mkdir(exist_ok=True)

        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("ULTIMATE SPRITE PIPELINE INSTALLATION LOG\n")
            f.write("=" * 50 + "\n\n")
            for entry in self.install_log:
                f.write(entry + "\n")

        self.log(f"   ✓ Installation log saved to {log_path}")
        return True

    def install(self) -> bool:
        """Run complete installation"""
        self.log("🚀 Starting Ultimate Sprite Pipeline Installation...")

        steps = [
            ("Setting up directories", self.setup_directory_structure),
            ("Installing Python dependencies", self.install_python_dependencies),
            ("Setting up ComfyUI integration", self.install_comfyui_integration),
            ("Downloading essential models", self.download_essential_models),
            ("Creating workflow files", self.create_workflow_files),
            ("Creating example files", self.create_example_files),
            ("Running installation tests", self.run_installation_tests),
            ("Saving installation log", self.save_installation_log)
        ]

        success = True
        for step_name, step_function in steps:
            self.log(f"\n📋 Step: {step_name}")
            if not step_function():
                self.log(f"❌ Step failed: {step_name}", "ERROR")
                success = False
                break
            else:
                self.log(f"✅ Step completed: {step_name}")

        if success:
            self.log("\n🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
            self.log("=" * 60)
            self.log("Next steps:")
            self.log("1. Place sprite images in the 'input/' directory")
            self.log("2. Run: python ultimate_sprite_pipeline.py")
            self.log("3. Check results in 'output/ultimate_sprites/'")
            self.log("=" * 60)
        else:
            self.log("\n❌ INSTALLATION FAILED!")
            self.log("Check the installation log for details.")

        return success


def main():
    """Main installation function"""
    installer = UltimatePipelineInstaller()

    # Ask for confirmation
    print("\nThis will install the Ultimate Sprite Processing Pipeline.")
    print("This includes downloading models and setting up ComfyUI integration.")
    print("Continue? (y/N): ", end="")

    if input().lower() in ['y', 'yes']:
        installer.install()
    else:
        print("Installation cancelled.")


if __name__ == "__main__":
    main()
