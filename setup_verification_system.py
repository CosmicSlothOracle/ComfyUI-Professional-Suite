#!/usr/bin/env python3
"""
Modern Pokemon Card Authenticity Verification System Setup
Installs all dependencies and creates proper directory structure
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True,
                                capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False


def create_directory_structure():
    """Create the required directory structure"""
    print("\n📁 Creating directory structure...")

    directories = [
        "reference_cards",
        "reference_cards/base_set",
        "reference_cards/jungle",
        "reference_cards/fossil",
        "reference_cards/team_rocket",
        "reference_cards/gym_heroes",
        "reference_cards/gym_challenge",
        "reference_cards/neo_genesis",
        "reference_cards/custom",
        "input_cards",
        "input_cards/to_verify",
        "input_cards/processed",
        "output",
        "output/reports",
        "output/visualizations",
        "output/confidence_maps",
        "models",
        "models/vision_transformer",
        "models/lora_adapters",
        "logs",
        "config"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")


def install_dependencies():
    """Install all required dependencies"""
    print("\n📦 Installing dependencies...")

    # Install main requirements
    if not run_command(
        f"{sys.executable} -m pip install -r requirements_modern_verification.txt",
        "Installing main requirements"
    ):
        return False

    # Install additional models and libraries
    commands = [
        ("python -m pip install --upgrade pip", "Upgrading pip"),
        ("python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121",
         "Installing PyTorch with CUDA support"),
    ]

    for command, description in commands:
        if not run_command(command, description):
            print(
                f"⚠️  Warning: {description} failed - continuing with CPU version")

    return True


def download_pretrained_models():
    """Download and cache required pretrained models"""
    print("\n🤖 Downloading pretrained models...")

    download_script = '''
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import DinForImageClassification
import timm

print("Downloading Vision Transformer models...")

# Download ViT-Large
processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
processor.save_pretrained('models/vision_transformer/vit-large')
model.save_pretrained('models/vision_transformer/vit-large')

# Download DINOv2
model_dino = DinForImageClassification.from_pretrained('facebook/dinov2-large')
model_dino.save_pretrained('models/vision_transformer/dinov2-large')

# Download ResNet from timm
model_resnet = timm.create_model('resnet50', pretrained=True)
torch.save(model_resnet.state_dict(), 'models/vision_transformer/resnet50_pretrained.pth')

print("✅ All models downloaded successfully!")
'''

    # Write and execute the download script
    with open('download_models.py', 'w') as f:
        f.write(download_script)

    if run_command(f"{sys.executable} download_models.py", "Downloading pretrained models"):
        os.remove('download_models.py')
        return True
    return False


def create_example_files():
    """Create example configuration and usage files"""
    print("\n📋 Creating example files...")

    # Create config file
    config_content = '''# Pokemon Card Authenticity Verification Configuration

[verification]
# Model settings
primary_model = "vit-large"
secondary_model = "dinov2-large"
ensemble_models = ["vit-large", "dinov2-large", "resnet50"]

# Processing settings
image_size = 512
batch_size = 1  # Single-card processing
confidence_threshold = 0.85
precision_tolerance = 0.001

# Output settings
save_visualizations = true
save_confidence_maps = true
generate_detailed_reports = true

[layout_analysis]
# Layout component weights
border_weight = 0.25
text_layout_weight = 0.20
artwork_weight = 0.30
color_accuracy_weight = 0.15
typography_weight = 0.10

# Tolerance settings (in pixels for 512x512 images)
border_tolerance = 2
text_position_tolerance = 5
color_delta_e_threshold = 2.3

[file_naming]
# Reference card naming convention
reference_format = "{set_name}_{card_number}_{card_name}_reference.{ext}"
# Input card naming convention
input_format = "{set_name}_{card_number}_{card_name}_generated.{ext}"
'''

    with open('config/verification_config.ini', 'w') as f:
        f.write(config_content)

    # Create README with file naming instructions
    readme_content = '''# Pokemon Card Authenticity Verification System

## File Naming Convention

### Reference Cards (Original/Official Cards)
Place in `reference_cards/{set_name}/` directory:

**Format:** `{set_name}_{card_number}_{card_name}_reference.{ext}`

**Examples:**
- `base_set_001_bulbasaur_reference.png`
- `jungle_007_pinsir_reference.jpg`
- `fossil_015_aerodactyl_reference.png`
- `team_rocket_083_dark_magneton_reference.png`

### Generated Cards (Cards to Verify)
Place in `input_cards/to_verify/` directory:

**Format:** `{set_name}_{card_number}_{card_name}_generated.{ext}`

**Examples:**
- `base_set_001_bulbasaur_generated.png`
- `jungle_007_pinsir_generated.jpg`
- `custom_001_charizard_generated.png`

### Supported Image Formats
- `.png` (recommended for highest quality)
- `.jpg` / `.jpeg`
- `.bmp`
- `.tiff`

### Directory Structure
```
pokemon_card_verification/
├── reference_cards/
│   ├── base_set/
│   │   ├── base_set_001_bulbasaur_reference.png
│   │   ├── base_set_004_charmander_reference.png
│   │   └── ...
│   ├── jungle/
│   │   ├── jungle_001_clefairy_reference.png
│   │   └── ...
│   ├── fossil/
│   └── custom/  # For custom or unknown sets
├── input_cards/
│   ├── to_verify/
│   │   ├── base_set_001_bulbasaur_generated.png
│   │   └── ...
│   └── processed/  # Automatically moved after verification
├── output/
│   ├── reports/
│   ├── visualizations/
│   └── confidence_maps/
└── models/
    └── vision_transformer/
```

## Quick Start

1. **Install dependencies:**
   ```bash
   python setup_verification_system.py
   ```

2. **Add your reference cards:**
   - Place official Pokemon cards in appropriate `reference_cards/{set}/` folders
   - Use the naming convention above

3. **Add cards to verify:**
   - Place AI-generated cards in `input_cards/to_verify/`
   - Ensure they match the reference card naming pattern

4. **Run verification:**
   ```bash
   python modern_card_verifier.py --input input_cards/to_verify/ --output output/
   ```

## File Matching

The system automatically matches generated cards with reference cards based on:
1. **Set name** (e.g., "base_set", "jungle")
2. **Card number** (e.g., "001", "007")
3. **Card name** (e.g., "bulbasaur", "pinsir")

Example match:
- Reference: `base_set_001_bulbasaur_reference.png`
- Generated: `base_set_001_bulbasaur_generated.png`
- ✅ **Match found** - verification will proceed

If no matching reference card is found, the system will:
1. Search for similar cards in the same set
2. Use the closest match with a confidence penalty
3. Generate a warning in the report
'''

    with open('README_FILE_NAMING.md', 'w') as f:
        f.write(readme_content)


def main():
    """Main setup function"""
    print("🚀 Setting up Modern Pokemon Card Authenticity Verification System")
    print("=" * 70)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)

    print(f"✅ Python version: {sys.version}")

    # Create directory structure
    create_directory_structure()

    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)

    # Download models
    if not download_pretrained_models():
        print("⚠️  Model download failed - you can download them manually later")

    # Create example files
    create_example_files()

    print("\n" + "=" * 70)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Read README_FILE_NAMING.md for file organization instructions")
    print(
        "2. Place your reference cards in reference_cards/{set_name}/ folders")
    print("3. Place generated cards to verify in input_cards/to_verify/")
    print("4. Run: python modern_card_verifier.py")
    print("\n💡 Example file names:")
    print("  Reference: reference_cards/base_set/base_set_001_bulbasaur_reference.png")
    print("  Generated: input_cards/to_verify/base_set_001_bulbasaur_generated.png")


if __name__ == "__main__":
    main()
