# 🚀 Ultimate Sprite Processing Pipeline

State-of-the-art AI-powered sprite processing with perfect background removal, pose analysis, motion detection, and intelligent upscaling.

## ✨ Features

### 🎭 Perfect Background Removal
- **BiRefNet-HR**: Best quality for complex hair and edges
- **BRIA RMBG 2.0**: Enterprise-grade background removal
- **InSPyReNet**: Academic state-of-the-art model
- **U2Net**: Reliable fallback option
- **Ensemble processing**: Automatically selects best result

### 🧠 AI-Powered Analysis
- **MediaPipe Pose Detection**: Advanced pose and body structure analysis
- **BLIP Image Captioning**: Automatic sprite description generation
- **DPT Depth Estimation**: 3D depth understanding for better processing
- **Pixel Density Analysis**: Detects inconsistent pixel clusters

### 🖌️ Intelligent Enhancement
- **Stable Diffusion Inpainting**: AI-powered pixel cluster filling
- **Motion Pattern Analysis**: Analyzes sprite movement for intelligent GIF timing
- **Quality Assessment**: Comprehensive quality scoring system

### 📈 State-of-the-Art Upscaling
- **SUPIR**: Ultimate quality 2x upscaling
- **Real-ESRGAN**: High-performance neural upscaling
- **Fallback Options**: OpenCV cubic interpolation with sharpening

### 📊 Four Output Formats
1. **Single Frames (1x)**: Perfect transparency, original resolution
2. **Single Frames (2x)**: Upscaled frames with perfect transparency
3. **Animated GIFs (1x)**: Intelligent timing based on motion analysis
4. **Animated GIFs (2x)**: Upscaled animations with intelligent timing

## 🔧 Installation

### Prerequisites
- **Python 3.8+**
- **8GB+ RAM** (16GB recommended)
- **CUDA GPU** with 4GB+ VRAM (recommended, CPU fallback available)
- **Windows 10+** (Linux/Mac compatible with minor modifications)

### Quick Installation
1. **Download the pipeline**:
   ```bash
   git clone https://github.com/your-repo/ultimate-sprite-pipeline.git
   cd ultimate-sprite-pipeline
   ```

2. **Run the automated installer**:
   ```bash
   python install_ultimate_pipeline.py
   ```

3. **Follow the installation prompts** - the installer will:
   - Install all Python dependencies
   - Download essential AI models
   - Set up ComfyUI integration
   - Create directory structure
   - Verify installation

### Manual Installation
If the automated installer fails:

1. **Install PyTorch with CUDA support**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements_ultimate.txt
   ```

3. **Create directories**:
   ```bash
   mkdir -p input output models workflows
   ```

## 🎮 Usage

### Quick Start
1. **Place sprite images** in the `input/` directory
2. **Run the pipeline**:
   ```bash
   python ultimate_sprite_pipeline.py
   ```
3. **Choose quality preset** when prompted
4. **Check results** in `output/ultimate_sprites/`

### Command Line Options
```bash
# Professional quality (recommended)
python ultimate_sprite_pipeline.py --quality professional

# Ultimate quality (best results, slower)
python ultimate_sprite_pipeline.py --quality ultimate

# Specific input directory
python ultimate_sprite_pipeline.py --input my_sprites/

# Fast processing
python ultimate_sprite_pipeline.py --quality fast
```

### Windows Batch Script
For convenience, use the included batch script:
```cmd
run_ultimate_pipeline.bat
```

## ⚙️ Configuration

### Quality Presets

| Preset | Processing Time | Quality | Description |
|--------|----------------|---------|-------------|
| **Fast** | ~30s per sprite | Good | Basic processing, CPU-friendly |
| **Balanced** | ~60s per sprite | High | Good balance of speed and quality |
| **Professional** | ~120s per sprite | Excellent | Recommended for most users |
| **Ultimate** | ~300s per sprite | Maximum | Best possible quality, requires GPU |

### Advanced Configuration
Edit `config_ultimate.json` for fine-tuning:

```json
{
  "pipeline": {
    "quality": "professional",
    "device": "cuda",
    "batch_size": 4,
    "enable_ai_analysis": true,
    "enable_inpainting": true,
    "enable_motion_analysis": true
  },
  "background_removal": {
    "models": ["birefnet", "bria", "inspyrenet", "u2net"],
    "ensemble_mode": true,
    "confidence_threshold": 0.7
  },
  "upscaling": {
    "preferred_model": "supir",
    "fallback_model": "realesrgan",
    "scale_factor": 2
  },
  "output": {
    "formats": ["frames_1x", "frames_2x", "gif_1x", "gif_2x"],
    "transparency_mode": "zero_background",
    "gif_optimization": true
  }
}
```

## 📁 Directory Structure

```
ultimate-sprite-pipeline/
├── input/                          # Place your sprite images here
├── output/
│   └── ultimate_sprites/
│       ├── sprite_name/
│       │   ├── frames_1x/          # Individual frames (1x)
│       │   ├── frames_2x/          # Individual frames (2x)
│       │   ├── sprite_name_animated.gif     # Animation (1x)
│       │   └── sprite_name_animated_2x.gif  # Animation (2x)
│       └── processing_report.json  # Detailed processing report
├── models/                         # AI models and weights
├── workflows/                      # ComfyUI workflow files
├── logs/                          # Processing logs
└── temp/                          # Temporary processing files
```

## 🔬 ComfyUI Integration

### Workflow File
The pipeline includes a complete ComfyUI workflow: `comfyui_workflow_ultimate_sprites.json`

### Node Requirements
- **ComfyUI-BRIA_AI-RMBG**: BRIA background removal
- **ComfyUI-BiRefNet**: BiRefNet-HR implementation
- **ComfyUI-SUPIR**: SUPIR upscaling nodes
- **ComfyUI_essentials**: Essential image processing nodes
- **ComfyUI-MediaPipe**: Pose detection nodes
- **ComfyUI-BLIP**: Image captioning nodes

### Manual ComfyUI Setup
1. Install ComfyUI custom nodes via ComfyUI Manager
2. Load the workflow file in ComfyUI
3. Place sprite images in ComfyUI input directory
4. Run the workflow

## 📊 Output Quality

### Transparency Guarantee
- **Zero Background Opacity**: All outputs have perfectly transparent backgrounds
- **Edge Preservation**: Advanced algorithms preserve fine details and smooth edges
- **Alpha Channel Optimization**: Clean alpha channels without artifacts

### Quality Metrics
The pipeline generates detailed quality reports including:
- Background removal confidence scores
- AI analysis quality ratings
- Motion detection accuracy
- Frame extraction statistics
- Processing time benchmarks

## 🛠️ Troubleshooting

### Common Issues

#### GPU Out of Memory
**Solution**: Reduce quality preset or enable CPU fallback
```bash
python ultimate_sprite_pipeline.py --quality fast --device cpu
```

#### Models Not Downloading
**Solution**: Manual model download
1. Check internet connection
2. Clear model cache: `rm -rf models/cache/`
3. Re-run installer: `python install_ultimate_pipeline.py`

#### Poor Background Removal
**Solutions**:
- Ensure high contrast between sprite and background
- Try different quality presets
- Check input image quality (minimum 256x256 recommended)

#### ComfyUI Integration Issues
**Solutions**:
1. Update ComfyUI to latest version
2. Install missing custom nodes via ComfyUI Manager
3. Check workflow file compatibility

### Performance Optimization

#### For Better Speed
- Use `fast` or `balanced` quality presets
- Enable GPU acceleration
- Reduce batch size if running out of memory
- Close other GPU-intensive applications

#### For Better Quality
- Use `professional` or `ultimate` quality presets
- Ensure high-quality input images
- Enable all AI analysis features
- Use ensemble background removal mode

### Log Analysis
Check processing logs for detailed diagnostics:
```bash
# View latest log
cat logs/processing_$(date +%Y%m%d).log

# Check installation log
cat logs/installation.log
```

## 🔧 System Requirements

### Minimum Requirements
- **CPU**: Intel i5 or AMD Ryzen 5 (4+ cores)
- **RAM**: 8GB
- **Storage**: 10GB free space
- **GPU**: Optional, improves speed significantly

### Recommended Requirements
- **CPU**: Intel i7 or AMD Ryzen 7 (8+ cores)
- **RAM**: 16GB+
- **Storage**: 20GB+ free space (SSD recommended)
- **GPU**: NVIDIA RTX 3060 or better (6GB+ VRAM)

### Ultimate Quality Requirements
- **CPU**: Intel i9 or AMD Ryzen 9
- **RAM**: 32GB+
- **Storage**: 50GB+ free space (NVMe SSD)
- **GPU**: NVIDIA RTX 4080 or better (12GB+ VRAM)

## 📝 Advanced Usage

### Batch Processing
Process multiple sprites automatically:
```python
from ultimate_sprite_pipeline import UltimateSpriteProcessor

processor = UltimateSpriteProcessor(config)
results = processor.process_directory("input/sprites/")
```

### Custom Quality Settings
```python
config = ProcessingConfig(
    quality_preset=QualityPreset.PROFESSIONAL,
    enable_ai_analysis=True,
    enable_inpainting=True,
    upscale_factor=4,  # 4x upscaling
    custom_models={
        "background_removal": "birefnet",
        "upscaling": "supir",
        "pose_detection": "mediapipe"
    }
)
```

### Integration with Other Tools
The pipeline outputs standard PNG and GIF files compatible with:
- **Game Engines**: Unity, Unreal Engine, Godot
- **Animation Software**: Adobe After Effects, Blender
- **Image Editors**: Photoshop, GIMP, Aseprite

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- Additional background removal models
- New upscaling algorithms
- Enhanced motion analysis
- Mobile device optimization
- Web interface development

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BiRefNet**: Advanced background removal research
- **BRIA AI**: Commercial-grade background removal
- **SUPIR**: State-of-the-art upscaling technology
- **ComfyUI**: Excellent node-based AI workflow platform
- **Hugging Face**: Extensive AI model ecosystem

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)

---

## 🎯 Quick Reference

### Command Summary
```bash
# Install
python install_ultimate_pipeline.py

# Process sprites (recommended)
python ultimate_sprite_pipeline.py --quality professional

# Fast processing
python ultimate_sprite_pipeline.py --quality fast

# Ultimate quality
python ultimate_sprite_pipeline.py --quality ultimate

# Custom input directory
python ultimate_sprite_pipeline.py --input my_sprites/
```

### Output Locations
- **Frames 1x**: `output/ultimate_sprites/sprite_name/frames_1x/`
- **Frames 2x**: `output/ultimate_sprites/sprite_name/frames_2x/`
- **GIF 1x**: `output/ultimate_sprites/sprite_name/sprite_name_animated.gif`
- **GIF 2x**: `output/ultimate_sprites/sprite_name/sprite_name_animated_2x.gif`

### Quality vs Speed
- **Fast**: 30s, CPU-friendly, good quality
- **Balanced**: 60s, balanced approach
- **Professional**: 120s, excellent quality ⭐ **Recommended**
- **Ultimate**: 300s, maximum quality, requires GPU

---

*Ultimate Sprite Processing Pipeline - Making sprite processing effortless with AI* 🚀