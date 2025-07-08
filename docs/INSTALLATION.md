# Installation Guide

## System Requirements

- Python 3.10 or higher
- Git installed
- 8GB RAM minimum (16GB recommended)
- NVIDIA GPU recommended (for optimal performance)

## Step-by-Step Installation

### 1. Basic Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Windows users
install_extensions.bat

# Linux/Mac users
python install_extensions.py
```

### 2. Verify Installation

After installation, you should see:
- New nodes in your ComfyUI interface
- ComfyUI-Manager in the node menu
- Additional workflow options

### 3. Common Issues

#### Windows Users
- If Python is not recognized, make sure it's added to your PATH
- Run as Administrator if you encounter permission issues
- Use Command Prompt rather than PowerShell for best compatibility

#### Linux Users
- Make sure you have python3-venv installed
- Check file permissions if installation fails
- Use `sudo` if required for system-wide installation

#### Mac Users
- Install Python from python.org rather than Homebrew
- Use Terminal for installation
- Make sure Xcode command line tools are installed

### 4. Updating

- Use ComfyUI-Manager to update extensions
- Or run the installation script again
- Check GitHub releases for major updates

### 5. Next Steps

- Read the [Usage Guide](USAGE.md)
- Try example workflows
- Join the community

## Need Help?

- Open an issue on GitHub
- Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
- Ask in community discussions