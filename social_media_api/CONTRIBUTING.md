# Contributing to Social Media Video Generation API

Thank you for considering contributing to this project! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:
- Clear description of the bug
- Steps to reproduce
- Expected behavior
- Screenshots if applicable
- Environment details (OS, Python version, etc.)

### Suggesting Enhancements

For feature requests, please create an issue with:
- Clear description of the feature
- Rationale for the feature
- Any implementation ideas you have

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests to ensure they pass
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature`)
7. Open a Pull Request

## Development Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/social-media-video-generator.git
cd social-media-video-generator
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up configuration
```bash
cp social_media_api/config/google_credentials.json.example social_media_api/config/google_credentials.json
# Edit the credentials file with your API keys
```

## Project Structure

- `api_server/` - FastAPI server implementation
- `app/` - Application code and UI
- `comfy_integration/` - ComfyUI integration
- `config/` - Configuration files
- `examples/` - Example scripts and outputs
- `docs/` - Documentation

## Coding Standards

- Follow PEP 8 style guide
- Write docstrings for all functions, classes, and modules
- Include type hints
- Write tests for new functionality

## Testing

Run tests with:
```bash
pytest
```

## Documentation

Update documentation for any changes you make:
- Update README.md if necessary
- Add docstrings to new code
- Update architecture diagrams if needed

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.