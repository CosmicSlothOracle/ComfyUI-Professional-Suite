#!/bin/bash

echo "Initializing Git repository for Social Media Video Generation API..."
echo

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Please install Git."
    exit 1
fi

# Initialize Git repository
git init

# Add .gitignore
git add .gitignore

# Add README and documentation files
git add README.md
git add CONTRIBUTING.md
git add LICENSE
git add docs/architecture.md

# Add configuration files
git add config/dashboard_config.json
git add config/google_credentials.json.example
git add pytest.ini
git add requirements.txt
git add setup.py

# Add source code
git add api_server/
git add app/
git add comfy_integration/
git add __init__.py
git add run_api.py
git add run_api.bat

# Add examples
git add examples/

# Add tests
git add tests/

# Add GitHub workflows
git add .github/

# Make initial commit
git commit -m "Initial commit: Social Media Video Generation API"

echo
echo "Git repository initialized successfully!"
echo
echo "Next steps:"
echo "1. Create a repository on GitHub"
echo "2. Run: git remote add origin https://github.com/yourusername/social-media-video-generator.git"
echo "3. Run: git push -u origin main"
echo