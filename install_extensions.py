import os
import sys
import json
import requests
from pathlib import Path

# List of useful extensions for development
EXTENSIONS = [
    "https://github.com/ltdrdata/ComfyUI-Manager",  # Main manager
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts",  # Workflow enhancements
    "https://github.com/pythongosssss/ComfyUI-WD14-Tagger",  # Auto tagging
    "https://github.com/pythongosssss/ComfyUI-Studio",  # UI improvements
    "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet",  # Advanced control
    "https://github.com/Gourieff/comfyui-reactor-node",  # Face swap and restoration
    "https://github.com/melMass/comfy_mtb",  # Additional nodes
    "https://github.com/crystian/ComfyUI-Crystools",  # Development tools
    "https://github.com/thecooltechguy/ComfyUI-Endpoint-Control",  # API endpoints
    "https://github.com/11cafe/comfyui-workspace-manager",  # Workspace organization
    "https://github.com/chrisgoringe/cg-use-everywhere",  # Node reuse
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",  # Video processing
    "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes",  # Custom nodes
    "https://github.com/ssitu/ComfyUI_UltimateSDUpscale",  # Upscaling
]


def install_extension(repo_url):
    try:
        # Extract repo name from URL
        repo_name = repo_url.split('/')[-1]

        # Create custom_nodes directory if it doesn't exist
        custom_nodes_dir = Path("custom_nodes")
        custom_nodes_dir.mkdir(exist_ok=True)

        # Clone the repository
        target_dir = custom_nodes_dir / repo_name
        if not target_dir.exists():
            os.system(f'git clone {repo_url} "{target_dir}"')
            print(f"✓ Installed {repo_name}")

            # Install requirements if they exist
            requirements_file = target_dir / "requirements.txt"
            if requirements_file.exists():
                os.system(f'pip install -r "{requirements_file}"')
                print(f"✓ Installed requirements for {repo_name}")
        else:
            print(f"! {repo_name} already exists, updating...")
            os.system(f'cd "{target_dir}" && git pull')

    except Exception as e:
        print(f"× Error installing {repo_url}: {str(e)}")


def main():
    print("Installing ComfyUI extensions...")
    for extension in EXTENSIONS:
        install_extension(extension)
    print("\nInstallation complete! Please restart ComfyUI to load the new extensions.")


if __name__ == "__main__":
    main()
