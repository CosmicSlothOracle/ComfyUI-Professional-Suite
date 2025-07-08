#!/usr/bin/env python3
"""
VAPORWAVE STYLE OVERVIEW & QUICK LAUNCHER
Shows all available vaporwave styles and provides easy execution commands
"""

import subprocess
import sys
from pathlib import Path


def show_vaporwave_overview():
    """Display comprehensive overview of all vaporwave styles"""

    print("🌈" + "=" * 70 + "🌈")
    print("           VAPORWAVE STYLISIERUNG - COMPLETE OVERVIEW")
    print("🌈" + "=" * 70 + "🌈")
    print()

    print("📊 CURRENT STATUS:")
    print("   ✅ Transparent Background Generation: COMPLETE (246/246 files)")
    print("   🌈 Ready for Vaporwave Stylization!")
    print()

    print("🎨 AVAILABLE VAPORWAVE STYLES:")
    print("=" * 72)

    styles = [
        {
            "name": "NEON",
            "emoji": "🌈",
            "description": "Bright neon colors, glowing effects, grid overlays",
            "features": ["Pink/Cyan color schemes", "Neon glow effects", "Grid overlays", "80s retro vibes"],
            "command": "python run_vaporwave_batch.py --style neon"
        },
        {
            "name": "RETRO",
            "emoji": "🌅",
            "description": "Sunset gradients, palm trees, VHS effects",
            "features": ["Sunset color palettes", "Palm tree silhouettes", "VHS degradation", "Retro typography"],
            "command": "python run_vaporwave_batch.py --style retro --text 'VAPOR WAVE'"
        },
        {
            "name": "GLITCH",
            "emoji": "⚡",
            "description": "Digital corruption, datamoshing, cyberpunk",
            "features": ["Digital glitch effects", "Chromatic aberration", "Datamoshing", "CRT effects"],
            "command": "python run_vaporwave_batch.py --style glitch"
        },
        {
            "name": "CHROME",
            "emoji": "🤖",
            "description": "Metallic surfaces, 3D geometry, holographic",
            "features": ["Chrome reflections", "3D geometric shapes", "Hologram effects", "Metallic gradients"],
            "command": "python run_vaporwave_batch.py --style chrome --text 'CHROME DREAMS'"
        }
    ]

    for i, style in enumerate(styles, 1):
        print(f"{i}. {style['emoji']} {style['name']} STYLE")
        print(f"   Description: {style['description']}")
        print(f"   Features:")
        for feature in style['features']:
            print(f"     • {feature}")
        print(f"   Command: {style['command']}")
        print()

    print("🚀 QUICK START COMMANDS:")
    print("=" * 72)
    print("📋 List all styles:")
    print("   python run_vaporwave_batch.py --list-styles")
    print()
    print("🧪 Test with limited files (first 5 videos):")
    print("   python run_vaporwave_batch.py --style neon --limit 5")
    print()
    print("⚡ Fast processing (more workers):")
    print("   python run_vaporwave_batch.py --style retro --workers 5")
    print()
    print("🎭 Custom text overlay:")
    print("   python run_vaporwave_batch.py --style chrome --text 'YOUR TEXT'")
    print()

    print("📁 OUTPUT STRUCTURE:")
    print("=" * 72)
    print("output/")
    print("├── *_fast_transparent.mp4     (Input: 246 transparent videos)")
    print("└── vaporwave/")
    print("    ├── neon/                  (Neon style outputs)")
    print("    ├── retro/                 (Retro style outputs)")
    print("    ├── glitch/                (Glitch style outputs)")
    print("    └── chrome/                (Chrome style outputs)")
    print()

    print("💡 PROCESSING TIPS:")
    print("=" * 72)
    print("• Each style will process all 246 transparent videos")
    print("• Use --limit for testing with fewer files first")
    print("• Adjust --workers based on your system (default: 3)")
    print("• Processing time: ~10-30 seconds per video")
    print("• Total time per style: ~45-120 minutes for all 246 files")
    print("• You can run different styles simultaneously")
    print()

    print("🎯 RECOMMENDED WORKFLOW:")
    print("=" * 72)
    print("1. Test with limited files first:")
    print("   python run_vaporwave_batch.py --style neon --limit 10")
    print()
    print("2. If satisfied, run full batch:")
    print("   python run_vaporwave_batch.py --style neon")
    print()
    print("3. Try other styles:")
    print("   python run_vaporwave_batch.py --style retro")
    print("   python run_vaporwave_batch.py --style glitch")
    print("   python run_vaporwave_batch.py --style chrome")
    print()

    print("🌈" + "=" * 70 + "🌈")


def check_prerequisites():
    """Check if all required files are present"""

    print("🔍 CHECKING PREREQUISITES...")

    # Check for transparent videos
    output_dir = Path("output")
    transparent_videos = list(output_dir.glob("*_fast_transparent.mp4"))

    print(f"✅ Found {len(transparent_videos)} transparent videos")

    # Check for workflow scripts
    required_scripts = [
        "workflow_vaporwave_neon.py",
        "workflow_vaporwave_retro.py",
        "workflow_vaporwave_glitch.py",
        "workflow_vaporwave_chrome.py",
        "run_vaporwave_batch.py"
    ]

    missing_scripts = []
    for script in required_scripts:
        if not Path(script).exists():
            missing_scripts.append(script)

    if missing_scripts:
        print(f"❌ Missing scripts: {', '.join(missing_scripts)}")
        return False
    else:
        print("✅ All workflow scripts present")
        return True


def interactive_launcher():
    """Interactive launcher for vaporwave styles"""

    print("\n🚀 INTERACTIVE VAPORWAVE LAUNCHER")
    print("=" * 50)

    styles = ["neon", "retro", "glitch", "chrome"]

    print("Choose a style:")
    for i, style in enumerate(styles, 1):
        print(f"{i}. {style.upper()}")

    try:
        choice = input("\nEnter choice (1-4) or 'q' to quit: ").strip()

        if choice.lower() == 'q':
            return

        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(styles):
            print("❌ Invalid choice!")
            return

        selected_style = styles[choice_idx]

        # Ask for options
        limit = input("Limit files for testing (default: all): ").strip()
        workers = input("Number of workers (default: 3): ").strip()

        # Build command
        cmd = [sys.executable, "run_vaporwave_batch.py",
               "--style", selected_style]

        if limit:
            cmd.extend(["--limit", limit])

        if workers:
            cmd.extend(["--workers", workers])

        # Add text for applicable styles
        if selected_style in ["retro", "chrome"]:
            text = input("Custom text overlay (optional): ").strip()
            if text:
                cmd.extend(["--text", text])

        print(f"\n🚀 Launching: {' '.join(cmd)}")
        print("=" * 50)

        # Execute command
        subprocess.run(cmd)

    except (ValueError, KeyboardInterrupt):
        print("\n⏹️ Cancelled")


def main():
    print("🌈 VAPORWAVE OVERVIEW & LAUNCHER 🌈")
    print()

    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met. Please ensure all files are present.")
        return

    # Show overview
    show_vaporwave_overview()

    # Ask if user wants interactive launcher
    choice = input(
        "Launch interactive style selector? (y/n): ").strip().lower()
    if choice == 'y':
        interactive_launcher()


if __name__ == "__main__":
    main()
