#!/usr/bin/env python3
"""
VAPORWAVE ANALYSIS & TEST
Analysiert die bestehenden Dateien und testet verschiedene Workflows
"""

import os
import sys
from pathlib import Path
import shutil
import time


def analyze_video_files():
    """Analysiere die beiden angegebenen Dateien"""

    print("🔍 VAPORWAVE DATEI-ANALYSE")
    print("=" * 60)

    # Originaldatei im Input-Ordner
    original_file = Path(
        "input/837b898b4d1eb49036dfce89c30cba59_fast_transparent.mp4")

    # Bearbeitete Datei (vom User angegeben)
    edited_file = Path(
        "C:/Users/skank/Downloads/Unbenanntes Video – Mit Clipchamp erstellt.mp4")

    print("📁 ANALYSIERTE DATEIEN:")
    print(f"1. ORIGINAL: {original_file}")
    print(
        f"   Status: {'✅ EXISTS' if original_file.exists() else '❌ NOT FOUND'}")
    if original_file.exists():
        size_mb = original_file.stat().st_size / (1024 * 1024)
        print(f"   Größe: {size_mb:.1f} MB")

    print(f"\n2. BEARBEITET: {edited_file}")
    print(
        f"   Status: {'✅ EXISTS' if edited_file.exists() else '❌ NOT FOUND'}")
    if edited_file.exists():
        size_mb = edited_file.stat().st_size / (1024 * 1024)
        print(f"   Größe: {size_mb:.1f} MB")

    return original_file.exists(), edited_file.exists()


def compare_with_existing_outputs():
    """Vergleiche mit den bestehenden Vaporwave-Outputs"""

    print("\n🎨 VERGLEICH MIT BESTEHENDEN VAPORWAVE-OUTPUTS")
    print("=" * 60)

    # Schaue welche Outputs bereits existieren
    test_dirs = [
        "output/vaporwave_test_comfy/neon",
        "output/vaporwave_test_comfy/retro",
        "output/vaporwave_test_comfy/glitch"
    ]

    target_file = "837b898b4d1eb49036dfce89c30cba59_fast_transparent"

    for test_dir in test_dirs:
        test_path = Path(test_dir)
        style = test_path.name

        print(f"\n🌈 {style.upper()} STYLE:")

        if test_path.exists():
            # Suche nach der Zieldatei
            matching_files = list(test_path.glob(f"*{target_file}*"))

            if matching_files:
                for file in matching_files:
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"   ✅ {file.name} ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ Keine Datei für {target_file} gefunden")
        else:
            print(f"   ❌ Verzeichnis nicht gefunden: {test_dir}")


def create_workflow_recommendations():
    """Erstelle Workflow-Empfehlungen basierend auf der Analyse"""

    print("\n💡 WORKFLOW-EMPFEHLUNGEN")
    print("=" * 60)

    print("BASIEREND AUF DER ANALYSE DER BESTEHENDEN WORKFLOWS:")
    print()

    print("❌ PROBLEME DER AKTUELLEN WORKFLOWS (1-3):")
    print("   • Nur Simulations-Scripts (kopieren nur Dateien)")
    print("   • Keine echte Bildverarbeitung")
    print("   • ComfyUI-Nodes existieren nicht")
    print("   • FFmpeg-Implementierung zu simpel")
    print()

    print("✅ LÖSUNG: ULTIMATE VAPORWAVE WORKFLOW")
    print("   • Echte FFmpeg-basierte Effekte")
    print("   • Multi-Pass-Verarbeitung")
    print("   • Authentische Vaporwave-Ästhetik")
    print()

    print("🎯 EMPFOHLENE WORKFLOW-PARAMETER FÜR IHR VIDEO:")
    print()

    print("1. 🌈 NEON SYNTHWAVE:")
    print("   • Hohe Sättigung (saturation=2.5)")
    print("   • Pink/Magenta Farbverschiebung (hue=300)")
    print("   • Neon-Glow-Effekte (gblur + unsharp)")
    print("   • Grid-Overlay (drawgrid)")
    print("   • Farb-Balance: Pink/Cyan")
    print()

    print("2. 🌅 RETRO SUNSET:")
    print("   • Warme Farbtemperatur (hue=20)")
    print("   • Sunset-Kurven (separate RGB-Kurven)")
    print("   • VHS-Degradation (noise + unsharp)")
    print("   • Scanlines (geq-Filter)")
    print("   • Orange/Pink Balance")
    print()

    print("3. ⚡ GLITCH CYBERPUNK:")
    print("   • Extreme Kontraste (contrast=1.6)")
    print("   • Cyan/Blau Verschiebung (hue=180)")
    print("   • Digitale Korruption (noise + geq)")
    print("   • Chromatische Aberration (split/overlay)")
    print("   • Random Pixel-Korruption")


def suggest_test_workflow():
    """Schlage einen Testworkflow vor"""

    print("\n🚀 EMPFOHLENER TEST-WORKFLOW")
    print("=" * 60)

    print("DA FFMPEG NICHT INSTALLIERT IST, ALTERNATIVE ANSÄTZE:")
    print()

    print("OPTION 1: FFMPEG INSTALLIEREN")
    print("   1. FFmpeg herunterladen: https://ffmpeg.org/download.html")
    print("   2. Zu Windows PATH hinzufügen")
    print("   3. Ultimate Workflow ausführen:")
    print("      python workflow_vaporwave_ultimate.py \\")
    print("        --input \"input/837b898b4d1eb49036dfce89c30cba59_fast_transparent.mp4\" \\")
    print("        --output \"output/vaporwave_ultimate\" \\")
    print("        --style neon")
    print()

    print("OPTION 2: COMFYUI-BASIERTE LÖSUNG")
    print("   • Erstelle echte ComfyUI-Workflows")
    print("   • Verwende existierende ComfyUI-Nodes")
    print("   • Integriere in bestehende ComfyUI-Installation")
    print()

    print("OPTION 3: PYTHON PIL/OPENCV-BASIERTE LÖSUNG")
    print("   • Verwende Python-Libraries für Bildverarbeitung")
    print("   • Implementiere Vaporwave-Effekte in Python")
    print("   • Keine externe Software erforderlich")


def main():
    print("🌈" + "="*70 + "🌈")
    print("           ULTIMATE VAPORWAVE ANALYSIS & WORKFLOW EMPFEHLUNG")
    print("🌈" + "="*70 + "🌈")

    # Analysiere die Dateien
    original_exists, edited_exists = analyze_video_files()

    # Vergleiche mit bestehenden Outputs
    compare_with_existing_outputs()

    # Erstelle Empfehlungen
    create_workflow_recommendations()

    # Schlage Testworkflow vor
    suggest_test_workflow()

    print("\n" + "="*70)
    print("📊 ZUSAMMENFASSUNG:")
    print(f"   Original-Datei: {'✅' if original_exists else '❌'}")
    print(f"   Bearbeitete Datei: {'✅' if edited_exists else '❌'}")
    print("   Empfohlener Workflow: Ultimate Vaporwave (FFmpeg-basiert)")
    print("   Alternative: ComfyUI-Integration oder Python PIL/OpenCV")
    print("="*70)


if __name__ == "__main__":
    main()
