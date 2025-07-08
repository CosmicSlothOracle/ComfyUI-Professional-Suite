#!/usr/bin/env python3
"""
VIDEO NODES TEST
===============
Prüft welche Video-Nodes verfügbar sind
"""

import requests
import json


def check_video_nodes():
    """Prüfe verfügbare Video-Nodes"""
    print("🎬 PRÜFE VIDEO-NODES...")

    try:
        response = requests.get(
            "http://127.0.0.1:8188/object_info", timeout=10)
        if response.status_code != 200:
            print("❌ Kann Node-Info nicht abrufen")
            return

        nodes = response.json()

        # Suche Video-relevante Nodes
        video_keywords = ["video", "load", "save", "VHS", "av", "ffmpeg"]
        video_nodes = []

        for node_name, node_info in nodes.items():
            node_lower = node_name.lower()
            if any(keyword.lower() in node_lower for keyword in video_keywords):
                video_nodes.append(node_name)

        print(f"\n📊 VERFÜGBARE VIDEO-NODES ({len(video_nodes)}):")
        for node in sorted(video_nodes):
            print(f"   • {node}")

        # Prüfe spezifische Nodes
        critical_nodes = [
            "VHS_LoadVideo", "LoadVideo", "VideoLoad",
            "VHS_VideoCombine", "VideoCombine", "SaveVideo"
        ]

        print(f"\n🔍 KRITISCHE VIDEO-NODES:")
        for node in critical_nodes:
            if node in nodes:
                print(f"   ✅ {node}")
            else:
                print(f"   ❌ {node}")

        # Prüfe Upscaling-Nodes
        upscale_nodes = [
            "UpscaleModelLoader", "ImageUpscaleWithModel",
            "UltimateSDUpscale", "RealESRGAN"
        ]

        print(f"\n🚀 UPSCALING-NODES:")
        for node in upscale_nodes:
            if node in nodes:
                print(f"   ✅ {node}")
            else:
                print(f"   ❌ {node}")

        return video_nodes

    except Exception as e:
        print(f"❌ Fehler: {e}")
        return []


def create_simple_video_workflow():
    """Erstelle einfachsten Video-Workflow"""

    # Prüfe verfügbare Nodes erst
    video_nodes = check_video_nodes()

    print(f"\n🛠️ ERSTELLE EINFACHEN VIDEO-WORKFLOW...")

    # Basis-Workflow ohne spezifische Video-Nodes
    workflow = {
        "1": {
            "inputs": {
                "image": "comica1750462002773.mp4"
            },
            "class_type": "LoadImage"  # Versuche MP4 als Bild zu laden
        },
        "2": {
            "inputs": {
                "filename_prefix": "video_test",
                "images": ["1", 0]
            },
            "class_type": "SaveImage"
        }
    }

    return workflow


def test_simple_video():
    """Teste einfachsten Video-Workflow"""
    print(f"\n🧪 TESTE EINFACHEN VIDEO-WORKFLOW...")

    workflow = create_simple_video_workflow()

    try:
        response = requests.post(
            "http://127.0.0.1:8188/prompt",
            json={"prompt": workflow},
            timeout=15
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Einfacher Workflow akzeptiert!")
            return True
        else:
            print(f"❌ Workflow-Fehler: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Details: {error_data}")
            except:
                print(f"   Raw: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Test fehlgeschlagen: {e}")
        return False


def main():
    """Haupttest"""
    print("🎬 VIDEO-NODES DIAGNOSE")
    print("=" * 40)

    # 1. Prüfe verfügbare Video-Nodes
    video_nodes = check_video_nodes()

    # 2. Teste einfachen Workflow
    success = test_simple_video()

    if not success:
        print(f"\n💡 LÖSUNGSVORSCHLAG:")
        print(f"   1. Custom Video-Nodes installieren (VHS, AnimateDiff)")
        print(f"   2. Oder: Standalone Python-Script für Video-Verarbeitung")

    return success


if __name__ == "__main__":
    main()
    input("\nDrücken Sie Enter...")
