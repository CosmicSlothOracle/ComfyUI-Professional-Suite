#!/usr/bin/env python3
"""
EINFACHER COMFYUI TEST
====================
Testet ob ComfyUI grundsätzlich funktioniert
"""

import json
import requests
import time


def test_comfyui_basic():
    """Teste ob ComfyUI grundsätzlich läuft"""
    print("🔍 TESTE COMFYUI GRUNDFUNKTIONEN...")

    try:
        # Test Server
        response = requests.get(
            "http://127.0.0.1:8188/system_stats", timeout=5)
        if response.status_code == 200:
            print("✅ ComfyUI Server läuft")
            stats = response.json()
            print(f"   System: {stats}")
        else:
            print("❌ ComfyUI Server antwortet nicht richtig")
            return False
    except Exception as e:
        print(f"❌ ComfyUI Server nicht erreichbar: {e}")
        return False

    # Test verfügbare Nodes
    try:
        response = requests.get(
            "http://127.0.0.1:8188/object_info", timeout=10)
        if response.status_code == 200:
            nodes = response.json()
            print(f"✅ {len(nodes)} Nodes verfügbar")

            # Prüfe wichtige Nodes
            important_nodes = ["LoadImage",
                               "SaveImage", "VAEDecode", "KSampler"]
            missing = [node for node in important_nodes if node not in nodes]

            if missing:
                print(f"⚠️ Fehlende wichtige Nodes: {missing}")
            else:
                print("✅ Alle wichtigen Nodes verfügbar")

        else:
            print("❌ Kann Node-Liste nicht abrufen")
            return False

    except Exception as e:
        print(f"❌ Fehler beim Abrufen der Nodes: {e}")
        return False

    return True


def create_minimal_workflow():
    """Erstelle minimalsten möglichen Workflow"""
    workflow = {
        "1": {
            "inputs": {
                "image": "ComfyUI_00001_.png"
            },
            "class_type": "LoadImage"
        },
        "2": {
            "inputs": {
                "filename_prefix": "test_output",
                "images": ["1", 0]
            },
            "class_type": "SaveImage"
        }
    }
    return workflow


def test_minimal_workflow():
    """Teste minimalsten Workflow"""
    print("\n🧪 TESTE MINIMALEN WORKFLOW...")

    workflow = create_minimal_workflow()

    try:
        response = requests.post(
            "http://127.0.0.1:8188/prompt",
            json={"prompt": workflow},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Workflow akzeptiert: {result}")
            return True
        else:
            print(f"❌ Workflow-Fehler: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Workflow-Test fehlgeschlagen: {e}")
        return False


def main():
    """Haupttest"""
    print("🔧 COMFYUI DIAGNOSE-TEST")
    print("=" * 40)

    # 1. Grundfunktionen testen
    if not test_comfyui_basic():
        print("\n💥 GRUNDPROBLEM: ComfyUI funktioniert nicht!")
        print("   Lösung: Abhängigkeiten installieren und Server neu starten")
        return False

    # 2. Minimalen Workflow testen
    if not test_minimal_workflow():
        print("\n💥 WORKFLOW-PROBLEM: JSON-Format oder Node-Fehler!")
        print("   Lösung: Workflow-JSON überprüfen")
        return False

    print("\n✅ COMFYUI GRUNDFUNKTIONEN OK!")
    print("   Bereit für komplexere Workflows")
    return True


if __name__ == "__main__":
    success = main()
    input("\nDrücken Sie Enter...")
