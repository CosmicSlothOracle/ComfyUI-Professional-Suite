#!/usr/bin/env python3
"""
🎬 STABLE VAPORWAVE WORKFLOW EXECUTOR
===================================
Robuste Ausführung des Vaporwave-Workflows
"""

import time
import json
import requests
from pathlib import Path


def wait_for_server():
    """Warte bis ComfyUI Server bereit ist"""
    print("⏳ Warte auf ComfyUI Server...")

    for attempt in range(30):
        try:
            response = requests.get(
                "http://127.0.0.1:8188/system_stats", timeout=2)
            if response.status_code == 200:
                print("✅ ComfyUI Server ist bereit!")
                return True
        except:
            pass

        print(f"   Versuch {attempt + 1}/30...", end="\r")
        time.sleep(2)

    print("\n❌ Server nicht erreichbar")
    return False


def load_workflow():
    """Lade Workflow aus JSON-Datei"""
    workflow_path = Path("workflows/video_to_vaporwave_gifs_workflow.json")

    if not workflow_path.exists():
        print(f"❌ Workflow nicht gefunden: {workflow_path}")
        return None

    try:
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        print("✅ Workflow geladen")
        return workflow
    except Exception as e:
        print(f"❌ Workflow-Fehler: {e}")
        return None


def execute_workflow(workflow):
    """Führe Workflow aus"""
    print("\n🎬 STARTE VAPORWAVE-TRANSFORMATION:")
    print("   📹 Video: comica1750462002773.mp4")
    print("   🚀 4x Upscaling (512x512 → 2048x2048)")
    print("   🎨 Vaporwave-Farbharmonisierung")
    print("   ✨ VHS-Retro-Effekte")
    print("   🎬 3 GIF-Sequenzen erstellen")
    print()

    try:
        # Sende Workflow
        response = requests.post(
            "http://127.0.0.1:8188/prompt",
            json={"prompt": workflow},
            timeout=30
        )

        if response.status_code != 200:
            print(f"❌ Server-Fehler: {response.status_code}")
            try:
                error_data = response.json()
                if 'error' in error_data:
                    print(f"   Details: {error_data['error']}")
                if 'node_errors' in error_data:
                    for node_id, error in error_data['node_errors'].items():
                        print(f"   Node {node_id}: {error}")
            except:
                print(f"   Raw Response: {response.text}")
            return False

        result = response.json()
        prompt_id = result.get("prompt_id")

        if not prompt_id:
            print("❌ Keine Prompt-ID erhalten")
            return False

        print(f"✅ Workflow gestartet (ID: {prompt_id[:8]}...)")

        # Überwache Ausführung
        return monitor_execution(prompt_id)

    except requests.exceptions.Timeout:
        print("❌ Timeout beim Senden des Workflows")
        return False
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        return False


def monitor_execution(prompt_id):
    """Überwache Workflow-Ausführung"""
    start_time = time.time()
    last_progress = ""

    print("🔄 Überwache Fortschritt...")

    while True:
        try:
            # Prüfe Queue-Status
            queue_resp = requests.get("http://127.0.0.1:8188/queue", timeout=5)
            if queue_resp.status_code == 200:
                queue_data = queue_resp.json()
                running = len(queue_data.get("queue_running", []))
                pending = len(queue_data.get("queue_pending", []))

                progress = f"Running: {running}, Pending: {pending}"
                if progress != last_progress:
                    elapsed = time.time() - start_time
                    print(f"   Status ({elapsed:.1f}s): {progress}")
                    last_progress = progress

            # Prüfe Completion
            history_resp = requests.get(
                f"http://127.0.0.1:8188/history/{prompt_id}", timeout=5)
            if history_resp.status_code == 200:
                history = history_resp.json()

                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status", {})

                    if status.get("completed", False):
                        elapsed = time.time() - start_time
                        print(
                            f"\n✅ VERARBEITUNG ABGESCHLOSSEN! ({elapsed:.1f}s)")

                        # Zeige Ergebnisse
                        outputs = entry.get("outputs", {})
                        if outputs:
                            print("\n📊 ERGEBNISSE:")
                            total_images = 0
                            for node_id, output in outputs.items():
                                if "images" in output:
                                    images = output["images"]
                                    total_images += len(images)
                                    print(
                                        f"   🖼️ Node {node_id}: {len(images)} Bilder")
                            print(
                                f"\n🎉 TOTAL: {total_images} Bilder erstellt!")

                        return True

                    elif "error" in status:
                        print(f"\n❌ Workflow-Fehler: {status['error']}")
                        return False

            time.sleep(3)

        except KeyboardInterrupt:
            print("\n⏹️ Abgebrochen durch Benutzer")
            return False
        except Exception as e:
            print(f"\n⚠️ Monitoring-Problem: {e}")
            time.sleep(5)


def show_final_results():
    """Zeige finale Ergebnisse"""
    print("\n" + "=" * 60)
    print("🎊 VAPORWAVE-TRANSFORMATION ABGESCHLOSSEN!")
    print("=" * 60)

    output_dirs = [
        Path("ComfyUI_engine/output"),
        Path("ComfyUI_engine/output/vaporwave_gifs"),
        Path("ComfyUI_engine/output/vaporwave_frames")
    ]

    total_files = 0
    for output_dir in output_dirs:
        if output_dir.exists():
            files = [f for f in output_dir.iterdir() if f.is_file()]
            if files:
                total_files += len(files)
                print(f"\n📁 {output_dir.name.upper()}: {len(files)} Dateien")
                for i, file in enumerate(sorted(files)[:3]):
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"   • {file.name} ({size_mb:.1f} MB)")
                if len(files) > 3:
                    print(f"   • ... und {len(files) - 3} weitere")

    print(f"\n🚀 ERFOLG! {total_files} Dateien erstellt")
    print(f"📂 Hauptordner: ComfyUI_engine/output/")
    print(f"🎬 Ihre Vaporwave-GIFs sind bereit!")
    print(f"🌐 ComfyUI Interface: http://127.0.0.1:8188")


def main():
    """Hauptausführung"""
    print("🎬 STABLE VAPORWAVE WORKFLOW EXECUTOR")
    print("=" * 50)

    # 1. Warte auf Server
    if not wait_for_server():
        print("💥 Server nicht verfügbar!")
        return False

    # 2. Lade Workflow
    workflow = load_workflow()
    if not workflow:
        print("💥 Workflow-Problem!")
        return False

    # 3. Führe Workflow aus
    success = execute_workflow(workflow)

    if success:
        show_final_results()
        print("\n🎊 MISSION ERFOLGREICH!")
        return True
    else:
        print("\n💥 WORKFLOW FEHLGESCHLAGEN!")
        return False


if __name__ == "__main__":
    try:
        success = main()
        input("\nDrücken Sie Enter zum Beenden...")
    except KeyboardInterrupt:
        print("\n⏹️ Abgebrochen")
    except Exception as e:
        print(f"\n💥 Fehler: {e}")
        input("\nDrücken Sie Enter zum Beenden...")
