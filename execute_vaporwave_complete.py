#!/usr/bin/env python3
"""
🎬 COMPLETE VAPORWAVE EXECUTION SCRIPT
=====================================
Startet ComfyUI und führt den kompletten Vaporwave-Workflow aus
"""

import os
import sys
import json
import time
import requests
import subprocess
import threading
from pathlib import Path


class VaporwaveExecutor:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8188"
        self.comfy_dir = Path("ComfyUI_engine")
        self.workflow_path = Path(
            "workflows/video_to_vaporwave_gifs_workflow.json")
        self.server_process = None

        print("🎬 VAPORWAVE COMPLETE EXECUTOR - CUDA EDITION")
        print("=" * 50)

    def check_gpu_setup(self):
        """Prüfe GPU und CUDA-Setup"""
        try:
            import torch
            print(f"🔥 PyTorch: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"🚀 CUDA: ✅ Aktiv")
                print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
                print(
                    f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                return True
            else:
                print("⚠️ CUDA: Nicht verfügbar, nutze CPU")
                return False
        except ImportError:
            print("❌ PyTorch nicht gefunden")
            return False

    def start_comfyui_server(self):
        """Starte ComfyUI Server"""
        print("🚀 Starte ComfyUI Server...")

        # GPU-Setup prüfen
        has_gpu = self.check_gpu_setup()

        # Wechsel ins ComfyUI_engine Verzeichnis
        os.chdir(self.comfy_dir)

        # Server-Kommando mit GPU-Optimierungen
        cmd = [sys.executable, "main.py", "--listen", "--port", "8188"]

        if has_gpu:
            cmd.extend(["--highvram", "--force-fp16"])  # GPU-Optimierungen
            print("🎯 GPU-Modus: High-VRAM + FP16 für maximale Performance")
        else:
            cmd.append("--cpu")
            print("🐌 CPU-Modus: Langsamere Verarbeitung")

        try:
            # Umgehe xformers-Problem
            env = os.environ.copy()
            env["XFORMERS_DISABLED"] = "1"

            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )

            print("⏳ Warte auf Server-Initialisierung...")

            # Warte auf Server-Bereitschaft
            max_attempts = 60  # 60 Sekunden warten
            for attempt in range(max_attempts):
                try:
                    response = requests.get(
                        f"{self.base_url}/system_stats", timeout=2)
                    if response.status_code == 200:
                        print("✅ ComfyUI Server ist bereit!")
                        # Zurück ins Hauptverzeichnis
                        os.chdir("..")
                        return True
                except requests.RequestException:
                    pass

                print(f"⏳ Warte... ({attempt + 1}/{max_attempts})", end="\r")
                time.sleep(1)

            print("\n❌ Server-Start fehlgeschlagen (Timeout)")
            return False

        except Exception as e:
            print(f"❌ Fehler beim Server-Start: {e}")
            return False

    def load_workflow(self):
        """Lade und validiere Workflow"""
        print("📋 Lade Workflow...")

        if not self.workflow_path.exists():
            print(f"❌ Workflow nicht gefunden: {self.workflow_path}")
            return None

        try:
            with open(self.workflow_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)

            print("✅ Workflow geladen")
            return workflow

        except Exception as e:
            print(f"❌ Workflow-Laden fehlgeschlagen: {e}")
            return None

    def create_output_dirs(self):
        """Erstelle Output-Verzeichnisse"""
        print("📁 Erstelle Output-Verzeichnisse...")

        dirs = [
            "ComfyUI_engine/output/vaporwave_gifs",
            "ComfyUI_engine/output/vaporwave_frames",
            "ComfyUI_engine/temp_processing"
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        print("✅ Verzeichnisse erstellt")

    def execute_workflow(self, workflow):
        """Führe Workflow aus"""
        print("🎬 Führe CUDA-beschleunigten Vaporwave-Workflow aus...")
        print("   Schritte:")
        print("   1. 📹 Video laden & Frames extrahieren")
        print("   2. 🚀 4x GPU-Upscaling mit RealESRGAN")
        print("   3. 🎨 Vaporwave-Farbharmonisierung")
        print("   4. ✨ VHS-Retro-Effekte")
        print("   5. ✂️ Sequenzen segmentieren")
        print("   6. 🎬 GIFs erstellen")
        print()

        try:
            # Sende Workflow an ComfyUI
            data = {"prompt": workflow}
            response = requests.post(
                f"{self.base_url}/prompt", json=data, timeout=10)

            if response.status_code != 200:
                print(f"❌ Workflow-Fehler: {response.status_code}")
                print(f"Response: {response.text}")
                return False

            result = response.json()
            prompt_id = result.get("prompt_id")

            if not prompt_id:
                print("❌ Keine Prompt-ID erhalten")
                return False

            print(f"✅ Workflow gestartet (ID: {prompt_id})")
            print("🔥 GPU-Turbo aktiviert! Verarbeitung läuft...")

            # Überwache Fortschritt
            return self.monitor_progress(prompt_id)

        except Exception as e:
            print(f"❌ Workflow-Ausführung fehlgeschlagen: {e}")
            return False

    def monitor_progress(self, prompt_id):
        """Überwache Workflow-Fortschritt"""
        start_time = time.time()
        last_status = ""

        while True:
            try:
                # Queue-Status prüfen
                queue_response = requests.get(
                    f"{self.base_url}/queue", timeout=5)
                if queue_response.status_code == 200:
                    queue_data = queue_response.json()
                    queue_running = queue_data.get("queue_running", [])
                    queue_pending = queue_data.get("queue_pending", [])

                    current_status = f"Running: {len(queue_running)}, Pending: {len(queue_pending)}"
                    if current_status != last_status:
                        elapsed = time.time() - start_time
                        print(f"🔄 Status ({elapsed:.1f}s): {current_status}")
                        last_status = current_status

                # History prüfen für Completion
                history_response = requests.get(
                    f"{self.base_url}/history/{prompt_id}", timeout=5)
                if history_response.status_code == 200:
                    history = history_response.json()

                    if prompt_id in history:
                        entry = history[prompt_id]
                        status = entry.get("status", {})

                        if status.get("completed", False):
                            elapsed = time.time() - start_time
                            print(
                                f"✅ CUDA-Verarbeitung abgeschlossen! ({elapsed:.1f}s)")

                            # Zeige Ausgaben
                            outputs = entry.get("outputs", {})
                            if outputs:
                                print("\n📊 ERGEBNISSE:")
                                for node_id, output in outputs.items():
                                    if "images" in output:
                                        images = output["images"]
                                        print(
                                            f"   Node {node_id}: {len(images)} Bilder erstellt")
                                        for img in images[:3]:  # Zeige erste 3
                                            print(
                                                f"     • {img.get('filename', 'unknown')}")

                            return True

                        elif "error" in status:
                            print(f"❌ Workflow-Fehler: {status['error']}")
                            return False

                time.sleep(2)

            except KeyboardInterrupt:
                print("\n⏹️ Verarbeitung abgebrochen")
                return False
            except Exception as e:
                print(f"⚠️ Monitoring-Fehler: {e}")
                time.sleep(5)

    def show_results(self):
        """Zeige finale Ergebnisse"""
        print("\n🎉 CUDA-BESCHLEUNIGTE VAPORWAVE-VERARBEITUNG ABGESCHLOSSEN!")
        print("=" * 60)

        # Prüfe Output-Verzeichnisse
        output_dirs = {
            "🎬 GIFs": Path("ComfyUI_engine/output/vaporwave_gifs"),
            "🖼️ Frames": Path("ComfyUI_engine/output/vaporwave_frames"),
            "📁 ComfyUI Output": Path("ComfyUI_engine/output")
        }

        total_files = 0
        for name, path in output_dirs.items():
            if path.exists():
                files = list(path.glob("*"))
                if files:
                    total_files += len(files)
                    print(f"\n📁 {name}: {len(files)} Dateien")
                    for i, file in enumerate(files[:3]):  # Zeige erste 3
                        size_mb = file.stat().st_size / (1024 * 1024)
                        print(f"   • {file.name} ({size_mb:.1f} MB)")
                    if len(files) > 3:
                        print(f"   • ... und {len(files) - 3} weitere")

        print(f"\n🚀 GPU-POWER UNLEASHED! {total_files} Dateien erstellt!")
        print(f"✨ IHRE VAPORWAVE-GIFS SIND BEREIT!")
        print(f"📂 Hauptordner: ComfyUI_engine/output/")
        print(f"🌐 ComfyUI Interface: {self.base_url}")

    def cleanup(self):
        """Aufräumen"""
        if self.server_process:
            print("\n🛑 Stoppe ComfyUI Server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()

    def run(self):
        """Hauptausführung"""
        try:
            # 1. Output-Verzeichnisse erstellen
            self.create_output_dirs()

            # 2. ComfyUI Server starten
            if not self.start_comfyui_server():
                return False

            # 3. Workflow laden
            workflow = self.load_workflow()
            if not workflow:
                return False

            # 4. Workflow ausführen
            success = self.execute_workflow(workflow)

            if success:
                # 5. Ergebnisse anzeigen
                self.show_results()
                return True
            else:
                print("💥 Workflow-Ausführung fehlgeschlagen!")
                return False

        except KeyboardInterrupt:
            print("\n⏹️ Ausführung abgebrochen")
            return False
        except Exception as e:
            print(f"💥 Unerwarteter Fehler: {e}")
            return False
        finally:
            self.cleanup()


def main():
    """Hauptfunktion"""
    print("""
    ██╗   ██╗ █████╗ ██████╗  ██████╗ ██████╗ ██╗    ██╗ █████╗ ██╗   ██╗███████╗
    ██║   ██║██╔══██╗██╔══██╗██╔═══██╗██╔══██╗██║    ██║██╔══██╗██║   ██║██╔════╝
    ██║   ██║███████║██████╔╝██║   ██║██████╔╝██║ █╗ ██║███████║██║   ██║█████╗
    ╚██╗ ██╔╝██╔══██║██╔═══╝ ██║   ██║██╔══██╗██║███╗██║██╔══██║╚██╗ ██╔╝██╔══╝
     ╚████╔╝ ██║  ██║██║     ╚██████╔╝██║  ██║╚███╔███╔╝██║  ██║ ╚████╔╝ ███████╗
      ╚═══╝  ╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝  ╚═══╝  ╚══════╝

                    🔥 CUDA EDITION - GPU TURBO ACTIVATED 🔥
    """)

    executor = VaporwaveExecutor()
    success = executor.run()

    if success:
        print("\n🎊 MISSION ERFOLGREICH!")
        print("🚀 Ihre CUDA-beschleunigten Vaporwave-GIFs sind bereit!")
        input("\nDrücken Sie Enter zum Beenden...")
        sys.exit(0)
    else:
        print("\n💥 MISSION FEHLGESCHLAGEN!")
        input("\nDrücken Sie Enter zum Beenden...")
        sys.exit(1)


if __name__ == "__main__":
    main()
