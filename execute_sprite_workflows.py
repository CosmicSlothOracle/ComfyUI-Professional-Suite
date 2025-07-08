#!/usr/bin/env python3
"""
Direkte Ausführung der Sprite-Processing Workflows
Verwendet ComfyUI API für automatisierte Verarbeitung
"""

import json
import os
import sys
import time
import requests
import subprocess
from pathlib import Path


class SpriteWorkflowExecutor:
    def __init__(self):
        self.comfyui_url = "http://127.0.0.1:8188"
        self.workflows_dir = Path("workflows/sprite_processing")
        self.output_dir = Path("output/styled_sprites")
        self.server_process = None

    def start_comfyui_server(self):
        """Startet ComfyUI Server im Hintergrund"""
        print("🚀 Starte ComfyUI Server...")
        try:
            # Server im Hintergrund starten
            self.server_process = subprocess.Popen([
                sys.executable, "main.py",
                "--listen", "--port", "8188", "--cpu"
            ], cwd="ComfyUI_engine", stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Warten bis Server bereit ist
            max_attempts = 30
            for attempt in range(max_attempts):
                try:
                    response = requests.get(
                        f"{self.comfyui_url}/system_stats", timeout=2)
                    if response.status_code == 200:
                        print("✅ ComfyUI Server ist bereit!")
                        return True
                except:
                    pass
                print(f"⏳ Warte auf Server... ({attempt+1}/{max_attempts})")
                time.sleep(2)

            print("❌ Server konnte nicht gestartet werden")
            return False
        except Exception as e:
            print(f"❌ Fehler beim Starten des Servers: {e}")
            return False

    def stop_comfyui_server(self):
        """Stoppt ComfyUI Server"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            print("🛑 ComfyUI Server gestoppt")

    def queue_workflow(self, workflow_path):
        """Fügt Workflow zur Verarbeitungsqueue hinzu"""
        try:
            # Workflow laden
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)

            # Workflow an ComfyUI senden
            response = requests.post(
                f"{self.comfyui_url}/prompt", json={"prompt": workflow})

            if response.status_code == 200:
                result = response.json()
                prompt_id = result.get("prompt_id")
                print(
                    f"✅ Workflow gestartet: {workflow_path.name} (ID: {prompt_id})")
                return prompt_id
            else:
                print(
                    f"❌ Fehler beim Senden von {workflow_path.name}: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Fehler beim Verarbeiten von {workflow_path}: {e}")
            return None

    def wait_for_completion(self, prompt_id):
        """Wartet auf Abschluss der Verarbeitung"""
        print(f"⏳ Warte auf Abschluss von Prompt {prompt_id}...")

        max_attempts = 300  # 5 Minuten Timeout
        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    f"{self.comfyui_url}/history/{prompt_id}")
                if response.status_code == 200:
                    history = response.json()
                    if prompt_id in history:
                        status = history[prompt_id].get("status", {})
                        if status.get("completed", False):
                            print(f"✅ Prompt {prompt_id} abgeschlossen!")
                            return True
                        elif "error" in status:
                            print(
                                f"❌ Fehler in Prompt {prompt_id}: {status['error']}")
                            return False

            except Exception as e:
                print(f"⚠️ Fehler beim Überprüfen des Status: {e}")

            time.sleep(1)

        print(f"⏰ Timeout für Prompt {prompt_id}")
        return False

    def execute_all_workflows(self):
        """Führt alle Sprite-Processing Workflows aus"""
        print("🎮 STARTE AUTOMATISIERTE SPRITE-VERARBEITUNG")
        print("=" * 60)

        # Server starten
        if not self.start_comfyui_server():
            return False

        try:
            # Alle Workflow-Dateien finden
            workflow_files = list(self.workflows_dir.glob("*.json"))
            if not workflow_files:
                print("❌ Keine Workflow-Dateien gefunden!")
                return False

            print(f"📋 Gefunden: {len(workflow_files)} Workflows")

            successful = 0
            failed = 0

            for workflow_file in workflow_files:
                print(f"\n🔄 Verarbeite: {workflow_file.name}")

                # Workflow zur Queue hinzufügen
                prompt_id = self.queue_workflow(workflow_file)
                if prompt_id:
                    # Auf Abschluss warten
                    if self.wait_for_completion(prompt_id):
                        successful += 1
                        print(f"✅ {workflow_file.name} erfolgreich verarbeitet")
                    else:
                        failed += 1
                        print(f"❌ {workflow_file.name} fehlgeschlagen")
                else:
                    failed += 1

            print("\n" + "=" * 60)
            print(f"📊 VERARBEITUNG ABGESCHLOSSEN")
            print(f"✅ Erfolgreich: {successful}")
            print(f"❌ Fehlgeschlagen: {failed}")
            print(f"📁 Ausgabe in: {self.output_dir}")

            return successful > 0

        finally:
            # Server stoppen
            self.stop_comfyui_server()

    def execute_single_workflow(self, workflow_name):
        """Führt einen einzelnen Workflow aus"""
        workflow_path = self.workflows_dir / f"{workflow_name}.json"
        if not workflow_path.exists():
            print(f"❌ Workflow nicht gefunden: {workflow_path}")
            return False

        print(f"🎮 STARTE EINZELNEN WORKFLOW: {workflow_name}")
        print("=" * 60)

        # Server starten
        if not self.start_comfyui_server():
            return False

        try:
            # Workflow ausführen
            prompt_id = self.queue_workflow(workflow_path)
            if prompt_id and self.wait_for_completion(prompt_id):
                print(f"✅ Workflow {workflow_name} erfolgreich abgeschlossen!")
                return True
            else:
                print(f"❌ Workflow {workflow_name} fehlgeschlagen!")
                return False

        finally:
            # Server stoppen
            self.stop_comfyui_server()


def main():
    """Hauptfunktion"""
    executor = SpriteWorkflowExecutor()

    if len(sys.argv) > 1:
        # Einzelnen Workflow ausführen
        workflow_name = sys.argv[1]
        executor.execute_single_workflow(workflow_name)
    else:
        # Alle Workflows ausführen
        executor.execute_all_workflows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Verarbeitung abgebrochen")
    except Exception as e:
        print(f"\n💥 Unerwarteter Fehler: {e}")
