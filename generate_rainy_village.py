#!/usr/bin/env python3
"""
AUTOMATISCHE GENERIERUNG: VERREGNETES DORF
=========================================

Generiert ein atmosphärisches GIF eines verregneten Dorfs mit ComfyUI
"""

import json
import requests
import time
import logging
import websocket
import threading
import uuid
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RainyVillageGenerator:
    """Generator für verregnete Dorf-GIFs"""

    def __init__(self, server_address="127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.generation_status = {"completed": False, "progress": 0}

    def load_workflow(self):
        """Lädt den Workflow für das verregnete Dorf"""
        workflow_path = Path("workflows/rainy_village_workflow.json")

        if not workflow_path.exists():
            logger.error(f"Workflow nicht gefunden: {workflow_path}")
            return None

        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)

            # Optimiere Workflow-Parameter für bessere Qualität
            workflow = self.optimize_workflow_parameters(workflow)

            logger.info("Workflow erfolgreich geladen und optimiert")
            return workflow

        except Exception as e:
            logger.error(f"Fehler beim Laden des Workflows: {e}")
            return None

    def optimize_workflow_parameters(self, workflow):
        """Optimiert Workflow-Parameter für höchste Qualität"""

        # Erhöhe Auflösung für bessere Details
        if "7" in workflow:
            workflow["7"]["inputs"]["width"] = 896
            workflow["7"]["inputs"]["height"] = 576
            # Mehr Frames für flüssigere Animation
            workflow["7"]["inputs"]["batch_size"] = 20

        # Verbessere Sampling-Parameter
        if "6" in workflow:
            # Mehr Steps für bessere Qualität
            workflow["6"]["inputs"]["steps"] = 30
            workflow["6"]["inputs"]["cfg"] = 7.5   # Optimierter CFG-Wert
            workflow["6"]["inputs"]["sampler_name"] = "dpmpp_2m"
            workflow["6"]["inputs"]["scheduler"] = "karras"

        # Optimiere GIF-Ausgabe
        if "9" in workflow:
            workflow["9"]["inputs"]["fps"] = 12    # Flüssigere Animation
            workflow["9"]["inputs"]["filename_prefix"] = "atmospheric_rainy_village"

        # Verbessere Prompt für atmosphärische Qualität
        if "2" in workflow:
            enhanced_prompt = (
                "masterpiece, best quality, ultra detailed, cinematic masterpiece, "
                "(atmospheric rainy village:1.4), (medieval german architecture:1.3), "
                "cobblestone streets glistening with rain, (dramatic rain droplets:1.3), "
                "(realistic water puddles reflecting warm lights:1.2), "
                "overcast stormy sky with dark clouds, traditional timber frame houses, "
                "gothic church spire silhouette, (warm glowing windows:1.2), "
                "vintage street lamps casting soft light, heavy mist and fog, "
                "(cinematic lighting:1.2), dramatic weather atmosphere, "
                "photorealistic, 8k ultra resolution, professional photography, "
                "moody color grading, depth of field, volumetric lighting"
            )
            workflow["2"]["inputs"]["text"] = enhanced_prompt

        logger.info("Workflow-Parameter für maximale Qualität optimiert")
        return workflow

    def queue_prompt(self, workflow):
        """Sendet den Workflow an ComfyUI"""
        try:
            prompt_data = {"prompt": workflow, "client_id": self.client_id}
            data = json.dumps(prompt_data).encode('utf-8')

            response = requests.post(
                f"http://{self.server_address}/prompt", data=data)
            return response.json()

        except Exception as e:
            logger.error(f"Fehler beim Senden des Prompts: {e}")
            return None

    def setup_websocket_monitoring(self):
        """Richtet WebSocket-Monitoring für Fortschrittsverfolgung ein"""

        def on_message(ws, message):
            try:
                data = json.loads(message)

                if data['type'] == 'progress':
                    progress = data['data']
                    value = progress.get('value', 0)
                    maximum = progress.get('max', 100)
                    node = progress.get('node', 'Unknown')

                    percentage = (value / maximum * 100) if maximum > 0 else 0
                    self.generation_status["progress"] = percentage

                    logger.info(
                        f"Fortschritt: {percentage:.1f}% - Node: {node} ({value}/{maximum})")

                elif data['type'] == 'executed':
                    node_id = data['data']['node']
                    logger.info(f"Node abgeschlossen: {node_id}")

                elif data['type'] == 'execution_complete':
                    self.generation_status["completed"] = True
                    logger.info("Generierung abgeschlossen!")

            except Exception as e:
                logger.debug(f"WebSocket-Nachricht-Fehler: {e}")

        def on_error(ws, error):
            logger.warning(f"WebSocket-Fehler: {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket-Verbindung geschlossen")

        def on_open(ws):
            logger.info("WebSocket-Verbindung geöffnet - Monitoring aktiv")

        try:
            ws_url = f"ws://{self.server_address}/ws?clientId={self.client_id}"
            ws = websocket.WebSocketApp(ws_url,
                                        on_message=on_message,
                                        on_error=on_error,
                                        on_close=on_close,
                                        on_open=on_open)

            # WebSocket in separatem Thread starten
            wst = threading.Thread(target=ws.run_forever)
            wst.daemon = True
            wst.start()

            return ws

        except Exception as e:
            logger.error(f"WebSocket-Setup fehlgeschlagen: {e}")
            return None

    def wait_for_completion(self, prompt_id, max_wait_time=600):
        """Wartet auf Abschluss der Generierung"""

        start_time = time.time()
        last_progress = -1

        logger.info("Warte auf Abschluss der Generierung...")
        logger.info("Geschätzte Zeit: 5-15 Minuten (abhängig von Hardware)")

        while time.time() - start_time < max_wait_time:
            # Prüfe Historie
            try:
                response = requests.get(
                    f"http://{self.server_address}/history/{prompt_id}")
                history = response.json()

                if prompt_id in history:
                    logger.info("Generierung erfolgreich abgeschlossen!")
                    return history[prompt_id]

            except Exception as e:
                logger.debug(f"Historie-Abfrage-Fehler: {e}")

            # Zeige Fortschritt an
            current_progress = self.generation_status["progress"]
            if current_progress != last_progress:
                logger.info(f"Gesamtfortschritt: {current_progress:.1f}%")
                last_progress = current_progress

            time.sleep(5)  # Warte 5 Sekunden

        logger.error(f"Zeitüberschreitung nach {max_wait_time} Sekunden")
        return None

    def download_results(self, history_data):
        """Lädt die generierten GIFs herunter"""

        downloaded_files = []

        try:
            outputs = history_data.get('outputs', {})

            for node_id, node_output in outputs.items():
                # Suche nach GIF-Ausgaben
                if 'gifs' in node_output:
                    for gif_info in node_output['gifs']:
                        filename = gif_info['filename']
                        subfolder = gif_info.get('subfolder', '')

                        logger.info(f"Lade herunter: {filename}")

                        # Download-URL erstellen
                        params = {
                            'filename': filename,
                            'subfolder': subfolder,
                            'type': 'output'
                        }
                        url_params = "&".join(
                            [f"{k}={v}" for k, v in params.items()])
                        download_url = f"http://{self.server_address}/view?{url_params}"

                        # Datei herunterladen
                        response = requests.get(download_url)

                        if response.status_code == 200:
                            output_path = Path("output") / filename
                            output_path.parent.mkdir(exist_ok=True)

                            with open(output_path, 'wb') as f:
                                f.write(response.content)

                            downloaded_files.append(str(output_path))

                            # Dateigröße anzeigen
                            file_size = len(response.content) / (1024 * 1024)
                            logger.info(
                                f"Gespeichert: {output_path} ({file_size:.1f} MB)")
                        else:
                            logger.error(
                                f"Download fehlgeschlagen: {filename}")

                # Suche nach Bildern
                elif 'images' in node_output:
                    for img_info in node_output['images']:
                        filename = img_info['filename']
                        logger.info(f"Bild generiert: {filename}")

        except Exception as e:
            logger.error(f"Download-Fehler: {e}")

        return downloaded_files

    def generate_rainy_village(self):
        """Hauptfunktion für die Generierung"""

        logger.info("=" * 60)
        logger.info("VERREGNETES DORF - GENERIERUNG GESTARTET")
        logger.info("=" * 60)

        # 1. Workflow laden
        workflow = self.load_workflow()
        if not workflow:
            return False

        # 2. WebSocket-Monitoring einrichten
        ws = self.setup_websocket_monitoring()

        # 3. Generierung starten
        logger.info("Sende Generierungsauftrag an ComfyUI...")
        result = self.queue_prompt(workflow)

        if not result or 'prompt_id' not in result:
            logger.error("Fehler beim Starten der Generierung")
            return False

        prompt_id = result['prompt_id']
        logger.info(f"Generierung gestartet! Prompt-ID: {prompt_id}")

        # 4. Auf Abschluss warten
        history_data = self.wait_for_completion(prompt_id)

        if not history_data:
            logger.error("Generierung fehlgeschlagen oder Zeitüberschreitung")
            return False

        # 5. Ergebnisse herunterladen
        logger.info("Lade Ergebnisse herunter...")
        downloaded_files = self.download_results(history_data)

        if downloaded_files:
            logger.info("=" * 60)
            logger.info("GENERIERUNG ERFOLGREICH ABGESCHLOSSEN!")
            logger.info("=" * 60)
            logger.info("Generierte Dateien:")
            for file_path in downloaded_files:
                logger.info(f"  - {file_path}")
            logger.info("=" * 60)
            return True
        else:
            logger.error("Keine Dateien heruntergeladen")
            return False


def main():
    """Hauptfunktion"""

    print("🌧️  VERREGNETES DORF - GIF GENERATOR")
    print("====================================")

    generator = RainyVillageGenerator()
    success = generator.generate_rainy_village()

    if success:
        print("\n✅ Erfolgreich! Das verregnete Dorf-GIF wurde generiert.")
        print("📁 Überprüfen Sie den 'output'-Ordner für die Ergebnisse.")
    else:
        print("\n❌ Generierung fehlgeschlagen. Überprüfen Sie die Logs.")

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
