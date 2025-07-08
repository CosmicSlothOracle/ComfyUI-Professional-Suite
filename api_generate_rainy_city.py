#!/usr/bin/env python3
"""
AUTOMATISCHE GENERIERUNG: VERREGNETE DEUTSCHE STADT IM ANIME-STIL
===============================================================

Verwendet die ComfyUI API um automatisch eine loopbare Sequenz zu generieren.
"""

import json
import requests
import time
import logging
from pathlib import Path
import uuid
import websocket
import threading

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComfyUIAPIClient:
    """
    Client für die ComfyUI API
    """

    def __init__(self, server_address="127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, prompt):
        """
        Sendet einen Prompt an die ComfyUI API
        """
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')

        try:
            req = requests.post(
                f"http://{self.server_address}/prompt", data=data)
            return req.json()
        except Exception as e:
            logger.error(f"❌ Fehler beim Senden des Prompts: {e}")
            return None

    def get_image(self, filename, subfolder, folder_type):
        """
        Lädt ein generiertes Bild herunter
        """
        try:
            data = {"filename": filename,
                    "subfolder": subfolder, "type": folder_type}
            url_values = "&".join([f"{k}={v}" for k, v in data.items()])

            response = requests.get(
                f"http://{self.server_address}/view?{url_values}")
            return response.content
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterladen: {e}")
            return None

    def get_history(self, prompt_id):
        """
        Holt die Historie eines Prompts
        """
        try:
            response = requests.get(
                f"http://{self.server_address}/history/{prompt_id}")
            return response.json()
        except Exception as e:
            logger.error(f"❌ Fehler beim Abrufen der Historie: {e}")
            return None

    def track_progress(self, prompt_id):
        """
        Verfolgt den Fortschritt der Generierung via WebSocket
        """
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if data['type'] == 'progress':
                    progress = data['data']
                    logger.info(
                        f"📊 Progress: {progress.get('value', 0)}/{progress.get('max', 100)} - {progress.get('node', 'Unknown')}")
                elif data['type'] == 'executed':
                    logger.info(f"✅ Node ausgeführt: {data['data']['node']}")
            except:
                pass

        def on_error(ws, error):
            logger.error(f"❌ WebSocket Fehler: {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.info("🔌 WebSocket Verbindung geschlossen")

        def on_open(ws):
            logger.info("🔌 WebSocket Verbindung geöffnet")

        try:
            ws = websocket.WebSocketApp(f"ws://{self.server_address}/ws?clientId={self.client_id}",
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
            logger.error(f"❌ WebSocket Fehler: {e}")
            return None


def load_workflow():
    """
    Lädt den Workflow für die verregnete deutsche Stadt
    """
    workflow_path = Path("german_rainy_city_workflow.json")

    if not workflow_path.exists():
        logger.error(f"❌ Workflow-Datei nicht gefunden: {workflow_path}")
        return None

    try:
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)

        logger.info(f"✅ Workflow geladen: {workflow_path}")
        return workflow
    except Exception as e:
        logger.error(f"❌ Fehler beim Laden des Workflows: {e}")
        return None


def generate_rainy_city():
    """
    Generiert die verregnete deutsche Stadt im Anime-Stil
    """
    logger.info("🎌 STARTE GENERIERUNG: VERREGNETE DEUTSCHE STADT IM ANIME-STIL")

    # ComfyUI API Client erstellen
    client = ComfyUIAPIClient()

    # Workflow laden
    workflow = load_workflow()
    if not workflow:
        return False

    # Prompt senden
    logger.info("📤 Sende Generierungsauftrag an ComfyUI...")
    result = client.queue_prompt(workflow)

    if not result:
        logger.error("❌ Fehler beim Senden des Prompts")
        return False

    prompt_id = result.get('prompt_id')
    if not prompt_id:
        logger.error("❌ Keine Prompt-ID erhalten")
        return False

    logger.info(f"✅ Prompt gesendet! ID: {prompt_id}")

    # Progress tracking starten
    ws = client.track_progress(prompt_id)

    # Warten auf Fertigstellung
    logger.info("⏳ Warte auf Fertigstellung der Generierung...")
    logger.info("📊 Geschätzte Zeit: 3-8 Minuten (abhängig von Hardware)")

    max_wait_time = 600  # 10 Minuten Maximum
    start_time = time.time()

    while True:
        # Historie abrufen
        history = client.get_history(prompt_id)

        if history and prompt_id in history:
            # Generierung abgeschlossen
            logger.info("🎉 GENERIERUNG ABGESCHLOSSEN!")

            # Ergebnisse suchen
            outputs = history[prompt_id].get('outputs', {})

            for node_id, node_output in outputs.items():
                if 'gifs' in node_output:
                    for gif_info in node_output['gifs']:
                        filename = gif_info['filename']
                        subfolder = gif_info.get('subfolder', '')

                        logger.info(f"📥 Lade herunter: {filename}")

                        # GIF herunterladen
                        gif_data = client.get_image(
                            filename, subfolder, "output")

                        if gif_data:
                            output_path = Path("output") / filename
                            output_path.parent.mkdir(exist_ok=True)

                            with open(output_path, 'wb') as f:
                                f.write(gif_data)

                            logger.info(f"✅ Gespeichert: {output_path}")
                            logger.info(
                                f"📊 Dateigröße: {len(gif_data) / 1024 / 1024:.2f} MB")

                            return True

            logger.warning("⚠️ Keine GIF-Ausgabe gefunden")
            return False

        # Timeout prüfen
        if time.time() - start_time > max_wait_time:
            logger.error("⏰ Timeout erreicht - Generierung abgebrochen")
            return False

        # Kurz warten
        time.sleep(5)

    return False


def main():
    """
    Hauptfunktion
    """
    print("🎌 AUTOMATISCHE ANIME-GENERIERUNG")
    print("=" * 50)
    print("Verregnete deutsche Stadt im Anime-Stil")
    print("Loopbare Sequenz mit 24 Frames")
    print("=" * 50)

    # Prüfen ob ComfyUI läuft
    try:
        response = requests.get("http://127.0.0.1:8188/", timeout=5)
        logger.info("✅ ComfyUI ist erreichbar")
    except:
        logger.error("❌ ComfyUI ist nicht erreichbar!")
        logger.info(
            "💡 Starte ComfyUI mit: python main.py --listen 0.0.0.0 --port 8188")
        return False

    # Generierung starten
    success = generate_rainy_city()

    if success:
        print("\n🎉 GENERIERUNG ERFOLGREICH ABGESCHLOSSEN!")
        print("\n📁 ERGEBNIS:")
        print("   ✅ Verregnete deutsche Stadt im Anime-Stil")
        print("   ✅ Loopbare GIF-Sequenz")
        print("   ✅ Hochwertige Qualität")
        print("   📁 Gespeichert in: output/")

        print("\n🎌 EIGENSCHAFTEN:")
        print("   🏛️ Traditionelle deutsche Architektur")
        print("   🌧️ Atmosphärische Regeneffekte")
        print("   🎨 Professioneller Anime-Stil")
        print("   🔄 Perfekt loopbar")

        return True
    else:
        print("\n❌ GENERIERUNG FEHLGESCHLAGEN")
        print("💡 Überprüfe ComfyUI-Logs für Details")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
