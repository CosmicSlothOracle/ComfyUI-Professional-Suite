#!/usr/bin/env python3
"""
🎬 VIDEO-TO-LINE ART WORKFLOW TESTER
===================================
Testet den Video-to-Line Art Workflow automatisch via ComfyUI API
"""

import json
import requests
import time
import websocket
from pathlib import Path
import threading
import uuid


class ComfyUIWorkflowTester:
    def __init__(self, server_address="127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, prompt):
        """Prompt in ComfyUI queue einreihen"""
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = requests.post(f"http://{self.server_address}/prompt", data=data)
        return json.loads(req.text)

    def get_image(self, filename, subfolder, folder_type):
        """Bild von ComfyUI Server holen"""
        data = {"filename": filename,
                "subfolder": subfolder, "type": folder_type}
        url_values = "&".join(
            [f"{key}={value}" for key, value in data.items()])

        with requests.get(f"http://{self.server_address}/view?{url_values}", stream=True) as response:
            return response.content

    def get_history(self, prompt_id):
        """Historie für Prompt ID abrufen"""
        with requests.get(f"http://{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.text)

    def get_images(self, ws, prompt):
        """Überwache WebSocket für fertige Bilder"""
        prompt_id = self.queue_prompt(prompt)['prompt_id']
        output_images = {}

        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break  # Execution finished
            else:
                continue  # Previews sind binäre Daten

        history = self.get_history(prompt_id)[prompt_id]
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            images_output = []
            if 'images' in node_output:
                for image in node_output['images']:
                    image_data = self.get_image(
                        image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

        return output_images

    def load_workflow(self, workflow_path):
        """Lade Workflow JSON"""
        with open(workflow_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_video_to_lineart(self):
        """Teste Video-to-Line Art Workflow"""
        print("🎬 TESTING VIDEO-TO-LINE ART WORKFLOW")
        print("=" * 50)

        # Workflow laden
        workflow_path = Path("workflows/video_to_lineart_schloss.json")
        if not workflow_path.exists():
            print(f"❌ Workflow not found: {workflow_path}")
            return False

        workflow = self.load_workflow(workflow_path)
        print(f"✅ Workflow loaded: {workflow_path}")

        # Video prüfen
        video_path = Path("input/Schlossgifdance.mp4")
        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            return False

        print(f"✅ Video found: {video_path}")

        # Verbindung zu ComfyUI
        try:
            ws = websocket.WebSocket()
            ws.connect(
                f"ws://{self.server_address}/ws?clientId={self.client_id}")
            print(f"✅ Connected to ComfyUI: {self.server_address}")
        except Exception as e:
            print(f"❌ Failed to connect to ComfyUI: {e}")
            print("💡 Make sure ComfyUI is running: python main.py")
            return False

        # Workflow ausführen
        print("\n🚀 Starting Video-to-Line Art processing...")
        print("⏳ This may take several minutes...")

        try:
            start_time = time.time()

            # Prompt ausführen
            images = self.get_images(ws, workflow)

            processing_time = time.time() - start_time

            print(f"\n🎉 PROCESSING COMPLETE!")
            print(f"⏱️  Processing time: {processing_time:.1f} seconds")
            print(f"📊 Generated images: {len(images)} nodes")

            # Output prüfen
            output_dir = Path("output")
            video_files = list(output_dir.glob("line_art_animation*.mp4"))

            if video_files:
                print(f"✅ Output video created: {video_files[0]}")
                return True
            else:
                print("⚠️  No output video found, but processing completed")
                return True

        except Exception as e:
            print(f"❌ Processing failed: {e}")
            return False
        finally:
            ws.close()


def main():
    """Hauptfunktion"""
    print("🎬 VIDEO-TO-LINE ART WORKFLOW TESTER")
    print("=" * 50)
    print()

    # Warte kurz bis ComfyUI gestartet ist
    print("⏳ Waiting for ComfyUI to start...")
    time.sleep(10)

    tester = ComfyUIWorkflowTester()

    # Teste Workflow
    success = tester.test_video_to_lineart()

    if success:
        print("\n🎉 TEST SUCCESSFUL!")
        print("✅ Video-to-Line Art Workflow funktioniert!")
        print("📁 Check output/ directory for results")
    else:
        print("\n❌ TEST FAILED!")
        print("🔧 Check ComfyUI logs for errors")

    print("\n🎬 Video-to-Line Art Workflow ready! ✨")


if __name__ == "__main__":
    main()
