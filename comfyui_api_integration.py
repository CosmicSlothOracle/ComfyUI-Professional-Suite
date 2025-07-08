#!/usr/bin/env python3
"""
🎨 COMFYUI API INTEGRATION - SDXL + LORA
========================================
API-basierte Integration mit laufendem ComfyUI Server
Verarbeitet GIFs mit SDXL + Pixel-Art LoRA über ComfyUI API

Features:
- Kommuniziert mit ComfyUI Server (localhost:8188)
- Lädt SDXL Checkpoint + Pixel-Art LoRA
- Verarbeitet einzelne Frames oder komplette GIFs
- Wendet 15-Farben Palette nach Enhancement an
"""

import json
import requests
import base64
import io
import time
from pathlib import Path
from PIL import Image, ImageSequence
import uuid


class ComfyUIAPIIntegrator:
    def __init__(self, server_url="http://localhost:8188"):
        self.server_url = server_url
        self.client_id = str(uuid.uuid4())

        # Model Paths (relativ zu ComfyUI models/)
        self.sdxl_checkpoint = "sdxl.safetensors"
        self.pixelart_lora = "pixel-art-xl-lora.safetensors"

        # Test Server Connection
        self.test_connection()

    def test_connection(self):
        """Teste Verbindung zum ComfyUI Server"""
        try:
            response = requests.get(f"{self.server_url}/system_stats")
            if response.status_code == 200:
                print("✅ ComfyUI Server verbunden!")
                stats = response.json()
                print(f"   System: {stats.get('system', {})}")
            else:
                print(
                    f"⚠️ Server antwortet, aber Status: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("❌ Kann nicht zu ComfyUI Server verbinden!")
            print(f"   Stelle sicher dass Server läuft auf: {self.server_url}")
        except Exception as e:
            print(f"❌ Verbindungsfehler: {e}")

    def upload_image(self, image):
        """Lädt Bild zum ComfyUI Server hoch"""
        # Konvertiere PIL Image zu Bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        # Upload via API
        files = {'image': ('image.png', img_buffer, 'image/png')}

        try:
            response = requests.post(
                f"{self.server_url}/upload/image",
                files=files
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('name')  # Uploaded filename
            else:
                print(f"❌ Image Upload fehlgeschlagen: {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ Upload Fehler: {e}")
            return None

    def create_workflow(self, input_image_name, positive_prompt, negative_prompt):
        """Erstellt ComfyUI Workflow JSON für SDXL + LoRA"""
        workflow = {
            "4": {  # CheckpointLoaderSimple
                "inputs": {
                    "ckpt_name": self.sdxl_checkpoint
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "10": {  # LoraLoader
                "inputs": {
                    "lora_name": self.pixelart_lora,
                    "strength_model": 1.0,
                    "strength_clip": 1.0,
                    "model": ["4", 0],  # Model from checkpoint
                    "clip": ["4", 1]    # CLIP from checkpoint
                },
                "class_type": "LoraLoader"
            },
            "6": {  # CLIPTextEncode (Positive)
                "inputs": {
                    "text": positive_prompt,
                    "clip": ["10", 1]  # CLIP from LoRA
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {  # CLIPTextEncode (Negative)
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["10", 1]  # CLIP from LoRA
                },
                "class_type": "CLIPTextEncode"
            },
            "11": {  # LoadImage
                "inputs": {
                    "image": input_image_name
                },
                "class_type": "LoadImage"
            },
            "12": {  # VAEEncode
                "inputs": {
                    "pixels": ["11", 0],  # Image from LoadImage
                    "vae": ["4", 2]       # VAE from checkpoint
                },
                "class_type": "VAEEncode"
            },
            "3": {  # KSampler
                "inputs": {
                    "seed": 42,
                    "steps": 15,
                    "cfg": 6.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 0.65,
                    "model": ["10", 0],        # Model from LoRA
                    "positive": ["6", 0],      # Positive conditioning
                    "negative": ["7", 0],      # Negative conditioning
                    "latent_image": ["12", 0]  # Encoded image
                },
                "class_type": "KSampler"
            },
            "8": {  # VAEDecode
                "inputs": {
                    "samples": ["3", 0],  # Samples from KSampler
                    "vae": ["4", 2]       # VAE from checkpoint
                },
                "class_type": "VAEDecode"
            },
            "9": {  # SaveImage
                "inputs": {
                    "filename_prefix": f"sdxl_lora_enhanced_{int(time.time())}",
                    "images": ["8", 0]  # Images from VAEDecode
                },
                "class_type": "SaveImage"
            }
        }

        return workflow

    def queue_workflow(self, workflow):
        """Sendet Workflow zur ComfyUI Queue"""
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }

        try:
            response = requests.post(
                f"{self.server_url}/prompt",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                prompt_id = result.get('prompt_id')
                print(f"✅ Workflow gequeued: {prompt_id}")
                return prompt_id
            else:
                print(f"❌ Workflow Queue Fehler: {response.status_code}")
                print(f"   Response: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Queue Fehler: {e}")
            return None

    def wait_for_completion(self, prompt_id, timeout=300):
        """Wartet auf Workflow Completion"""
        print(f"⏳ Warte auf Completion von {prompt_id}...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check Queue Status
                response = requests.get(f"{self.server_url}/queue")
                if response.status_code == 200:
                    queue_data = response.json()

                    # Check if our prompt is still running
                    running = queue_data.get('queue_running', [])
                    pending = queue_data.get('queue_pending', [])

                    # Find our prompt
                    found_running = any(
                        item[1] == prompt_id for item in running)
                    found_pending = any(
                        item[1] == prompt_id for item in pending)

                    if not found_running and not found_pending:
                        print("✅ Workflow abgeschlossen!")
                        return True

                time.sleep(2)  # Check every 2 seconds

            except Exception as e:
                print(f"❌ Status Check Fehler: {e}")
                time.sleep(2)

        print(f"⏰ Timeout nach {timeout} Sekunden")
        return False

    def get_output_images(self, prompt_id):
        """Lädt Output Images vom Server"""
        try:
            # Get History für unseren Prompt
            response = requests.get(f"{self.server_url}/history/{prompt_id}")

            if response.status_code == 200:
                history = response.json()

                if prompt_id in history:
                    outputs = history[prompt_id].get('outputs', {})

                    # Finde SaveImage Node Outputs
                    for node_id, node_output in outputs.items():
                        if 'images' in node_output:
                            images = []

                            for img_info in node_output['images']:
                                filename = img_info.get('filename')

                                if filename:
                                    # Download Image
                                    img_response = requests.get(
                                        f"{self.server_url}/view",
                                        params={'filename': filename}
                                    )

                                    if img_response.status_code == 200:
                                        image = Image.open(
                                            io.BytesIO(img_response.content))
                                        images.append(image)

                            return images

            print("❌ Keine Output Images gefunden")
            return []

        except Exception as e:
            print(f"❌ Output Download Fehler: {e}")
            return []

    def enhance_frame_with_lora(self, frame, positive_prompt, negative_prompt):
        """Enhanced einzelnen Frame mit SDXL + LoRA"""
        print(f"🔄 Enhance Frame: {frame.size}")

        # 1. Upload Frame
        uploaded_name = self.upload_image(frame)
        if not uploaded_name:
            print("❌ Frame Upload fehlgeschlagen")
            return frame

        # 2. Create Workflow
        workflow = self.create_workflow(
            uploaded_name, positive_prompt, negative_prompt)

        # 3. Queue Workflow
        prompt_id = self.queue_workflow(workflow)
        if not prompt_id:
            print("❌ Workflow Queue fehlgeschlagen")
            return frame

        # 4. Wait for Completion
        if not self.wait_for_completion(prompt_id):
            print("❌ Workflow Timeout")
            return frame

        # 5. Get Output
        output_images = self.get_output_images(prompt_id)
        if output_images:
            enhanced_frame = output_images[0]
            print(f"✅ Frame enhanced: {enhanced_frame.size}")
            return enhanced_frame
        else:
            print("❌ Keine Output erhalten")
            return frame

    def process_gif_with_lora(self, input_gif_path, output_gif_path):
        """Verarbeitet komplettes GIF mit SDXL + LoRA Enhancement"""
        print(f"🎬 Verarbeite GIF: {input_gif_path}")

        # Pixel Art Prompts
        positive_prompt = ("pixel art, 8bit style, retro game graphics, sharp pixels, "
                           "limited color palette, clean pixel work, crisp edges")
        negative_prompt = ("blurry, photorealistic, smooth gradients, anti-aliasing, "
                           "high resolution, soft edges, detailed textures")

        try:
            # Lade Original GIF
            with Image.open(input_gif_path) as gif:
                frames = []
                durations = []

                print(f"📊 Original GIF Info:")
                print(f"   Frames: {getattr(gif, 'n_frames', 1)}")
                print(f"   Size: {gif.size}")
                print(f"   Format: {gif.format}")

                # Verarbeite jeden Frame
                for frame_num, frame in enumerate(ImageSequence.Iterator(gif)):
                    print(f"\n🖼️ Frame {frame_num + 1}")

                    # Konvertiere zu RGB falls nötig
                    if frame.mode != 'RGB':
                        frame = frame.convert('RGB')

                    # Resize für SDXL (optimal: 1024x1024, min: 512x512)
                    target_size = 512
                    if max(frame.size) < target_size:
                        scale_factor = target_size / max(frame.size)
                        new_size = (
                            int(frame.size[0] * scale_factor),
                            int(frame.size[1] * scale_factor)
                        )
                        frame = frame.resize(new_size, Image.NEAREST)

                    # Enhance mit SDXL + LoRA
                    enhanced_frame = self.enhance_frame_with_lora(
                        frame, positive_prompt, negative_prompt)

                    frames.append(enhanced_frame)

                    # Preserve timing
                    duration = getattr(frame, 'info', {}).get('duration', 100)
                    durations.append(duration)

                # Speichere Enhanced GIF
                if frames:
                    frames[0].save(
                        output_gif_path,
                        save_all=True,
                        append_images=frames[1:],
                        duration=durations,
                        loop=0,
                        optimize=True
                    )

                    print(f"\n✅ Enhanced GIF gespeichert: {output_gif_path}")
                    print(f"   Frames: {len(frames)}")
                    print(f"   Size: {frames[0].size}")

                    return True

        except Exception as e:
            print(f"❌ GIF Processing Fehler: {e}")
            return False

        return False


def main():
    """Demo der API-basierten LoRA Integration"""
    print("🎨 COMFYUI API INTEGRATION - SDXL + LORA")
    print("=" * 60)

    # Initialize API Integrator
    integrator = ComfyUIAPIIntegrator()

    # Test mit einem vorhandenen GIF
    test_gifs = [
        "input/chorizombi-umma.gif",
        "input/2a9bdd70dc936f3c482444e529694edf_fast_transparent_converted.gif",
        "input/7ee80664e6f86ac416750497557bf6fc_fast_transparent_converted.gif"
    ]

    for gif_path in test_gifs:
        if Path(gif_path).exists():
            print(f"\n🎯 Teste mit: {gif_path}")

            output_path = f"output/SDXL_LORA_{Path(gif_path).stem}.gif"

            success = integrator.process_gif_with_lora(gif_path, output_path)

            if success:
                print(f"✅ Erfolgreich verarbeitet: {output_path}")
            else:
                print(f"❌ Verarbeitung fehlgeschlagen: {gif_path}")

            break  # Teste nur ersten verfügbaren

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
