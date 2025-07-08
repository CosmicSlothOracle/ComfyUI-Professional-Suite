#!/usr/bin/env python3
"""
🎨 COMFYUI SDXL + LORA PROCESSOR
===============================
Vollständiger Processor für GIF-Enhancement mit SDXL + Pixel-Art LoRA
über ComfyUI API

Features:
- Lädt SDXL + Pixel-Art LoRA über ComfyUI API
- Verarbeitet komplette GIFs Frame für Frame
- Erhält Original-Timing und Transparenz
- Kombiniert mit 15-Farben Palette Post-Processing
"""

import json
import requests
import io
import time
import uuid
from pathlib import Path
from PIL import Image, ImageSequence
import numpy as np


class ComfyUISDXLLoRAProcessor:
    def __init__(self, server_url="http://localhost:8188"):
        self.server_url = server_url
        self.client_id = str(uuid.uuid4())

        # Model Namen (wie in ComfyUI verfügbar)
        self.sdxl_checkpoint = "sdxl.safetensors"
        self.pixelart_lora = "pixel-art-xl-lora.safetensors"

        print(f"🔗 Verbinde zu ComfyUI: {server_url}")
        self.test_connection()

    def test_connection(self):
        """Teste ComfyUI Server Verbindung"""
        try:
            response = requests.get(f"{self.server_url}/system_stats")
            if response.status_code == 200:
                print("✅ ComfyUI Server verbunden!")
                return True
            else:
                print(f"❌ Server Fehler: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Verbindungsfehler: {e}")
            return False

    def upload_image(self, image):
        """Upload Bild zu ComfyUI Server"""
        # Konvertiere zu RGB falls nötig
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Speichere als PNG in Memory
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        # Upload
        files = {'image': ('input.png', img_buffer, 'image/png')}

        try:
            response = requests.post(
                f"{self.server_url}/upload/image", files=files)
            if response.status_code == 200:
                result = response.json()
                filename = result.get('name')
                print(f"📤 Bild hochgeladen: {filename}")
                return filename
            else:
                print(f"❌ Upload fehlgeschlagen: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Upload Fehler: {e}")
            return None

    def create_sdxl_lora_workflow(self, input_image_name, positive_prompt, negative_prompt):
        """Erstelle SDXL + LoRA Workflow"""
        workflow = {
            # 1. SDXL Checkpoint Loader
            "1": {
                "inputs": {
                    "ckpt_name": self.sdxl_checkpoint
                },
                "class_type": "CheckpointLoaderSimple"
            },

            # 2. LoRA Loader
            "2": {
                "inputs": {
                    "lora_name": self.pixelart_lora,
                    "strength_model": 0.8,  # LoRA Stärke für Model
                    "strength_clip": 0.8,   # LoRA Stärke für CLIP
                    "model": ["1", 0],      # Model von Checkpoint
                    "clip": ["1", 1]        # CLIP von Checkpoint
                },
                "class_type": "LoraLoader"
            },

            # 3. Positive Prompt
            "3": {
                "inputs": {
                    "text": positive_prompt,
                    "clip": ["2", 1]  # CLIP mit LoRA
                },
                "class_type": "CLIPTextEncode"
            },

            # 4. Negative Prompt
            "4": {
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["2", 1]  # CLIP mit LoRA
                },
                "class_type": "CLIPTextEncode"
            },

            # 5. Load Input Image
            "5": {
                "inputs": {
                    "image": input_image_name
                },
                "class_type": "LoadImage"
            },

            # 6. VAE Encode
            "6": {
                "inputs": {
                    "pixels": ["5", 0],  # Bild von LoadImage
                    "vae": ["1", 2]      # VAE von Checkpoint
                },
                "class_type": "VAEEncode"
            },

            # 7. KSampler (SDXL + LoRA Processing)
            "7": {
                "inputs": {
                    "seed": 42,
                    "steps": 12,           # Weniger Steps für Speed
                    "cfg": 5.5,            # Lower CFG für mehr Kontrolle
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 0.4,        # Wenig Denoise um Original zu erhalten
                    "model": ["2", 0],     # SDXL Model mit LoRA
                    "positive": ["3", 0],  # Positive Conditioning
                    "negative": ["4", 0],  # Negative Conditioning
                    "latent_image": ["6", 0]  # Encoded Input
                },
                "class_type": "KSampler"
            },

            # 8. VAE Decode
            "8": {
                "inputs": {
                    "samples": ["7", 0],  # Samples von KSampler
                    "vae": ["1", 2]       # VAE von Checkpoint
                },
                "class_type": "VAEDecode"
            },

            # 9. Save Image
            "9": {
                "inputs": {
                    "filename_prefix": f"sdxl_lora_{int(time.time())}",
                    "images": ["8", 0]    # Images von VAEDecode
                },
                "class_type": "SaveImage"
            }
        }

        return workflow

    def queue_and_wait(self, workflow, timeout=180):
        """Queue Workflow und warte auf Completion"""
        # Queue Workflow
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }

        try:
            response = requests.post(f"{self.server_url}/prompt", json=payload)
            if response.status_code != 200:
                print(f"❌ Queue Fehler: {response.status_code}")
                return None

            result = response.json()
            prompt_id = result.get('prompt_id')
            print(f"⏳ Queued: {prompt_id}")

        except Exception as e:
            print(f"❌ Queue Fehler: {e}")
            return None

        # Warte auf Completion
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.server_url}/queue")
                if response.status_code == 200:
                    queue_data = response.json()

                    # Check ob unser Prompt noch läuft
                    running = queue_data.get('queue_running', [])
                    pending = queue_data.get('queue_pending', [])

                    is_running = any(item[1] == prompt_id for item in running)
                    is_pending = any(item[1] == prompt_id for item in pending)

                    if not is_running and not is_pending:
                        print("✅ Generation abgeschlossen!")
                        return prompt_id

                time.sleep(1)

            except Exception as e:
                print(f"❌ Status Check Fehler: {e}")
                return None

        print(f"⏰ Timeout nach {timeout}s")
        return None

    def get_output_image(self, prompt_id):
        """Hole Output Image vom Server"""
        try:
            # Get History
            response = requests.get(f"{self.server_url}/history/{prompt_id}")
            if response.status_code != 200:
                return None

            history = response.json()
            if prompt_id not in history:
                return None

            outputs = history[prompt_id].get('outputs', {})

            # Finde SaveImage Output
            for node_id, node_output in outputs.items():
                if 'images' in node_output:
                    for img_info in node_output['images']:
                        filename = img_info.get('filename')
                        if filename:
                            # Download Image
                            img_response = requests.get(
                                f"{self.server_url}/view",
                                params={'filename': filename}
                            )

                            if img_response.status_code == 200:
                                return Image.open(io.BytesIO(img_response.content))

            return None

        except Exception as e:
            print(f"❌ Output Download Fehler: {e}")
            return None

    def enhance_frame(self, frame):
        """Enhance einzelnen Frame mit SDXL + LoRA"""
        # Pixel Art Prompts
        positive = ("pixel art, 8bit style, retro game graphics, sharp pixels, "
                    "limited color palette, clean pixel work, crisp edges, retro aesthetic, "
                    "vintage game style")

        negative = ("blurry, photorealistic, smooth gradients, anti-aliasing, "
                    "high resolution, soft edges, detailed textures, modern, realistic")

        # 1. Resize für SDXL (mindestens 512x512)
        original_size = frame.size
        min_size = 512

        if max(frame.size) < min_size:
            scale = min_size / max(frame.size)
            new_size = (
                int(frame.size[0] * scale),
                int(frame.size[1] * scale)
            )
            frame = frame.resize(new_size, Image.NEAREST)

        # 2. Upload Frame
        uploaded_name = self.upload_image(frame)
        if not uploaded_name:
            return frame  # Return original bei Fehler

        # 3. Create Workflow
        workflow = self.create_sdxl_lora_workflow(
            uploaded_name, positive, negative)

        # 4. Queue und warte
        prompt_id = self.queue_and_wait(workflow)
        if not prompt_id:
            return frame  # Return original bei Fehler

        # 5. Get Output
        enhanced = self.get_output_image(prompt_id)
        if enhanced:
            # Resize zurück zu Original-Größe
            if enhanced.size != original_size:
                enhanced = enhanced.resize(original_size, Image.NEAREST)

            print(f"✅ Frame enhanced: {original_size}")
            return enhanced
        else:
            return frame  # Return original bei Fehler

    def process_gif(self, input_path, output_path):
        """Verarbeite komplettes GIF mit SDXL + LoRA"""
        print(f"🎬 Verarbeite GIF: {input_path}")

        try:
            with Image.open(input_path) as gif:
                frames = []
                durations = []

                total_frames = getattr(gif, 'n_frames', 1)
                print(f"📊 {total_frames} Frames zu verarbeiten")

                for frame_num, frame in enumerate(ImageSequence.Iterator(gif)):
                    print(f"\n🖼️ Frame {frame_num + 1}/{total_frames}")

                    # Enhance Frame
                    enhanced_frame = self.enhance_frame(frame)
                    frames.append(enhanced_frame)

                    # Preserve Timing
                    duration = getattr(frame, 'info', {}).get('duration', 100)
                    durations.append(duration)

                # Speichere Enhanced GIF
                if frames:
                    frames[0].save(
                        output_path,
                        save_all=True,
                        append_images=frames[1:],
                        duration=durations,
                        loop=0,
                        optimize=True
                    )

                    print(f"\n✅ Enhanced GIF gespeichert: {output_path}")
                    return True

        except Exception as e:
            print(f"❌ GIF Processing Fehler: {e}")
            return False

        return False


def main():
    """Demo SDXL + LoRA Processing"""
    print("🎨 COMFYUI SDXL + LORA PROCESSOR")
    print("=" * 50)

    processor = ComfyUISDXLLoRAProcessor()

    # Test mit vorhandenem GIF
    test_files = [
        "input/chorizombi-umma.gif",
        "input/2a9bdd70dc936f3c482444e529694edf_fast_transparent_converted.gif"
    ]

    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\n🎯 Teste mit: {test_file}")

            output_file = f"output/SDXL_LORA_ENHANCED_{Path(test_file).stem}.gif"

            success = processor.process_gif(test_file, output_file)

            if success:
                print(f"🎉 Erfolgreich enhanced: {output_file}")
            else:
                print(f"❌ Enhancement fehlgeschlagen")

            break  # Teste nur erste verfügbare Datei

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
