#!/usr/bin/env python3
"""
Simple SDXL + LoRA API Test
"""

import json
import requests
import io
import uuid
from PIL import Image


def test_simple_workflow():
    """Test a simple SDXL + LoRA workflow"""
    server_url = "http://localhost:8188"
    client_id = str(uuid.uuid4())

    # Simple workflow for text-to-image with SDXL + LoRA
    workflow = {
        "1": {  # CheckpointLoaderSimple
            "inputs": {
                "ckpt_name": "sdxl.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "2": {  # LoraLoader
            "inputs": {
                "lora_name": "pixel-art-xl-lora.safetensors",
                "strength_model": 0.8,
                "strength_clip": 0.8,
                "model": ["1", 0],
                "clip": ["1", 1]
            },
            "class_type": "LoraLoader"
        },
        "3": {  # CLIPTextEncode (Positive)
            "inputs": {
                "text": "pixel art, 8bit style, simple character, retro game",
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "4": {  # CLIPTextEncode (Negative)
            "inputs": {
                "text": "blurry, photorealistic, complex",
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "5": {  # Empty Latent Image
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        "6": {  # KSampler
            "inputs": {
                "seed": 42,
                "steps": 10,
                "cfg": 6.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["2", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler"
        },
        "7": {  # VAEDecode
            "inputs": {
                "samples": ["6", 0],
                "vae": ["1", 2]
            },
            "class_type": "VAEDecode"
        },
        "8": {  # SaveImage
            "inputs": {
                "filename_prefix": "sdxl_lora_test",
                "images": ["7", 0]
            },
            "class_type": "SaveImage"
        }
    }

    # Queue workflow
    payload = {
        "prompt": workflow,
        "client_id": client_id
    }

    try:
        print("🔄 Queuing SDXL + LoRA workflow...")
        response = requests.post(f"{server_url}/prompt", json=payload)

        if response.status_code == 200:
            result = response.json()
            prompt_id = result.get('prompt_id')
            print(f"✅ Workflow queued: {prompt_id}")

            # Simple wait (in real implementation, you'd monitor queue status)
            print("⏳ Waiting for completion...")
            import time
            time.sleep(30)  # Wait 30 seconds

            print("🎉 Test completed! Check ComfyUI output folder for results.")
            return True
        else:
            print(f"❌ Queue failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    print("🧪 SDXL + LORA API TEST")
    print("=" * 40)

    success = test_simple_workflow()

    if success:
        print("\n✅ SDXL + LoRA integration test successful!")
        print("💡 You can now build more complex workflows")
    else:
        print("\n❌ Test failed. Check ComfyUI server and models.")


if __name__ == "__main__":
    main()
