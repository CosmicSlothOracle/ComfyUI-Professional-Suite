#!/usr/bin/env python3
"""
Test ComfyUI API Connection and Model Availability
"""

import requests
import json


def test_api_connection():
    """Test basic API connection"""
    try:
        response = requests.get("http://localhost:8188/system_stats")
        if response.status_code == 200:
            data = response.json()
            print("✅ ComfyUI API Connection: OK")
            print(f"   ComfyUI Version: {data['system']['comfyui_version']}")
            print(f"   Python Version: {data['system']['python_version']}")
            print(f"   PyTorch Version: {data['system']['pytorch_version']}")
            return True
        else:
            print(f"❌ API Connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Connection error: {e}")
        return False


def check_models():
    """Check available models"""
    try:
        # Check checkpoints
        response = requests.get("http://localhost:8188/object_info")
        if response.status_code == 200:
            data = response.json()

            # Find CheckpointLoaderSimple node info
            checkpoint_node = data.get("CheckpointLoaderSimple", {})
            if checkpoint_node:
                inputs = checkpoint_node.get("input", {})
                required_inputs = inputs.get("required", {})
                ckpt_info = required_inputs.get("ckpt_name", {})

                if isinstance(ckpt_info, list) and len(ckpt_info) > 0:
                    available_checkpoints = ckpt_info[0]
                    print(
                        f"\n📁 Available Checkpoints ({len(available_checkpoints)}):")
                    for ckpt in available_checkpoints:
                        print(f"   - {ckpt}")
                        if "sdxl" in ckpt.lower():
                            print(f"     ✅ SDXL Model found!")

            # Find LoraLoader node info
            lora_node = data.get("LoraLoader", {})
            if lora_node:
                inputs = lora_node.get("input", {})
                required_inputs = inputs.get("required", {})
                lora_info = required_inputs.get("lora_name", {})

                if isinstance(lora_info, list) and len(lora_info) > 0:
                    available_loras = lora_info[0]
                    print(f"\n📁 Available LoRAs ({len(available_loras)}):")
                    for lora in available_loras:
                        print(f"   - {lora}")
                        if "pixel" in lora.lower():
                            print(f"     ✅ Pixel-Art LoRA found!")

            return True
        else:
            print(f"❌ Model check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model check error: {e}")
        return False


def test_queue():
    """Test basic queue functionality"""
    try:
        response = requests.get("http://localhost:8188/queue")
        if response.status_code == 200:
            data = response.json()
            running = data.get("queue_running", [])
            pending = data.get("queue_pending", [])

            print(f"\n🔄 Queue Status:")
            print(f"   Running: {len(running)}")
            print(f"   Pending: {len(pending)}")

            return True
        else:
            print(f"❌ Queue check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Queue check error: {e}")
        return False


def main():
    print("🧪 COMFYUI API TEST")
    print("=" * 40)

    # Test API connection
    if not test_api_connection():
        return

    # Check available models
    if not check_models():
        return

    # Test queue
    if not test_queue():
        return

    print("\n✅ All API tests passed!")
    print("🚀 Ready for SDXL + LoRA integration!")


if __name__ == "__main__":
    main()
