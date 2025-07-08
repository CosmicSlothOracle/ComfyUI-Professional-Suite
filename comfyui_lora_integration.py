#!/usr/bin/env python3
"""
🎨 COMFYUI SDXL + LORA INTEGRATION
=================================
Echte Integration von SDXL Checkpoint und Pixel-Art LoRA in ComfyUI

Dieser Script zeigt, wie die LoRA-Integration funktioniert:
1. Lädt SDXL Checkpoint
2. Lädt Pixel-Art LoRA
3. Wendet LoRA auf Model an
4. Verwendet Enhanced Model für Processing
"""

import sys
import os
from pathlib import Path

# Füge ComfyUI zum Python Path hinzu
comfyui_path = Path(__file__).parent
if str(comfyui_path) not in sys.path:
    sys.path.insert(0, str(comfyui_path))

try:
    # ComfyUI Core Imports
    import folder_paths
    import comfy.model_management as model_management
    import comfy.sd as sd
    import comfy.utils as utils
    from comfy.model_loader import load_lora_for_models

    # Nodes
    from nodes import CheckpointLoaderSimple, LoraLoader, CLIPTextEncode, KSampler, VAEDecode, VAEEncode

    COMFYUI_AVAILABLE = True
    print("✅ ComfyUI modules erfolgreich geladen!")

except ImportError as e:
    COMFYUI_AVAILABLE = False
    print(f"❌ ComfyUI Import Fehler: {e}")
    print("💡 Stelle sicher, dass ComfyUI korrekt installiert ist")


class ComfyUILoraIntegrator:
    def __init__(self):
        self.sdxl_checkpoint_path = "sdxl.safetensors"
        self.pixelart_lora_path = "pixel-art-xl-lora.safetensors"

        # Model States
        self.model = None
        self.clip = None
        self.vae = None
        self.lora_loaded = False

    def load_sdxl_checkpoint(self):
        """Lädt SDXL Checkpoint"""
        if not COMFYUI_AVAILABLE:
            print("❌ ComfyUI nicht verfügbar für Checkpoint-Loading")
            return False

        try:
            print(f"🔄 Lade SDXL Checkpoint: {self.sdxl_checkpoint_path}")

            # CheckpointLoaderSimple Node
            checkpoint_loader = CheckpointLoaderSimple()
            result = checkpoint_loader.load_checkpoint(
                self.sdxl_checkpoint_path)

            self.model, self.clip, self.vae = result

            print("✅ SDXL Checkpoint erfolgreich geladen!")
            print(f"   Model: {type(self.model)}")
            print(f"   CLIP: {type(self.clip)}")
            print(f"   VAE: {type(self.vae)}")

            return True

        except Exception as e:
            print(f"❌ Fehler beim Laden des SDXL Checkpoints: {e}")
            return False

    def load_pixelart_lora(self):
        """Lädt und wendet Pixel-Art LoRA an"""
        if not COMFYUI_AVAILABLE or self.model is None:
            print("❌ SDXL Model muss zuerst geladen werden")
            return False

        try:
            print(f"🔄 Lade Pixel-Art LoRA: {self.pixelart_lora_path}")

            # LoraLoader Node
            lora_loader = LoraLoader()

            # Lade LoRA mit Stärke 1.0 für Model und CLIP
            result = lora_loader.load_lora(
                model=self.model,
                clip=self.clip,
                lora_name=self.pixelart_lora_path,
                strength_model=1.0,
                strength_clip=1.0
            )

            self.model, self.clip = result
            self.lora_loaded = True

            print("✅ Pixel-Art LoRA erfolgreich angewendet!")
            print("🎨 Model ist jetzt für Pixel-Art optimiert")

            return True

        except Exception as e:
            print(f"❌ Fehler beim Laden der LoRA: {e}")
            return False

    def encode_prompts(self, positive_prompt, negative_prompt):
        """Enkodiert Prompts mit CLIP"""
        if not COMFYUI_AVAILABLE or self.clip is None:
            print("❌ CLIP Model nicht verfügbar")
            return None, None

        try:
            print("🔄 Enkodiere Prompts...")

            # CLIPTextEncode Nodes
            clip_encoder = CLIPTextEncode()

            positive_conditioning = clip_encoder.encode(
                self.clip, positive_prompt)
            negative_conditioning = clip_encoder.encode(
                self.clip, negative_prompt)

            print(f"✅ Prompts enkodiert:")
            print(f"   Positive: \"{positive_prompt[:50]}...\"")
            print(f"   Negative: \"{negative_prompt[:50]}...\"")

            return positive_conditioning[0], negative_conditioning[0]

        except Exception as e:
            print(f"❌ Fehler beim Enkodieren der Prompts: {e}")
            return None, None

    def generate_with_lora(self, input_image, positive_prompt, negative_prompt):
        """Generiert Enhanced Version mit SDXL + LoRA"""
        if not COMFYUI_AVAILABLE:
            print("❌ ComfyUI nicht verfügbar für Generation")
            return None

        if not self.lora_loaded:
            print("❌ LoRA muss zuerst geladen werden")
            return None

        try:
            print("🎬 Starte SDXL + LoRA Generation...")

            # Encode Prompts
            positive_cond, negative_cond = self.encode_prompts(
                positive_prompt, negative_prompt)
            if positive_cond is None:
                return None

            # VAE Encode - konvertiere PIL Image zu latent
            vae_encoder = VAEEncode()

            # Konvertiere PIL zu Tensor (ComfyUI Format)
            import torch
            import numpy as np

            # PIL -> Numpy -> Tensor
            img_array = np.array(input_image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(
                img_array).unsqueeze(0)  # Batch dimension

            latents = vae_encoder.encode(self.vae, img_tensor)[0]

            # KSampler - SDXL + LoRA Processing
            sampler = KSampler()

            processed_latents = sampler.sample(
                model=self.model,
                seed=42,
                steps=20,
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                positive=positive_cond,
                negative=negative_cond,
                latent_image=latents,
                denoise=0.75  # Weniger Denoise um Original-Struktur zu erhalten
            )[0]

            # VAE Decode
            vae_decoder = VAEDecode()
            output_tensor = vae_decoder.decode(self.vae, processed_latents)[0]

            # Tensor -> PIL
            output_array = output_tensor.squeeze(0).cpu().numpy()
            output_array = (output_array * 255).astype(np.uint8)

            from PIL import Image
            output_image = Image.fromarray(output_array)

            print("✅ SDXL + LoRA Generation abgeschlossen!")

            return output_image

        except Exception as e:
            print(f"❌ Fehler bei der Generation: {e}")
            return None

    def setup_complete_pipeline(self):
        """Lädt komplette SDXL + LoRA Pipeline"""
        print("🚀 Setup SDXL + LoRA Pipeline...")
        print("=" * 50)

        # Schritt 1: SDXL Checkpoint laden
        if not self.load_sdxl_checkpoint():
            return False

        # Schritt 2: LoRA laden und anwenden
        if not self.load_pixelart_lora():
            return False

        print("=" * 50)
        print("🎯 SDXL + LoRA Pipeline bereit!")
        print("💫 Kann jetzt Enhanced Pixel Art generieren")

        return True


def main():
    """Demo der LoRA Integration"""
    print("🎨 COMFYUI SDXL + LORA INTEGRATION DEMO")
    print("=" * 60)

    integrator = ComfyUILoraIntegrator()

    # Setup Pipeline
    success = integrator.setup_complete_pipeline()

    if success:
        print("\\n🎯 Pipeline Setup erfolgreich!")
        print("💡 Jetzt können GIFs mit SDXL + LoRA verarbeitet werden")

        # Beispiel Prompts
        positive = "pixel art, 8bit style, retro game graphics, sharp pixels, limited color palette"
        negative = "blurry, photorealistic, smooth gradients, high resolution"

        print(f"\\n📝 Beispiel Prompts:")
        print(f"   Positive: {positive}")
        print(f"   Negative: {negative}")

    else:
        print("\\n❌ Pipeline Setup fehlgeschlagen")
        print("💡 Prüfe SDXL und LoRA Dateien in models/checkpoints/")

    print("\\n" + "=" * 60)


if __name__ == "__main__":
    main()
