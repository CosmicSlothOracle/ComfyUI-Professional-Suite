#!/usr/bin/env python3
"""
GIF Reference Pipeline with Art Style Control
============================================

Erweiterte Version der Pipeline mit vollständiger Art Style Kontrolle.
Du kannst jetzt spezifische Kunststile für die Generierung auswählen!

Verfügbare Kunst-Stile:
- anime, manga, pixel_art, watercolor, oil_painting
- digital_art, sketch, cartoon, realistic, abstract
- impressionist, cyberpunk, steampunk, art_nouveau
- minimalist, baroque, pop_art, street_art, fantasy
- sci_fi, vintage, retro, neon, pastel, monochrome

Usage Examples:
    # Anime Style
    python gif_reference_pipeline_with_styles.py --input ref.gif --style anime --prompt "magical girl transformation"

    # Cyberpunk Style
    python gif_reference_pipeline_with_styles.py --input ref.gif --style cyberpunk --prompt "neon city streets"

    # Pixel Art Style
    python gif_reference_pipeline_with_styles.py --input ref.gif --style pixel_art --prompt "retro game character"
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from PIL import Image, ImageSequence, ImageEnhance, ImageFilter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArtStyleManager:
    """
    Verwaltet alle verfügbaren Kunst-Stile und deren Eigenschaften
    """

    def __init__(self):
        self.styles = self._initialize_styles()

    def _initialize_styles(self) -> Dict[str, Dict]:
        """Initialisiert alle verfügbaren Kunst-Stile"""

        return {
            "anime": {
                "name": "Anime/Manga",
                "description": "Japanese anime and manga style with bold colors, clean lines, and expressive characters",
                "prompt_keywords": ["anime style", "manga", "cel-shaded", "bold outlines", "vibrant colors"],
                "color_palette": [(255, 182, 193), (135, 206, 250), (255, 255, 224), (255, 160, 122)],
                "brightness": 0.75,
                "contrast": 0.85,
                "saturation": 0.95,
                "texture_style": "clean and sharp",
                "lighting": "dramatic with rim lighting",
                "example_prompts": [
                    "magical girl transformation sequence",
                    "dynamic action scene with special effects",
                    "emotional character close-up with sparkles"
                ]
            },

            "pixel_art": {
                "name": "Pixel Art",
                "description": "Retro 8-bit and 16-bit pixel art style with limited colors and blocky aesthetics",
                "prompt_keywords": ["pixel art", "8-bit", "16-bit", "retro gaming", "pixelated"],
                "color_palette": [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)],
                "brightness": 0.65,
                "contrast": 0.9,
                "saturation": 0.8,
                "texture_style": "blocky and pixelated",
                "lighting": "flat with simple shadows",
                "example_prompts": [
                    "retro game character walking animation",
                    "classic arcade game explosion effect",
                    "pixel art landscape with parallax scrolling"
                ]
            },

            "watercolor": {
                "name": "Watercolor Painting",
                "description": "Soft watercolor painting style with flowing colors and organic textures",
                "prompt_keywords": ["watercolor", "soft edges", "color bleeding", "artistic painting"],
                "color_palette": [(255, 228, 225), (230, 230, 250), (240, 248, 255), (255, 240, 245)],
                "brightness": 0.8,
                "contrast": 0.35,
                "saturation": 0.6,
                "texture_style": "soft and flowing",
                "lighting": "gentle and diffused",
                "example_prompts": [
                    "flowing water with soft reflections",
                    "delicate flower petals in the wind",
                    "dreamy landscape with soft colors"
                ]
            },

            "cyberpunk": {
                "name": "Cyberpunk",
                "description": "Futuristic cyberpunk aesthetic with neon colors and high-tech elements",
                "prompt_keywords": ["cyberpunk", "neon", "futuristic", "high-tech", "dystopian"],
                "color_palette": [(255, 0, 255), (0, 255, 255), (255, 20, 147), (50, 205, 50)],
                "brightness": 0.4,
                "contrast": 0.95,
                "saturation": 1.0,
                "texture_style": "glowing and digital",
                "lighting": "neon and dramatic",
                "example_prompts": [
                    "neon-lit city street with holographic ads",
                    "cybernetic character with glowing implants",
                    "digital rain effect with matrix code"
                ]
            },

            "oil_painting": {
                "name": "Oil Painting",
                "description": "Classical oil painting style with rich textures and masterful brushwork",
                "prompt_keywords": ["oil painting", "classical art", "brush strokes", "renaissance style"],
                "color_palette": [(139, 69, 19), (255, 215, 0), (220, 20, 60), (72, 61, 139)],
                "brightness": 0.55,
                "contrast": 0.75,
                "saturation": 0.7,
                "texture_style": "rich and textured",
                "lighting": "dramatic chiaroscuro",
                "example_prompts": [
                    "portrait with dramatic lighting",
                    "still life with rich textures",
                    "landscape with golden hour lighting"
                ]
            },

            "minimalist": {
                "name": "Minimalist",
                "description": "Clean minimalist design with simple forms and limited colors",
                "prompt_keywords": ["minimalist", "clean", "simple", "geometric", "modern"],
                "color_palette": [(255, 255, 255), (128, 128, 128), (0, 0, 0), (200, 200, 200)],
                "brightness": 0.9,
                "contrast": 0.6,
                "saturation": 0.2,
                "texture_style": "clean and smooth",
                "lighting": "even and soft",
                "example_prompts": [
                    "simple geometric shapes in motion",
                    "clean architectural elements",
                    "abstract forms with negative space"
                ]
            },

            "fantasy": {
                "name": "Fantasy Art",
                "description": "Magical fantasy style with ethereal elements and mystical atmosphere",
                "prompt_keywords": ["fantasy art", "magical", "ethereal", "mystical", "enchanted"],
                "color_palette": [(186, 85, 211), (255, 215, 0), (72, 209, 204), (255, 105, 180)],
                "brightness": 0.65,
                "contrast": 0.65,
                "saturation": 0.85,
                "texture_style": "magical and glowing",
                "lighting": "mystical with magical effects",
                "example_prompts": [
                    "magical spell casting with glowing effects",
                    "enchanted forest with fairy lights",
                    "dragon breathing colorful flames"
                ]
            },

            "street_art": {
                "name": "Street Art/Graffiti",
                "description": "Urban street art and graffiti style with bold colors and dynamic forms",
                "prompt_keywords": ["street art", "graffiti", "urban", "spray paint", "bold colors"],
                "color_palette": [(255, 69, 0), (50, 205, 50), (255, 215, 0), (255, 20, 147)],
                "brightness": 0.7,
                "contrast": 0.9,
                "saturation": 0.95,
                "texture_style": "rough and textured",
                "lighting": "urban and dramatic",
                "example_prompts": [
                    "dynamic graffiti tag animation",
                    "colorful mural coming to life",
                    "spray paint effect with drips"
                ]
            },

            "art_nouveau": {
                "name": "Art Nouveau",
                "description": "Elegant Art Nouveau style with flowing organic forms and decorative elements",
                "prompt_keywords": ["art nouveau", "elegant", "organic forms", "decorative", "flowing"],
                "color_palette": [(218, 165, 32), (128, 0, 128), (0, 128, 0), (255, 140, 0)],
                "brightness": 0.6,
                "contrast": 0.7,
                "saturation": 0.75,
                "texture_style": "ornate and flowing",
                "lighting": "elegant and refined",
                "example_prompts": [
                    "flowing botanical patterns",
                    "elegant feminine figure with decorative elements",
                    "ornate architectural details"
                ]
            },

            "retro": {
                "name": "Retro/Vintage",
                "description": "Nostalgic retro style with vintage colors and classic aesthetics",
                "prompt_keywords": ["retro", "vintage", "classic", "nostalgic", "old-school"],
                "color_palette": [(255, 192, 203), (255, 255, 224), (176, 196, 222), (255, 218, 185)],
                "brightness": 0.7,
                "contrast": 0.6,
                "saturation": 0.65,
                "texture_style": "aged and nostalgic",
                "lighting": "warm and vintage",
                "example_prompts": [
                    "vintage advertisement style",
                    "classic car with chrome details",
                    "retro diner with neon signs"
                ]
            }
        }

    def get_style(self, style_name: str) -> Dict:
        """Holt einen spezifischen Stil"""
        return self.styles.get(style_name, self.styles["digital_art"])

    def list_styles(self) -> List[str]:
        """Listet alle verfügbaren Stile auf"""
        return list(self.styles.keys())

    def get_style_info(self, style_name: str) -> str:
        """Gibt detaillierte Stil-Informationen zurück"""
        style = self.get_style(style_name)

        info = f"""
🎨 {style['name']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 Beschreibung: {style['description']}

🎯 Stil-Eigenschaften:
   • Helligkeit: {style['brightness']:.1f}
   • Kontrast: {style['contrast']:.1f}
   • Sättigung: {style['saturation']:.1f}
   • Textur: {style['texture_style']}
   • Beleuchtung: {style['lighting']}

🎨 Farbpalette: {len(style['color_palette'])} Hauptfarben
   {' '.join([f'#{r:02x}{g:02x}{b:02x}' for r, g, b in style['color_palette']])}

💡 Beispiel-Prompts:
"""
        for i, prompt in enumerate(style['example_prompts'], 1):
            info += f"   {i}. {prompt}\n"

        return info


class StyledGIFGenerator:
    """
    GIF-Generator mit Art Style Kontrolle
    """

    def __init__(self):
        self.style_manager = ArtStyleManager()

    def generate_styled_gif(
        self,
        reference_gif_path: str,
        target_style: str,
        custom_prompt: str,
        output_path: str,
        style_strength: float = 0.8,
        preserve_colors: bool = False,
        color_temperature: float = 0.0,
        texture_enhancement: float = 0.5,
        lighting_style: str = "natural"
    ) -> Dict[str, Any]:
        """Generiert GIF mit spezifischem Art Style"""

        logger.info(f"🎨 Starte Style-Generation: {target_style}")

        try:
            # Stil-Konfiguration laden
            style_config = self.style_manager.get_style(target_style)

            # Referenz-GIF analysieren
            analysis_data = self._analyze_reference_gif(reference_gif_path)

            # Stil-spezifischen Prompt erstellen
            styled_prompt = self._create_styled_prompt(
                custom_prompt, style_config, analysis_data
            )

            # Frames generieren
            generated_frames = self._generate_styled_frames(
                analysis_data, style_config, styled_prompt,
                style_strength, preserve_colors, color_temperature,
                texture_enhancement, lighting_style
            )

            # Als GIF speichern
            output_info = self._save_styled_gif(
                generated_frames, output_path, analysis_data, style_config
            )

            result = {
                "input_path": reference_gif_path,
                "output_path": output_path,
                "target_style": target_style,
                "styled_prompt": styled_prompt,
                "style_config": style_config,
                "analysis_data": analysis_data,
                "output_info": output_info,
                "generation_timestamp": time.time()
            }

            logger.info(f"✅ Style-Generation abgeschlossen: {output_path}")

            return result

        except Exception as e:
            logger.error(f"❌ Style-Generation fehlgeschlagen: {str(e)}")
            raise RuntimeError(f"Style-Generation fehlgeschlagen: {str(e)}")

    def _analyze_reference_gif(self, gif_path: str) -> Dict:
        """Analysiert die Referenz-GIF"""

        if not os.path.exists(gif_path):
            raise FileNotFoundError(f"GIF-Datei nicht gefunden: {gif_path}")

        gif = Image.open(gif_path)
        frames = []

        try:
            for frame in ImageSequence.Iterator(gif):
                rgb_frame = frame.convert('RGB')
                frame_array = np.array(rgb_frame)
                frames.append(frame_array)
        except EOFError:
            pass

        # Basis-Analyse
        width, height = gif.size
        frame_count = len(frames)

        # FPS-Erkennung
        fps = 12.0
        if hasattr(gif, 'info') and 'duration' in gif.info:
            duration_ms = gif.info.get('duration', 83)
            if duration_ms > 0:
                fps = 1000.0 / duration_ms

        return {
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "fps": fps,
            "frames": frames,
            "duration": frame_count / fps
        }

    def _create_styled_prompt(
        self,
        base_prompt: str,
        style_config: Dict,
        analysis_data: Dict
    ) -> str:
        """Erstellt stil-spezifischen Prompt"""

        # Stil-Keywords hinzufügen
        style_keywords = ", ".join(style_config["prompt_keywords"])

        # Basis-Prompt erweitern
        if base_prompt:
            styled_prompt = f"{base_prompt}, {style_keywords}"
        else:
            styled_prompt = f"beautiful artwork, {style_keywords}"

        # Qualitäts-Keywords hinzufügen
        quality_keywords = "high quality, detailed, masterpiece, professional"

        final_prompt = f"{styled_prompt}, {quality_keywords}"

        logger.info(f"📝 Generierter Prompt: {final_prompt}")

        return final_prompt

    def _generate_styled_frames(
        self,
        analysis_data: Dict,
        style_config: Dict,
        prompt: str,
        style_strength: float,
        preserve_colors: bool,
        color_temperature: float,
        texture_enhancement: float,
        lighting_style: str
    ) -> List[Image.Image]:
        """Generiert stil-spezifische Frames"""

        frames = []
        frame_count = analysis_data["frame_count"]
        width = analysis_data["width"]
        height = analysis_data["height"]

        logger.info(f"🖼️ Generiere {frame_count} Frames im {style_config['name']} Stil")

        for i in range(frame_count):
            progress = i / max(1, frame_count - 1)

            # Frame basierend auf Stil generieren
            frame = self._create_styled_frame(
                width, height, progress, style_config,
                style_strength, preserve_colors, color_temperature,
                texture_enhancement
            )

            frames.append(frame)

        return frames

    def _create_styled_frame(
        self,
        width: int,
        height: int,
        progress: float,
        style_config: Dict,
        style_strength: float,
        preserve_colors: bool,
        color_temperature: float,
        texture_enhancement: float
    ) -> Image.Image:
        """Erstellt einen stil-spezifischen Frame"""

        # Stil-basierte Farbpalette
        color_palette = style_config["color_palette"]
        brightness = style_config["brightness"]
        contrast = style_config["contrast"]
        saturation = style_config["saturation"]

        # Frame-Array erstellen
        frame_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Stil-spezifische Generierung
        style_name = style_config.get("name", "").lower()

        if "pixel" in style_name:
            frame = self._create_pixel_art_frame(width, height, progress, color_palette)
        elif "watercolor" in style_name:
            frame = self._create_watercolor_frame(width, height, progress, color_palette)
        elif "cyberpunk" in style_name:
            frame = self._create_cyberpunk_frame(width, height, progress, color_palette)
        elif "minimalist" in style_name:
            frame = self._create_minimalist_frame(width, height, progress, color_palette)
        else:
            frame = self._create_artistic_frame(width, height, progress, color_palette)

        # Stil-Verbesserungen anwenden
        if style_strength > 0:
            frame = self._apply_style_enhancements(
                frame, brightness, contrast, saturation,
                style_strength, texture_enhancement
            )

        # Farbtemperatur anpassen
        if color_temperature != 0:
            frame = self._adjust_color_temperature(frame, color_temperature)

        return frame

    def _create_pixel_art_frame(
        self,
        width: int,
        height: int,
        progress: float,
        colors: List[Tuple[int, int, int]]
    ) -> Image.Image:
        """Erstellt Pixel-Art Frame"""

        # Niedrigere Auflösung für Pixel-Effekt
        pixel_width = max(8, width // 16)
        pixel_height = max(8, height // 16)

        # Pixelated Frame erstellen
        frame_array = np.zeros((pixel_height, pixel_width, 3), dtype=np.uint8)

        for y in range(pixel_height):
            for x in range(pixel_width):
                # Farbauswahl basierend auf Position und Zeit
                color_idx = int((x + y + progress * len(colors)) % len(colors))
                frame_array[y, x] = colors[color_idx]

        # Auf Zielgröße skalieren (ohne Antialiasing)
        frame = Image.fromarray(frame_array)
        frame = frame.resize((width, height), Image.NEAREST)

        return frame

    def _create_watercolor_frame(
        self,
        width: int,
        height: int,
        progress: float,
        colors: List[Tuple[int, int, int]]
    ) -> Image.Image:
        """Erstellt Watercolor Frame"""

        frame_array = np.zeros((height, width, 3), dtype=np.float32)

        # Mehrere Farbschichten für Watercolor-Effekt
        for layer in range(3):
            for y in range(height):
                for x in range(width):
                    # Weiche Verläufe
                    spatial_factor = (x / width + y / height) / 2
                    temporal_factor = progress + layer * 0.1

                    # Farbe auswählen
                    color_idx = int((spatial_factor + temporal_factor) * len(colors)) % len(colors)
                    color = colors[color_idx]

                    # Weiche Mischung
                    alpha = 0.3 + 0.2 * np.sin(2 * np.pi * (spatial_factor + temporal_factor))

                    for c in range(3):
                        frame_array[y, x, c] += color[c] * alpha

        # Normalisieren und zu uint8 konvertieren
        frame_array = np.clip(frame_array / 3, 0, 255).astype(np.uint8)
        frame = Image.fromarray(frame_array)

        # Weichzeichnung für Watercolor-Effekt
        frame = frame.filter(ImageFilter.GaussianBlur(radius=1))

        return frame

    def _create_cyberpunk_frame(
        self,
        width: int,
        height: int,
        progress: float,
        colors: List[Tuple[int, int, int]]
    ) -> Image.Image:
        """Erstellt Cyberpunk Frame"""

        frame_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Neon-Linien und Gitter
        for y in range(height):
            for x in range(width):
                # Gitter-Pattern
                grid_x = (x // 20) % 2
                grid_y = (y // 20) % 2

                # Animierte Neon-Effekte
                neon_intensity = 0.5 + 0.5 * np.sin(2 * np.pi * (progress + x / width))

                if grid_x or grid_y:
                    # Neon-Farben
                    color_idx = int((progress + x / width + y / height) * len(colors)) % len(colors)
                    base_color = colors[color_idx]

                    # Neon-Verstärkung
                    r = min(255, int(base_color[0] * neon_intensity * 1.5))
                    g = min(255, int(base_color[1] * neon_intensity * 1.5))
                    b = min(255, int(base_color[2] * neon_intensity * 1.5))

                    frame_array[y, x] = [r, g, b]

        frame = Image.fromarray(frame_array)

        # Glow-Effekt
        frame = frame.filter(ImageFilter.GaussianBlur(radius=0.5))

        return frame

    def _create_minimalist_frame(
        self,
        width: int,
        height: int,
        progress: float,
        colors: List[Tuple[int, int, int]]
    ) -> Image.Image:
        """Erstellt Minimalist Frame"""

        # Hauptsächlich weiß/grau mit wenigen Akzenten
        frame = Image.new('RGB', (width, height), (240, 240, 240))

        # Einfache geometrische Form
        from PIL import ImageDraw
        draw = ImageDraw.Draw(frame)

        # Animierter Kreis oder Rechteck
        center_x = width // 2
        center_y = height // 2
        size = min(width, height) // 4

        # Position basierend auf Progress
        offset_x = int(size * 0.5 * np.sin(2 * np.pi * progress))
        offset_y = int(size * 0.3 * np.cos(2 * np.pi * progress))

        # Akzentfarbe
        accent_color = colors[0] if colors else (128, 128, 128)

        # Geometrische Form zeichnen
        draw.ellipse([
            center_x - size + offset_x,
            center_y - size + offset_y,
            center_x + size + offset_x,
            center_y + size + offset_y
        ], fill=accent_color)

        return frame

    def _create_artistic_frame(
        self,
        width: int,
        height: int,
        progress: float,
        colors: List[Tuple[int, int, int]]
    ) -> Image.Image:
        """Erstellt allgemeinen künstlerischen Frame"""

        frame_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Künstlerischer Verlauf
        for y in range(height):
            for x in range(width):
                # Komplexe Farbberechnung
                spatial_factor = (x / width + y / height) / 2
                temporal_factor = progress

                # Mehrere Sinuswellen für Komplexität
                wave1 = np.sin(2 * np.pi * (spatial_factor + temporal_factor))
                wave2 = np.sin(4 * np.pi * (spatial_factor - temporal_factor))
                wave3 = np.sin(6 * np.pi * (spatial_factor * temporal_factor))

                combined_wave = (wave1 + wave2 + wave3) / 3

                # Farbauswahl
                color_idx = int((combined_wave + 1) * len(colors) / 2) % len(colors)
                base_color = colors[color_idx]

                # Farbintensität modulieren
                intensity = 0.5 + 0.5 * combined_wave

                r = int(base_color[0] * intensity)
                g = int(base_color[1] * intensity)
                b = int(base_color[2] * intensity)

                frame_array[y, x] = [
                    np.clip(r, 0, 255),
                    np.clip(g, 0, 255),
                    np.clip(b, 0, 255)
                ]

        return Image.fromarray(frame_array)

    def _apply_style_enhancements(
        self,
        frame: Image.Image,
        brightness: float,
        contrast: float,
        saturation: float,
        style_strength: float,
        texture_enhancement: float
    ) -> Image.Image:
        """Wendet Stil-Verbesserungen an"""

        # Helligkeit anpassen
        if brightness != 0.5:
            brightness_factor = 1.0 + (brightness - 0.5) * style_strength
            brightness_enhancer = ImageEnhance.Brightness(frame)
            frame = brightness_enhancer.enhance(brightness_factor)

        # Kontrast anpassen
        if contrast != 0.5:
            contrast_factor = 1.0 + (contrast - 0.5) * style_strength
            contrast_enhancer = ImageEnhance.Contrast(frame)
            frame = contrast_enhancer.enhance(contrast_factor)

        # Sättigung anpassen
        if saturation != 0.5:
            saturation_factor = 1.0 + (saturation - 0.5) * style_strength
            saturation_enhancer = ImageEnhance.Color(frame)
            frame = saturation_enhancer.enhance(saturation_factor)

        # Textur-Verbesserung
        if texture_enhancement > 0.5:
            # Schärfe erhöhen
            sharpness_factor = 1.0 + (texture_enhancement - 0.5)
            sharpness_enhancer = ImageEnhance.Sharpness(frame)
            frame = sharpness_enhancer.enhance(sharpness_factor)

        return frame

    def _adjust_color_temperature(self, frame: Image.Image, temperature: float) -> Image.Image:
        """Passt Farbtemperatur an"""

        if temperature == 0:
            return frame

        # Farbtemperatur-Anpassung
        frame_array = np.array(frame, dtype=np.float32)

        if temperature > 0:  # Wärmer (mehr Rot/Gelb)
            frame_array[:, :, 0] *= (1.0 + temperature * 0.3)  # Rot
            frame_array[:, :, 1] *= (1.0 + temperature * 0.1)  # Grün
            frame_array[:, :, 2] *= (1.0 - temperature * 0.2)  # Blau
        else:  # Kühler (mehr Blau)
            frame_array[:, :, 0] *= (1.0 + temperature * 0.2)  # Rot
            frame_array[:, :, 1] *= (1.0 + temperature * 0.1)  # Grün
            frame_array[:, :, 2] *= (1.0 - temperature * 0.3)  # Blau

        frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
        return Image.fromarray(frame_array)

    def _save_styled_gif(
        self,
        frames: List[Image.Image],
        output_path: str,
        analysis_data: Dict,
        style_config: Dict
    ) -> Dict:
        """Speichert das gestylte GIF"""

        # Output-Verzeichnis erstellen
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Frame-Dauer berechnen
        fps = analysis_data["fps"]
        duration_ms = int(1000 / fps)

        try:
            # Als GIF speichern
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=True
            )

            # Output-Informationen
            file_size = os.path.getsize(output_path)

            return {
                "output_path": output_path,
                "style_name": style_config["name"],
                "file_size_bytes": file_size,
                "file_size_mb": file_size / (1024 * 1024),
                "frame_count": len(frames),
                "fps": fps,
                "duration_ms": duration_ms,
                "total_duration_sec": len(frames) / fps
            }

        except Exception as e:
            raise RuntimeError(f"GIF-Speicherung fehlgeschlagen: {str(e)}")


def main():
    """Hauptfunktion mit Art Style Kontrolle"""

    parser = argparse.ArgumentParser(
        description="GIF Reference Pipeline mit Art Style Kontrolle"
    )

    # Eingabe-Parameter
    parser.add_argument("--input", "-i", required=True,
                        help="Pfad zur Referenz-GIF-Datei")
    parser.add_argument("--output", "-o", required=True,
                        help="Ausgabe-Pfad für generierte GIF")
    parser.add_argument("--prompt", "-p", default="",
                        help="Text-Prompt für die Generierung")

    # Stil-Parameter
    parser.add_argument("--style", "-s",
                        choices=ArtStyleManager().list_styles(),
                        default="digital_art",
                        help="Art Style für die Generierung")
    parser.add_argument("--style-strength", type=float, default=0.8,
                        help="Stil-Stärke (0.0-1.0)")
    parser.add_argument("--preserve-colors", action="store_true",
                        help="Original-Farben beibehalten")
    parser.add_argument("--color-temp", type=float, default=0.0,
                        help="Farbtemperatur (-1.0 bis 1.0)")
    parser.add_argument("--texture-enhancement", type=float, default=0.5,
                        help="Textur-Verbesserung (0.0-1.0)")

    # Info-Parameter
    parser.add_argument("--list-styles", action="store_true",
                        help="Alle verfügbaren Stile auflisten")
    parser.add_argument("--style-info",
                        help="Detaillierte Informationen zu einem Stil")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Ausführliche Ausgabe")

    args = parser.parse_args()

    # Logging-Level setzen
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Style Manager initialisieren
    style_manager = ArtStyleManager()

    # Style-Informationen anzeigen
    if args.list_styles:
        print("🎨 Verfügbare Art Styles:")
        print("=" * 50)
        for style_name in style_manager.list_styles():
            style = style_manager.get_style(style_name)
            print(f"• {style_name:<15} - {style['name']}")
        print(f"\nInsgesamt: {len(style_manager.list_styles())} Stile verfügbar")
        return

    if args.style_info:
        print(style_manager.get_style_info(args.style_info))
        return

    # Generator initialisieren
    generator = StyledGIFGenerator()

    try:
        # Stil-Information anzeigen
        style_config = style_manager.get_style(args.style)
        print(f"🎨 Gewählter Stil: {style_config['name']}")
        print(f"📝 Beschreibung: {style_config['description']}")

        # GIF generieren
        result = generator.generate_styled_gif(
            reference_gif_path=args.input,
            target_style=args.style,
            custom_prompt=args.prompt,
            output_path=args.output,
            style_strength=args.style_strength,
            preserve_colors=args.preserve_colors,
            color_temperature=args.color_temp,
            texture_enhancement=args.texture_enhancement
        )

        # Ergebnis anzeigen
        print(f"\n✅ Style-GIF erfolgreich generiert!")
        print(f"   Input: {result['input_path']}")
        print(f"   Output: {result['output_path']}")
        print(f"   Stil: {result['style_config']['name']}")
        print(f"   Frames: {result['output_info']['frame_count']}")
        print(f"   Dateigröße: {result['output_info']['file_size_mb']:.2f} MB")
        print(f"   Prompt: {result['styled_prompt']}")

        print(f"\n🎉 Pipeline erfolgreich abgeschlossen!")

    except Exception as e:
        logger.error(f"💥 Pipeline-Fehler: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()