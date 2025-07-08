#!/usr/bin/env python3
"""
Custom Manga Batch Generator
============================

Generiert spezifische 14 GIFs im Manga Art Style basierend auf Nutzerwahl.
Optimierte Version mit besserer Fehlerbehandlung.
"""

import os
import sys
import json
import time
import numpy as np
from PIL import Image, ImageSequence, ImageEnhance, ImageFilter, ImageDraw
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CustomMangaBatchGenerator:
    """
    Spezieller Generator für die vom Nutzer ausgewählten GIFs
    """

    def __init__(self):
        self.manga_config = self._initialize_manga_style()
        self.output_dir = "output/custom_manga_batch"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _initialize_manga_style(self) -> Dict:
        """Verbesserte Manga-Stil-Konfiguration"""

        return {
            "name": "Enhanced Manga Art Style",
            "description": "Optimized Japanese manga style with improved rendering",
            "color_palette": [
                (255, 255, 255),  # Weiß
                (0, 0, 0),        # Schwarz
                (255, 182, 193),  # Rosa
                (135, 206, 250),  # Himmelblau
                (255, 215, 0),    # Gold
                (255, 105, 180),  # Hot Pink
                (50, 205, 50),    # Grün
                (255, 69, 0),     # Orange-Rot
                (138, 43, 226),   # Lila
                (255, 20, 147)    # Deep Pink
            ],
            "style_properties": {
                "brightness": 0.85,
                "contrast": 0.95,
                "saturation": 0.9,
                "outline_strength": 0.9,
                "dramatic_effects": True,
                "speed_lines": True,
                "emotion_emphasis": True,
                "magical_effects": True
            }
        }

    def generate_custom_batch(self) -> List[Dict]:
        """Generiert die spezifischen 14 GIFs"""

        # Die vom Nutzer angegebenen GIFs
        target_gifs = [
            {
                "path": "input/tenor_fast_transparent_converted.gif",
                "prompt": "dynamic manga character with expressive movements",
                "style": "character_animation"
            },
            {
                "path": "input/ffcb41ab727135955c859e88bc286c54_fast_transparent_converted.gif",
                "prompt": "manga action sequence with dramatic effects",
                "style": "action_sequence"
            },
            {
                "path": "input/0af40433ddb755bfee5a1738717c7028_fast_transparent_converted.gif",
                "prompt": "emotional manga scene with sparkles",
                "style": "emotional_scene"
            },
            {
                "path": "input/tumblr_3c7ddc41f9d033983af0359360d773cf_38fa1d69_540_fast_transparent_converted.gif",
                "prompt": "cute manga character with big eyes",
                "style": "character_animation"
            },
            {
                "path": "input/tableware-milk-fill-only-34_fast_transparent_converted.gif",
                "prompt": "manga food animation with magical effects",
                "style": "transformation"
            },
            {
                "path": "input/48443c9ff5614de637efc09bcede2f90_fast_transparent_converted.gif",
                "prompt": "manga character transformation sequence",
                "style": "transformation"
            },
            {
                "path": "input/8995d65c9054dddf1036afccc5e13359_fast_transparent_converted.gif",
                "prompt": "manga sprite with clean animations",
                "style": "character_animation"
            },
            {
                "path": "input/022f076b146c9ffdc0805d383b7a2f32_fast_transparent_converted.gif",
                "prompt": "dramatic manga action with speed lines",
                "style": "action_sequence"
            },
            {
                "path": "input/00dd8a85ebe872350d8ffda6435903a1_fast_transparent_converted.gif",
                "prompt": "elegant manga character movement",
                "style": "character_animation"
            },
            {
                "path": "input/styled_purple_flame_archer_fast_transparent_converted.gif",
                "prompt": "manga archer with purple flame effects",
                "style": "magical_combat"
            },
            {
                "path": "input/e0b9c377238ff883cf0d8f76e5499a63_fast_transparent_converted.gif",
                "prompt": "mystical manga character with aura",
                "style": "transformation"
            },
            {
                "path": "input/styled_lightning_archer_demonslayer_fast_transparent_converted.gif",
                "prompt": "manga demon slayer with lightning effects",
                "style": "magical_combat"
            },
            {
                "path": "input/michkartonmitantrieb_fast_transparent_converted.gif",
                "prompt": "manga mechanical character animation",
                "style": "action_sequence"
            },
            {
                "path": "input/tumblr_inline_nfpj8uucP11s6lw3t540_fast_transparent_converted.gif",
                "prompt": "cute manga character with emotional expressions",
                "style": "emotional_scene"
            }
        ]

        results = []
        successful = 0
        failed = 0

        logger.info("🎌 Starte Custom Manga Batch-Generierung")
        logger.info(f"📁 Output-Verzeichnis: {self.output_dir}")
        logger.info(f"🎯 Zu verarbeiten: {len(target_gifs)} GIFs")

        for i, gif_config in enumerate(target_gifs, 1):
            filename = Path(gif_config["path"]).name
            logger.info(f"\n🎨 Generiere {i}/{len(target_gifs)}: {filename}")

            try:
                # Prüfen ob Datei existiert
                if not os.path.exists(gif_config["path"]):
                    logger.warning(f"⚠️ Datei nicht gefunden: {gif_config['path']}")
                    results.append({
                        "input_file": filename,
                        "error": "File not found",
                        "success": False
                    })
                    failed += 1
                    continue

                result = self._generate_single_manga_gif(gif_config, i)
                results.append(result)
                successful += 1
                logger.info(f"✅ Erfolgreich: {Path(result['output_path']).name}")

            except Exception as e:
                logger.error(f"❌ Fehler bei {filename}: {str(e)}")
                results.append({
                    "input_file": filename,
                    "error": str(e),
                    "success": False
                })
                failed += 1

        # Zusammenfassung
        logger.info(f"\n📊 Batch abgeschlossen:")
        logger.info(f"   ✅ Erfolgreich: {successful}")
        logger.info(f"   ❌ Fehlgeschlagen: {failed}")
        logger.info(f"   📈 Erfolgsrate: {successful/(successful+failed)*100:.1f}%")

        # Report erstellen
        self._create_enhanced_report(results, successful, failed)

        return results

    def _generate_single_manga_gif(self, gif_config: Dict, index: int) -> Dict:
        """Generiert ein einzelnes optimiertes Manga-GIF"""

        input_path = gif_config["path"]
        filename = Path(input_path).stem
        output_filename = f"manga_{index:02d}_{filename}.gif"
        output_path = os.path.join(self.output_dir, output_filename)

        # Analyse mit besserer Fehlerbehandlung
        analysis = self._analyze_gif_robust(input_path)

        # Frames generieren
        manga_frames = self._generate_enhanced_manga_frames(
            analysis, gif_config['prompt'], gif_config['style']
        )

        # GIF speichern
        output_info = self._save_optimized_gif(manga_frames, output_path, analysis)

        return {
            "input_file": Path(input_path).name,
            "output_path": output_path,
            "prompt": gif_config['prompt'],
            "style_focus": gif_config['style'],
            "analysis_summary": {
                "frames": analysis["frame_count"],
                "resolution": f"{analysis['width']}x{analysis['height']}",
                "fps": analysis["fps"],
                "motion": analysis.get("motion_intensity", 0)
            },
            "output_info": output_info,
            "success": True,
            "generation_time": time.time()
        }

    def _analyze_gif_robust(self, gif_path: str) -> Dict:
        """Robuste GIF-Analyse mit Fehlerbehandlung"""

        try:
            gif = Image.open(gif_path)
            frames = []

            # Frame-Extraktion mit Limit
            frame_count = 0
            max_frames = 30  # Limit für Performance

            for frame in ImageSequence.Iterator(gif):
                if frame_count >= max_frames:
                    break
                try:
                    rgb_frame = frame.convert('RGB')
                    frame_array = np.array(rgb_frame)
                    frames.append(frame_array)
                    frame_count += 1
                except Exception as e:
                    logger.warning(f"Frame {frame_count} übersprungen: {str(e)}")
                    continue

            if not frames:
                raise ValueError("Keine gültigen Frames gefunden")

            width, height = gif.size

            # FPS-Erkennung mit Fallback
            fps = 12.0
            try:
                if hasattr(gif, 'info') and 'duration' in gif.info:
                    duration_ms = gif.info.get('duration', 83)
                    if duration_ms > 0:
                        fps = min(30.0, max(1.0, 1000.0 / duration_ms))
            except:
                pass

            # Bewegungsanalyse
            motion_intensity = 0.0
            if len(frames) > 1:
                try:
                    motion_intensity = self._calculate_motion_safely(frames)
                except:
                    logger.warning("Bewegungsanalyse fehlgeschlagen, verwende Standard-Wert")

            return {
                "width": width,
                "height": height,
                "frame_count": len(frames),
                "fps": fps,
                "frames": frames,
                "motion_intensity": motion_intensity,
                "duration": len(frames) / fps,
                "file_size": os.path.getsize(gif_path)
            }

        except Exception as e:
            logger.error(f"GIF-Analyse fehlgeschlagen: {str(e)}")
            raise RuntimeError(f"Konnte GIF nicht analysieren: {str(e)}")

    def _calculate_motion_safely(self, frames: List[np.ndarray]) -> float:
        """Sichere Bewegungsberechnung"""

        motion_values = []
        for i in range(1, min(len(frames), 10)):  # Max 10 Frames für Performance
            try:
                prev_frame = frames[i-1].astype(np.float32)
                curr_frame = frames[i].astype(np.float32)
                diff = np.mean(np.abs(curr_frame - prev_frame))
                motion_values.append(diff)
            except:
                continue

        return float(np.mean(motion_values)) if motion_values else 5.0

    def _generate_enhanced_manga_frames(
        self,
        analysis: Dict,
        prompt: str,
        style_focus: str
    ) -> List[Image.Image]:
        """Generiert verbesserte Manga-Frames"""

        frames = []
        frame_count = analysis["frame_count"]
        width = analysis["width"]
        height = analysis["height"]
        motion_intensity = analysis.get("motion_intensity", 5.0)

        logger.info(f"📐 Generiere {frame_count} Enhanced Manga-Frames ({width}x{height})")
        logger.info(f"🎯 Stil: {style_focus}")
        logger.info(f"🏃 Motion: {motion_intensity:.2f}")

        for i in range(frame_count):
            progress = i / max(1, frame_count - 1)

            try:
                # Frame basierend auf Stil generieren
                if style_focus == "magical_combat":
                    frame = self._create_magical_combat_frame(width, height, progress, motion_intensity)
                elif style_focus == "character_animation":
                    frame = self._create_enhanced_character_frame(width, height, progress, motion_intensity)
                elif style_focus == "action_sequence":
                    frame = self._create_enhanced_action_frame(width, height, progress, motion_intensity)
                elif style_focus == "emotional_scene":
                    frame = self._create_enhanced_emotional_frame(width, height, progress, motion_intensity)
                elif style_focus == "transformation":
                    frame = self._create_enhanced_transformation_frame(width, height, progress, motion_intensity)
                else:
                    frame = self._create_enhanced_general_frame(width, height, progress, motion_intensity)

                # Manga-Effekte anwenden
                frame = self._apply_enhanced_manga_effects(frame, style_focus, progress)
                frames.append(frame)

            except Exception as e:
                logger.warning(f"Frame {i} fehlgeschlagen, verwende Fallback: {str(e)}")
                # Fallback-Frame
                fallback_frame = self._create_fallback_frame(width, height, progress)
                frames.append(fallback_frame)

        return frames

    def _create_magical_combat_frame(self, width: int, height: int, progress: float, motion: float) -> Image.Image:
        """Erstellt magischen Kampf-Frame"""

        # Dunkler Hintergrund für Drama
        base_color = (
            int(20 + 30 * np.sin(2 * np.pi * progress)),
            int(10 + 20 * np.sin(2 * np.pi * progress + 1)),
            int(40 + 60 * np.sin(2 * np.pi * progress + 2))
        )
        frame = Image.new('RGB', (width, height), base_color)
        draw = ImageDraw.Draw(frame)

        center_x = width // 2
        center_y = height // 2

        # Magische Energie-Aura
        for ring in range(4):
            radius = int(20 + ring * 15 + 10 * np.sin(4 * np.pi * progress + ring))
            intensity = 1.0 - (ring * 0.2)

            # Energie-Ring
            color = (
                int(255 * intensity),
                int(100 * intensity),
                int(255 * intensity)
            )

            draw.ellipse([
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius
            ], outline=color, width=max(1, int(3 * intensity)))

        # Blitz-Effekte
        if motion > 15:
            for lightning in range(6):
                angle = lightning * 60 + progress * 180
                angle_rad = np.radians(angle)

                start_radius = 30
                end_radius = min(width, height) // 3

                start_x = center_x + int(start_radius * np.cos(angle_rad))
                start_y = center_y + int(start_radius * np.sin(angle_rad))
                end_x = center_x + int(end_radius * np.cos(angle_rad))
                end_y = center_y + int(end_radius * np.sin(angle_rad))

                # Zickzack-Blitz
                mid_x = (start_x + end_x) // 2 + int(10 * np.sin(8 * np.pi * progress))
                mid_y = (start_y + end_y) // 2 + int(10 * np.cos(8 * np.pi * progress))

                draw.line([start_x, start_y, mid_x, mid_y], fill=(255, 255, 0), width=3)
                draw.line([mid_x, mid_y, end_x, end_y], fill=(255, 255, 0), width=3)

        return frame

    def _create_enhanced_character_frame(self, width: int, height: int, progress: float, motion: float) -> Image.Image:
        """Verbesserte Character-Animation"""

        frame = Image.new('RGB', (width, height), (248, 248, 255))
        draw = ImageDraw.Draw(frame)

        char_x = width // 2
        char_y = height // 2

        # Sanfte Animation
        offset_x = int(15 * np.sin(2 * np.pi * progress))
        offset_y = int(8 * np.cos(3 * np.pi * progress))

        # Charakter-Silhouette (verbessert)
        head_size = min(width, height) // 10

        # Kopf mit Schatten
        shadow_offset = 3
        draw.ellipse([
            char_x - head_size + offset_x + shadow_offset,
            char_y - head_size * 2 + offset_y + shadow_offset,
            char_x + head_size + offset_x + shadow_offset,
            char_y + offset_y + shadow_offset
        ], fill=(200, 200, 200))

        # Hauptkopf
        draw.ellipse([
            char_x - head_size + offset_x,
            char_y - head_size * 2 + offset_y,
            char_x + head_size + offset_x,
            char_y + offset_y
        ], fill=(255, 220, 200), outline=(0, 0, 0), width=3)

        # Verbesserte Augen
        eye_size = head_size // 2
        pupil_size = eye_size // 3

        # Linkes Auge
        draw.ellipse([
            char_x - eye_size - 3 + offset_x,
            char_y - head_size + offset_y,
            char_x - 3 + offset_x,
            char_y - head_size + eye_size + offset_y
        ], fill=(135, 206, 250), outline=(0, 0, 0), width=2)

        # Pupille links
        draw.ellipse([
            char_x - pupil_size - 3 + offset_x,
            char_y - head_size + pupil_size//2 + offset_y,
            char_x - 3 + pupil_size + offset_x,
            char_y - head_size + pupil_size + pupil_size//2 + offset_y
        ], fill=(0, 0, 0))

        # Rechtes Auge
        draw.ellipse([
            char_x + 3 + offset_x,
            char_y - head_size + offset_y,
            char_x + eye_size + 3 + offset_x,
            char_y - head_size + eye_size + offset_y
        ], fill=(135, 206, 250), outline=(0, 0, 0), width=2)

        # Pupille rechts
        draw.ellipse([
            char_x + 3 + offset_x,
            char_y - head_size + pupil_size//2 + offset_y,
            char_x + 3 + pupil_size + offset_x,
            char_y - head_size + pupil_size + pupil_size//2 + offset_y
        ], fill=(0, 0, 0))

        # Glitzer-Effekte
        if progress > 0.2:
            for glitter in range(3):
                glitter_x = char_x + int(20 * np.cos(2 * np.pi * progress + glitter)) + offset_x
                glitter_y = char_y - head_size + int(10 * np.sin(2 * np.pi * progress + glitter)) + offset_y

                draw.ellipse([
                    glitter_x - 2, glitter_y - 2,
                    glitter_x + 2, glitter_y + 2
                ], fill=(255, 255, 255))

        return frame

    def _create_enhanced_action_frame(self, width: int, height: int, progress: float, motion: float) -> Image.Image:
        """Verbesserte Action-Sequenz"""

        frame = Image.new('RGB', (width, height), (255, 240, 240))
        draw = ImageDraw.Draw(frame)

        center_x = width // 2
        center_y = height // 2

        # Dynamische Speed Lines
        line_count = max(8, int(motion / 2))
        for i in range(line_count):
            angle = (i * 360 / line_count + progress * 720) % 360
            angle_rad = np.radians(angle)

            start_radius = 40
            end_radius = min(width, height) // 2

            start_x = center_x + int(start_radius * np.cos(angle_rad))
            start_y = center_y + int(start_radius * np.sin(angle_rad))
            end_x = center_x + int(end_radius * np.cos(angle_rad))
            end_y = center_y + int(end_radius * np.sin(angle_rad))

            line_width = max(1, int(motion / 8))
            draw.line([start_x, start_y, end_x, end_y], fill=(0, 0, 0), width=line_width)

        # Zentrale Action
        action_size = int(25 + 15 * np.sin(6 * np.pi * progress))

        # Explosions-Kern
        draw.ellipse([
            center_x - action_size,
            center_y - action_size,
            center_x + action_size,
            center_y + action_size
        ], fill=(255, 69, 0), outline=(255, 215, 0), width=4)

        # Funken-Effekte
        if motion > 10:
            for spark in range(12):
                spark_angle = spark * 30 + progress * 360
                spark_rad = np.radians(spark_angle)
                spark_dist = action_size + int(20 * np.random.random())

                spark_x = center_x + int(spark_dist * np.cos(spark_rad))
                spark_y = center_y + int(spark_dist * np.sin(spark_rad))

                draw.ellipse([
                    spark_x - 2, spark_y - 2,
                    spark_x + 2, spark_y + 2
                ], fill=(255, 255, 0))

        return frame

    def _create_enhanced_emotional_frame(self, width: int, height: int, progress: float, motion: float) -> Image.Image:
        """Verbesserte emotionale Szene"""

        # Sanfter Farbverlauf-Hintergrund
        base_r = int(255 * (0.9 + 0.1 * np.sin(2 * np.pi * progress)))
        base_g = int(240 * (0.9 + 0.1 * np.sin(2 * np.pi * progress + 1)))
        base_b = int(245 * (0.9 + 0.1 * np.sin(2 * np.pi * progress + 2)))

        frame = Image.new('RGB', (width, height), (base_r, base_g, base_b))
        draw = ImageDraw.Draw(frame)

        # Mehr Sterne für emotionale Intensität
        star_count = max(20, int(motion))
        for i in range(star_count):
            star_x = int((i * 67 + progress * 300) % width)
            star_y = int((i * 89 + progress * 200) % height)

            # Verschiedene Stern-Größen
            star_size = int(2 + 3 * np.sin(2 * np.pi * progress + i * 0.5))

            # Stern-Farben variieren
            colors = [(255, 182, 193), (255, 105, 180), (255, 20, 147), (255, 255, 255)]
            color = colors[i % len(colors)]

            # 4-zackiger Stern
            points = []
            for j in range(4):
                angle = j * 90
                angle_rad = np.radians(angle)
                point_x = star_x + int(star_size * np.cos(angle_rad))
                point_y = star_y + int(star_size * np.sin(angle_rad))
                points.extend([point_x, point_y])

            if len(points) >= 6:
                draw.polygon(points, fill=color)

        # Zentrales pulsierendes Herz
        center_x = width // 2
        center_y = height // 2
        heart_size = int(25 + 15 * np.sin(4 * np.pi * progress))

        # Herz-Schatten
        shadow_offset = 3
        self._draw_heart(draw, center_x + shadow_offset, center_y + shadow_offset,
                        heart_size, (200, 150, 150))

        # Hauptherz
        self._draw_heart(draw, center_x, center_y, heart_size, (255, 105, 180))

        return frame

    def _create_enhanced_transformation_frame(self, width: int, height: int, progress: float, motion: float) -> Image.Image:
        """Verbesserte Transformation"""

        # Magischer Hintergrund mit Farbwechsel
        base_color = (
            int(30 + 120 * progress),
            int(10 + 100 * progress),
            int(80 + 175 * progress)
        )
        frame = Image.new('RGB', (width, height), base_color)
        draw = ImageDraw.Draw(frame)

        center_x = width // 2
        center_y = height // 2

        # Mehrere magische Kreise
        for ring in range(5):
            radius = int(25 + ring * 20 + 15 * np.sin(3 * np.pi * progress + ring))

            # Ring-Farbe basierend auf Progress
            ring_color = (
                int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * progress + ring))),
                int(215 * (0.7 + 0.3 * np.cos(2 * np.pi * progress + ring))),
                int(100 + 155 * (0.6 + 0.4 * np.sin(2 * np.pi * progress + ring)))
            )

            draw.ellipse([
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius
            ], outline=ring_color, width=3)

            # Magische Runen auf den Ringen
            rune_count = 6 + ring * 2
            for rune in range(rune_count):
                angle = rune * (360 / rune_count) + progress * 180
                angle_rad = np.radians(angle)

                rune_x = center_x + int(radius * np.cos(angle_rad))
                rune_y = center_y + int(radius * np.sin(angle_rad))

                # Kleine magische Symbole
                symbol_size = 2 + ring
                draw.rectangle([
                    rune_x - symbol_size, rune_y - symbol_size,
                    rune_x + symbol_size, rune_y + symbol_size
                ], fill=(255, 255, 255))

        # Zentrale Transformation-Energie
        energy_size = int(20 + 30 * progress)
        draw.ellipse([
            center_x - energy_size, center_y - energy_size,
            center_x + energy_size, center_y + energy_size
        ], fill=(255, 255, 255), outline=(255, 215, 0), width=3)

        # Energie-Strahlen
        if progress > 0.2:
            beam_count = 8
            for beam in range(beam_count):
                angle = beam * (360 / beam_count)
                angle_rad = np.radians(angle)
                beam_length = int(40 + 40 * progress)

                end_x = center_x + int(beam_length * np.cos(angle_rad))
                end_y = center_y + int(beam_length * np.sin(angle_rad))

                draw.line([center_x, center_y, end_x, end_y],
                         fill=(255, 255, 255), width=4)

        return frame

    def _create_enhanced_general_frame(self, width: int, height: int, progress: float, motion: float) -> Image.Image:
        """Verbesserter allgemeiner Frame"""

        frame = Image.new('RGB', (width, height), (248, 248, 255))
        draw = ImageDraw.Draw(frame)

        center_x = width // 2
        center_y = height // 2

        # Animiertes Element mit Schatten
        size = int(35 + 20 * np.sin(2 * np.pi * progress))

        # Schatten
        shadow_offset = 5
        draw.ellipse([
            center_x - size + shadow_offset,
            center_y - size + shadow_offset,
            center_x + size + shadow_offset,
            center_y + size + shadow_offset
        ], fill=(180, 180, 180))

        # Hauptelement
        draw.ellipse([
            center_x - size, center_y - size,
            center_x + size, center_y + size
        ], fill=(135, 206, 250), outline=(0, 0, 0), width=4)

        # Highlight
        highlight_size = size // 2
        draw.ellipse([
            center_x - highlight_size, center_y - highlight_size,
            center_x, center_y
        ], fill=(255, 255, 255))

        return frame

    def _create_fallback_frame(self, width: int, height: int, progress: float) -> Image.Image:
        """Fallback-Frame bei Fehlern"""

        frame = Image.new('RGB', (width, height), (240, 240, 240))
        draw = ImageDraw.Draw(frame)

        # Einfacher animierter Kreis
        center_x = width // 2
        center_y = height // 2
        size = int(20 + 10 * np.sin(2 * np.pi * progress))

        draw.ellipse([
            center_x - size, center_y - size,
            center_x + size, center_y + size
        ], fill=(100, 100, 100), outline=(0, 0, 0), width=2)

        return frame

    def _draw_heart(self, draw, x: int, y: int, size: int, color: tuple):
        """Hilfsfunktion zum Zeichnen eines Herzens"""

        # Vereinfachtes Herz aus zwei Kreisen und einem Dreieck
        half_size = size // 2

        # Linker Kreis
        draw.ellipse([
            x - size, y - half_size,
            x, y + half_size
        ], fill=color)

        # Rechter Kreis
        draw.ellipse([
            x, y - half_size,
            x + size, y + half_size
        ], fill=color)

        # Untere Spitze
        draw.polygon([
            x, y + half_size,
            x - half_size, y,
            x + half_size, y
        ], fill=color)

    def _apply_enhanced_manga_effects(self, frame: Image.Image, style_focus: str, progress: float) -> Image.Image:
        """Verbesserte Manga-Effekte"""

        # Basis-Verbesserungen
        enhancer = ImageEnhance.Contrast(frame)
        frame = enhancer.enhance(1.4)

        enhancer = ImageEnhance.Color(frame)
        frame = enhancer.enhance(1.3)

        enhancer = ImageEnhance.Sharpness(frame)
        frame = enhancer.enhance(1.2)

        # Stil-spezifische Effekte
        if style_focus == "magical_combat":
            # Zusätzlicher Kontrast für Drama
            enhancer = ImageEnhance.Contrast(frame)
            frame = enhancer.enhance(1.2)

        elif style_focus == "emotional_scene":
            # Weichere Effekte
            enhancer = ImageEnhance.Brightness(frame)
            frame = enhancer.enhance(1.1)

        return frame

    def _save_optimized_gif(self, frames: List[Image.Image], output_path: str, analysis: Dict) -> Dict:
        """Optimierte GIF-Speicherung"""

        fps = analysis["fps"]
        duration_ms = max(40, min(200, int(1000 / fps)))  # 40-200ms Range

        try:
            # Optimierte Speicherung
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=True,
                disposal=2  # Restore to background
            )

            file_size = os.path.getsize(output_path)

            return {
                "output_path": output_path,
                "file_size_bytes": file_size,
                "file_size_kb": file_size / 1024,
                "file_size_mb": file_size / (1024 * 1024),
                "frame_count": len(frames),
                "fps": fps,
                "duration_ms": duration_ms,
                "optimization": "enabled"
            }

        except Exception as e:
            raise RuntimeError(f"GIF-Speicherung fehlgeschlagen: {str(e)}")

    def _create_enhanced_report(self, results: List[Dict], successful: int, failed: int):
        """Erstellt erweiterten Report"""

        # JSON-sichere Daten für Report
        safe_results = []
        for result in results:
            safe_result = {
                "input_file": result.get("input_file", "unknown"),
                "success": result.get("success", False),
                "prompt": result.get("prompt", ""),
                "style_focus": result.get("style_focus", ""),
                "generation_time": result.get("generation_time", 0)
            }

            if result.get("success"):
                safe_result.update({
                    "output_path": result.get("output_path", ""),
                    "analysis_summary": result.get("analysis_summary", {}),
                    "output_info": result.get("output_info", {})
                })
            else:
                safe_result["error"] = result.get("error", "Unknown error")

            safe_results.append(safe_result)

        report = {
            "generation_timestamp": time.time(),
            "total_processed": len(results),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(results) if results else 0,
            "output_directory": self.output_dir,
            "manga_style_config": self.manga_config,
            "results": safe_results
        }

        report_path = os.path.join(self.output_dir, "custom_manga_report.json")

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"📄 Report gespeichert: {report_path}")
        except Exception as e:
            logger.warning(f"Report-Speicherung fehlgeschlagen: {str(e)}")

        # Konsolen-Ausgabe
        print(f"\n🎌 CUSTOM MANGA BATCH - ABGESCHLOSSEN")
        print(f"═══════════════════════════════════════════════════")
        print(f"📊 Verarbeitet: {len(results)}")
        print(f"✅ Erfolgreich: {successful}")
        print(f"❌ Fehlgeschlagen: {failed}")
        print(f"📈 Erfolgsrate: {successful/len(results)*100:.1f}%")
        print(f"📁 Output: {self.output_dir}")

        if successful > 0:
            print(f"\n🎨 Generierte Manga-GIFs:")
            for result in results:
                if result.get("success"):
                    filename = Path(result["output_path"]).name
                    output_info = result.get("output_info", {})
                    size_kb = output_info.get("file_size_kb", 0)
                    frames = output_info.get("frame_count", 0)
                    print(f"   • {filename}")
                    print(f"     Style: {result['style_focus']}, Frames: {frames}, Size: {size_kb:.1f}KB")


def main():
    """Hauptfunktion"""

    print("🎌 CUSTOM MANGA BATCH GENERATOR")
    print("═══════════════════════════════════════════════════")
    print("Generiert 14 spezifische GIFs im optimierten Manga Style")
    print("mit verbesserter Fehlerbehandlung und Effekten.")
    print("═══════════════════════════════════════════════════\n")

    try:
        generator = CustomMangaBatchGenerator()
        results = generator.generate_custom_batch()

        successful = len([r for r in results if r.get("success", False)])
        if successful > 0:
            print(f"\n🎉 {successful} Manga-GIFs erfolgreich generiert!")
        else:
            print(f"\n⚠️ Keine GIFs konnten generiert werden.")

    except Exception as e:
        logger.error(f"💥 Batch-Generation fehlgeschlagen: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()