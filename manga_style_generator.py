#!/usr/bin/env python3
"""
Manga Style GIF Generator
========================

Generiert 5 ausgewählte GIFs aus dem Input-Verzeichnis im Manga Art Style.
Basiert auf den Recherchen zur GIF-Referenz-Pipeline.
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
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MangaStyleGenerator:
    """
    Spezialisierter Generator für Manga Art Style GIFs
    """

    def __init__(self):
        self.manga_config = self._initialize_manga_style()
        self.output_dir = "output/manga_style_generation"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _initialize_manga_style(self) -> Dict:
        """Initialisiert Manga-spezifische Stil-Konfiguration"""

        return {
            "name": "Manga Art Style",
            "description": "Japanese manga style with bold outlines, dramatic expressions, and dynamic compositions",
            "color_palette": [
                (255, 255, 255),  # Weiß für Highlights
                (0, 0, 0),        # Schwarz für Outlines
                (255, 182, 193),  # Rosa für Emotionen
                (135, 206, 250),  # Himmelblau
                (255, 215, 0),    # Gold für Effekte
                (255, 105, 180),  # Hot Pink für Drama
                (50, 205, 50),    # Grün für Natur
                (255, 69, 0)      # Orange-Rot für Action
            ],
            "style_properties": {
                "brightness": 0.8,
                "contrast": 0.9,
                "saturation": 0.85,
                "outline_strength": 0.8,
                "dramatic_effects": True,
                "speed_lines": True,
                "emotion_emphasis": True
            },
            "prompts": {
                "character_animation": "dynamic manga character with expressive eyes and bold outlines",
                "action_sequence": "manga action scene with speed lines and dramatic effects",
                "emotional_scene": "emotional manga moment with sparkles and dramatic lighting",
                "transformation": "magical manga transformation with glowing effects",
                "general": "manga style artwork with clean lines and vibrant colors"
            }
        }

    def generate_manga_batch(self) -> List[Dict]:
        """Generiert 5 ausgewählte GIFs im Manga Style"""

        # Ausgewählte GIFs für Manga-Konvertierung
        selected_gifs = [
            {
                "filename": "eleni_fast_transparent_converted.gif",
                "prompt": "elegant manga character with flowing hair and graceful movements",
                "style_focus": "character_animation"
            },
            {
                "filename": "9f720323126213.56047641e9c83_fast_transparent_converted.gif",
                "prompt": "dynamic manga action sequence with dramatic poses",
                "style_focus": "action_sequence"
            },
            {
                "filename": "animated_sprite_fast_transparent_converted.gif",
                "prompt": "cute manga sprite character with expressive animations",
                "style_focus": "character_animation"
            },
            {
                "filename": "chaplin_dance_fast_transparent_converted.gif",
                "prompt": "comedic manga character with exaggerated expressions and movements",
                "style_focus": "emotional_scene"
            },
            {
                "filename": "1_fast_transparent_converted.gif",
                "prompt": "mystical manga transformation with magical effects",
                "style_focus": "transformation"
            }
        ]

        results = []

        logger.info("🎌 Starte Manga Style Batch-Generierung")
        logger.info(f"📁 Output-Verzeichnis: {self.output_dir}")

        for i, gif_config in enumerate(selected_gifs, 1):
            logger.info(f"\n🎨 Generiere {i}/5: {gif_config['filename']}")

            try:
                result = self._generate_single_manga_gif(gif_config, i)
                results.append(result)
                logger.info(
                    f"✅ Erfolgreich generiert: {result['output_path']}")

            except Exception as e:
                logger.error(
                    f"❌ Fehler bei {gif_config['filename']}: {str(e)}")
                results.append({
                    "input_file": gif_config['filename'],
                    "error": str(e),
                    "success": False
                })

        # Batch-Report erstellen
        self._create_batch_report(results)

        return results

    def _generate_single_manga_gif(self, gif_config: Dict, index: int) -> Dict:
        """Generiert ein einzelnes Manga-Style GIF"""

        input_path = f"input/{gif_config['filename']}"
        output_filename = f"manga_style_{index:02d}_{Path(gif_config['filename']).stem}.gif"
        output_path = os.path.join(self.output_dir, output_filename)

        # Referenz-GIF analysieren
        analysis = self._analyze_reference_gif(input_path)

        # Manga-Style Frames generieren
        manga_frames = self._generate_manga_frames(
            analysis, gif_config['prompt'], gif_config['style_focus']
        )

        # Als GIF speichern
        output_info = self._save_manga_gif(manga_frames, output_path, analysis)

        return {
            "input_file": gif_config['filename'],
            "output_path": output_path,
            "prompt": gif_config['prompt'],
            "style_focus": gif_config['style_focus'],
            "analysis": analysis,
            "output_info": output_info,
            "success": True,
            "generation_time": time.time()
        }

    def _analyze_reference_gif(self, gif_path: str) -> Dict:
        """Analysiert die Referenz-GIF für Manga-Konvertierung"""

        if not os.path.exists(gif_path):
            raise FileNotFoundError(f"GIF nicht gefunden: {gif_path}")

        gif = Image.open(gif_path)
        frames = []

        try:
            frame_count = 0
            for frame in ImageSequence.Iterator(gif):
                if frame_count >= 50:  # Limit für Performance
                    break
                rgb_frame = frame.convert('RGB')
                frame_array = np.array(rgb_frame)
                frames.append(frame_array)
                frame_count += 1
        except EOFError:
            pass

        width, height = gif.size

        # FPS-Erkennung
        fps = 12.0
        if hasattr(gif, 'info') and 'duration' in gif.info:
            duration_ms = gif.info.get('duration', 83)
            if duration_ms > 0:
                fps = 1000.0 / duration_ms

        # Bewegungsanalyse (vereinfacht)
        motion_intensity = self._analyze_motion_simple(frames)

        # Farbanalyse
        color_analysis = self._analyze_colors_simple(frames)

        return {
            "width": width,
            "height": height,
            "frame_count": len(frames),
            "fps": fps,
            "frames": frames,
            "motion_intensity": motion_intensity,
            "color_analysis": color_analysis,
            "duration": len(frames) / fps
        }

    def _analyze_motion_simple(self, frames: List[np.ndarray]) -> float:
        """Einfache Bewegungsanalyse"""

        if len(frames) < 2:
            return 0.0

        motion_values = []
        for i in range(1, len(frames)):
            # Frame-Differenz berechnen
            diff = np.mean(np.abs(frames[i].astype(
                float) - frames[i-1].astype(float)))
            motion_values.append(diff)

        return float(np.mean(motion_values)) if motion_values else 0.0

    def _analyze_colors_simple(self, frames: List[np.ndarray]) -> Dict:
        """Einfache Farbanalyse"""

        if not frames:
            return {"dominant_colors": [], "brightness": 0.5}

        # Sample-Frame für Analyse
        sample_frame = frames[len(frames) // 2]

        # Durchschnittliche Helligkeit
        brightness = np.mean(sample_frame) / 255.0

        # Dominante Farben (vereinfacht)
        reshaped = sample_frame.reshape(-1, 3)
        unique_colors = np.unique(reshaped, axis=0)

        # Top 5 Farben basierend auf Häufigkeit
        dominant_colors = []
        for color in unique_colors[:5]:
            dominant_colors.append(tuple(color))

        return {
            "dominant_colors": dominant_colors,
            "brightness": float(brightness)
        }

    def _generate_manga_frames(
        self,
        analysis: Dict,
        prompt: str,
        style_focus: str
    ) -> List[Image.Image]:
        """Generiert Manga-Style Frames"""

        frames = []
        frame_count = analysis["frame_count"]
        width = analysis["width"]
        height = analysis["height"]
        motion_intensity = analysis["motion_intensity"]

        logger.info(
            f"📐 Generiere {frame_count} Manga-Frames ({width}x{height})")
        logger.info(f"🎯 Stil-Fokus: {style_focus}")
        logger.info(f"🏃 Bewegungsintensität: {motion_intensity:.2f}")

        for i in range(frame_count):
            progress = i / max(1, frame_count - 1)

            # Manga-Frame basierend auf Stil-Fokus generieren
            if style_focus == "character_animation":
                frame = self._create_character_manga_frame(
                    width, height, progress, motion_intensity)
            elif style_focus == "action_sequence":
                frame = self._create_action_manga_frame(
                    width, height, progress, motion_intensity)
            elif style_focus == "emotional_scene":
                frame = self._create_emotional_manga_frame(
                    width, height, progress, motion_intensity)
            elif style_focus == "transformation":
                frame = self._create_transformation_manga_frame(
                    width, height, progress, motion_intensity)
            else:
                frame = self._create_general_manga_frame(
                    width, height, progress, motion_intensity)

            # Manga-spezifische Effekte anwenden
            frame = self._apply_manga_effects(
                frame, style_focus, progress, motion_intensity)

            frames.append(frame)

        return frames

    def _create_character_manga_frame(
        self, width: int, height: int, progress: float, motion: float
    ) -> Image.Image:
        """Erstellt Character-Animation Manga Frame"""

        # Basis-Frame mit sanften Farben
        frame = Image.new('RGB', (width, height),
                          (250, 248, 255))  # Sehr helles Violett
        draw = ImageDraw.Draw(frame)

        # Charakter-Silhouette (vereinfacht)
        char_x = width // 2
        char_y = height // 2

        # Animierte Bewegung
        offset_x = int(20 * np.sin(2 * np.pi * progress))
        offset_y = int(10 * np.cos(4 * np.pi * progress))

        # Kopf
        head_size = min(width, height) // 8
        draw.ellipse([
            char_x - head_size + offset_x,
            char_y - head_size * 2 + offset_y,
            char_x + head_size + offset_x,
            char_y + offset_y
        ], fill=(255, 220, 200), outline=(0, 0, 0), width=3)

        # Augen (große Manga-Augen)
        eye_size = head_size // 3
        # Linkes Auge
        draw.ellipse([
            char_x - eye_size - 5 + offset_x,
            char_y - head_size + offset_y,
            char_x - 5 + offset_x,
            char_y - head_size + eye_size + offset_y
        ], fill=(135, 206, 250), outline=(0, 0, 0), width=2)

        # Rechtes Auge
        draw.ellipse([
            char_x + 5 + offset_x,
            char_y - head_size + offset_y,
            char_x + eye_size + 5 + offset_x,
            char_y - head_size + eye_size + offset_y
        ], fill=(135, 206, 250), outline=(0, 0, 0), width=2)

        # Glitzer in den Augen (Manga-typisch)
        if progress > 0.3:
            draw.ellipse([
                char_x - eye_size//2 + offset_x,
                char_y - head_size + eye_size//2 + offset_y,
                char_x - eye_size//2 + 3 + offset_x,
                char_y - head_size + eye_size//2 + 3 + offset_y
            ], fill=(255, 255, 255))

        return frame

    def _create_action_manga_frame(
        self, width: int, height: int, progress: float, motion: float
    ) -> Image.Image:
        """Erstellt Action-Sequenz Manga Frame"""

        # Dynamischer Hintergrund
        frame = Image.new('RGB', (width, height),
                          (255, 240, 240))  # Leicht rosa
        draw = ImageDraw.Draw(frame)

        # Speed Lines (Manga-typisch)
        center_x = width // 2
        center_y = height // 2

        # Animierte Speed Lines
        for i in range(12):
            angle = (i * 30 + progress * 360) % 360
            angle_rad = np.radians(angle)

            start_x = center_x + int(50 * np.cos(angle_rad))
            start_y = center_y + int(50 * np.sin(angle_rad))
            end_x = center_x + int(150 * np.cos(angle_rad))
            end_y = center_y + int(150 * np.sin(angle_rad))

            # Speed Line Dicke basierend auf Motion
            line_width = max(1, int(3 * motion / 10))

            draw.line([start_x, start_y, end_x, end_y],
                      fill=(0, 0, 0), width=line_width)

        # Zentrale Action-Form
        action_size = int(30 + 20 * np.sin(4 * np.pi * progress))
        draw.ellipse([
            center_x - action_size,
            center_y - action_size,
            center_x + action_size,
            center_y + action_size
        ], fill=(255, 69, 0), outline=(0, 0, 0), width=4)

        # Explosions-Effekt
        if progress > 0.5:
            explosion_points = []
            for i in range(8):
                angle = i * 45
                angle_rad = np.radians(angle)
                exp_x = center_x + int(action_size * 1.5 * np.cos(angle_rad))
                exp_y = center_y + int(action_size * 1.5 * np.sin(angle_rad))
                explosion_points.extend([exp_x, exp_y])

            if len(explosion_points) >= 6:
                draw.polygon(explosion_points, fill=(255, 215, 0),
                             outline=(255, 69, 0), width=2)

        return frame

    def _create_emotional_manga_frame(
        self, width: int, height: int, progress: float, motion: float
    ) -> Image.Image:
        """Erstellt Emotional-Szene Manga Frame"""

        # Sanfter Hintergrund mit Farbverlauf-Simulation
        frame = Image.new('RGB', (width, height), (255, 240, 245))  # Rosa-Weiß
        draw = ImageDraw.Draw(frame)

        # Emotionale Sparkles/Sterne
        for i in range(15):
            star_x = int((i * 47 + progress * 200) % width)
            star_y = int((i * 73 + progress * 150) % height)

            # Stern-Größe animiert
            star_size = int(3 + 2 * np.sin(2 * np.pi * progress + i))

            # Stern zeichnen
            star_points = []
            for j in range(5):
                angle = j * 72 - 90  # -90 für aufrechten Stern
                angle_rad = np.radians(angle)
                outer_x = star_x + int(star_size * np.cos(angle_rad))
                outer_y = star_y + int(star_size * np.sin(angle_rad))
                star_points.extend([outer_x, outer_y])

                # Innere Punkte
                inner_angle = angle + 36
                inner_angle_rad = np.radians(inner_angle)
                inner_x = star_x + \
                    int(star_size * 0.5 * np.cos(inner_angle_rad))
                inner_y = star_y + \
                    int(star_size * 0.5 * np.sin(inner_angle_rad))
                star_points.extend([inner_x, inner_y])

            if len(star_points) >= 6:
                draw.polygon(star_points, fill=(255, 182, 193))

        # Emotionales Zentrum
        center_x = width // 2
        center_y = height // 2

        # Pulsierendes Herz
        heart_size = int(20 + 10 * np.sin(6 * np.pi * progress))

        # Vereinfachtes Herz
        draw.ellipse([
            center_x - heart_size,
            center_y - heart_size//2,
            center_x,
            center_y + heart_size//2
        ], fill=(255, 105, 180))

        draw.ellipse([
            center_x,
            center_y - heart_size//2,
            center_x + heart_size,
            center_y + heart_size//2
        ], fill=(255, 105, 180))

        # Herz-Spitze
        draw.polygon([
            center_x, center_y + heart_size//2,
            center_x - heart_size//2, center_y,
            center_x + heart_size//2, center_y
        ], fill=(255, 105, 180))

        return frame

    def _create_transformation_manga_frame(
        self, width: int, height: int, progress: float, motion: float
    ) -> Image.Image:
        """Erstellt Transformation Manga Frame"""

        # Magischer Hintergrund
        base_color = (
            int(50 + 100 * progress),
            int(20 + 80 * progress),
            int(100 + 155 * progress)
        )
        frame = Image.new('RGB', (width, height), base_color)
        draw = ImageDraw.Draw(frame)

        center_x = width // 2
        center_y = height // 2

        # Magische Kreise
        for ring in range(3):
            radius = int(30 + ring * 25 + 20 *
                         np.sin(2 * np.pi * progress + ring))

            # Kreis-Outline
            draw.ellipse([
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius
            ], outline=(255, 215, 0), width=3)

            # Magische Symbole auf dem Kreis
            for symbol in range(8):
                angle = symbol * 45 + progress * 180
                angle_rad = np.radians(angle)
                symbol_x = center_x + int(radius * np.cos(angle_rad))
                symbol_y = center_y + int(radius * np.sin(angle_rad))

                # Kleiner magischer Punkt
                draw.ellipse([
                    symbol_x - 3, symbol_y - 3,
                    symbol_x + 3, symbol_y + 3
                ], fill=(255, 255, 255))

        # Zentrale Transformation
        transform_size = int(15 + 25 * progress)
        draw.ellipse([
            center_x - transform_size,
            center_y - transform_size,
            center_x + transform_size,
            center_y + transform_size
        ], fill=(255, 255, 255), outline=(255, 215, 0), width=2)

        # Energiestrahlen
        if progress > 0.3:
            for beam in range(6):
                angle = beam * 60
                angle_rad = np.radians(angle)
                beam_length = int(50 + 30 * np.sin(4 * np.pi * progress))

                end_x = center_x + int(beam_length * np.cos(angle_rad))
                end_y = center_y + int(beam_length * np.sin(angle_rad))

                draw.line([center_x, center_y, end_x, end_y],
                          fill=(255, 255, 255), width=3)

        return frame

    def _create_general_manga_frame(
        self, width: int, height: int, progress: float, motion: float
    ) -> Image.Image:
        """Erstellt allgemeinen Manga Frame"""

        # Standard Manga-Hintergrund
        frame = Image.new('RGB', (width, height), (248, 248, 255))
        draw = ImageDraw.Draw(frame)

        # Einfaches animiertes Element
        center_x = width // 2
        center_y = height // 2

        # Animierter Kreis mit Manga-Outline
        size = int(30 + 15 * np.sin(2 * np.pi * progress))

        draw.ellipse([
            center_x - size,
            center_y - size,
            center_x + size,
            center_y + size
        ], fill=(135, 206, 250), outline=(0, 0, 0), width=4)

        # Manga-typische Highlight
        highlight_size = size // 3
        draw.ellipse([
            center_x - highlight_size,
            center_y - highlight_size,
            center_x,
            center_y
        ], fill=(255, 255, 255))

        return frame

    def _apply_manga_effects(
        self,
        frame: Image.Image,
        style_focus: str,
        progress: float,
        motion: float
    ) -> Image.Image:
        """Wendet Manga-spezifische Effekte an"""

        # Kontrast und Sättigung erhöhen (Manga-typisch)
        enhancer = ImageEnhance.Contrast(frame)
        frame = enhancer.enhance(1.3)

        enhancer = ImageEnhance.Color(frame)
        frame = enhancer.enhance(1.2)

        # Leichte Schärfung für klare Linien
        enhancer = ImageEnhance.Sharpness(frame)
        frame = enhancer.enhance(1.1)

        return frame

    def _save_manga_gif(
        self,
        frames: List[Image.Image],
        output_path: str,
        analysis: Dict
    ) -> Dict:
        """Speichert das Manga-Style GIF"""

        fps = analysis["fps"]
        duration_ms = max(50, int(1000 / fps))  # Mindestens 50ms pro Frame

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

            file_size = os.path.getsize(output_path)

            return {
                "output_path": output_path,
                "file_size_bytes": file_size,
                "file_size_mb": file_size / (1024 * 1024),
                "frame_count": len(frames),
                "fps": fps,
                "duration_ms": duration_ms
            }

        except Exception as e:
            raise RuntimeError(f"GIF-Speicherung fehlgeschlagen: {str(e)}")

    def _create_batch_report(self, results: List[Dict]):
        """Erstellt Batch-Report"""

        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]

        report = {
            "generation_timestamp": time.time(),
            "total_processed": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) if results else 0,
            "output_directory": self.output_dir,
            "manga_style_config": self.manga_config,
            "results": results
        }

        report_path = os.path.join(
            self.output_dir, "manga_generation_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"📄 Batch-Report gespeichert: {report_path}")

        # Zusammenfassung ausgeben
        print(f"\n🎌 MANGA STYLE BATCH-GENERIERUNG ABGESCHLOSSEN")
        print(f"═══════════════════════════════════════════════════════")
        print(f"📊 Verarbeitet: {len(results)}")
        print(f"✅ Erfolgreich: {len(successful)}")
        print(f"❌ Fehlgeschlagen: {len(failed)}")
        print(f"📈 Erfolgsrate: {len(successful)/len(results)*100:.1f}%")
        print(f"📁 Output-Verzeichnis: {self.output_dir}")

        if successful:
            print(f"\n🎨 Generierte Manga-GIFs:")
            for result in successful:
                output_info = result.get("output_info", {})
                file_size = output_info.get("file_size_mb", 0)
                frame_count = output_info.get("frame_count", 0)
                print(f"   • {Path(result['output_path']).name}")
                print(f"     Prompt: {result['prompt']}")
                print(f"     Frames: {frame_count}, Größe: {file_size:.2f} MB")


def main():
    """Hauptfunktion für Manga Style Generation"""

    print("🎌 MANGA STYLE GIF GENERATOR")
    print("═══════════════════════════════════════════════════════")
    print("Generiert 5 ausgewählte GIFs aus dem Input-Verzeichnis")
    print("im authentischen Manga Art Style mit:")
    print("• Bold Outlines & Clean Lines")
    print("• Dramatic Expressions & Effects")
    print("• Speed Lines & Action Elements")
    print("• Emotional Sparkles & Highlights")
    print("• Magical Transformation Effects")
    print("═══════════════════════════════════════════════════════\n")

    try:
        generator = MangaStyleGenerator()
        results = generator.generate_manga_batch()

        print(f"\n🎉 Manga Style Generation erfolgreich abgeschlossen!")

    except Exception as e:
        logger.error(f"💥 Generation fehlgeschlagen: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
