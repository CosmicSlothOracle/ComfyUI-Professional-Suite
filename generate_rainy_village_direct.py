#!/usr/bin/env python3
"""
VERREGNETES DORF - DIREKTGENERIERUNG
===================================

Generiert ein atmosphärisches GIF eines verregneten Dorfs ohne ComfyUI-Server
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import random

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RainyVillageDirectGenerator:
    """Direkter Generator für verregnete Dorf-GIFs ohne ComfyUI-Server"""

    def __init__(self):
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

        # Basis-Parameter für die Generierung
        self.width = 896
        self.height = 576
        self.frames = 24
        self.fps = 12

        # Farbpalette für verregnetes Dorf
        self.color_palette = {
            'sky_dark': (45, 52, 64),
            'sky_storm': (67, 76, 94),
            'clouds': (88, 91, 112),
            'rain': (136, 192, 208),
            'stone_dark': (76, 68, 64),
            'stone_light': (94, 84, 78),
            'timber_dark': (62, 39, 35),
            'timber_light': (94, 63, 54),
            'roof_red': (139, 69, 69),
            'roof_dark': (85, 32, 32),
            'window_warm': (255, 215, 160),
            'lamp_glow': (255, 235, 190),
            'puddle': (88, 91, 112),
            'reflection': (180, 190, 200)
        }

    def create_village_background(self):
        """Erstellt den Hintergrund des Dorfs"""

        # Basis-Canvas
        img = Image.new('RGB', (self.width, self.height),
                        self.color_palette['sky_dark'])
        draw = ImageDraw.Draw(img)

        # Himmel mit Wolken
        self.draw_stormy_sky(draw)

        # Mittelalterliche Gebäude
        self.draw_medieval_buildings(draw)

        # Kirche im Hintergrund
        self.draw_church_silhouette(draw)

        # Straßenbeleuchtung
        self.draw_street_lamps(draw)

        # Kopfsteinpflaster
        self.draw_cobblestone_street(draw)

        return img

    def draw_stormy_sky(self, draw):
        """Zeichnet einen stürmischen Himmel"""

        # Gradient von dunkel zu weniger dunkel
        for y in range(0, self.height // 2):
            intensity = int(y / (self.height // 2) * 40)
            color = (
                self.color_palette['sky_dark'][0] + intensity,
                self.color_palette['sky_dark'][1] + intensity,
                self.color_palette['sky_dark'][2] + intensity
            )
            draw.rectangle([0, y, self.width, y+1], fill=color)

        # Dunkle Wolken
        cloud_positions = [
            (150, 80, 350, 140),
            (400, 60, 600, 120),
            (650, 90, 850, 150),
            (50, 120, 250, 180),
            (300, 140, 500, 200)
        ]

        for x1, y1, x2, y2 in cloud_positions:
            draw.ellipse([x1, y1, x2, y2], fill=self.color_palette['clouds'])

    def draw_medieval_buildings(self, draw):
        """Zeichnet mittelalterliche Gebäude"""

        # Hauptgebäude-Strukturen
        buildings = [
            {'x': 100, 'y': 280, 'w': 120, 'h': 180, 'type': 'timber'},
            {'x': 240, 'y': 300, 'w': 100, 'h': 160, 'type': 'stone'},
            {'x': 360, 'y': 270, 'w': 140, 'h': 190, 'type': 'timber'},
            {'x': 520, 'y': 290, 'w': 110, 'h': 170, 'type': 'stone'},
            {'x': 650, 'y': 260, 'w': 130, 'h': 200, 'type': 'timber'},
        ]

        for building in buildings:
            x, y, w, h = building['x'], building['y'], building['w'], building['h']

            if building['type'] == 'timber':
                # Fachwerkhaus
                # Basis
                draw.rectangle([x, y, x+w, y+h],
                               fill=self.color_palette['stone_light'])

                # Fachwerk-Balken
                for i in range(3):
                    beam_y = y + (i+1) * h // 4
                    draw.rectangle([x, beam_y-3, x+w, beam_y+3],
                                   fill=self.color_palette['timber_dark'])

                # Vertikale Balken
                for i in range(2):
                    beam_x = x + (i+1) * w // 3
                    draw.rectangle([beam_x-2, y, beam_x+2, y+h],
                                   fill=self.color_palette['timber_dark'])

                # Dach
                roof_points = [(x-10, y), (x+w//2, y-30), (x+w+10, y)]
                draw.polygon(roof_points, fill=self.color_palette['roof_red'])

            else:
                # Steinhaus
                draw.rectangle([x, y, x+w, y+h],
                               fill=self.color_palette['stone_dark'])

                # Steinmuster
                for row in range(h // 20):
                    for col in range(w // 30):
                        stone_x = x + col * 30
                        stone_y = y + row * 20
                        draw.rectangle([stone_x, stone_y, stone_x+28, stone_y+18],
                                       outline=self.color_palette['stone_light'])

                # Dach
                roof_points = [(x-5, y), (x+w//2, y-20), (x+w+5, y)]
                draw.polygon(roof_points, fill=self.color_palette['roof_dark'])

            # Fenster mit warmem Licht
            windows = [
                (x + w//4, y + h//3, x + w//4 + 20, y + h//3 + 25),
                (x + 3*w//4 - 10, y + h//3, x + 3*w//4 + 10, y + h//3 + 25),
                (x + w//4, y + 2*h//3, x + w//4 + 20, y + 2*h//3 + 25),
                (x + 3*w//4 - 10, y + 2*h//3, x + 3*w//4 + 10, y + 2*h//3 + 25)
            ]

            for wx1, wy1, wx2, wy2 in windows:
                draw.rectangle([wx1, wy1, wx2, wy2],
                               fill=self.color_palette['window_warm'])

    def draw_church_silhouette(self, draw):
        """Zeichnet eine Kirchensilhouette im Hintergrund"""

        church_x = 750
        church_y = 200
        church_w = 100
        church_h = 150

        # Kirchengebäude
        draw.rectangle([church_x, church_y, church_x+church_w, church_y+church_h],
                       fill=self.color_palette['stone_dark'])

        # Kirchturm
        tower_w = 40
        tower_h = 100
        tower_x = church_x + church_w//2 - tower_w//2
        tower_y = church_y - tower_h

        draw.rectangle([tower_x, tower_y, tower_x+tower_w, tower_y+tower_h],
                       fill=self.color_palette['stone_dark'])

        # Turmspitze
        spire_points = [(tower_x+tower_w//2, tower_y-30),
                        (tower_x, tower_y),
                        (tower_x+tower_w, tower_y)]
        draw.polygon(spire_points, fill=self.color_palette['roof_dark'])

    def draw_street_lamps(self, draw):
        """Zeichnet Straßenbeleuchtung"""

        lamp_positions = [200, 400, 600]

        for lamp_x in lamp_positions:
            lamp_y = 420

            # Laternenpfahl
            draw.rectangle([lamp_x-3, lamp_y, lamp_x+3, lamp_y+120],
                           fill=self.color_palette['timber_dark'])

            # Laterne
            draw.ellipse([lamp_x-15, lamp_y-20, lamp_x+15, lamp_y+10],
                         fill=self.color_palette['lamp_glow'])

    def draw_cobblestone_street(self, draw):
        """Zeichnet Kopfsteinpflaster"""

        street_y = 460

        # Straßenbasis
        draw.rectangle([0, street_y, self.width, self.height],
                       fill=self.color_palette['stone_dark'])

        # Kopfsteine
        for y in range(street_y, self.height, 15):
            for x in range(0, self.width, 18):
                # Kleine Variation für natürliches Aussehen
                offset_x = random.randint(-2, 2)
                offset_y = random.randint(-1, 1)

                stone_x = x + offset_x
                stone_y = y + offset_y

                draw.ellipse([stone_x, stone_y, stone_x+12, stone_y+10],
                             fill=self.color_palette['stone_light'],
                             outline=self.color_palette['stone_dark'])

    def add_rain_effect(self, img, frame_number):
        """Fügt Regeneffekt hinzu"""

        draw = ImageDraw.Draw(img)

        # Regenlinien
        num_raindrops = 150

        for _ in range(num_raindrops):
            # Zufällige Position
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)

            # Bewegung basierend auf Frame
            rain_offset = (frame_number * 8) % self.height
            y = (y + rain_offset) % self.height

            # Regentropfen-Linie
            length = random.randint(8, 15)
            thickness = random.randint(1, 2)

            draw.line([x, y, x-2, y+length],
                      fill=self.color_palette['rain'], width=thickness)

        return img

    def add_puddles_and_reflections(self, img):
        """Fügt Pfützen und Reflexionen hinzu"""

        draw = ImageDraw.Draw(img)

        # Pfützen auf der Straße
        puddle_positions = [
            (150, 480, 200, 520),
            (350, 490, 420, 530),
            (550, 485, 600, 525),
            (750, 475, 810, 515)
        ]

        for px1, py1, px2, py2 in puddle_positions:
            # Pfütze
            draw.ellipse([px1, py1, px2, py2],
                         fill=self.color_palette['puddle'])

            # Reflexion der Beleuchtung
            reflection_x = px1 + (px2-px1)//2
            reflection_y = py1 + (py2-py1)//2
            draw.ellipse([reflection_x-5, reflection_y-3, reflection_x+5, reflection_y+3],
                         fill=self.color_palette['reflection'])

        return img

    def add_atmospheric_effects(self, img):
        """Fügt atmosphärische Effekte hinzu (Nebel, Dunst)"""

        # Leichter Nebel-Effekt
        overlay = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        fog_draw = ImageDraw.Draw(overlay)

        # Nebelschwaden
        for i in range(5):
            fog_x = random.randint(0, self.width)
            fog_y = random.randint(self.height//2, self.height-100)
            fog_w = random.randint(100, 200)
            fog_h = random.randint(30, 60)

            fog_draw.ellipse([fog_x, fog_y, fog_x+fog_w, fog_y+fog_h],
                             fill=(200, 200, 220, 30))

        # Nebel mit Basisbild kombinieren
        img = Image.alpha_composite(
            img.convert('RGBA'), overlay).convert('RGB')

        return img

    def enhance_cinematic_quality(self, img):
        """Verbessert die kinematische Qualität"""

        # Kontrast erhöhen
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)

        # Sättigung leicht reduzieren für düstere Stimmung
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.9)

        # Helligkeit anpassen
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.85)

        # Leichter Unschärfe-Effekt für Tiefe
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        return img

    def generate_frame(self, frame_number):
        """Generiert einen einzelnen Frame"""

        logger.info(f"Generiere Frame {frame_number + 1}/{self.frames}")

        # Basis-Dorfszene
        img = self.create_village_background()

        # Regeneffekt hinzufügen
        img = self.add_rain_effect(img, frame_number)

        # Pfützen und Reflexionen
        img = self.add_puddles_and_reflections(img)

        # Atmosphärische Effekte
        img = self.add_atmospheric_effects(img)

        # Kinematische Verbesserungen
        img = self.enhance_cinematic_quality(img)

        return img

    def create_gif(self, frames, output_path):
        """Erstellt das finale GIF"""

        logger.info(f"Erstelle GIF mit {len(frames)} Frames...")

        # GIF speichern
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000/self.fps),  # Millisekunden pro Frame
            loop=0,
            optimize=True
        )

        # Dateigröße anzeigen
        file_size = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"GIF gespeichert: {output_path} ({file_size:.1f} MB)")

    def generate_rainy_village(self):
        """Hauptfunktion für die Generierung"""

        logger.info("=" * 60)
        logger.info("VERREGNETES DORF - DIREKTGENERIERUNG GESTARTET")
        logger.info("=" * 60)
        logger.info(
            f"Parameter: {self.width}x{self.height}, {self.frames} Frames, {self.fps} FPS")

        start_time = time.time()

        # Alle Frames generieren
        frames = []
        for frame_num in range(self.frames):
            frame = self.generate_frame(frame_num)
            frames.append(frame)

            # Fortschritt anzeigen
            progress = (frame_num + 1) / self.frames * 100
            logger.info(f"Fortschritt: {progress:.1f}%")

        # GIF erstellen
        timestamp = int(time.time())
        output_filename = f"atmospheric_rainy_village_{timestamp}.gif"
        output_path = self.output_dir / output_filename

        self.create_gif(frames, output_path)

        # Statistiken
        generation_time = time.time() - start_time

        logger.info("=" * 60)
        logger.info("GENERIERUNG ERFOLGREICH ABGESCHLOSSEN!")
        logger.info("=" * 60)
        logger.info(f"Ausgabedatei: {output_path}")
        logger.info(f"Generierungszeit: {generation_time:.1f} Sekunden")
        logger.info(f"Frames: {self.frames}")
        logger.info(f"FPS: {self.fps}")
        logger.info(f"Auflösung: {self.width}x{self.height}")
        logger.info("=" * 60)

        return str(output_path)


def main():
    """Hauptfunktion"""

    print("🌧️  VERREGNETES DORF - DIREKTGENERIERUNG")
    print("========================================")
    print("Generiert ein atmosphärisches GIF ohne ComfyUI-Server")
    print()

    try:
        generator = RainyVillageDirectGenerator()
        output_file = generator.generate_rainy_village()

        print(f"\n✅ Erfolgreich! Das verregnete Dorf-GIF wurde generiert:")
        print(f"📁 {output_file}")
        print(f"\n🎬 Das GIF zeigt:")
        print("   - Mittelalterliche deutsche Architektur")
        print("   - Atmosphärischen Regen mit Animation")
        print("   - Warme Fensterbeleuchtung")
        print("   - Kopfsteinpflaster mit Pfützen")
        print("   - Stürmischer Himmel mit Wolken")
        print("   - Kinematische Qualität")

        return True

    except Exception as e:
        print(f"\n❌ Fehler bei der Generierung: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
