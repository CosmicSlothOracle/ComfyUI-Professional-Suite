#!/usr/bin/env python3
"""
Direkte Sprite-Verarbeitung: Extrahiert Frames aus Sprite-Sheets
Bereitet sie für AI-Processing vor
"""

import os
import json
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np


class DirectSpriteProcessor:
    def __init__(self):
        self.input_dir = Path("input")
        self.output_dir = Path("output")
        self.frames_dir = self.output_dir / "extracted_frames"
        self.pose_dir = self.output_dir / "pose_detection"
        self.styled_dir = self.output_dir / "styled_sprites"

        # Verzeichnisse erstellen
        for directory in [self.frames_dir, self.pose_dir, self.styled_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def parse_sprite_filename(self, filename):
        """Analysiert Sprite-Dateinamen für Grid-Informationen"""
        # Format: action_frames_size_gridXxY.png
        # Beispiel: idle_9_512x512_grid9x1.png
        parts = filename.stem.split('_')

        if len(parts) >= 4:
            action = parts[0]
            frames = int(parts[1])
            size = parts[2]  # z.B. "512x512"
            grid_info = parts[3]  # z.B. "grid9x1"

            # Grid-Größe extrahieren
            if grid_info.startswith('grid'):
                grid_part = grid_info[4:]  # Entferne 'grid'
                if 'x' in grid_part:
                    cols, rows = map(int, grid_part.split('x'))
                    return {
                        'action': action,
                        'frames': frames,
                        'size': size,
                        'cols': cols,
                        'rows': rows
                    }

        return None

    def extract_frames_from_sprite(self, sprite_path, sprite_info):
        """Extrahiert einzelne Frames aus einem Sprite-Sheet"""
        print(f"🎮 Verarbeite: {sprite_path.name}")
        print(f"   📐 Grid: {sprite_info['cols']}x{sprite_info['rows']}")
        print(f"   🎯 Frames: {sprite_info['frames']}")

        # Sprite-Sheet laden
        sprite_image = Image.open(sprite_path)
        width, height = sprite_image.size

        # Frame-Größe berechnen
        frame_width = width // sprite_info['cols']
        frame_height = height // sprite_info['rows']

        print(f"   📏 Frame-Größe: {frame_width}x{frame_height}")

        frames = []
        frame_paths = []

        # Frames extrahieren
        frame_count = 0
        for row in range(sprite_info['rows']):
            for col in range(sprite_info['cols']):
                if frame_count >= sprite_info['frames']:
                    break

                # Frame-Position berechnen
                left = col * frame_width
                top = row * frame_height
                right = left + frame_width
                bottom = top + frame_height

                # Frame extrahieren
                frame = sprite_image.crop((left, top, right, bottom))
                frames.append(frame)

                # Frame speichern
                frame_filename = f"{sprite_info['action']}_frame_{frame_count:03d}.png"
                frame_path = self.frames_dir / frame_filename
                frame.save(frame_path)
                frame_paths.append(frame_path)

                frame_count += 1

            if frame_count >= sprite_info['frames']:
                break

        print(f"   ✅ {len(frames)} Frames extrahiert")
        return frames, frame_paths

    def create_pose_detection_placeholder(self, frames, sprite_info):
        """Erstellt Pose-Detection Platzhalter (vereinfacht)"""
        print(f"   🤖 Erstelle Pose-Detection Daten...")

        pose_data = []

        for i, frame in enumerate(frames):
            # Vereinfachte "Pose-Detection" - findet Bildmittelpunkt und Konturen
            # In der echten Implementierung würde hier OpenPose verwendet

            # Frame zu numpy array konvertieren
            frame_array = np.array(frame)

            # Einfache Schwerpunkt-Berechnung
            if len(frame_array.shape) == 3:
                # RGB zu Graustufen
                gray = np.mean(frame_array, axis=2)
            else:
                gray = frame_array

            # Finde Pixel mit Inhalt (nicht transparent/weiß)
            content_pixels = np.where(gray < 240)  # Nicht-weiße Pixel

            if len(content_pixels[0]) > 0:
                center_y = int(np.mean(content_pixels[0]))
                center_x = int(np.mean(content_pixels[1]))
            else:
                center_y, center_x = frame.height // 2, frame.width // 2

            # Pose-Daten (vereinfacht)
            pose_info = {
                'frame': i,
                'center': [center_x, center_y],
                'action': sprite_info['action'],
                'confidence': 0.8  # Platzhalter-Wert
            }
            pose_data.append(pose_info)

            # Visualisierung erstellen
            pose_vis = frame.copy()
            draw = ImageDraw.Draw(pose_vis)

            # Zentrum markieren
            radius = 5
            draw.ellipse([
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius
            ], fill='red', outline='darkred', width=2)

            # Pose-Visualisierung speichern
            pose_filename = f"{sprite_info['action']}_pose_{i:03d}.png"
            pose_path = self.pose_dir / pose_filename
            pose_vis.save(pose_path)

        # Pose-Daten als JSON speichern
        pose_json_path = self.pose_dir / f"{sprite_info['action']}_poses.json"
        with open(pose_json_path, 'w', encoding='utf-8') as f:
            json.dump(pose_data, f, indent=2, ensure_ascii=False)

        print(f"   ✅ Pose-Daten erstellt: {len(pose_data)} Posen")
        return pose_data

    def create_style_transfer_setup(self, sprite_info, pose_data):
        """Erstellt Setup für Style-Transfer"""
        print(f"   🎨 Erstelle Style-Transfer Setup...")

        # Style-Varianten definieren
        styles = {
            'anime': {
                'prompt': f'anime character, {sprite_info["action"]} pose, cel shading, vibrant colors',
                'negative': 'blurry, low quality, western cartoon'
            },
            'pixel_art': {
                'prompt': f'pixel art character, {sprite_info["action"]} animation, 16-bit style, retro game',
                'negative': 'blurry, smooth, anti-aliased, high resolution'
            },
            'enhanced': {
                'prompt': f'detailed character, {sprite_info["action"]} pose, high quality, professional artwork',
                'negative': 'blurry, low quality, amateur'
            }
        }

        # Setup-Informationen speichern
        setup_info = {
            'sprite_info': sprite_info,
            'pose_data': pose_data,
            'styles': styles,
            'processing_notes': [
                f"Verwende {len(pose_data)} extrahierte Frames",
                f"Action: {sprite_info['action']}",
                f"Original-Grid: {sprite_info['cols']}x{sprite_info['rows']}",
                "Für ComfyUI: Lade Frames einzeln und verwende Pose-Daten"
            ]
        }

        setup_path = self.styled_dir / \
            f"{sprite_info['action']}_style_setup.json"
        with open(setup_path, 'w', encoding='utf-8') as f:
            json.dump(setup_info, f, indent=2, ensure_ascii=False)

        print(f"   ✅ Style-Setup erstellt")
        return setup_info

    def process_all_sprites(self):
        """Verarbeitet alle Sprite-Sheets im Input-Verzeichnis"""
        print("🎮 DIREKTE SPRITE-VERARBEITUNG")
        print("=" * 60)

        # Sprite-Dateien finden
        sprite_files = []
        for pattern in ['*.png', '*.jpg', '*.jpeg']:
            sprite_files.extend(self.input_dir.glob(pattern))

        # Nur Sprite-Sheets mit erkennbarem Format
        valid_sprites = []
        for sprite_file in sprite_files:
            sprite_info = self.parse_sprite_filename(sprite_file)
            if sprite_info:
                valid_sprites.append((sprite_file, sprite_info))

        if not valid_sprites:
            print("❌ Keine gültigen Sprite-Sheets gefunden!")
            print("💡 Erwartetes Format: action_frames_size_gridXxY.png")
            return

        print(f"📋 Gefundene Sprite-Sheets: {len(valid_sprites)}")
        print()

        processed_results = []

        for sprite_path, sprite_info in valid_sprites:
            print(f"🔄 Verarbeite: {sprite_path.name}")

            try:
                # 1. Frames extrahieren
                frames, frame_paths = self.extract_frames_from_sprite(
                    sprite_path, sprite_info)

                # 2. Pose-Detection (vereinfacht)
                pose_data = self.create_pose_detection_placeholder(
                    frames, sprite_info)

                # 3. Style-Transfer Setup
                setup_info = self.create_style_transfer_setup(
                    sprite_info, pose_data)

                processed_results.append({
                    'sprite': sprite_path.name,
                    'info': sprite_info,
                    'frames_extracted': len(frames),
                    'pose_data': len(pose_data),
                    'setup_created': True
                })

                print(f"   ✅ Verarbeitung abgeschlossen")
                print()

            except Exception as e:
                print(f"   ❌ Fehler: {e}")
                print()

        # Zusammenfassung erstellen
        summary = {
            'processed_sprites': len(processed_results),
            'total_frames': sum(r['frames_extracted'] for r in processed_results),
            'results': processed_results,
            'next_steps': [
                "1. Öffne ComfyUI Interface: http://localhost:8188",
                "2. Lade einen der erstellten Workflows",
                "3. Ersetze Input-Bilder mit extrahierten Frames",
                "4. Starte AI-Processing für gewünschte Styles"
            ]
        }

        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("=" * 60)
        print("📊 VERARBEITUNG ABGESCHLOSSEN")
        print("=" * 60)
        print(f"✅ Sprite-Sheets verarbeitet: {len(processed_results)}")
        print(f"🎯 Frames extrahiert: {summary['total_frames']}")
        print(f"📁 Ausgabe-Verzeichnisse:")
        print(f"   🖼️  Frames: {self.frames_dir}")
        print(f"   🤖 Pose-Daten: {self.pose_dir}")
        print(f"   🎨 Style-Setups: {self.styled_dir}")
        print()
        print("🚀 NÄCHSTE SCHRITTE:")
        for step in summary['next_steps']:
            print(f"   {step}")


def main():
    """Hauptfunktion"""
    processor = DirectSpriteProcessor()
    processor.process_all_sprites()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Verarbeitung abgebrochen")
    except Exception as e:
        print(f"\n💥 Fehler: {e}")
        import traceback
        traceback.print_exc()
