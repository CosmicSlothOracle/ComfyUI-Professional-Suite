#!/usr/bin/env python3
"""
Sprite-Sheet Format Checker für AI-Optimierung
=============================================
Prüft ob Sprite-Sheets optimal für AI-Verarbeitung formatiert sind.
"""

import os
from pathlib import Path
from PIL import Image
import math


def check_sprite_format(image_path):
    """Prüfe ein einzelnes Sprite-Sheet auf AI-Optimierung"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            format_type = img.format
            mode = img.mode

            # Grundlegende Informationen
            info = {
                'filename': Path(image_path).name,
                'size': f"{width}x{height}",
                'format': format_type,
                'mode': mode,
                'has_transparency': mode in ['RGBA', 'LA'] or 'transparency' in img.info,
                'issues': [],
                'recommendations': [],
                'score': 0
            }

            # Bewertung der Dimensionen
            if width == height:
                info['score'] += 20
                info['recommendations'].append(
                    "✅ Quadratisches Format ist gut für AI")
            else:
                info['issues'].append(
                    "⚠️ Nicht-quadratisches Format kann AI-Probleme verursachen")

            # Bewertung der Größe
            total_pixels = width * height
            if 512*512 <= total_pixels <= 2048*2048:
                info['score'] += 30
                info['recommendations'].append(
                    "✅ Optimale Größe für AI-Verarbeitung")
            elif total_pixels < 256*256:
                info['issues'].append(
                    "❌ Zu klein - AI benötigt mindestens 256x256")
            elif total_pixels > 4096*4096:
                info['issues'].append("⚠️ Sehr groß - könnte langsam werden")
            else:
                info['score'] += 15
                info['recommendations'].append("🔶 Akzeptable Größe")

            # Bewertung möglicher Grid-Aufteilung
            possible_grids = []
            for grid_size in [2, 3, 4, 6, 8, 12, 16]:
                if width % grid_size == 0 and height % grid_size == 0:
                    frame_w = width // grid_size
                    frame_h = height // grid_size
                    if 32 <= frame_w <= 512 and 32 <= frame_h <= 512:
                        possible_grids.append(
                            f"{grid_size}x{grid_size} ({frame_w}x{frame_h} pro Frame)")

            # Bewertung rechteckiger Grids
            for rows in [1, 2, 3, 4, 6, 8]:
                for cols in [2, 3, 4, 6, 8, 12, 16]:
                    if width % cols == 0 and height % rows == 0:
                        frame_w = width // cols
                        frame_h = height // rows
                        if 32 <= frame_w <= 512 and 32 <= frame_h <= 512 and f"{cols}x{rows}" not in [g.split()[0] for g in possible_grids]:
                            possible_grids.append(
                                f"{cols}x{rows} ({frame_w}x{frame_h} pro Frame)")

            if possible_grids:
                info['score'] += 25
                info['possible_grids'] = possible_grids[:5]  # Top 5
                info['recommendations'].append(
                    f"✅ {len(possible_grids)} mögliche Grid-Aufteilungen gefunden")
            else:
                info['issues'].append(
                    "❌ Keine sinnvolle Grid-Aufteilung möglich")

            # Format-Bewertung
            if format_type == 'PNG':
                info['score'] += 15
                info['recommendations'].append(
                    "✅ PNG-Format ist optimal für AI")
            elif format_type in ['JPEG', 'JPG']:
                info['score'] += 5
                info['issues'].append("🔶 JPG-Format - PNG wäre besser für AI")
            else:
                info['issues'].append(
                    f"⚠️ Ungewöhnliches Format: {format_type}")

            # Transparenz-Bewertung
            if info['has_transparency']:
                info['score'] += 10
                info['recommendations'].append(
                    "✅ Transparenz hilft bei Pose-Erkennung")
            else:
                info['recommendations'].append(
                    "🔶 Transparenz würde Pose-Erkennung verbessern")

            # Gesamtbewertung
            if info['score'] >= 80:
                info['rating'] = "🥇 OPTIMAL"
                info['ai_ready'] = True
            elif info['score'] >= 60:
                info['rating'] = "🥈 GUT"
                info['ai_ready'] = True
            elif info['score'] >= 40:
                info['rating'] = "🥉 AKZEPTABEL"
                info['ai_ready'] = True
            else:
                info['rating'] = "❌ PROBLEMATISCH"
                info['ai_ready'] = False

            return info

    except Exception as e:
        return {
            'filename': Path(image_path).name,
            'error': f"Fehler beim Laden: {e}",
            'ai_ready': False
        }


def suggest_improvements(info):
    """Schlage konkrete Verbesserungen vor"""
    suggestions = []

    if not info.get('ai_ready', False):
        suggestions.append("🔧 VERBESSERUNGSVORSCHLÄGE:")

        # Größenempfehlungen
        if 'Zu klein' in str(info.get('issues', [])):
            suggestions.append("   📏 Vergrößere auf mindestens 512x512 Pixel")
        elif 'Sehr groß' in str(info.get('issues', [])):
            suggestions.append("   📏 Verkleinere auf maximal 2048x2048 Pixel")

        # Format-Empfehlungen
        if any('JPG' in issue for issue in info.get('issues', [])):
            suggestions.append("   🖼️ Konvertiere zu PNG-Format")

        # Grid-Empfehlungen
        if not info.get('possible_grids'):
            suggestions.append(
                "   🔲 Passe Dimensionen für gleichmäßige Grid-Aufteilung an")
            suggestions.append(
                "   💡 Empfohlen: 512x512 (4x4), 1024x1024 (8x8), 768x768 (6x6)")

        # Transparenz-Empfehlung
        if not info.get('has_transparency'):
            suggestions.append(
                "   🎭 Füge transparenten Hintergrund hinzu (hilft AI bei Pose-Erkennung)")

    return suggestions


def scan_sprite_directory(directory="input/sprite_sheets"):
    """Scanne alle Sprite-Sheets im Verzeichnis"""
    sprite_dir = Path(directory)

    if not sprite_dir.exists():
        print(f"❌ Verzeichnis nicht gefunden: {sprite_dir}")
        return

    # Unterstützte Formate
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']

    # Finde alle Bilddateien
    sprite_files = []
    for ext in image_extensions:
        sprite_files.extend(sprite_dir.glob(f"*{ext}"))
        sprite_files.extend(sprite_dir.glob(f"*{ext.upper()}"))

    if not sprite_files:
        print(f"📁 Keine Sprite-Sheets gefunden in: {sprite_dir}")
        print("💡 Lege deine Sprite-Sheets (.png, .jpg) in dieses Verzeichnis!")
        return

    print(f"🔍 SPRITE-SHEET FORMAT ANALYSE")
    print(f"Verzeichnis: {sprite_dir}")
    print(f"Gefundene Dateien: {len(sprite_files)}")
    print("=" * 60)

    results = []
    optimal_count = 0

    for sprite_file in sprite_files:
        print(f"\n📄 {sprite_file.name}")
        print("-" * 40)

        info = check_sprite_format(sprite_file)
        results.append(info)

        if 'error' in info:
            print(f"❌ {info['error']}")
            continue

        # Grundinfo
        print(f"📐 Größe: {info['size']}")
        print(f"📁 Format: {info['format']}")
        print(f"🎭 Transparenz: {'Ja' if info['has_transparency'] else 'Nein'}")
        print(f"⭐ Bewertung: {info['rating']} ({info['score']}/100)")

        # Mögliche Grids
        if info.get('possible_grids'):
            print(f"🔲 Mögliche Grids: {', '.join(info['possible_grids'][:3])}")

        # Empfehlungen
        if info['recommendations']:
            print("✅ Positive Aspekte:")
            for rec in info['recommendations'][:3]:
                print(f"   {rec}")

        # Probleme
        if info['issues']:
            print("⚠️ Verbesserungsmöglichkeiten:")
            for issue in info['issues']:
                print(f"   {issue}")

        # Verbesserungsvorschläge
        suggestions = suggest_improvements(info)
        if suggestions:
            print()
            for suggestion in suggestions[:5]:
                print(suggestion)

        if info['ai_ready'] and info['score'] >= 80:
            optimal_count += 1

    # Zusammenfassung
    print("\n" + "=" * 60)
    print("📊 ZUSAMMENFASSUNG")
    print("=" * 60)
    print(f"Analysierte Dateien: {len(results)}")
    print(f"Optimal für AI: {optimal_count}")
    print(
        f"AI-kompatibel: {sum(1 for r in results if r.get('ai_ready', False))}")

    if optimal_count == len(results) and len(results) > 0:
        print("\n🎉 ALLE SPRITE-SHEETS SIND OPTIMAL FÜR AI!")
        print("🚀 Du kannst direkt mit der AI-Verarbeitung beginnen!")
    elif any(r.get('ai_ready', False) for r in results):
        print("\n🔶 EINIGE SPRITES SIND AI-BEREIT")
        print("💡 Optimiere die anderen für beste Ergebnisse")
    else:
        print("\n⚠️ SPRITES BENÖTIGEN OPTIMIERUNG")
        print("📖 Siehe Anleitung: input/sprite_sheets/SPRITE_FORMAT_ANLEITUNG.md")


def main():
    print("🎮 SPRITE-SHEET FORMAT CHECKER")
    print("Prüft ob deine Sprites optimal für AI-Verarbeitung sind")
    print()

    # Stelle sicher dass das Input-Verzeichnis existiert
    input_dir = Path("input/sprite_sheets")
    input_dir.mkdir(parents=True, exist_ok=True)

    scan_sprite_directory()

    print()
    print("💡 NÄCHSTE SCHRITTE:")
    print("1. Optimiere deine Sprites basierend auf den Empfehlungen")
    print("2. Starte ComfyUI: python main.py --listen")
    print("3. Lade Workflow: workflows/sprite_processing/sprite_extractor.json")
    print("4. Beginne AI-Processing!")


if __name__ == "__main__":
    main()
