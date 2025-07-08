#!/usr/bin/env python3
"""
AI RESULTS ANALYSIS - SERIOUSLY WEIRD STUFF CHECK
Analysiert alle AI-generierten Ergebnisse
"""

import os
import json
from pathlib import Path
from collections import defaultdict


def analyze_ai_results():
    """Analysiert alle AI-Ergebnisse"""
    print("🤖 AI RESULTS ANALYSIS - SERIOUSLY WEIRD STUFF")
    print("=" * 60)

    output_dir = Path("ComfyUI_engine/output")

    # Alle AI-generierten Dateien finden
    ai_files = list(output_dir.glob("AI_*.png"))
    other_files = [f for f in output_dir.glob(
        "*.png") if not f.name.startswith("AI_")]

    print(f"📊 GEFUNDENE DATEIEN:")
    print(f"   🤖 AI-generierte Sprites: {len(ai_files)}")
    print(f"   📁 Andere Bilder: {len(other_files)}")
    print(f"   📈 TOTAL: {len(ai_files) + len(other_files)} Bilder")
    print()

    # AI-Dateien nach Sprite und Style kategorisieren
    sprite_analysis = defaultdict(lambda: defaultdict(list))

    for ai_file in ai_files:
        # Parse filename: AI_sprite_style__number_.png
        name_parts = ai_file.stem.split('_')
        if len(name_parts) >= 3:
            sprite = name_parts[1]  # idle, walk, jump, attack, intro
            style = name_parts[2]   # anime, pixel, enhanced

            file_size = ai_file.stat().st_size
            sprite_analysis[sprite][style].append({
                'filename': ai_file.name,
                'size_kb': round(file_size / 1024, 1),
                'path': str(ai_file)
            })

    # Detaillierte Analyse
    print("🎮 AI-SPRITE ANALYSE:")
    print("=" * 60)

    total_ai_images = 0

    for sprite in sorted(sprite_analysis.keys()):
        print(f"\n🎯 SPRITE: {sprite.upper()}")
        print("-" * 30)

        for style in sorted(sprite_analysis[sprite].keys()):
            files = sprite_analysis[sprite][style]
            total_size = sum(f['size_kb'] for f in files)

            print(f"   🎨 {style}: {len(files)} Bilder ({total_size:.1f} KB)")

            for file_info in files:
                print(
                    f"      📄 {file_info['filename']} ({file_info['size_kb']} KB)")

            total_ai_images += len(files)

    # Andere interessante Dateien
    if other_files:
        print(f"\n🔍 ANDERE GENERIERTE BILDER:")
        print("-" * 30)

        for other_file in other_files[:10]:  # Zeige erste 10
            file_size = other_file.stat().st_size / 1024
            print(f"   📄 {other_file.name} ({file_size:.1f} KB)")

        if len(other_files) > 10:
            print(f"   ... und {len(other_files) - 10} weitere")

    # SERIOUSLY WEIRD STUFF BEWERTUNG
    print(f"\n🤯 SERIOUSLY WEIRD STUFF BEWERTUNG:")
    print("=" * 60)

    weirdness_score = 0

    # Kriterium 1: Anzahl AI-Generierungen
    if total_ai_images >= 15:
        weirdness_score += 50
        print(f"✅ KRASS: {total_ai_images} AI-Sprites generiert (+50 Punkte)")

    # Kriterium 2: Verschiedene Styles
    unique_styles = set()
    for sprite_data in sprite_analysis.values():
        unique_styles.update(sprite_data.keys())

    if len(unique_styles) >= 3:
        weirdness_score += 30
        print(
            f"✅ VIELFALT: {len(unique_styles)} verschiedene Styles (+30 Punkte)")

    # Kriterium 3: Alle Sprite-Typen abgedeckt
    if len(sprite_analysis) >= 4:
        weirdness_score += 20
        print(
            f"✅ VOLLSTÄNDIG: {len(sprite_analysis)} Sprite-Typen (+20 Punkte)")

    # Kriterium 4: Dateigröße deutet auf echte AI-Bilder hin
    avg_size = sum(f['size_kb'] for sprite_data in sprite_analysis.values()
                   for style_data in sprite_data.values()
                   for f in style_data) / total_ai_images if total_ai_images > 0 else 0

    if avg_size > 200:
        weirdness_score += 25
        print(
            f"✅ QUALITÄT: Ø {avg_size:.1f} KB pro Bild (echte AI-Qualität) (+25 Punkte)")

    # Kriterium 5: Zusätzliche Generierungen
    if len(other_files) > 50:
        weirdness_score += 15
        print(
            f"✅ BONUS: {len(other_files)} zusätzliche Generierungen (+15 Punkte)")

    # Gesamtbewertung
    print(f"\n🏆 GESAMTBEWERTUNG:")
    print(f"   🎯 Weirdness Score: {weirdness_score}/140")

    if weirdness_score >= 120:
        rating = "🤯 SERIOUSLY FUCKING WEIRD!"
        comment = "Das ist absolut verrückt! Die AI hat eine MASSIVE Menge hochwertiger Sprites generiert!"
    elif weirdness_score >= 100:
        rating = "🔥 EXTREMELY WEIRD!"
        comment = "Beeindruckende AI-Leistung mit vielen verschiedenen Sprites!"
    elif weirdness_score >= 80:
        rating = "😱 VERY WEIRD!"
        comment = "Solide AI-Generierung mit guter Vielfalt!"
    elif weirdness_score >= 60:
        rating = "🤔 PRETTY WEIRD!"
        comment = "Gute AI-Ergebnisse, aber könnte mehr sein!"
    else:
        rating = "😐 NOT THAT WEIRD"
        comment = "Standard AI-Output, nichts Besonderes."

    print(f"   📈 Rating: {rating}")
    print(f"   💬 Kommentar: {comment}")

    # Empfehlungen
    print(f"\n🚀 NÄCHSTE SCHRITTE:")
    print("-" * 30)
    print("1. 🖼️  Öffne ComfyUI_engine/output/ und schaue dir die Bilder an!")
    print("2. 🎨 Vergleiche die verschiedenen Styles (anime, pixel, enhanced)")
    print("3. 🎮 Verwende die besten Sprites für dein Spiel")
    print("4. 🔄 Starte weitere AI-Generierungen für mehr Variationen")

    # Ergebnisse speichern
    results_summary = {
        'total_ai_sprites': total_ai_images,
        'sprite_types': len(sprite_analysis),
        'styles': list(unique_styles),
        'weirdness_score': weirdness_score,
        'rating': rating,
        'sprite_breakdown': dict(sprite_analysis),
        'recommendations': [
            "Schaue dir die generierten Bilder an",
            "Vergleiche Styles",
            "Wähle beste Sprites aus",
            "Starte weitere Generierungen"
        ]
    }

    results_file = Path("ai_results_summary.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2,
                  ensure_ascii=False, default=str)

    print(f"\n📄 Detaillierte Analyse gespeichert: {results_file}")


if __name__ == "__main__":
    analyze_ai_results()
