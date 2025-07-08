#!/usr/bin/env python3
"""
GENTLE VAPORWAVE WORKFLOW - SANFTE PARAMETER
Reduzierte Intensität zur Vermeidung von:
- Zu intensive Kanten
- Verschwimmende Grenzen
- Fragment-Bildung
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import json


def analyze_video_properties(video_path):
    """Analysiere Video-Eigenschaften für sanfte Parameter"""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    # Analysiere erste 5 Frames (weniger für schnellere Analyse)
    frames_analyzed = 0
    brightness_values = []
    contrast_values = []
    saturation_values = []
    edge_density_values = []

    while frames_analyzed < 5:
        ret, frame = cap.read()
        if not ret:
            break

        # Konvertiere zu verschiedenen Farbräumen
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Berechne Statistiken
        brightness = np.mean(lab[:, :, 0])  # L-Kanal
        contrast = np.std(lab[:, :, 0])     # Kontrast als Standardabweichung
        saturation = np.mean(hsv[:, :, 1])  # S-Kanal

        # Kanten-Dichte analysieren
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        brightness_values.append(brightness)
        contrast_values.append(contrast)
        saturation_values.append(saturation)
        edge_density_values.append(edge_density)

        frames_analyzed += 1

    cap.release()

    if not brightness_values:
        return None

    # Berechne Durchschnittswerte
    analysis = {
        'brightness': np.mean(brightness_values),      # 0-255
        'contrast': np.mean(contrast_values),          # Höher = mehr Kontrast
        'saturation': np.mean(saturation_values),      # 0-255
        # 0-1, mehr = mehr Details
        'edge_density': np.mean(edge_density_values),
        'is_bright': np.mean(brightness_values) > 140,  # Erhöhte Schwelle
        # Reduzierte Schwelle
        'is_high_contrast': np.mean(contrast_values) > 40,
        # Reduzierte Schwelle
        'is_saturated': np.mean(saturation_values) > 80,
        'has_many_edges': np.mean(edge_density_values) > 0.1  # Neue Metric
    }

    return analysis


def calculate_gentle_parameters(analysis, target_style):
    """Berechne sanfte Parameter zur Vermeidung von Artefakten"""

    if not analysis:
        return get_gentle_defaults(target_style)

    params = {}

    if target_style == "neon":
        # SANFTE NEON-Parameter
        base_brightness = 0.1 if analysis['is_bright'] else 0.15  # Reduziert!
        # Stark reduziert!
        base_contrast = 1.2 if analysis['is_high_contrast'] else 1.15
        # Stark reduziert!
        base_saturation = 1.3 if analysis['is_saturated'] else 1.5

        # Weitere Reduktion bei vielen Kanten
        if analysis['has_many_edges']:
            base_contrast *= 0.9  # Weitere Reduktion
            base_brightness *= 0.8

        params.update({
            'brightness_adjust': base_brightness,
            'contrast_adjust': base_contrast,
            'saturation_adjust': base_saturation,
            'hue_shift': 280,  # Sanfter Pink-Shift
            'glow_intensity': 0.3,  # Stark reduziert!
            'edge_preservation': 0.8,  # Neue Parameter für Kanten-Erhaltung
            'smoothing_factor': 0.5   # Sanfte Glättung
        })

    elif target_style == "retro":
        # SANFTE RETRO-Parameter
        params.update({
            'brightness_adjust': 0.08,   # Sehr sanft
            'contrast_adjust': 1.1,      # Minimal
            'saturation_adjust': 1.4,    # Reduziert
            'hue_shift': 15,             # Sanfter Warm-Shift
            'vhs_noise': 0.1,            # Stark reduziert
            'edge_preservation': 0.9,
            'vintage_softness': 0.3      # Neue sanfte Vintage-Effekte
        })

    elif target_style == "glitch":
        # SANFTE GLITCH-Parameter (Oxymoron, aber machbar)
        params.update({
            'brightness_adjust': 0.05,
            'contrast_adjust': 1.25,     # Reduziert
            'saturation_adjust': 1.6,    # Stark reduziert
            'hue_shift': 160,            # Sanfter Cyan
            'glitch_intensity': 0.2,     # Sehr sanft
            'noise_level': 0.15,         # Reduziert
            'edge_preservation': 0.7
        })

    return params


def get_gentle_defaults(style):
    """Sanfte Fallback-Parameter"""

    defaults = {
        "neon": {
            'brightness_adjust': 0.12,
            'contrast_adjust': 1.15,
            'saturation_adjust': 1.4,
            'hue_shift': 280,
            'glow_intensity': 0.3,
            'edge_preservation': 0.8,
            'smoothing_factor': 0.5
        },
        "retro": {
            'brightness_adjust': 0.08,
            'contrast_adjust': 1.1,
            'saturation_adjust': 1.3,
            'hue_shift': 15,
            'vhs_noise': 0.1,
            'edge_preservation': 0.9,
            'vintage_softness': 0.3
        },
        "glitch": {
            'brightness_adjust': 0.05,
            'contrast_adjust': 1.2,
            'saturation_adjust': 1.5,
            'hue_shift': 160,
            'glitch_intensity': 0.2,
            'noise_level': 0.1,
            'edge_preservation': 0.8
        }
    }

    return defaults.get(style, defaults["neon"])


def apply_gentle_neon_effects(frame, params):
    """Sanfte Neon-Effekte mit Kanten-Erhaltung"""

    original = frame.copy()

    # Konvertiere zu HSV für sanfte Farbmanipulation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    # SANFTE Sättigung (mit Begrenzung)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params['saturation_adjust'], 0, 255)

    # SANFTE Helligkeit
    hsv[:, :, 2] = np.clip(
        hsv[:, :, 2] * (1 + params['brightness_adjust']), 0, 255)

    # SANFTER Hue-Shift
    hsv[:, :, 0] = (hsv[:, :, 0] + params['hue_shift'] //
                    4) % 180  # Reduziert!

    # Zurück zu BGR
    hsv = hsv.astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # SANFTER Kontrast mit Begrenzung
    result = result.astype(np.float32)
    result = np.clip(result * params['contrast_adjust'], 0, 255)
    result = result.astype(np.uint8)

    # SEHR SANFTER Glow-Effekt
    if params['glow_intensity'] > 0:
        # Kleinerer Kernel für weniger Verschwimmen
        blurred = cv2.GaussianBlur(result, (7, 7), 1.0)  # Reduziert!
        glow_strength = params['glow_intensity']
        result = cv2.addWeighted(
            result, 1-glow_strength*0.3, blurred, glow_strength*0.3, 0)

    # Kanten-Erhaltung durch selektives Mischen
    if params.get('edge_preservation', 0) > 0:
        # Erkenne Kanten
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_orig, 30, 100)  # Sanftere Kanten-Erkennung
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        edges_normalized = edges.astype(np.float32) / 255.0

        # Mische Original und Effekt basierend auf Kanten
        preservation = params['edge_preservation']
        for c in range(3):
            result[:, :, c] = (result[:, :, c] * (1 - edges_normalized * preservation) +
                               original[:, :, c] * edges_normalized * preservation)

    return result.astype(np.uint8)


def apply_gentle_effects(frame, style, params):
    """Hauptfunktion für sanfte Effekte"""

    if style == "neon":
        return apply_gentle_neon_effects(frame, params)
    elif style == "retro":
        return apply_gentle_retro_effects(frame, params)
    elif style == "glitch":
        return apply_gentle_glitch_effects(frame, params)
    else:
        return frame


def apply_gentle_retro_effects(frame, params):
    """Sanfte Retro-Effekte"""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Sanfte Retro-Anpassungen
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params['saturation_adjust'], 0, 255)
    hsv[:, :, 2] = np.clip(
        hsv[:, :, 2] * (1 + params['brightness_adjust']), 0, 255)
    hsv[:, :, 0] = (hsv[:, :, 0] + params['hue_shift'] //
                    6) % 180  # Sehr sanft

    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Sanftes VHS-Rauschen
    if params['vhs_noise'] > 0:
        noise = np.random.normal(
            0, params['vhs_noise'] * 5, result.shape)  # Reduziert
        result = np.clip(result + noise, 0, 255).astype(np.uint8)

    return result


def apply_gentle_glitch_effects(frame, params):
    """Sanfte Glitch-Effekte"""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Sanfte Cyberpunk-Farben
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params['saturation_adjust'], 0, 255)
    hsv[:, :, 0] = (hsv[:, :, 0] + 60) % 180  # Sanfter Cyan

    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Sanfter Kontrast
    result = np.clip(
        result * params['contrast_adjust'], 0, 255).astype(np.uint8)

    # Minimales digitales Rauschen
    if params['noise_level'] > 0:
        noise = np.random.randint(
            0, int(params['noise_level'] * 20), result.shape)
        result = np.clip(result + noise, 0, 255).astype(np.uint8)

    return result


def process_video_gentle(input_file, output_dir, style):
    """Verarbeite Video mit sanften Parametern"""

    input_path = Path(input_file)
    if not input_path.exists():
        return False, f"❌ Eingabedatei nicht gefunden: {input_file}"

    output_file = output_dir / f"{input_path.stem}_gentle_{style}.mp4"

    print(f"🌸 GENTLE VAPORWAVE ({style.upper()}): {input_path.name}")

    # Video analysieren
    print("   🔍 Analysiere für sanfte Parameter...")
    analysis = analyze_video_properties(input_path)

    if analysis:
        print(
            f"   📊 Helligkeit: {analysis['brightness']:.1f}, Kontrast: {analysis['contrast']:.1f}")
        print(
            f"   🎨 Sättigung: {analysis['saturation']:.1f}, Kanten: {analysis['edge_density']:.3f}")
        print(f"   🏷️ Tags: {'Hell' if analysis['is_bright'] else 'Dunkel'}, "
              f"{'Kontrast' if analysis['is_high_contrast'] else 'Flach'}, "
              f"{'Detailreich' if analysis['has_many_edges'] else 'Glatt'}")

    # Sanfte Parameter berechnen
    print("   ⚙️ Berechne sanfte Parameter...")
    params = calculate_gentle_parameters(analysis, style)

    print(f"   📝 Sanfte Parameter: {json.dumps(params, indent=6)}")

    # Video verarbeiten
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return False, "Kann Video nicht öffnen"

    # Video-Eigenschaften
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video-Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    frame_count = 0

    try:
        print("   🎨 Wende sanfte Vaporwave-Effekte an...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sanfte Effekte anwenden
            processed_frame = apply_gentle_effects(frame, style, params)

            out.write(processed_frame)
            frame_count += 1

            if frame_count % 20 == 0:  # Progress
                progress = (frame_count / total_frames) * 100
                print(
                    f"   Verarbeitet: {frame_count}/{total_frames} Frames ({progress:.1f}%) - Sanft!")

        return True, f"✅ {frame_count} Frames sanft verarbeitet"

    except Exception as e:
        return False, f"❌ Fehler: {str(e)}"

    finally:
        cap.release()
        out.release()


def main():
    parser = argparse.ArgumentParser(description="Gentle Vaporwave Processing")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--style", choices=["neon", "retro", "glitch"], default="neon",
                        help="Vaporwave style (gentle version)")

    args = parser.parse_args()

    # Prüfe OpenCV
    try:
        import cv2
    except ImportError:
        print("❌ OpenCV nicht installiert! Installiere mit: pip install opencv-python")
        sys.exit(1)

    input_file = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    success, message = process_video_gentle(input_file, output_dir, args.style)

    print(f"\n{message}")

    if success:
        print(f"✅ GENTLE VAPORWAVE ERFOLGREICH!")
        print(f"📁 Ausgabe: {output_dir}")
        print("🌸 Sanfte Parameter vermeiden Kanten-Probleme und Verschwimmen")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
