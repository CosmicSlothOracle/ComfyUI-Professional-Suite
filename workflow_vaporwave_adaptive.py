#!/usr/bin/env python3
"""
ADAPTIVE VAPORWAVE WORKFLOW - INTELLIGENT PARAMETER ADJUSTMENT
Analysiert jedes Video und passt die Vaporwave-Parameter automatisch an
Verhindert Überanpassung und funktioniert mit verschiedenen Video-Typen
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import json


def analyze_video_properties(video_path):
    """Analysiere Video-Eigenschaften für adaptive Parameter"""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    # Analysiere erste 10 Frames
    frames_analyzed = 0
    brightness_values = []
    contrast_values = []
    saturation_values = []
    dominant_colors = []

    while frames_analyzed < 10:
        ret, frame = cap.read()
        if not ret:
            break

        # Konvertiere zu verschiedenen Farbräumen
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # Berechne Statistiken
        brightness = np.mean(lab[:, :, 0])  # L-Kanal
        contrast = np.std(lab[:, :, 0])     # Kontrast als Standardabweichung
        saturation = np.mean(hsv[:, :, 1])  # S-Kanal

        brightness_values.append(brightness)
        contrast_values.append(contrast)
        saturation_values.append(saturation)

        # Dominante Farbe
        pixels = frame.reshape(-1, 3)
        dominant_color = np.mean(pixels, axis=0)
        dominant_colors.append(dominant_color)

        frames_analyzed += 1

    cap.release()

    if not brightness_values:
        return None

    # Berechne Durchschnittswerte
    analysis = {
        'brightness': np.mean(brightness_values),      # 0-255
        'contrast': np.mean(contrast_values),          # Höher = mehr Kontrast
        'saturation': np.mean(saturation_values),      # 0-255
        'dominant_color': np.mean(dominant_colors, axis=0),  # [B, G, R]
        'is_bright': np.mean(brightness_values) > 128,
        'is_high_contrast': np.mean(contrast_values) > 50,
        'is_saturated': np.mean(saturation_values) > 100
    }

    return analysis


def calculate_adaptive_parameters(analysis, target_style):
    """Berechne adaptive Parameter basierend auf Video-Analyse"""

    if not analysis:
        # Fallback: Moderate Parameter
        return get_default_parameters(target_style)

    params = {}

    if target_style == "neon":
        # NEON: Anpassung an Helligkeit und Sättigung
        if analysis['is_bright']:
            # Helles Video: Weniger Aufhellung, mehr Kontrast
            params['brightness_adjust'] = 0.1
            params['contrast_adjust'] = 1.6
        else:
            # Dunkles Video: Mehr Aufhellung, moderater Kontrast
            params['brightness_adjust'] = 0.3
            params['contrast_adjust'] = 1.4

        if analysis['is_saturated']:
            # Bereits gesättigt: Moderate Sättigung
            params['saturation_adjust'] = 1.8
        else:
            # Wenig gesättigt: Starke Sättigung
            params['saturation_adjust'] = 2.5

        # Neon-spezifisch
        params['hue_shift'] = 300  # Pink/Magenta
        params['glow_intensity'] = 0.8
        params['grid_opacity'] = 0.3

    elif target_style == "retro":
        # RETRO: Anpassung an Farbtemperatur
        dominant_b, dominant_g, dominant_r = analysis['dominant_color']

        if dominant_b > dominant_r:  # Eher kalt
            # Kaltes Video: Mehr Wärme hinzufügen
            params['warmth_adjust'] = 0.4
            params['hue_shift'] = 30  # Richtung Orange
        else:  # Eher warm
            # Warmes Video: Moderate Wärme
            params['warmth_adjust'] = 0.2
            params['hue_shift'] = 20

        params['vhs_noise'] = 0.3 if analysis['is_high_contrast'] else 0.4
        params['saturation_adjust'] = 1.8

    elif target_style == "glitch":
        # GLITCH: Anpassung an Kontrast
        if analysis['is_high_contrast']:
            # Hoher Kontrast: Moderate Verstärkung
            params['contrast_adjust'] = 1.4
            params['glitch_intensity'] = 0.5
        else:
            # Niedriger Kontrast: Starke Verstärkung
            params['contrast_adjust'] = 1.7
            params['glitch_intensity'] = 0.7

        params['hue_shift'] = 180  # Cyan
        params['noise_level'] = 0.4
        params['saturation_adjust'] = 2.2

    return params


def get_default_parameters(style):
    """Fallback-Parameter falls Analyse fehlschlägt"""

    defaults = {
        "neon": {
            'brightness_adjust': 0.2,
            'contrast_adjust': 1.4,
            'saturation_adjust': 2.0,
            'hue_shift': 300,
            'glow_intensity': 0.7,
            'grid_opacity': 0.3
        },
        "retro": {
            'warmth_adjust': 0.3,
            'hue_shift': 25,
            'vhs_noise': 0.35,
            'saturation_adjust': 1.8,
            'contrast_adjust': 1.2
        },
        "glitch": {
            'contrast_adjust': 1.5,
            'glitch_intensity': 0.6,
            'hue_shift': 180,
            'noise_level': 0.4,
            'saturation_adjust': 2.2
        }
    }

    return defaults.get(style, defaults["neon"])


def apply_vaporwave_effects_opencv(input_path, output_path, style, params):
    """Wende Vaporwave-Effekte mit OpenCV an"""

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
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Wende Effekte basierend auf Stil an
            if style == "neon":
                processed_frame = apply_neon_effects(frame, params)
            elif style == "retro":
                processed_frame = apply_retro_effects(frame, params)
            elif style == "glitch":
                processed_frame = apply_glitch_effects(frame, params)
            else:
                processed_frame = frame

            out.write(processed_frame)
            frame_count += 1

            if frame_count % 30 == 0:  # Progress
                progress = (frame_count / total_frames) * 100
                print(
                    f"   Verarbeitet: {frame_count}/{total_frames} Frames ({progress:.1f}%)")

        return True, f"✅ {frame_count} Frames verarbeitet"

    except Exception as e:
        return False, f"❌ Fehler bei Verarbeitung: {str(e)}"

    finally:
        cap.release()
        out.release()


def apply_neon_effects(frame, params):
    """Wende Neon-Effekte auf Frame an"""

    # Konvertiere zu HSV für bessere Farbmanipulation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Anpassungen
    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1] * params['saturation_adjust'], 0, 255)  # Sättigung
    hsv[:, :, 2] = np.clip(
        hsv[:, :, 2] * (1 + params['brightness_adjust']), 0, 255)  # Helligkeit

    # Hue-Shift für Pink/Magenta
    hsv[:, :, 0] = (hsv[:, :, 0] + params['hue_shift'] // 2) % 180

    # Zurück zu BGR
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Kontrast anpassen
    result = np.clip(
        result * params['contrast_adjust'], 0, 255).astype(np.uint8)

    # Glow-Effekt (Gaussian Blur + Overlay)
    if params['glow_intensity'] > 0:
        blurred = cv2.GaussianBlur(result, (15, 15), 0)
        result = cv2.addWeighted(result, 0.7, blurred, 0.3, 0)

    return result


def apply_retro_effects(frame, params):
    """Wende Retro-Effekte auf Frame an"""

    # Konvertiere zu HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Retro-Farbanpassungen
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params['saturation_adjust'], 0, 255)
    hsv[:, :, 0] = (hsv[:, :, 0] + params['hue_shift'] //
                    2) % 180  # Warm shift

    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # VHS-Rauschen
    if params['vhs_noise'] > 0:
        noise = np.random.normal(0, params['vhs_noise'] * 10, result.shape)
        result = np.clip(result + noise, 0, 255).astype(np.uint8)

    return result


def apply_glitch_effects(frame, params):
    """Wende Glitch-Effekte auf Frame an"""

    # Konvertiere zu HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Cyberpunk-Farben
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params['saturation_adjust'], 0, 255)
    hsv[:, :, 0] = (hsv[:, :, 0] + 90) % 180  # Cyan shift

    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Extremer Kontrast
    result = np.clip(
        result * params['contrast_adjust'], 0, 255).astype(np.uint8)

    # Digitales Rauschen
    if params['noise_level'] > 0:
        noise = np.random.randint(
            0, int(params['noise_level'] * 50), result.shape)
        result = np.clip(result + noise, 0, 255).astype(np.uint8)

    return result


def process_video_adaptive(input_file, output_dir, style):
    """Verarbeite Video mit adaptiven Parametern"""

    input_path = Path(input_file)
    if not input_path.exists():
        return False, f"❌ Eingabedatei nicht gefunden: {input_file}"

    output_file = output_dir / f"{input_path.stem}_adaptive_{style}.mp4"

    print(f"🎯 ADAPTIVE VAPORWAVE ({style.upper()}): {input_path.name}")

    # Schritt 1: Video analysieren
    print("   🔍 Analysiere Video-Eigenschaften...")
    analysis = analyze_video_properties(input_path)

    if analysis:
        print(
            f"   📊 Helligkeit: {analysis['brightness']:.1f}, Kontrast: {analysis['contrast']:.1f}")
        print(f"   🎨 Sättigung: {analysis['saturation']:.1f}")
        print(f"   🏷️ Tags: {'Hell' if analysis['is_bright'] else 'Dunkel'}, "
              f"{'Kontrast' if analysis['is_high_contrast'] else 'Flach'}, "
              f"{'Gesättigt' if analysis['is_saturated'] else 'Blass'}")

    # Schritt 2: Parameter berechnen
    print("   ⚙️ Berechne adaptive Parameter...")
    params = calculate_adaptive_parameters(analysis, style)

    print(f"   📝 Parameter: {json.dumps(params, indent=6)}")

    # Schritt 3: Effekte anwenden
    print("   🎨 Wende Vaporwave-Effekte an...")
    success, message = apply_vaporwave_effects_opencv(
        input_path, output_file, style, params)

    return success, message


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Vaporwave Processing")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--style", choices=["neon", "retro", "glitch"], default="neon",
                        help="Vaporwave style")

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

    success, message = process_video_adaptive(
        input_file, output_dir, args.style)

    print(f"\n{message}")

    if success:
        print(f"✅ ADAPTIVE VAPORWAVE ERFOLGREICH!")
        print(f"📁 Ausgabe: {output_dir}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
