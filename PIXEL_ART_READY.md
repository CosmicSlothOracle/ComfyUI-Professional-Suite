# 🎨 Pixel-Art Automation - Bereit zur Nutzung!

## ✅ Setup erfolgreich abgeschlossen

Das Pixel-Art Automation System ist jetzt funktionsbereit und kann verwendet werden.

## 🚀 Sofort nutzbar

### Windows (Einfach)

```batch
# Basis-Nutzung
scripts\run_pixel_art.bat input\mein_bild.jpg modern

# Mit eigenem Output-Namen
scripts\run_pixel_art.bat input\foto.png retro mein_pixel_art

# Game Boy Style
scripts\run_pixel_art.bat input\bild.jpg gameboy
```

### Python (Direkt)

```bash
# Modern Style (empfohlen)
python scripts\standalone_pixel_art_converter.py --input input\bild.jpg --output output\result.png --style modern

# Retro Style mit NES-Farben
python scripts\standalone_pixel_art_converter.py --input input\bild.jpg --output output\retro.png --style retro

# Game Boy Style
python scripts\standalone_pixel_art_converter.py --input input\bild.jpg --output output\gameboy.png --style gameboy

# High Resolution
python scripts\standalone_pixel_art_converter.py --input input\bild.jpg --output output\hires.png --style high_res
```

## 🎯 Verfügbare Styles

| Style        | Auflösung | Farben | Charakteristika                        |
| ------------ | --------- | ------ | -------------------------------------- |
| **modern**   | 512x512   | 64     | Ausgewogen, moderne Pixelart           |
| **retro**    | 256x256   | 16     | NES-Palette, Floyd-Steinberg Dithering |
| **gameboy**  | 320x320   | 4      | Game Boy Grüntöne, nostalgisch         |
| **high_res** | 768x768   | 128    | Hochauflösend, detailreich             |

## 📁 Verzeichnisstruktur

```
ComfyUI-master/
├── input/          # Legen Sie hier Ihre Bilder ab
├── output/         # Hier finden Sie die Ergebnisse
├── scripts/        # Automation-Scripts
│   ├── run_pixel_art.bat                    # Windows Batch-Script
│   ├── standalone_pixel_art_converter.py    # Python Converter
│   └── setup_pixel_art_automation.py       # Setup-Script
└── temp_processing/ # Temporäre Dateien
```

## 🔧 System-Status

- ✅ Python 3.13.1 - Kompatibel
- ✅ PIL (Pillow) - Installiert
- ✅ OpenCV - Installiert
- ✅ NumPy - Installiert
- ✅ Verzeichnisse - Erstellt
- ✅ Test-Bild - Verfügbar
- ✅ Standalone-Converter - Funktionsfähig
- ⚠️ ffmpeg - Optional (für Video-Features)

## 🎮 Beispiel-Nutzung

### 1. Bild in input/ ablegen

```
# Kopieren Sie Ihr Bild nach:
input\mein_foto.jpg
```

### 2. Konvertierung starten

```batch
# Windows Batch (Doppelklick oder Kommandozeile)
scripts\run_pixel_art.bat input\mein_foto.jpg modern

# Oder Python direkt
python scripts\standalone_pixel_art_converter.py --input input\mein_foto.jpg --output output\pixel_modern.png --style modern
```

### 3. Ergebnis prüfen

```
# Ergebnis finden Sie in:
output\mein_foto_pixel_modern.png
```

## 🛠️ Erweiterte Funktionen

### Batch-Verarbeitung (Mehrere Bilder)

```batch
for %%f in (input\*.jpg) do (
    scripts\run_pixel_art.bat "%%f" modern
)
```

### Alle Styles testen

```batch
scripts\run_pixel_art.bat input\test.jpg modern
scripts\run_pixel_art.bat input\test.jpg retro
scripts\run_pixel_art.bat input\test.jpg gameboy
scripts\run_pixel_art.bat input\test.jpg high_res
```

## 📋 Unterstützte Formate

**Input:** JPG, PNG, BMP, TIFF, WebP
**Output:** PNG (optimiert für Pixel-Art)

## 🎯 Tipps für beste Ergebnisse

1. **Bildgröße**: Verwenden Sie Bilder zwischen 200x200 und 1024x1024 Pixel
2. **Kontrast**: Bilder mit gutem Kontrast ergeben bessere Pixel-Art
3. **Style-Wahl**:
   - `modern`: Für zeitgemäße Pixel-Art
   - `retro`: Für nostalgische 8-Bit Ästhetik
   - `gameboy`: Für monochrome Grüntöne
   - `high_res`: Für detailreiche Arbeiten

## 🚨 Fehlerbehebung

### "Bild kann nicht geladen werden"

- Prüfen Sie den Dateipfad
- Stellen Sie sicher, dass das Format unterstützt wird

### "Python nicht gefunden"

- Stellen Sie sicher, dass die virtuelle Umgebung aktiviert ist
- Verwenden Sie: `.venv\Scripts\activate`

### "Modul nicht gefunden"

- Führen Sie das Setup erneut aus: `python scripts\setup_pixel_art_automation.py`

## 🎉 Viel Erfolg!

Das System ist jetzt bereit für die Pixel-Art-Erstellung. Experimentieren Sie mit verschiedenen Styles und Bildern!
