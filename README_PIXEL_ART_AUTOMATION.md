# 🧰 ComfyUI Pixel-Art Automation System

Vollständig automatisierte Pixel-Art-Generation ohne GUI-Abhängigkeiten. Dieses System verwandelt Bilder und Videos in hochwertige Pixel-Art mit verschiedenen Styles.

## ✨ Features

- **🚀 Vollständig CLI-gesteuert** - Keine GUI nötig
- **🎨 5 vordefinierte Styles** - Modern, Retro, High-Res, Sprite, Animated
- **🎬 Video-zu-GIF Pipeline** - Automatische Frame-Extraktion und -verarbeitung
- **📦 Batch-Processing** - Verarbeitung ganzer Verzeichnisse
- **🔄 Reproduzierbar** - Seed-basierte Konsistenz
- **⚙️ Konfigurierbar** - JSON-basierte Style-Konfiguration
- **🌐 Plattformübergreifend** - Windows (.bat) und Unix (.sh) Scripts

## 📦 Installation & Setup

### Voraussetzungen

1. **ComfyUI** bereits installiert und funktionsfähig
2. **Python 3.8+**
3. **ffmpeg** (für Video-Processing)
4. **ComfyUI-PixelArt-Detector** Custom Node (bereits installiert)

### Dependencies installieren

```bash
# Zusätzliche Python-Pakete
pip install pillow requests opencv-python

# ffmpeg (je nach System)
# Windows: choco install ffmpeg
# macOS: brew install ffmpeg
# Ubuntu: apt install ffmpeg
```

## 🚀 Schnellstart

### Windows

```cmd
# Einzelbild
scripts\run_pixel_art.bat --style modern

# Batch-Processing
scripts\run_pixel_art.bat --batch --style retro

# Video zu Pixel-Art-GIF
scripts\run_pixel_art.bat --video "mein_video.mp4" --fps 10 --style animated
```

### Unix/Linux/macOS

```bash
# Ausführbar machen
chmod +x scripts/run_pixel_art.sh

# Einzelbild
./scripts/run_pixel_art.sh --style modern

# Batch-Processing
./scripts/run_pixel_art.sh --batch --style retro

# Video zu Pixel-Art-GIF
./scripts/run_pixel_art.sh --video "mein_video.mp4" --fps 10 --style animated
```

## 🎨 Verfügbare Styles

| Style        | Auflösung | Farben | Beschreibung                            |
| ------------ | --------- | ------ | --------------------------------------- |
| **modern**   | 512x512   | 64     | Moderne, hochauflösende Pixel-Art       |
| **retro**    | 256x256   | 16     | Klassischer 8-Bit NES/GameBoy Look      |
| **high_res** | 768x768   | 128    | Detaillierte, hochauflösende Pixel-Art  |
| **sprite**   | 64x64     | 32     | Optimiert für Sprite-Sheets             |
| **animated** | 512x512   | 48     | Konsistent für Frame-by-Frame Animation |

## 📁 Projektstruktur

```
ComfyUI-master/
├── scripts/
│   ├── pixel_art_cli.py         # Haupt-CLI-Script
│   ├── batch_video_processor.py # Video-Processing
│   ├── run_pixel_art.bat        # Windows Wrapper
│   └── run_pixel_art.sh         # Unix Wrapper
├── configs/
│   └── pixel_art_styles.json    # Style-Konfigurationen
├── workflows/
│   └── modern_pixel_art.json    # ComfyUI-Workflow-Templates
├── input/                       # Input-Bilder/Videos
├── output/                      # Generierte Pixel-Art
└── temp_processing/             # Temporäre Dateien
```

## 💻 CLI-Referenz

### Basis-CLI (pixel_art_cli.py)

```bash
python scripts/pixel_art_cli.py [OPTIONS]

Optionen:
  --input, -i PATH      Input-Datei oder -Verzeichnis (erforderlich)
  --output, -o NAME     Output-Name (erforderlich)
  --style, -s STYLE     Pixel-Art-Style [modern|retro|high_res|sprite|animated]
  --seed NUMBER         Seed für Reproduzierbarkeit (Standard: 42)
  --comfy-path PATH     Pfad zur ComfyUI-Installation (Standard: .)
```

### Video-Batch-Processor

```bash
python scripts/batch_video_processor.py [OPTIONS]

Optionen:
  --video, -v PATH           Input-Video-Datei (erforderlich)
  --output, -o DIR           Output-Verzeichnis (erforderlich)
  --style, -s STYLE          Pixel-Art-Style
  --fps FLOAT                FPS für Frame-Extraktion
  --max-frames INT           Maximale Anzahl Frames
  --parallel-workers INT     Anzahl parallele Worker (Standard: 2)
  --create-gif               Erstelle GIF aus verarbeiteten Frames
  --gif-fps FLOAT            FPS für Output-GIF (Standard: 10)
  --create-sprite            Erstelle Sprite-Sheet
  --sprite-cols INT          Spalten für Sprite-Sheet (Standard: 8)
  --cleanup                  Bereinige temporäre Dateien
```

## 🔧 Konfiguration

### Style-Anpassung

Bearbeiten Sie `configs/pixel_art_styles.json` um:

- Neue Styles hinzuzufügen
- Bestehende Parameter zu ändern
- Eigene Paletten zu definieren

```json
{
  "styles": {
    "mein_style": {
      "name": "Mein Custom Style",
      "resolution": [400, 400],
      "colors_max": 32,
      "dither": "bayer-4",
      "palette": "NES",
      "cleanup": true,
      "cleanup_threshold": 0.03
    }
  }
}
```

### Workflow-Anpassung

Workflow-Templates in `workflows/` können angepasst werden für:

- Andere Auflösungen
- Verschiedene Dithering-Methoden
- Custom Node-Ketten
- Spezielle Output-Formate

## 🎯 Anwendungsbeispiele

### 1. Einzelbild zu moderner Pixel-Art

```bash
# Windows
scripts\run_pixel_art.bat --input "mein_foto.jpg" --style modern

# Unix
./scripts/run_pixel_art.sh --input "mein_foto.jpg" --style modern
```

### 2. Batch-Verarbeitung ganzer Verzeichnisse

```bash
# Alle Bilder im Input-Verzeichnis
./scripts/run_pixel_art.sh --batch --style retro
```

### 3. Video zu animierter Pixel-Art

```bash
# Video-Frames extrahieren und verarbeiten
./scripts/run_pixel_art.sh --video "gameplay.mp4" --fps 12 --style animated

# Mit GIF-Erstellung
python scripts/batch_video_processor.py \
  --video "gameplay.mp4" \
  --output "./pixel_animation" \
  --style animated \
  --create-gif \
  --gif-fps 12
```

### 4. Sprite-Sheet Erstellung

```bash
# Frame-Extraktion aus Video für Sprite-Sheet
python scripts/batch_video_processor.py \
  --video "character_animation.mp4" \
  --output "./sprites" \
  --style sprite \
  --max-frames 16 \
  --create-sprite \
  --sprite-cols 4
```

## 🔬 Erweiterte Features

### Reproduzierbare Ergebnisse

```bash
# Gleicher Seed = identische Ergebnisse
./scripts/run_pixel_art.sh --input "test.jpg" --seed 1337 --style modern
```

### Parallele Video-Verarbeitung

```bash
# Mehr Worker für schnellere Verarbeitung (GPU-abhängig)
python scripts/batch_video_processor.py \
  --video "long_video.mp4" \
  --parallel-workers 4 \
  --style modern
```

### Custom Paletten

Erweitern Sie `configs/pixel_art_styles.json` um eigene Farbpaletten:

```json
{
  "palettes": {
    "meine_palette": {
      "type": "fixed",
      "colors": 16,
      "description": "Meine eigene Farbpalette"
    }
  }
}
```

## 🐛 Troubleshooting

### Häufige Probleme

1. **"Python nicht gefunden"**

   - Stellen Sie sicher, dass Python im PATH ist
   - Verwenden Sie `python3` statt `python` auf Unix-Systemen

2. **"ffmpeg nicht gefunden"**

   - Installieren Sie ffmpeg für Video-Processing
   - Stellen Sie sicher, dass ffmpeg im PATH ist

3. **"ComfyUI-Module nicht gefunden"**

   - Stellen Sie sicher, dass Sie im ComfyUI-Hauptverzeichnis sind
   - Prüfen Sie `--comfy-path` Parameter

4. **"Keine Output-Dateien"**
   - Prüfen Sie Logs in `pixel_art_automation.log`
   - Validieren Sie Input-Dateiformate (PNG, JPG, WEBP, BMP)

### Debug-Modus

```bash
# Erhöhte Logging-Ausgabe
export COMFYUI_LOG_LEVEL=DEBUG
./scripts/run_pixel_art.sh --style modern
```

## 📊 Performance-Optimierung

### GPU-Optimierung

- **VRAM**: Moderne Styles benötigen ~2-4GB VRAM
- **Batch-Größe**: Bei wenig VRAM `--parallel-workers 1` verwenden
- **Auflösung**: Kleinere Auflösungen für schnellere Verarbeitung

### Disk-Space

- Temporäre Dateien werden in `temp_processing/` gespeichert
- Verwenden Sie `--cleanup` für automatische Bereinigung
- Video-Frames können viel Speicher benötigen

## 🚀 Integration & Automation

### CI/CD Integration

```yaml
# GitHub Actions Beispiel
- name: Generate Pixel Art
  run: |
    python scripts/pixel_art_cli.py \
      --input ./assets/images \
      --output pixel_art_batch \
      --style modern \
      --seed ${{ github.run_number }}
```

### API-Integration

Das System kann als Service-Wrapper verwendet werden:

```python
# Custom Integration
from scripts.pixel_art_cli import PixelArtProcessor

processor = PixelArtProcessor('.')
processor.process_image('input.jpg', 'output', 'modern', 42)
```

## 📈 Roadmap

- [ ] **Web-API** für Remote-Processing
- [ ] **Kubernetes Deployment** für Cloud-Skalierung
- [ ] **ControlNet Integration** für bessere Kantenkontrolle
- [ ] **Upscaling Pipeline** für höhere Auflösungen
- [ ] **Style-Transfer** zwischen verschiedenen Pixel-Art-Arten

## 📄 Lizenz

Dieses Projekt verwendet die gleichen Lizenzbedingungen wie ComfyUI. Siehe [ComfyUI License](https://github.com/comfyanonymous/ComfyUI/blob/master/LICENSE) für Details.

## 🤝 Beitragen

Contributions willkommen! Bitte:

1. Fork das Repository
2. Erstellen Sie einen Feature-Branch
3. Committen Sie Ihre Änderungen
4. Senden Sie einen Pull Request

---

**🎨 Viel Spaß beim Erstellen von Pixel-Art! 🎮**
