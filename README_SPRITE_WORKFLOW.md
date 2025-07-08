# 🎨 ComfyUI Sprite-Sheet Workflow - Vollautomatisches Setup

> **Extrahiere Frames aus Sprite-Sheets und transformiere sie in neue Kunststile mit Pose-Bewahrung**

## 🎯 Was macht dieser Workflow?

Dieser vollautomatisierte Workflow ermöglicht es dir:

1. **📋 Sprite-Sheet Frame-Extraktion**: Automatische Zerlegung von Sprite-Sheets in Einzelframes
2. **🕺 Pose-Erkennung**: Extraktion der Körperpose mit OpenPose ControlNet
3. **🎨 Style-Transfer**: Transformation in neue Kunststile (Anime, Pixel-Art, Realistisch)
4. **⚡ Pose-Bewahrung**: Exakte Reproduktion der ursprünglichen Posen
5. **🔄 Batch-Processing**: Verarbeitung ganzer Sprite-Sheet-Sammlungen

## 🚀 Schnellstart (3 Schritte)

### 1. Installation ausführen
```cmd
# Doppelklick auf:
install_sprite_workflow.bat
```

### 2. Modelle herunterladen
```cmd
# Optional - für automatischen Download:
python download_models.py

# Oder manuell von den angegebenen Links
```

### 3. ComfyUI starten
```cmd
# Doppelklick auf:
start_sprite_workflow.bat
```

## 📦 Was wird automatisch installiert?

### 🔧 Custom Nodes
- **ComfyUI-Manager** - Node-Verwaltung
- **ComfyUI-AnimateDiff-Evolved** - Animation-Tools
- **ComfyUI-Advanced-ControlNet** - Erweiterte Pose-Kontrolle
- **ComfyUI_IPAdapter_plus** - Image Prompting
- **ComfyUI-Impact-Pack** - Erweiterte Bildverarbeitung
- **ComfyUI-KJNodes** - Utility-Nodes
- **ComfyUI_essentials** - Grundlegende Tools
- **ComfyUI-PixelArt-Detector** - Pixel-Art-Spezifische Tools
- **comfy_mtb** - Batch-Processing-Tools

### 🤖 AI-Modelle (Automatisch heruntergeladen)
- **DreamShaper XL Turbo** (6.46 GB) - Schnelle, hochqualitative Generierung
- **RealVisXL v4.0** (6.46 GB) - Fotorealistische Ergebnisse
- **OpenPose ControlNet** (1.45 GB) - Pose-Erkennung und -Kontrolle
- **Depth ControlNet** (1.45 GB) - Tiefenkarten-Kontrolle
- **Lineart ControlNet** (1.45 GB) - Linienerkennung
- **Pixel Art XL LoRA** (144 MB) - Retro Pixel-Art Style
- **Anime Style XL LoRA** (144 MB) - Anime/Manga Style
- **ACE++ LoRA** (144 MB) - Allgemeine Style-Verbesserung
- **Standard VAE Models** (335 MB) - Bildencoding/Decoding

**💾 Gesamtgröße: ~18 GB**

## 📋 Vorgefertigte Workflows

### 1. 🎭 Sprite Frame Extractor
**Datei**: `workflows/sprite_processing/sprite_extractor.json`

**Funktionen**:
- Lädt Sprite-Sheets (PNG/JPG/GIF)
- Extrahiert Einzelframes automatisch
- Erkennt Körperposen mit OpenPose
- Speichert Pose-Daten für spätere Verwendung

**Verwendung**:
1. Sprite-Sheet in `input/sprite_sheets/` ablegen
2. Workflow in ComfyUI laden
3. Frame-Größe anpassen (Standard: 64x64)
4. Queue ausführen

### 2. 🎨 Style Transfer mit Pose-Preservation
**Datei**: `workflows/sprite_processing/style_transfer.json`

**Funktionen**:
- Lädt extrahierte Character-Frames
- Verwendet Pose-Referenzen für Kontrolle
- Wendet gewählte Kunstrichtung an
- Behält exakte Körperpose bei
- Unterstützt verschiedene Style-Presets

**Style-Presets**:
- **Anime**: Vibrant, cel-shaded, manga-style
- **Pixel Art**: 8-bit, retro, blocky design
- **Realistic**: Photorealistic, detailed textures

## 🛠️ Konfiguration

### Sprite-Einstellungen
**Datei**: `config.json`

```json
{
  "sprite_settings": {
    "default_frame_size": [64, 64],
    "supported_formats": ["png", "jpg", "gif"],
    "output_format": "png"
  }
}
```

### Style-Presets anpassen
```json
{
  "style_presets": {
    "anime": {
      "lora": "anime_style_xl.safetensors",
      "strength": 0.8,
      "prompt_prefix": "anime style, vibrant colors, "
    }
  }
}
```

## 📁 Verzeichnisstruktur

```
ComfyUI-master/
├── input/
│   └── sprite_sheets/          # Deine Sprite-Sheets hier ablegen
├── output/
│   ├── extracted_poses/        # Extrahierte Pose-Daten
│   └── styled_character/       # Transformierte Charaktere
├── workflows/
│   └── sprite_processing/      # Vorgefertigte Workflows
├── models/                     # AI-Modelle (automatisch heruntergeladen)
└── custom_nodes/              # Installierte Extensions
```

## 🎮 Praktische Beispiele

### Beispiel 1: RPG Character Sprite
```
Input: character_walk_cycle.png (8 Frames à 64x64)
Style: Anime
Output: 8 anime-style Frames mit exakt gleichen Posen
```

### Beispiel 2: Fighting Game Sprite
```
Input: fighter_combo.png (12 Frames à 128x128)
Style: Pixel Art → Realistic
Output: Photorealistische Kampf-Animation
```

### Beispiel 3: Batch-Processing
```
Input: Ganzer Ordner mit 50 verschiedenen Sprites
Styles: Alle 3 Style-Presets
Output: 150 transformierte Sprite-Sheets (50 × 3 Styles)
```

## ⚡ Performance-Optimierung

### Hardware-Empfehlungen
- **GPU**: NVIDIA RTX 3060 (12GB VRAM) oder besser
- **RAM**: 16GB System-RAM minimum
- **Storage**: 25GB freier SSD-Speicher
- **CPU**: Moderne Mehrkern-CPU (für Batch-Processing)

### Optimierungseinstellungen
```python
# Für schwächere GPUs:
# Im Startup-Script hinzufügen:
--lowvram --cpu-vae
```

## 🔧 Troubleshooting

### Häufige Probleme

**❌ "Custom Node XY nicht gefunden"**
```
Lösung: ComfyUI Manager → Install Missing Nodes
```

**❌ "Model nicht gefunden"**
```
Lösung: Überprüfe models/ Verzeichnis, führe download_models.py erneut aus
```

**❌ "Out of Memory"**
```
Lösung: Batch-Size reduzieren, --lowvram Parameter verwenden
```

**❌ "Pose wird nicht erkannt"**
```
Lösung: Frame-Größe erhöhen (min. 256x256), Kontrast verbessern
```

## 📚 Erweiterte Verwendung

### Custom Style-Training
1. Eigene LoRA-Modelle trainieren
2. In `models/loras/` ablegen
3. Config.json entsprechend anpassen

### Automatisierung mit Python
```python
from sprite_processor import SpriteProcessor

processor = SpriteProcessor()
processor.process_batch('input/sprite_sheets/', style='anime')
```

### Integration in Game-Pipeline
- Export direkt in Unity/Unreal Engine Formate
- Automatische Sprite-Atlas-Generierung
- Batch-Rename für Asset-Pipeline

## 🎯 Best Practices

### Sprite-Sheet Vorbereitung
- **Mindestauflösung**: 256x256 für beste Pose-Erkennung
- **Einheitliche Frame-Größen**: Alle Frames gleich groß
- **Klare Körperformen**: Gute Kontraste für Pose-Detection
- **PNG-Format**: Transparenz-Unterstützung

### Workflow-Optimierung
- **Pose zuerst**: Extrahiere alle Posen vor Style-Transfer
- **Batch-Processing**: Verarbeite ähnliche Sprites zusammen
- **Style-Tests**: Teste neue Styles mit kleinen Samples
- **Backup**: Sichere Original-Sprites vor Verarbeitung

## 📈 Ergebnisqualität

### Erwartete Resultate
- **🎯 Pose-Accuracy**: 95%+ bei gut erkennbaren Charakteren
- **🎨 Style-Konsistenz**: Einheitlicher Look über alle Frames
- **⚡ Processing-Speed**: ~10-30 Sekunden pro Frame (RTX 4090)
- **💾 Output-Quality**: 512x512 Standard, bis 2048x2048 möglich

### Qualitätsfaktoren
- Input-Sprite Qualität und Auflösung
- Pose-Klarheit und Kontraste
- Gewählter AI-Modell und Style
- Hardware-Performance

## 🔄 Updates & Wartung

### Regelmäßige Updates
```cmd
# Custom Nodes aktualisieren:
cd custom_nodes/ComfyUI-Manager
git pull

# Neue Modelle checken:
python download_models.py --check-updates
```

### Community-Erweiterungen
- Neue ControlNet-Modelle integrieren
- Custom LoRA-Sammlungen hinzufügen
- Workflow-Varianten austauschen

## 🎉 Fertig!

Du hast jetzt ein vollständig automatisiertes System für:
- ✅ Sprite-Frame-Extraktion
- ✅ Pose-Erkennung und -Bewahrung
- ✅ Stil-Transformation
- ✅ Batch-Processing
- ✅ Qualitäts-Kontrolle

**🚀 Starte jetzt mit deinen ersten Sprite-Transformationen!**

---

## 📞 Support

Bei Problemen oder Fragen:
1. Prüfe die Troubleshooting-Sektion
2. Schaue in die ComfyUI-Community
3. Öffne ein GitHub Issue mit detaillierter Beschreibung

**Viel Erfolg mit deinem automatisierten Sprite-Workflow! 🎨✨**