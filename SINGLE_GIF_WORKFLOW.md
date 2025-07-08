# 🎬 Single GIF Workflow mit 15-Farben-Palette

## Übersicht

Dieser Workflow verarbeitet **genau 1 Input-GIF** zu **1 Output-GIF** unter Verwendung der extrahierten 15-Farben-Palette aus `pixel_modern_0ea53a3cdbfdcf14caf1c8cccdb60143.gif`.

## 🎨 Extrahierte Farbpalette

### Kategorien und Farben:

#### Dunkel & Monochrom (1-2)

- **#1**: `RGB(8, 12, 16)` - Tiefes Schwarz mit leichtem Blaustich
- **#2**: `RGB(25, 15, 35)` - Sehr dunkles Lila

#### Warme Töne (3-5)

- **#3**: `RGB(85, 25, 35)` - Dunkles Rot (weinrot)
- **#4**: `RGB(200, 100, 45)` - Sattes Orange
- **#5**: `RGB(235, 220, 180)` - Blasses Gelb/Beige

#### Grüntöne (6-7)

- **#6**: `RGB(145, 220, 85)` - Helles Apfelgrün
- **#7**: `RGB(45, 160, 95)` - Satteres Smaragdgrün

#### Blautöne (8-11)

- **#8**: `RGB(35, 85, 95)` - Dunkles Grün-Blau
- **#9**: `RGB(25, 45, 85)` - Navyblau
- **#10**: `RGB(65, 115, 180)` - Mittelblau
- **#11**: `RGB(85, 195, 215)` - Cyan / Türkis

#### Kühle Grautöne (12-14)

- **#12**: `RGB(95, 105, 125)` - Graublau
- **#13**: `RGB(85, 75, 95)` - Graulila
- **#14**: `RGB(45, 55, 75)` - Graphitblau

#### Letzter Ton (15)

- **#15**: `RGB(12, 8, 8)` - Fast Schwarz, leicht abweichend von #1

## 🚀 Nutzung

### Methode 1: Direkter Aufruf

```bash
python single_gif_processor.py "input/your_file.gif" "output/result.gif"
```

### Methode 2: Vereinfachtes Script

```bash
python scripts/process_single_gif.py "input/your_file.gif"
```

_(Output wird automatisch generiert als `output/single_processed_filename_15colors.gif`)_

### Methode 3: Interaktive Eingabe

```bash
python scripts/process_single_gif.py
# Dann Input-Pfad eingeben wenn gefragt
```

## ⚙️ Workflow-Konfiguration

```json
{
  "resolution": [512, 512],
  "pixelize_size": 4,
  "color_matching": "euclidean_distance",
  "preserve_animation": true,
  "optimize_gif": true
}
```

## 📋 Workflow-Schritte

1. **Input-Validierung**: Prüfe ob GIF-Datei existiert
2. **Frame-Extraktion**: Alle Frames aus der Input-GIF
3. **Größenanpassung**: Resize auf 512x512 Pixel
4. **Pixelisierung**: 4x4 Pixel-Blöcke für Retro-Look
5. **Farbzuordnung**: Jeder Pixel wird der nächsten Palette-Farbe zugeordnet
6. **GIF-Erstellung**: Animiertes GIF mit Original-Timing
7. **Optimierung**: Dateigröße-Optimierung

## 📊 Technische Details

### Farbzuordnung

- **Methode**: Euklidische Distanz im RGB-Raum
- **Formel**: `min(sqrt((r1-r2)² + (g1-g2)² + (b1-b2)²))`
- **Performance**: Pixel-für-Pixel Verarbeitung

### Output-Eigenschaften

- **Format**: Animiertes GIF
- **Auflösung**: 512x512 Pixel
- **Farben**: Exakt 15 Farben aus der Palette
- **Animation**: Original-Timing beibehalten
- **Optimierung**: Aktiviert für kleinere Dateigröße

## 📁 Dateien

- `single_gif_processor.py` - Haupt-Processor-Klasse
- `scripts/process_single_gif.py` - Vereinfachtes Interface
- `configs/extracted_15color_palette.json` - Palette-Konfiguration

## ✅ Beispiele

### Erfolgreiches Processing:

```
🎬 Verarbeite: example.gif
✅ GIF erstellt: output/result.gif (97 Frames)
🎨 15-Farben-Palette angewendet
📊 Dateigröße: 1.23 MB
```

### Input-Validierung:

```
❌ Input-Datei nicht gefunden: nonexistent.gif
```

## 🔧 Anpassungen

Für andere Paletten können die RGB-Werte in `single_gif_processor.py` in der `extracted_palette` Liste angepasst werden.
