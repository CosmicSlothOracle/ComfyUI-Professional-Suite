# 🎬 Manual Video-to-Line Art Workflow - Anleitung

## ✅ **Setup ist komplett! Jetzt manuell in ComfyUI verwenden:**

### 🚀 **1. ComfyUI starten:**

```bash
# ComfyUI ist bereits im Hintergrund gestartet
# Falls nicht, starten Sie es mit:
python main.py --listen
```

### 🌐 **2. Browser öffnen:**
- Navigieren Sie zu: **http://127.0.0.1:8188**
- ComfyUI Web Interface sollte laden

### 📄 **3. Workflow laden:**
1. Klicken Sie auf **"Load"** Button (oben links)
2. Wählen Sie: **`workflows/video_to_lineart_schloss.json`**
3. Der komplette Workflow wird geladen

### 🎯 **4. Workflow-Übersicht:**

Der Workflow besteht aus folgenden Hauptkomponenten:

#### **📹 Video Input Sektion:**
- **VHS_LoadVideo Node:** Lädt `Schlossgifdance.mp4` aus input/
- **Bereits konfiguriert** für Ihr Video

#### **🎛️ Control & Models Sektion:**
- **ControlNetLoader:** `control_v11p_sd15_lineart.pth` (✅ heruntergeladen)
- **CheckpointLoader:** `dreamshaper_8.safetensors` (✅ heruntergeladen)

#### **✏️ Line Art Processing:**
- **LineArtPreprocessor:** Extrahiert Line Art aus Video Frames
- **ControlNetApply:** Wendet Line Art Control an

#### **🎬 Generation:**
- **Positive Prompt:** "clean line art drawing, black and white, simple lines..."
- **Negative Prompt:** "blurry, low quality, noise, artifacts..."
- **K-Sampler:** 20 Steps, CFG 7.5, euler_ancestral

#### **📹 Output:**
- **VAEDecode:** Konvertiert zu Bildern
- **VHS_VideoCombine:** Erstellt finale MP4 Animation
- **PreviewImage:** Zeigt Vorschau

### ⚙️ **5. Parameter anpassen (optional):**

#### **Video Input anpassen:**
- **VHS_LoadVideo Node:** Ändern Sie den Dateinamen falls gewünscht
- Standard: `Schlossgifdance.mp4`

#### **Qualität anpassen:**
- **K-Sampler Steps:** 15-25 (höher = bessere Qualität)
- **CFG Scale:** 7.5 (Prompt-Stärke)
- **Resolution:** 512x512 (in EmptyLatentImage)

#### **Style anpassen:**
```
Positive Prompt Beispiele:
- "clean line art, vector graphics, minimal"
- "anime line art, manga style, cel animation"
- "sketch style, hand drawn, artistic"
- "comic book art, bold lines"
```

### 🚀 **6. Processing starten:**

1. **Queue Prompt** Button klicken (oben rechts)
2. **Progress** verfolgen in der Queue
3. **Preview** wird während Processing angezeigt
4. **Fertige Animation** wird in `output/` gespeichert

### ⏱️ **7. Erwartete Processing-Zeit:**

- **Hardware abhängig:** 2-10 Minuten
- **GTX 1660:** ~8-10 Minuten
- **RTX 3080:** ~3-5 Minuten
- **RTX 4090:** ~1-2 Minuten

### 📁 **8. Output prüfen:**

Nach Completion:
- **Output Verzeichnis:** `output/`
- **Dateiname:** `line_art_animation_XXXXX.mp4`
- **Format:** H.264 MP4, 24 FPS

### 🔧 **9. Troubleshooting:**

#### **Custom Nodes fehlen:**
Falls Fehler "Node type not found":
```bash
# Installation der benötigten Custom Nodes:
cd custom_nodes
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
git clone https://github.com/Fannovel16/ComfyUI_ControlNet_Aux.git
```

#### **Modelle fehlen:**
Falls "Model not found":
```bash
# Erneut Modelle downloaden:
python download_models.py
```

#### **VRAM Error:**
Falls "CUDA out of memory":
- Steps reduzieren: 15-20
- Resolution reduzieren: 512x512
- Andere Programme schließen

### 🎨 **10. Style Variations:**

#### **Clean Vector Style:**
```
Positive: "clean line art, vector graphics, minimal lines, high contrast"
Negative: "sketchy, rough, shading, color"
```

#### **Anime Style:**
```
Positive: "anime line art, manga style, cel animation, clean cartoon lines"
Negative: "realistic, photographic, western style"
```

#### **Sketch Style:**
```
Positive: "hand drawn sketch, rough lines, artistic drawing, charcoal style"
Negative: "clean, vector, digital, precise"
```

## 🎉 **Bereit für professionelle Line Art Animationen!**

Der Workflow ist **komplett eingerichtet** und **einsatzbereit**. Alle Modelle sind heruntergeladen, alle Pfade sind konfiguriert.

**Nächster Schritt:** Öffnen Sie ComfyUI im Browser und laden Sie den Workflow! 🎬✨

### 📞 **Support:**
- Workflow JSON: `workflows/video_to_lineart_schloss.json`
- Vollständige Anleitung: `README_VIDEO_LINEART.md`
- Quick Start: `QUICK_START.md`