# 🎬 Video-to-Line Art - Schnellstart

## 🚀 Sofort loslegen:

### 1. Modelle downloaden:
```bash
python download_models.py
```

### 2. ComfyUI starten:
```bash
python main.py
```

### 3. Workflow laden:
- Browser: http://127.0.0.1:8188
- Load → `workflows/video_to_lineart_schloss.json`

### 4. Video verarbeiten:
- Ihr Video ist bereits voreingestellt: `Schlossgifdance.mp4`
- Oder ändern Sie den Dateinamen im "Load Video" Node
- Queue Prompt klicken

## ⚙️ Parameter anpassen:

### Prompts:
- **Positive:** "clean line art, vector, minimal style"
- **Negative:** "blurry, color, shading, complex"

### Qualität:
- **Steps:** 15-25 (K-Sampler)
- **CFG:** 7.5 (Prompt-Stärke)
- **Resolution:** 512x512

## 📁 Dateien:
- **Input:** Ihr Video hier platzieren
- **Output:** Fertige Line Art Animation

## 🔧 Custom Nodes needed:
- ComfyUI-VideoHelperSuite
- ComfyUI_ControlNet_Aux

Viel Erfolg! 🎨✨
