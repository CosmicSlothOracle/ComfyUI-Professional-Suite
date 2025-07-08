# 🔑 Civitai API Key Setup

## 1. **API Key erhalten:**
1. Besuchen Sie [civitai.com](https://civitai.com)
2. Registrieren/Anmelden
3. Gehen Sie zu **Profil Settings** → **API Keys**
4. Erstellen Sie einen neuen API Key
5. **Kopieren Sie den Key**

## 2. **In ComfyUI-Lora-Manager einfügen:**
1. ComfyUI starten
2. Im Browser: `http://localhost:8188/loras` öffnen
3. **Settings** → **API Key** einfügen
4. **Speichern**

## 3. **Automatisch die restlichen LoRAs herunterladen:**
```bash
python scripts/automated_lora_downloader.py
```

---

## 🎯 **Alternative: Manuelle Downloads**

Falls Sie keinen API Key wollen, laden Sie manuell herunter:

### **Zumi Style PixAI** ✅ (bereits installiert)
- [Civitai Link](https://civitai.com/models/254043/zumi-style)

### **Pixel Art XL**
- [Civitai Link](https://civitai.com/models/43820/pixel-art-style)
- Datei nach `models/loras/` verschieben

### **Modern Anime Style**
- [Civitai Link](https://civitai.com/models/680645/madbears-best-anime-style-on-flux1-dev)
- Datei nach `models/loras/` verschieben