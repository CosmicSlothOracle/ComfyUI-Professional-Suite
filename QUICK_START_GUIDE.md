# 🃏 Modern Pokemon Card Authenticity Verification System
## Quick Start Guide

### 📦 Installation

**Step 1: Run the installer**
```bash
# On Windows, double-click:
install.bat

# Or run in PowerShell/Command Prompt:
./install.bat
```

This will:
- ✅ Create all necessary directories
- ✅ Install all Python dependencies including Vision Transformers
- ✅ Set up CUDA PyTorch (if available)

---

## 📁 File Organization & Naming

### **CRITICAL: File Naming Convention**

#### **Reference Cards (Original/Official Cards)**
Place your original Pokemon card images in the appropriate set folder:

**📍 Location:** `reference_cards/{set_name}/`
**📝 Format:** `{set_name}_{card_number}_{card_name}_reference.{ext}`

**Examples:**
```
reference_cards/base_set/base_set_001_bulbasaur_reference.png
reference_cards/base_set/base_set_004_charmander_reference.png
reference_cards/base_set/base_set_025_pikachu_reference.png
reference_cards/jungle/jungle_007_pinsir_reference.jpg
reference_cards/fossil/fossil_015_aerodactyl_reference.png
reference_cards/team_rocket/team_rocket_083_dark_magneton_reference.png
```

#### **Generated Cards (AI-Generated Cards to Verify)**
Place the AI-generated cards you want to verify:

**📍 Location:** `input_cards/to_verify/`
**📝 Format:** `{set_name}_{card_number}_{card_name}_generated.{ext}`

**Examples:**
```
input_cards/to_verify/base_set_001_bulbasaur_generated.png
input_cards/to_verify/base_set_004_charmander_generated.png
input_cards/to_verify/jungle_007_pinsir_generated.jpg
input_cards/to_verify/custom_001_my_pokemon_generated.png
```

### **🎯 Automatic Matching**
The system automatically matches:
- `base_set_001_bulbasaur_generated.png` → `base_set_001_bulbasaur_reference.png`
- `jungle_007_pinsir_generated.jpg` → `jungle_007_pinsir_reference.jpg`

---

## 🚀 Usage Examples

### **Example 1: Verify a Single Card**
```bash
python modern_card_verifier.py --single "input_cards/to_verify/base_set_001_bulbasaur_generated.png"
```

### **Example 2: Verify All Cards in Directory**
```bash
python modern_card_verifier.py --input "input_cards/to_verify"
```

### **Example 3: Specify Custom Reference**
```bash
python modern_card_verifier.py --single "my_generated_card.png" --reference "my_reference_card.png"
```

---

## 📋 Complete Setup Example

### **Step-by-Step Setup:**

1. **Install the system:**
   ```bash
   ./install.bat
   ```

2. **Add your reference cards:**
   ```
   reference_cards/
   ├── base_set/
   │   ├── base_set_001_bulbasaur_reference.png
   │   ├── base_set_004_charmander_reference.png
   │   └── base_set_025_pikachu_reference.png
   ├── jungle/
   │   └── jungle_007_pinsir_reference.png
   └── custom/
       └── custom_001_my_pokemon_reference.png
   ```

3. **Add generated cards to verify:**
   ```
   input_cards/to_verify/
   ├── base_set_001_bulbasaur_generated.png
   ├── base_set_004_charmander_generated.png
   └── jungle_007_pinsir_generated.png
   ```

4. **Run verification:**
   ```bash
   python modern_card_verifier.py
   ```

---

## 📊 Understanding Results

### **Authenticity Scores:**
- **0.85 - 1.00**: ✅ **HIGH AUTHENTICITY** (Likely genuine layout)
- **0.70 - 0.84**: ⚠️ **MODERATE AUTHENTICITY** (Suspicious elements)
- **0.00 - 0.69**: ❌ **LOW AUTHENTICITY** (Likely fake/AI-generated)

### **Output Files:**
After verification, you'll find:

```
output/
├── reports/
│   └── verification_base_set_001_bulbasaur_20250107_143022.json
├── visualizations/
│   └── verification_base_set_001_bulbasaur_20250107_143022.png
└── confidence_maps/
    └── (Advanced analysis files)
```

### **Sample Report:**
```json
{
  "authenticity_score": 0.923,
  "assessment": "AUTHENTIC",
  "component_scores": {
    "vit_similarity": 0.945,
    "dino_similarity": 0.912,
    "border_score": 0.889,
    "color_score": 0.934
  }
}
```

---

## 🔧 Supported Formats

### **Image Formats:**
- `.png` ← **Recommended** (highest quality)
- `.jpg` / `.jpeg`
- `.bmp`
- `.tiff`

### **Card Sets Supported:**
- `base_set` - Base Set (1st Edition)
- `jungle` - Jungle Expansion
- `fossil` - Fossil Expansion
- `team_rocket` - Team Rocket
- `gym_heroes` - Gym Heroes
- `gym_challenge` - Gym Challenge
- `neo_genesis` - Neo Genesis
- `custom` - Custom/Unknown sets

---

## ⚠️ Important Notes

### **File Naming Rules:**
1. ✅ Use **lowercase** for set names: `base_set` not `Base_Set`
2. ✅ Use **three-digit** card numbers: `001` not `1`
3. ✅ Use **underscores** to separate: `_` not `-` or spaces
4. ✅ Remove special characters from names: `mr_mime` not `mr._mime`

### **Naming Examples:**
```
✅ CORRECT:
base_set_001_bulbasaur_reference.png
jungle_007_pinsir_generated.jpg
team_rocket_083_dark_magneton_reference.png

❌ INCORRECT:
Base Set 001 Bulbasaur Reference.png
jungle-007-pinsir-generated.jpg
team_rocket_83_dark magneton_reference.png
```

---

## 🆘 Troubleshooting

### **Problem: "No reference card found"**
**Solution:** Check file naming matches exactly:
- Generated: `base_set_001_bulbasaur_generated.png`
- Reference: `base_set_001_bulbasaur_reference.png`

### **Problem: CUDA out of memory**
**Solution:** The system automatically falls back to CPU processing

### **Problem: Model download fails**
**Solution:** Models will download automatically on first run

---

## 🔬 Technical Details

### **AI Models Used:**
- **Vision Transformer (ViT-Large)**: Primary layout analysis
- **DINOv2**: Self-supervised feature learning
- **ResNet-50**: Traditional CNN features
- **SSIM**: Structural similarity analysis
- **Perceptual Hashing**: Quick similarity detection

### **Analysis Components:**
1. **Border Analysis** (25% weight): Frame consistency
2. **Color Analysis** (15% weight): Palette accuracy
3. **Layout Analysis** (30% weight): Element positioning
4. **Typography** (10% weight): Font consistency
5. **Feature Similarity** (20% weight): Deep learning features

---

## 🎯 Real-World Usage

### **Scenario 1: AI Artist Verification**
You've created AI-generated Pokemon cards and want to check layout authenticity:

1. Place original cards in `reference_cards/base_set/`
2. Place your AI cards in `input_cards/to_verify/`
3. Run verification to get authenticity scores

### **Scenario 2: Batch Processing**
You have 100 AI-generated cards to verify:

1. Name all files properly with the convention
2. Run: `python modern_card_verifier.py`
3. Check `output/reports/` for individual results

---

## ✅ Final Checklist

Before running verification:

- [ ] Installed dependencies with `install.bat`
- [ ] Reference cards placed in `reference_cards/{set}/`
- [ ] Generated cards placed in `input_cards/to_verify/`
- [ ] File names follow exact convention
- [ ] Image formats are supported (.png, .jpg, etc.)

**Ready to verify? Run:**
```bash
python modern_card_verifier.py
```

---

*For technical support or advanced configuration, check the logs in the `logs/` directory.*