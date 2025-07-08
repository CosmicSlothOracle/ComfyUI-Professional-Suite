@echo off
echo =========================================================
echo Modern Pokemon Card Authenticity Verification Setup
echo =========================================================

echo Creating directory structure...
mkdir reference_cards 2>nul
mkdir reference_cards\base_set 2>nul
mkdir reference_cards\jungle 2>nul
mkdir reference_cards\fossil 2>nul
mkdir reference_cards\team_rocket 2>nul
mkdir reference_cards\gym_heroes 2>nul
mkdir reference_cards\gym_challenge 2>nul
mkdir reference_cards\neo_genesis 2>nul
mkdir reference_cards\custom 2>nul
mkdir input_cards 2>nul
mkdir input_cards\to_verify 2>nul
mkdir input_cards\processed 2>nul
mkdir output 2>nul
mkdir output\reports 2>nul
mkdir output\visualizations 2>nul
mkdir output\confidence_maps 2>nul
mkdir models 2>nul
mkdir models\vision_transformer 2>nul
mkdir models\lora_adapters 2>nul
mkdir logs 2>nul
mkdir config 2>nul

echo Directory structure created successfully!

echo.
echo Installing Python dependencies...
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.35.0
pip install diffusers>=0.24.0
pip install timm>=0.9.12
pip install opencv-python>=4.8.1
pip install pillow>=10.1.0
pip install accelerate>=0.24.0
pip install datasets>=2.14.0
pip install peft>=0.6.0
pip install scikit-image>=0.21.0
pip install matplotlib>=3.8.0
pip install seaborn>=0.12.0
pip install numpy>=1.24.0
pip install scipy>=1.11.0
pip install colorspacious>=1.1.2
pip install colour-science>=0.4.3
pip install faiss-cpu>=1.7.4
pip install imagehash>=4.3.1
pip install tqdm>=4.66.0
pip install safetensors>=0.4.0

echo.
echo =========================================================
echo Setup completed successfully!
echo =========================================================
echo.
echo Next steps:
echo 1. Place reference cards in reference_cards\{set_name}\ folders
echo 2. Place generated cards in input_cards\to_verify\ folder
echo 3. Run: python modern_card_verifier.py
echo.
echo Example file names:
echo   Reference: reference_cards\base_set\base_set_001_bulbasaur_reference.png
echo   Generated: input_cards\to_verify\base_set_001_bulbasaur_generated.png
echo.
pause