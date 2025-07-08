#!/usr/bin/env python3
"""
AUTOMATIC MODEL DOWNLOADER - ULTIMATE ANIME PIPELINE
===================================================

Lädt automatisch alle essentiellen Modelle für die Anime Video Generation herunter.
Basierend auf umfassender Internet-Recherche und bewährten Technologien.
"""

import os
import sys
import requests
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse
import time

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EssentialModelDownloader:
    """
    Automatischer Download aller essentiellen Modelle
    """

    def __init__(self):
        self.models_dir = Path("models")
        self.download_timeout = 300  # 5 Minuten pro Datei

        # Essentielle Modelle (Research-basiert)
        self.essential_models = {
            "motion_modules": {
                "target_dir": "animatediff_models",
                "models": {
                    "v3_sd15_mm.ckpt": {
                        "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt",
                        "size_mb": 837,
                        "priority": 1,
                        "description": "AnimateDiff V3 Motion Module - Neueste Version"
                    },
                    "mm_sd_v15_v2.ckpt": {
                        "url": "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt",
                        "size_mb": 909,
                        "priority": 2,
                        "description": "AnimateDiff V2 Motion Module - Bewährt stabil"
                    }
                }
            },
            "checkpoints": {
                "target_dir": "checkpoints",
                "models": {
                    "counterfeitV30_v30.safetensors": {
                        "url": "https://civitai.com/api/download/models/57618",
                        "size_mb": 2000,
                        "priority": 1,
                        "description": "CounterfeitV3.0 - Hochwertige Anime-Generierung"
                    }
                }
            },
            "controlnet": {
                "target_dir": "controlnet",
                "models": {
                    "control_v11p_sd15_openpose.pth": {
                        "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth",
                        "size_mb": 1400,
                        "priority": 1,
                        "description": "OpenPose ControlNet für Pose-Kontrolle"
                    }
                }
            },
            "vae": {
                "target_dir": "vae",
                "models": {
                    "vae-ft-mse-840000-ema-pruned.safetensors": {
                        "url": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
                        "size_mb": 335,
                        "priority": 1,
                        "description": "Standard VAE für bessere Farben"
                    }
                }
            }
        }

    def create_directories(self):
        """
        Erstellt alle erforderlichen Verzeichnisse
        """
        for category, config in self.essential_models.items():
            target_dir = self.models_dir / config["target_dir"]
            target_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Verzeichnis erstellt: {target_dir}")

    def download_file(self, url: str, target_path: Path, expected_size_mb: int = None) -> bool:
        """
        Lädt eine einzelne Datei herunter
        """
        if target_path.exists():
            logger.info(
                f"⏭️  {target_path.name} bereits vorhanden, überspringe...")
            return True

        logger.info(f"📥 Lade herunter: {target_path.name}")
        logger.info(f"🔗 URL: {url}")

        try:
            response = requests.get(
                url, stream=True, timeout=self.download_timeout)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        # Progress anzeigen
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            if downloaded_size % (1024 * 1024 * 10) == 0:  # Alle 10MB
                                logger.info(
                                    f"📊 Progress: {progress:.1f}% ({downloaded_size // (1024*1024)}MB / {total_size // (1024*1024)}MB)")

            # Dateigröße überprüfen
            actual_size_mb = target_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"✅ Download abgeschlossen: {target_path.name} ({actual_size_mb:.1f}MB)")

            if expected_size_mb and abs(actual_size_mb - expected_size_mb) > 50:
                logger.warning(
                    f"⚠️  Unerwartete Dateigröße: {actual_size_mb:.1f}MB (erwartet: {expected_size_mb}MB)")

            return True

        except Exception as e:
            logger.error(
                f"❌ Download fehlgeschlagen für {target_path.name}: {e}")
            if target_path.exists():
                target_path.unlink()  # Defekte Datei löschen
            return False

    def download_all_models(self):
        """
        Lädt alle essentiellen Modelle herunter
        """
        logger.info("🚀 STARTE DOWNLOAD ALLER ESSENTIELLEN MODELLE")

        # Verzeichnisse erstellen
        self.create_directories()

        total_models = sum(len(config["models"])
                           for config in self.essential_models.values())
        downloaded_count = 0
        failed_downloads = []

        # Nach Priorität sortiert herunterladen
        for category, config in self.essential_models.items():
            logger.info(f"\n📂 KATEGORIE: {category.upper()}")
            target_base_dir = self.models_dir / config["target_dir"]

            # Modelle nach Priorität sortieren
            sorted_models = sorted(
                config["models"].items(),
                key=lambda x: x[1].get("priority", 999)
            )

            for model_name, model_config in sorted_models:
                target_path = target_base_dir / model_name

                logger.info(f"\n🎯 {model_config['description']}")
                success = self.download_file(
                    model_config["url"],
                    target_path,
                    model_config.get("size_mb")
                )

                if success:
                    downloaded_count += 1
                else:
                    failed_downloads.append((category, model_name))

                # Kurze Pause zwischen Downloads
                time.sleep(2)

        # Zusammenfassung
        logger.info(f"\n🎯 DOWNLOAD ZUSAMMENFASSUNG:")
        logger.info(f"✅ Erfolgreich: {downloaded_count}/{total_models}")
        logger.info(f"❌ Fehlgeschlagen: {len(failed_downloads)}")

        if failed_downloads:
            logger.warning(f"\n⚠️  FEHLGESCHLAGENE DOWNLOADS:")
            for category, model_name in failed_downloads:
                logger.warning(f"   - {category}: {model_name}")

        if downloaded_count == total_models:
            logger.info(f"\n🎉 ALLE MODELLE ERFOLGREICH HERUNTERGELADEN!")
            logger.info(
                f"📁 Modelle gespeichert in: {self.models_dir.absolute()}")
            return True
        else:
            logger.warning(f"\n⚠️  EINIGE DOWNLOADS FEHLGESCHLAGEN")
            logger.info(
                f"💡 Versuche fehlgeschlagene Downloads manuell oder mit besserer Internetverbindung")
            return False

    def verify_installation(self):
        """
        Überprüft die Installation aller Modelle
        """
        logger.info("\n🔍 ÜBERPRÜFE INSTALLATION...")

        missing_models = []
        total_size_gb = 0

        for category, config in self.essential_models.items():
            target_base_dir = self.models_dir / config["target_dir"]

            for model_name, model_config in config["models"].items():
                target_path = target_base_dir / model_name

                if target_path.exists():
                    size_mb = target_path.stat().st_size / (1024 * 1024)
                    total_size_gb += size_mb / 1024
                    logger.info(f"✅ {model_name} ({size_mb:.1f}MB)")
                else:
                    missing_models.append(f"{category}/{model_name}")
                    logger.warning(f"❌ {model_name} FEHLT")

        logger.info(f"\n📊 GESAMTGRÖSSE: {total_size_gb:.2f}GB")

        if not missing_models:
            logger.info("🎉 ALLE MODELLE VOLLSTÄNDIG INSTALLIERT!")
            return True
        else:
            logger.warning(f"⚠️  {len(missing_models)} MODELLE FEHLEN:")
            for model in missing_models:
                logger.warning(f"   - {model}")
            return False


def main():
    """
    Hauptfunktion für Model-Download
    """
    print("🎌 ULTIMATE ANIME PIPELINE - MODEL DOWNLOADER")
    print("=" * 60)
    print("Automatischer Download aller essentiellen Modelle")
    print("Basierend auf umfassender Internet-Recherche")
    print("=" * 60)

    try:
        downloader = EssentialModelDownloader()

        # Überprüfe verfügbaren Speicherplatz
        total_size_estimate = 6.5  # GB geschätzt
        print(f"\n📊 Geschätzte Download-Größe: ~{total_size_estimate}GB")
        print("⏱️  Geschätzte Zeit: 20-60 Minuten (abhängig von Internetverbindung)")

        # Starte Download
        success = downloader.download_all_models()

        # Überprüfe Installation
        verification_success = downloader.verify_installation()

        if success and verification_success:
            print("\n🎯 MODEL DOWNLOAD ERFOLGREICH ABGESCHLOSSEN!")
            print("\n📋 NÄCHSTE SCHRITTE:")
            print("   1. ComfyUI neu starten")
            print("   2. Workflow-Templates testen")
            print("   3. Erste Anime-Videos generieren!")
            return True
        else:
            print("\n⚠️  DOWNLOAD UNVOLLSTÄNDIG")
            print("💡 Überprüfe Internetverbindung und versuche erneut")
            return False

    except Exception as e:
        logger.error(f"❌ Kritischer Fehler: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
