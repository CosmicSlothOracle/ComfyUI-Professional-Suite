#!/usr/bin/env python3
"""
CUSTOM SPRITESHEET BATCH PROCESSOR
Verarbeitet eine spezifische Liste von Spritesheets mit dem verbesserten Original Workflow

Features:
- 2x Upscaling + Schärfung + Bildverbesserungen
- Kontrast und Schwarzwert erhöht
- Weißabgleich und Farbintensivierung
- Original Background Detection (4 Ecken)
- Frame Extraction mit 30px Padding
- Verstärkter Vaporwave Filter (37.5%)
- GIF Generation
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time

# Importiere das erweiterte Original Workflow System
from original_workflow_optimized import OriginalWorkflowOptimized


class CustomSpritesheetBatch:
    """Verarbeitet spezifische Spritesheets mit dem verbesserten Workflow"""

    def __init__(self):
        self.processor = OriginalWorkflowOptimized()
        self.target_files = [
            "Pixel-Art_Sprites_eines_entschlossenen_Mannes - Kopie.png",
            "Cartoon_Donald_Trump_Sprite_Sheet - Kopie.png",
            "Golfer_Schwingt_den_Ball_in_Bewegung - Kopie.png",
            "Sprung_und_Flip_in_Pixelkunst - Kopie.png",
            "Karikatur_eines_Golfspielers_in_Bewegung - Kopie.png",
            "Alterer_Kampfkunstler_im_Gehzyklus (1).png",
            "Alterer_Kampfkunstler_in_Bewegung.png",
            "alterer_Kampfkunstler_im_Gehzyklus.png",
            "Kampfkunstler_in_Aktion.png",
            "Alterer_Mann_mit_Stock_im_Gang.png",
            "Der_altere_Mann_mit_Gehstock.png",
            "Bartiger_Mann_im_Anzug_geht.png",
            "Charakterreferenzblatt_eines_alteren_Mannes.png",
            "Laufender_Mann_im_Anzug_mit_Bart.png",
            "Laufender_Krieger_mit_Holzstab.png",
            "Alterer_Mann_in_fliessender_Bewegung.png",
            "Bald_Mann_mit_Stock_in_Bewegung.png",
            "Barack_Obama_in_Bewegung.png",
            "Mann_mit_magischem_Rauchstoss.png",
            "Mann_zaubert_schwebenden_Rauch.png",
            "Energieausbruche_eines_Gelehrten.png",
            "Mann_mit_Rauchenergie-Explosionen.png",
            "Alterer_Krieger_im_Gangzyklus.png",
            "Mann_mit_magischen_Rauchwirbeln.png",
            "Pixel-Schritte_eines_Mannes.png",
            "Charakter-Drehbuch_in_verschiedenen_Ansichten.png",
            "Mann_im_Anzug_-_Verschiedene_Ansichten.png",
            "Schritt-fur-Schritt_Laufzyklus_eines_Mannes.png",
            "Mann_im_Anzug_im_Gehen.png",
            "Mimik_eines_Geschaftsmannes.png",
            "Elderlicher_Mann_in_verschiedenen_Posen.png",
            "Mann_mit_Schwertanimationen.png",
            "Mannlicher_Charakter_in_verschiedenen_Posen (1).png",
            "Verschiedene_Blickwinkel_eines_Charakters.png",
            "Vielfalt_der_Gesichtsausdrucke.png",
            "Mimikenausdrucke_eines_ukrainischen_Mannes.png",
            "Mann_mit_ernstem_Blick_in_verschiedenen_Posen.png",
            "Posen_des_Mannes_in_Olivgrun.png",
            "Manner_in_Bewegung_Sechs_Posen.png",
            "Frau_im_roten_Anzug_mit_Bier.png",
            "Junge_Frau_im_roten_Anzug (1).png",
            "Pixel_Art_Animations_einer_Frau.png",
            "Spritzen_mit_einem_Bierglas.png",
            "Charakterblatt_mit_rotem_Anzug_und_Bier.png",
            "Prasident_Obama_Gehzyklus_in_Cartoon-Stil.png",
            "Wandelnder_Mann_trifft_auf_Wurstchen.png",
            "Gehen_mit_uberraschendem_Wurst-Explosion.png",
            "Geschaftsmann_im_Gang_mit_Wurstexplosion.png",
            "Vektor-Sprite_Alterer_Mann_im_Anzug.png",
            "Mann_im_Anzug_in_Bewegung (1).png",
            "Funf_Ansichten_eines_seriosen_Mannes.png",
            "Emotionen_eines_alteren_Mannes.png",
            "Geschaftsmann_im_Gehen_-_Spritesheet.png",
            "Charakterdesign_eines_Matrosen_in_Posen.png",
            "Mann_in_Anzug_mit_Augenklappe.png",
            "Mann_im_Anzug_in_funf_Posen.png",
            "Bauer_im_Navalanzug.png",
            "Charakterblatt_eines_entschlossenen_Mannes.png",
            "Madchen_im_Regenmantel_im_Pixel-Stil.png",
            "Charakterreferenzblatt_eines_Madchens.png",
            "Jugendliches_Madchen_mit_verschiedenen_Ausdrucken.png",
            "Junges_Madchen_im_Regenmantel.png",
            "Madchen_im_Regenmantel_-_funf_Posen.png",
            "Wachsende_Pflanzen_um_den_Zauberer.png",
            "Magische_Pflanzengartnerei_in_Bewegung.png",
            "Naturkraft_in_acht_Bewegungen.png",
            "Mann_raucht_Zigarette_in_Serie.png",
            "Blauer_Energieball_und_rauchige_Entfaltung.png",
            "Mann_zundet_Feuer_mit_grossem_Feuerzeug.png",
            "Blitzzauber_eines_alteren_Mannes (1).png",
            "Blitzzauber_eines_alteren_Mannes.png",
            "Der_Zauber_des_alten_Mannes.png",
            "Zaubernder_alter_Mann_mit_Blitz.png",
            "Alterer_Mann_im_Anzug_mit_Handgeste.png",
            "Alterer_Mann_in_verschiedenen_Emotionen.png",
            "Gesichtsausdrucke_eines_alteren_Mannes.png",
            "Alterer_Mann_mit_verschiedenen_Gesichtsausdrucken.png",
            "Energieentfaltung_eines_Magiers.png",
            "Gehender_alterer_Mann_im_Anzug.png",
            "Alterer_Mann_im_Anzug_beim_Gehen.png",
            "Mannlicher_Charakter_in_verschiedenen_Posen.png",
            "Emotions_in_20_Ausdrucken.png",
            "Mann_mit_verschiedenen_Gesichtsausdrucken (2).png",
            "Mannlicher_Charakter_im_stilvollen_Anzug.png",
            "Mann_in_Anzug_mit_verschiedenen_Gesichtsausdrucken.png",
            "Mannlicher_Charakter_mit_verschiedenen_Emotionen.png",
            "Der_Wirbelsturm_der_grunen_Energie.png",
            "Kampfender_Gentleman_im_Pixel-Stil.png",
            "Kampfer_im_Anzug_mit_Schwertern (1).png",
            "Charakter-Referenzblatt_eines_Herren.png",
            "Kampfender_Mann_in_pixeliger_Aktion.png",
            "Charakterblatt_mit_verschiedenen_Gesichtsausdrucken.png",
            "Kampfender_Gentleman_im_Anzug.png",
            "Mann_mit_verschiedenen_Gesichtsausdrucken (1).png",
            "Kampfer_im_Anzug_mit_Schwertern.png",
            "Alterer_Mann_mit_Gehstock_in_Bewegung.png",
            "Mann_mit_Stock_im_Gehen.png",
            "Mann_in_Anzug_mit_Gehstock.png",
            "Mannlicher_Charakter_mit_Gehstock.png",
            "Ein_Mann_mit_Schwert_und_Anzug.png",
            "Mann_im_Anzug_in_Bewegung.png",
            "Karikatur_eines_Mannes_in_Aktion.png",
            "Mann_mit_Pistole_im_Gras.png",
            "Springender_Geschaftsmann_in_Bewegung.png",
            "Zombie-Erweckung_durch_Magier.png",
            "Verwandlung_in_eine_Limousine.png",
            "Mann_steigt_aus_Limousine_aus.png",
            "Kampfer_in_pixelart_umhullt_von_Energie.png",
            "Blauer_Energietrail_und_Doppel-Schlag.png",
            "KOS_Event_2025_Poster.png",
            "Sonnenbrille_abnehmen_und_aufsetzen.png",
            "Mann_mit_Sonnenbrille_in_Aktion.png",
            "Der_Mann_mit_den_Sonnenbrillen.png",
            "Wiederbelebung_mit_Energie_und_Energie.png",
            "Der_Mann_und_das_Skelett_beschworen.png",
            "Frau_im_roten_Anzug.png",
            "Energieangriff_einer_entschlossenen_Frau.png",
            "Energieangriff_in_Aktion.png",
            "Zwischenzeitliche_Aktionen_einer_Frau.png",
            "Junge_Frau_im_roten_Anzug.png",
            "Bier_und_Bewegung_Pixel-Animation.png",
            "Pixelart_Geschaftsmann_mit_erhobenem_Finger.png",
            "Kampferin_in_Pixelkunst.png",
            "Kampftechniken_in_Pixelkunst.png",
            "Pixelart_Charaktere_im_Anzug.png",
            "Alterer_Mann_mit_Leuchtarmbanduhr.png",
            "Muskuloser_Wandel_eines_Mannes - Kopie.png",
            "Mannlicher_Charakter_Ausdrucksvariationen.png",
            "Kampf_mit_dem_Bierfass.png",
            "Sprunganimation_einer_jungen_Frau.png",
            "Madchen_im_Regenmantel_Gehzyklus.png",
            "Schnellfeuernder_Wursttrager_im_Anzug.png",
            "Wurstschuter_in_Aktion.png",
            "Raketenangriff_einer_mutigen_Madchenfigur.png",
            "Mann_mit_Wurstwerfer_im_Animationstil.png",
            "Wurstchen-Versorger_im_Pixel-Art_Stil.png",
            "Mann_mit_Wurstkanone_in_Bewegung.png",
            "Elektrische_Energie_in_Pixel-Art.png",
            "Energiegeladener_Mann_mit_lila_Aura.png",
            "Madchen_mit_Pfeifenblatt-Wirbel.png",
            "Selbstbewusste_Geschaftsfrau_im_roten_Anzug.png",
            "Laufender_Madchen-Sprite_mit_Regenmantel.png",
            "Sprunganimation_eines_Businessmanns.png",
            "Sprung_und_Salto_im_Anzug.png",
            "Springender_Geschaftsmann_im_2D-Stil.png",
            "Pixel-Sprite_eines_Geschaftsmannes_im_Sprung.png",
            "Mannlicher_Charakter_beim_Springen.png",
            "Mann_im_Anzug_Funf_Perspektiven.png",
            "Sprunganimation_eines_schrulligen_Geschaftsmannes.png",
            "Sprungener_Mann_im_grauen_Anzug.png",
            "Kampfer_im_Anzug_mit_Waffe.png",
            "Mann_in_Anzug_in_Bewegung.png",
            "Geschaftsmann_in_Aktion.png",
            "Geschaftsmann_in_Action-Posen.png",
            "Gehen_Animation_im_Regenmantel.png",
            "Wasserwirbel_im_Regensturm.png",
            "Mann_im_grauen_Anzug.png",
            "Dunkle_Energie_in_Pixelkunst.png",
            "Korperwachter_mit_Kurbis_und_Pistole.png",
            "Madchen_beschwort_Regen_und_Wasserwirbel.png",
            "Der_Mann_mit_der_Schrotflinte.png",
            "Heldengestalt_in_Aktion.png",
            "Madchen_mit_Wasserpistole_im_Pixelstil.png",
            "Mann_mit_gelebtem_Schritt_und_Entschlossenheit.png",
            "Mann_mit_lila_Blitzenergie.png",
            "Feuersturm_und_rasante_Action.png",
            "Mann_mit_Smartphone_und_Schrotflinte.png",
            "Gehen_im_Cartoon-Stil.png",
            "Madchen_im_Regenmantel_kampft.png",
            "Machtiger_Mann_mit_Blitzenergie.png",
            "Karikatur_von_Trump_in_Aktion.png",
            "Trump_in_Action_Der_dynamische_Kampf.png",
            "Kartonhafte_Trump-Action-Szenen.png",
            "Trump_in_Action_mit_Golfclub_und_Pistole.png",
            "Trump_beim_Golfschwung.png",
            "Trump_beim_Golfspielen_im_Pixelstil.png",
            "Golfschlag_im_Pixel-Stil.png",
            "Machtiger_Zauberer_mit_lila_Blitzen.png",
            "Charakter_mit_Schrotflinte_in_Bewegung.png",
            "Pixelart-Angriff_und_Schlaganimation.png",
            "Vektorillustrationen_eines_vielseitigen_Charakters.png",
            "Sicherheitsmann_im_Einsatz_und_Tanz.png",
            "Charakter-Exploration_in_verschiedenen_Ansichten.png",
            "Kampfsport-Posen_eines_Mannes.png",
            "Bald_Mann_in_Anzug_-_Sprite_Sheet.png",
            "Kampfkunste_in_fliessenden_Bewegungen.png",
            "Sportlicher_Charakter_in_Aktion.png",
            "Mann_in_Anzug_-_Aktion_Pose.png",
            "Marchenhafte_Animation_eines_tapferen_Mannes.png",
            "Mann_im_Anzug_Animationen_und_Posen.png",
            "Mann_in_Bewegung_und_Fall.png",
            "Mann_in_verschiedenen_Bewegungsposen.png",
            "2D_Sprites_des_Mannes_im_Anzug.png",
            "Sprung_und_Balance_in_Aktion.png",
            "Mannlicher_Sprite-Charakter_in_Bewegung.png",
            "Akrobatische_Wurfbewegungen_im_Anzug.png",
            "Piratensuit_mit_Peitsche_und_Holzbein.png",
            "Doppelsprung_Animation_eines_Mannes.png",
            "Vektor-Grafik_Ausdrucksstarken_eines_Piraten.png",
            "Alter_Mann_mit_Augenklappe_und_Holztaille.png",
            "Piratenmann_mit_vielfaltigen_Emotionen.png",
            "Der_entschlossene_Piraten-Older.png",
            "Kraftpaket_im_Anzug.png",
            "Energieexplosion_in_goldenen_Strahlen.png",
            "Energieentladung_einer_Zauberin.png",
            "Mann_und_Bar_in_Aktion.png",
            "Barreiter_im_Kampfeinsatz.png",
            "Energieprojektion_in_goldenem_Licht.png",
            "Piratenkrieger_im_Kampfpose.png",
            "Sprite_Sheet_Layout_fur_Animationen.png",
            "Piraten-Charakter_im_Abenteuer-Modus.png",
            "Mann_mit_Sonnenbrille_im_Wandel (1).png",
            "Mann_im_Anzug_mit_Sonnenbrille (1).png",
            "Mann_mit_Sonnenbrille_im_Wandel.png",
            "Vektorillustration_eines_stilvollen_Mannes.png",
            "Pixel-Art_Charakter_im_Anzug.png",
            "Geschaftsmann_im_Pixel-Art-Stil.png",
            "Sonnenbrillenwechsel_in_stilvollem_Anzug.png",
            "Junger_Mann_im_blauen_Anzug.png",
            "Pirat_im_Kampf_und_Feuerdrill.png",
            "Charakteremoticons_im_formellen_Anzug.png",
            "Kampfkunst_in_Aktion.png",
            "Stylischer_Mann_mit_Sonnenbrille.png",
            "Kampftechniken_im_Sprite-Stil.png",
            "Kleriker_in_verschiedenen_Handlungsschritten.png",
            "Priester_mit_Gebet_und_Bewegung.png",
            "Priester_Ausdrucksserie_in_16_Bildern.png",
            "Clergyman_in_14_Gesichtsausdrucken.png",
            "Vater_im_Gebet_und_Handlungen.png",
            "Barreiten_im_Kampfmodul.png",
            "Bar_und_Krieger_in_Aktion.png",
            "Pixel-Art_Priester-Animationen.png",
            "Priester_im_Gebet_und_Lichtstrahl.png",
            "Mann_auf_Barenrucken_im_Kampf.png",
            "Barreitendes_Muskelpaket_in_Aktion.png",
            "Priester_Animation_Sheet.png",
            "Papstliche_Pixel-Action_in_goldenen_Strahlen.png",
            "Helle_Strahlen_des_Gebets.png",
            "Priester-Charakter_in_verschiedenen_Posen.png",
            "Priester_in_verschiedenen_Posen.png",
            "Piratenabenteuer_in_neun_Posen.png",
            "Schwertangriff_eines_Mannes_in_Anzug.png",
            "Kampfszene_eines_entschlossenen_Mannes.png",
            "Schwertangriff_in_Vintage-Animation.png",
            "Mittelalterlicher_Mann_im_Olivgrun.png",
            "Madchen_im_Regenmantel_Kampfposen.png",
            "Zelensky_in_verschiedenen_Gefuhlsausdrucken.png",
            "Mannerausdrucke_im_Militarstil.png",
            "Charakterausdrucke_eines_Mannes - Kopie.png",
            "Mann_mit_wechselnden_Gefuhlen - Kopie.png",
            "Schritte_eines_soldatischen_Gangzyklus - Kopie.png",
            "Marschender_Soldat_im_Sprite-Stil - Kopie.png",
            "Madchen_im_Sprungkick-Training - Kopie.png",
            "Ukrainischer_Kampfer_in_Aktion - Kopie.png",
            "Spinnende_Kampferin_im_Regenmantel - Kopie.png",
            "Kampfkunst_in_vier_Reihen - Kopie.png",
            "Handlungssequenz_eines_entschlossenen_Charakters - Kopie.png",
            "Sprung_und_Drehung_eines_Kampfers - Kopie.png",
            "Angriff_in_dynamischen_Bewegungen - Kopie.png",
            "Kraftvolle_Angriffshaltung_in_Bewegung - Kopie.png",
            "Pflanzenmagie_in_Aktion - Kopie.png",
            "Elektrische_Transformation_und_Roboterschopfung - Kopie.png",
            "Mech_und_Explosion_von_Blattern - Kopie.png",
            "Kampfer_im_Oliv-Anzug - Kopie.png",
            "Roboteranzug-Transformation_in_12_Schritten - Kopie.png",
            "Nervose_Gange_eines_Cartoon-Charakters - Kopie.png",
            "Mech_Pilot_im_Actionmodus - Kopie.png",
            "Magie_des_Wachstums_in_Aktion - Kopie.png",
            "Wurfender_Degen_in_Aktion - Kopie.png",
            "Messerwurf-Animation_im_Anzug - Kopie.png",
            "Der_Daggerwurf_in_12_Szenen - Kopie.png",
            "Magischer_Angriff_in_Aktion - Kopie.png",
            "Kampferische_Entladung_mit_Blitzen - Kopie.png",
            "Kampferische_Pose_Dynamische_Aktion - Kopie.png",
            "Karikaturhafte_Posen_eines_Mannes - Kopie.png",
            "Karikatur_der_wutenden_Mimik - Kopie.png",
            "Karikaturen_eines_wutenden_Charakters - Kopie.png",
            "Karikatur_eines_wutenden_Politikers - Kopie.png"
        ]

    def run_custom_batch(self):
        """Führt die Batch-Verarbeitung der spezifischen Dateien durch"""
        print("🎮 CUSTOM SPRITESHEET BATCH PROCESSOR")
        print("=" * 60)
        print(f"📝 Target files: {len(self.target_files)}")
        print("🔧 Workflow: Enhanced Original + Optimizations")
        print("   1. 🔍 2x Upscaling + Sharpening + Quality Enhancement")
        print("   2. 🎯 Original Background Detection (4 corners)")
        print("   3. 📦 Original Frame Extraction (30px padding)")
        print("   4. 🌈 Enhanced Vaporwave Filter (37.5% intensity)")
        print("   5. 🎬 GIF Generation")
        print("=" * 60)

        # Erstelle Session-Verzeichnis
        self.processor.create_session_directory()

        # Verarbeite jede Datei
        existing_files = []
        missing_files = []

        for filename in self.target_files:
            file_path = Path("input") / filename
            if file_path.exists():
                existing_files.append(file_path)
            else:
                missing_files.append(filename)

        print(f"\n📊 File Status:")
        print(f"   ✅ Found: {len(existing_files)} files")
        if missing_files:
            print(f"   ❌ Missing: {len(missing_files)} files")
            print("   Missing files:")
            for missing in missing_files[:5]:  # Show first 5 missing files
                print(f"      - {missing}")
            if len(missing_files) > 5:
                print(f"      ... and {len(missing_files) - 5} more")

        if not existing_files:
            print("❌ No files found to process!")
            return

        print(f"\n🚀 Processing {len(existing_files)} files...")
        self.processor.start_time = time.time()

        # Verarbeite alle vorhandenen Dateien
        for file_path in existing_files:
            try:
                self.processor.process_single_spritesheet(file_path)
                self.processor.processed_files.append(str(file_path))
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
                self.processor.failed_files.append(str(file_path))

        # Erstelle Abschlussbericht
        self.processor.create_master_report()

        # Erfolgsstatistiken
        total_time = time.time() - self.processor.start_time
        success_count = len(self.processor.processed_files)
        failed_count = len(self.processor.failed_files)

        print(f"\n🎉 CUSTOM BATCH PROCESSING COMPLETE!")
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        print(f"✅ Successfully processed: {success_count} files")
        print(f"❌ Failed: {failed_count} files")
        print(
            f"📦 Total frames extracted: {self.processor.total_frames_extracted}")
        print(f"🌈 All frames processed with enhanced Vaporwave filter (37.5%)")
        print(f"📁 Results saved to: {self.processor.session_dir}")


def main():
    """Hauptfunktion"""
    try:
        processor = CustomSpritesheetBatch()
        processor.run_custom_batch()
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
