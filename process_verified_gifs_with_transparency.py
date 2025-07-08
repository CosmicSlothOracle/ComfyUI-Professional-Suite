#!/usr/bin/env python3
"""
🎬 VERIFIZIERTE GIFs MIT TRANSPARENZ-ERHALTUNG
=============================================
Verarbeitet nur die vom User verifizierten GIF-Dateien
- Exakte 15-Farben-Palette
- ALLE Frames beibehalten
- TRANSPARENZ erhalten!
"""

from batch_15color_processor import Batch15ColorProcessor


def main():
    # VERIFIZIERTE Liste - alle Dateien existieren garantiert
    verified_file_list = [
        "C:/Users/Public/ComfyUI-master/input/444.gif",
        "C:/Users/Public/ComfyUI-master/input/44334.gif",
        "C:/Users/Public/ComfyUI-master/input/554.gif",
        "C:/Users/Public/ComfyUI-master/input/styled_energy_projectile_archer_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/styled_splash_archer_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/YZvS_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/styled_lightning_archer_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/FNNi_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/styled_dust_ring_archer_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/eugenia-gifmaker-me_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/quintuple-throw-effect-final_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/styled_electric_arc_archer_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/e0b9c377238ff883cf0d8f76e5499a63_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/VIR_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/VU4B_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/styled_lightning_bolt_archer_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/Td1h_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/styled_spark_explosion_archer_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/styled_lightning_archer_demonslayer_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/styled_water_spin_archer_demonslayer_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/styled_cosmic_archer_anime_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/styled_waterbeam_archer_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/styled_slash_archer_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/michkartonmitantrieb_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/styled_ripple_archer_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/stylized_vortex_portal_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/Download_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/Space_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/Download (1)_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/tumblr_ny4unqNuLW1r843z4o6_1280_gif (853×480)_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/portal_frame_001_simple_enhanced_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/m3xrxPO_gif (800×400)_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/portal_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/d6e2feca5d914ef8af32c57ce43c4e91_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/+_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/IFqaJCM_gif (800×400)_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/06c0bff5eca69e95d1b62b7ecb9d3092_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/00dd8a85ebe872350d8ffda6435903a1_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/00370bedfef25129fd8441c864f67bb9_fast_transparent_converted.gif"
    ]

    # 🎯 BATCH VERARBEITUNG MIT TRANSPARENZ-ERHALTUNG
    processor = Batch15ColorProcessor()

    total_files = len(verified_file_list)
    successful = 0
    failed = 0

    print("🎬 BATCH 15-FARBEN-PALETTE PROCESSOR MIT TRANSPARENZ")
    print("=" * 60)
    print(f"📁 {total_files} verifizierte GIF-Dateien zu verarbeiten")
    print(f"📂 Output: output/batch_15color_processed/")
    print("🎨 TRANSPARENZ wird erhalten!")
    print("=" * 60)

    import time
    start_time = time.time()

    for i, input_file in enumerate(verified_file_list, 1):
        try:
            filename = input_file.split('/')[-1]
            output_file = f"output/batch_15color_processed/TRANSPARENCY_15color_{filename}"

            print(f"🔄 [{i:3d}/{total_files}] {filename}")

            success, frame_count = processor.process_single_gif_batch(
                input_file, output_file)

            if success:
                successful += 1
                print(
                    f"   ✅ Erstellt: {frame_count} Frames (ALLE erhalten + TRANSPARENZ!)")
            else:
                failed += 1
                print(f"   ❌ Fehler bei: {filename}")

            # Progress info
            progress = (i / total_files) * 100
            elapsed = time.time() - start_time
            remaining = (elapsed / i) * (total_files - i) if i > 0 else 0

            print(
                f"   📊 Fortschritt: {progress:.1f}% | ✅{successful} ❌{failed} | ~{remaining/60:.1f}min verbleibend")

        except Exception as e:
            failed += 1
            print(f"   ❌ Fehler: {str(e)[:50]}...")

    # Final summary
    total_time = time.time() - start_time
    print("=" * 60)
    print("🎯 BATCH VERARBEITUNG MIT TRANSPARENZ ABGESCHLOSSEN")
    print(f"   ✅ Erfolgreich: {successful}")
    print(f"   ❌ Fehlgeschlagen: {failed}")
    print(f"   ⏱️  Gesamtzeit: {total_time/60:.1f} Minuten")
    print(f"   📁 Output: output\\batch_15color_processed")
    print("   🎨 TRANSPARENZ erhalten!")
    print("=" * 60)


if __name__ == "__main__":
    main()
