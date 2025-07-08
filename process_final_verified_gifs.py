#!/usr/bin/env python3
"""
🎬 FINALE VERIFIZIERTE GIF-BATCH VERARBEITUNG
=============================================
Verarbeitet die vom User final bereitgestellte Liste von GIF-Dateien
- Exakte 15-Farben-Palette
- ALLE Frames beibehalten
- TRANSPARENZ erhalten!
"""

from batch_15color_processor import Batch15ColorProcessor


def main():
    # FINALE VERIFIZIERTE Liste vom User
    final_verified_files = [
        "C:/Users/Public/ComfyUI-master/input/80c8b5e077a2b58aec45013484d5ba7f_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/81c2091067d61552af5bdccf26a7f477_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/82a55a7f9d9a3cab7545146c78146d9d_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/554.gif",
        "C:/Users/Public/ComfyUI-master/input/44334.gif",
        "C:/Users/Public/ComfyUI-master/input/444.gif",
        "C:/Users/Public/ComfyUI-master/input/333.gif",
        "C:/Users/Public/ComfyUI-master/input/123.gif",
        "C:/Users/Public/ComfyUI-master/input/12.gif",
        "C:/Users/Public/ComfyUI-master/input/1.gif",
        "C:/Users/Public/ComfyUI-master/input/giphy.gif",
        "C:/Users/Public/ComfyUI-master/input/chorizombi-umma.gif",
        "C:/Users/Public/ComfyUI-master/input/test_image.png",
        "C:/Users/Public/ComfyUI-master/input/output-onlinegiftools (2)_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/eleni_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/d1d71ff4514a99bfb0f0e93ef59e3575_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/7ee80664e6f86ac416750497557bf6fc_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/0e35f5b16b8ba60a10fdd360de075def_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/517K_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/1_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/1ccd5fdfa2791a1665dbca3420a37120_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/1OqPxD_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/2a9bdd70dc936f3c482444e529694edf_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/3d8b3736a14a1034e2badb0fd641566f_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/3e5e30ba640660145fec1041550e75f8_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/3f288d7d75e1a46a359b180e45d62c7c_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/4edc1b2c3a7513d7df2968e92bd75d09_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/5e4ff9f247f84ce5a09ddfe9d435dc67._fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/7c20c1d1a17442f7f3362241bf57e6f8_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/7eBQ_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/8b118732fae6c4b738c400d9d1687257_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/9be30323126227.560476b52e37e_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/9f9735a5c4d8bd0502ec3e64bdda6cfb_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/9f720323126213.56047641e9c83_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/15fe9e01d0aefa5f3da72da8c0dd9d3f_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/21f78d23126231.5604767a04f53_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/022f076b146c9ffdc0805d383b7a2f32_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/23_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/24ae4572aaab593bc1cc04383bc07591_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/26a09634994b96e38d5bdafd16fa9b75_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/30f0cd23127003.560476bf1c589_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/52de83165cfcec1ba2b2b49fe1c9d883_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/62ef326ab85cdd46ea19e268a4ba4dcf_fast_transparent_converted.gif",
        "C:/Users/Public/ComfyUI-master/input/64b6d3d0458eaaf8129a59e50327e77c_fast_transparent_converted.gif"
    ]

    # 🎯 BATCH VERARBEITUNG MIT TRANSPARENZ-ERHALTUNG
    processor = Batch15ColorProcessor()

    total_files = len(final_verified_files)
    successful = 0
    failed = 0

    print("🎬 FINALE BATCH 15-FARBEN-PALETTE VERARBEITUNG")
    print("=" * 60)
    print(f"📁 {total_files} finale verifizierte GIF-Dateien zu verarbeiten")
    print(f"📂 Output: output/batch_15color_processed/")
    print("🎨 TRANSPARENZ wird erhalten!")
    print("🎯 ALLE Frames werden beibehalten!")
    print("=" * 60)

    import time
    start_time = time.time()

    for i, input_file in enumerate(final_verified_files, 1):
        try:
            filename = input_file.split('/')[-1]
            output_file = f"output/batch_15color_processed/FINAL_15color_{filename}"

            print(f"🔄 [{i:3d}/{total_files}] {filename}")

            # Verarbeite mit Transparenz-Erhaltung
            processor.process_single_gif_batch(input_file, output_file)

            successful += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining_time = avg_time * (total_files - i)

            print(f"   ✅ ALLE erhalten + TRANSPARENZ!")
            print(
                f"   📊 Fortschritt: {(i/total_files)*100:.1f}% | ✅{successful} ❌{failed} | ~{remaining_time/60:.1f}min verbleibend")

        except Exception as e:
            print(f"   ❌ Fehler: {str(e)}")
            failed += 1

    total_time = time.time() - start_time

    print("=" * 60)
    print("🎯 FINALE BATCH VERARBEITUNG ABGESCHLOSSEN")
    print(f"   ✅ Erfolgreich: {successful}")
    print(f"   ❌ Fehlgeschlagen: {failed}")
    print(f"   ⏱️  Gesamtzeit: {total_time/60:.1f} Minuten")
    print(f"   📁 Output: output\\batch_15color_processed")
    print("=" * 60)
    print("🎯 VOLLSTÄNDIGE VERARBEITUNG ABGESCHLOSSEN!")
    print("✅ Alle GIFs mit exakter 15-Farben-Palette, erhaltenen Frames und Transparenz!")
    print("📂 Output: output/batch_15color_processed/FINAL_15color_*.gif")


if __name__ == "__main__":
    main()
