#!/usr/bin/env python3
"""
TARGETED BATCH PROCESSOR - SPECIFIC FILE LIST
FAST TRANSPARENT WORKFLOW - 100% TRANSPARENT BACKGROUNDS
"""

import subprocess
import os
import sys
import time
from pathlib import Path

# SPEZIFISCHE DATEILISTE VOM USER
TARGETED_FILES = [
    "e109a1a8c8324b38947ff23eded58d99..gif",
    "559c2e47e018ac820885a753d51c098e.gif",
    "26a09634994b96e38d5bdafd16fa9b75.gif",
    "d60abfb7ec3ba5dea74f4181782c8a37.gif",
    "bf28790973c87cf39ab2eda62d9653b3.gif",
    "d2f092a467a547f4eb80e92a58ec798e.gif",
    "62ef326ab85cdd46ea19e268a4ba4dcf.gif",
    "2a9bdd70dc936f3c482444e529694edf.gif",
    "022f076b146c9ffdc0805d383b7a2f32.gif",
    "ebb946e99e5ff654fdaf45112ddac4c7.gif",
    "a6108b31b391378d30856edba57172a4.gif",
    "68348466b7c0cdcd1c5ac628314a4020 (1).gif",
    "81c2091067d61552af5bdccf26a7f477.gif",
    "517b57598f117c4be4910f9db8e60536.gif",
    "c0a420e57c75f1f5863d48197fd19c3a.gif",
    "837b898b4d1eb49036dfce89c30cba59.gif",
    "1938caaca4055d456a9c12ef8648a057.gif",
    "1ccd5fdfa2791a1665dbca3420a37120.gif",
    "2161e0c91f4e326f104ffe30552232ac.gif",
    "642c8bf2af91afd9e44dec00a06914f6.gif",
    "8995d65c9054dddf1036afccc5e13359.gif",
    "9f9735a5c4d8bd0502ec3e64bdda6cfb.gif",
    "24ae4572aaab593bc1cc04383bc07591.gif",
    "R (3).gif",
    "spiral.gif",
    "P9vTI0.gif",
    "ab87d0968a07f8d7873e98d55e8a05aa.gif",
    "e675dd23126203.5604763779b3e.gif",
    "9f720323126213.56047641e9c83.gif",
    "b8c60c23126211.56047639576d7.gif",
    "4616bd23126229.560476b7483ce (1).gif",
    "761c9e23126197.5604763ee41d9.gif",
    "90533123125971.5604763d19db6.gif",
    "30f0cd23127003.560476bf1c589.gif",
    "82a55a7f9d9a3cab7545146c78146d9d.gif",
    "4616bd23126229.560476b7483ce.gif",
    "21f78d23126231.5604767a04f53.gif",
    "gwanwoo-tak-1490340467910.gif",
    "d74kefa-d3488cf5-a9d7-4e9d-9f57-7f9f19c708d8.gif",
    "+.gif",
    "9be30323126227.560476b52e37e.gif",
    "0ab95223125965.560476786fbe2.gif",
    "pouring-milk-into-a-bowl-62ocyu51a4wijtkj.gif",
    "Hr9Q7O.gif",
    "bf1cfd0c3ab46c304bdd71fe7daf0cbb.gif",
    "original-929f40cd2227e48d41c7aec5bb6be5e7.gif",
    "tableware-milk-fill-only-34.gif",
    "tumblr_3c7ddc41f9d033983af0359360d773cf_38fa1d69_540.gif",
    "c19d6274e1fd53c5ca46cdafccb4cbc9.gif",
    "2609a4ee571128f2079373b8d7b0a1a0.gif",
    "0af40433ddb755bfee5a1738717c7028.gif",
    "be28e91b47891c6861207edd5bca8e6c.gif",
    "uJ1Dg2.gif",
    "48443c9ff5614de637efc09bcede2f90.gif",
    "3f288d7d75e1a46a359b180e45d62c7c.gif",
    "3e5e30ba640660145fec1041550e75f8.gif",
    "16583.gif",
    "R (2).gif",
    "R (1).gif",
    "R.gif",
    "planet-animierte-gifs-014.gif",
    "yv80rldigcu61.gif",
    "giphy.gif",
    "ffcb41ab727135955c859e88bc286c54.gif",
    "tenor.gif",
    "tumblr_inline_nfpj8uucP11s6lw3t540.gif",
    "dh-11.gif",
    "dh-10.gif",
    "dh-09.gif",
    "dh-08.gif",
    "dh-07.gif",
    "dh-06.gif",
    "dh-05.gif",
    "dh-04.gif",
    "dh-03.gif",
    "dh-02.gif",
    "dh-01.gif",
    "dhds-VfOdi5X - Konvertiert1-11-f2cb3ba67f45e1dc516dfbb31bbe94f7.gif",
    "dhds-Td1h - Konvertiert1-09-e61937b9f2b91b4dcce893fa807a2fe9.gif",
    "dhds-quintuple-throw-effect-final - Konvertiert1-07-855c3016a58e8f32ee99907eb8928024.gif",
    "dhds-b588e7067a7676432635295ee5db43f5 - Konvertiert1-06-4739f342636e105cec778f27eb746e7c.gif",
    "dhds-369e9ceeb279785e7a86bed68490af92 - Konvertiert1-05-9fa83ac3e98946f002476e61355f1e3f.gif",
    "dhds-7ee80664e6f86ac416750497557bf6fc - Konvertiert1.mkv-04-eafb94011dc7841d192f9638d9655f5f.gif",
    "dhds-1OqPxD - Konvertiert1-03-5772c64aab58d01156bb55022b1fffb6.gif",
    "dhds-00dd8a85ebe872350d8ffda6435903a1 - Konvertiert1.mkv-02-853dab4179b25602029c55500ad79547.gif",
    "dhds-ZavF5qz - Konvertiert1.mkv-01-c54838a464defcf428e215268204bc15.gif",
    "portal.gif",
    "dsvgs.gif",
    "m3xrxPO_gif (800×400).gif",
    "portal_frame_001_simple_enhanced.gif",
    "deomnplanet.gif",
    "anotherspriteauramovieclip.gif",
    "6220502aa1db1db990dc03c15eb134e5.gif",
    "d1d71ff4514a99bfb0f0e93ef59e3575.gif",
    "00370bedfef25129fd8441c864f67bb9.gif",
    "7ee80664e6f86ac416750497557bf6fc.gif",
    "tumblr_ny4unqNuLW1r843z4o6_1280_gif (853×480).gif",
    "0e35f5b16b8ba60a10fdd360de075def.gif",
    "1644545_21bf9.gif",
    "517K.gif",
    "Space.gif",
    "Download (1).gif",
    "Download.gif",
    "a2e9c46e0d9fab0b7de0688aaf49f25f.gif",
    "stylized_vortex_portal.gif",
    "beginningofmilch.gif",
    "antrieb.gif",
    "michkartonmitantrieb.gif",
    "styled_ripple_archer.gif",
    "styled_waterbeam_archer.gif",
    "styled_slash_archer.gif",
    "styled_cosmic_archer_anime.gif",
    "styled_water_spin_archer_demonslayer.gif",
    "styled_lightning_archer_demonslayer.gif",
    "styled_spark_explosion_archer.gif",
    "Td1h.gif",
    "styled_lightning_bolt_archer.gif",
    "VU4B.gif",
    "VIR.gif",
    "e0b9c377238ff883cf0d8f76e5499a63.gif",
    "bc92a54fd52558c950378140d66059e3.gif",
    "styled_electric_arc_archer.gif",
    "quintuple-throw-effect-final.gif",
    "981572617a1436ecbd91801613823681.gif",
    "64b6d3d0458eaaf8129a59e50327e77c.gif",
    "eugenia-gifmaker-me.gif",
    "styled_dust_ring_archer.gif",
    "FNNi.gif",
    "styled_energy_projectile_archer.gif",
    "c5cd86843eaedd2a1ec8511e8c304b30.gif",
    "styled_lightning_archer.gif",
    "369e9ceeb279785e7a86bed68490af92.gif",
    "YZvS.gif",
    "52de83165cfcec1ba2b2b49fe1c9d883.gif",
    "styled_splash_archer.gif",
    "styled_purple_flame_archer.gif",
    "eeac95ccf445298bf822afed492d9b8a.gif",
    "WrqR.gif",
    "IuGx.gif",
    "styled_tornado_archer.gif",
    "a63cc5bbc877920b126c3ffe3137efce_w200.gif",
    "00dd8a85ebe872350d8ffda6435903a1.gif",
    "styled_portal_projectile_archer.gif",
    "lexan-cool-effect.gif",
    "b33d0666d4b65b2e92bfe804aaf68fa4.gif",
    "styled_projectile_archer.gif",
    "styled_explosion.gif",
    "styled_final_attack.gif",
    "output-onlinegiftools (6).gif",
    "final-attack-effect-3.gif",
    "VfOdi5X.gif",
    "DwfOrtv.gif",
    "ZavF5qz.gif",
    "325800_989f6.gif",
    "stimulate-effect.gif",
    "output-onlinegiftools (34).gif",
    "output-onlinegiftools (5).gif",
    "863b7fcc2b7b4c7d98478fe796490634.gif",
    "1OqPxD.gif",
    "tSfCbU.gif",
    "b588e7067a7676432635295ee5db43f5.gif",
    "7eBQ.gif",
    "output-onlinegiftools (2).gif",
    "dance_105bpm.gif",
    "ldance11_sharpened.gif",
    "Laurin dance..gif",
    "l.gif",
    "dance_combined.gif",
    "laurinbimsdnace.gif",
    "sprite_animation_precise.gif",
    "animated_sprite_corrected (1).gif",
    "punch_cropped.gif",
    "animated_sprite.gif",
    "punch.gif",
    "sprite_animation.gif",
    "1.gif",
    "oscar isaac dance ex machina.gif",
    "7c20c1d1a17442f7f3362241bf57e6f8.gif",
    "1464190168-giphy-16.gif",
    "story_club_night_.2.gif",
    "xi_bw.gif",
    "xi.gif",
    "stargazer_story.gif",
    "zoom_out_transition.gif",
    "as.gif",
    "eleni.gif",
    "dance_animation.gif",
    "transparent_character.gif",
    "dance_sprite.gif",
    "final_growth.gif",
    "magic_growth.gif",
    "plant_growth_magic.gif",
    "output-onlinegiftools (1).gif",
    "characters_final16.gif",
    "characters_12.gif",
    "characters.gif",
    "animation.gif",
    "xi_1.gif",
    "van_gogh_dancer_transparent.gif",
    "sprite_anim_v2 (1).gif",
    "combined_animation_optimized (1).gif",
    "final_dance_pingpong_slowed.gif",
    "final_dance_pingpong_transparent.gif",
    "neun_leben.gif",
    "flut_raum.gif",
    "surreal_aufwachen_1.gif",
    "chaplin_dance.gif",
    "villa_party.gif",
    "koi_rotate_1.gif",
    "rick-and-morty-fortnite.gif",
    "combined_jesus_aura_1.gif",
    "combined_animation_retry (2).gif",
    "erdoattackknife8.gif",
    "ooo.gif",
    "XIWALK.gif",
    "Intro_27_512x512.gif",
    "sel_1.gif",
    "aas.gif",
    "pirate.gif",
    "walk_obamf6_1.gif",
    "merkelflip10f_1_1.gif",
    "erdo.gif",
    "dodo_1.gif",
    "putbear.gif",
    "PEERKICK.GIF",
    "peer_4.gif",
    "gym-roshi_2.gif",
    "200w.gif",
    "spinning_vinyl_clean.gif",
    "535f7143cb7c6c78135f9a84b27d71ab.gif",
    "0be6dbb2639d41162f0a518c28994066.gif",
    "46675a00bc70fb3eb2e9f4c09d34dab2.gif",
    "13391684557ae6df587a5ef4d92d8366.gif",
    "3d8b3736a14a1034e2badb0fd641566f.gif",
    "0ea53a3cdbfdcf14caf1c8cccdb60143.gif",
    "952fa268bac222d795de5a2729ac11d2.gif",
    "814a64af2b852a4d3a847ff890091a51.gif",
    "15fe9e01d0aefa5f3da72da8c0dd9d3f.gif",
    "90db9de284da8af70d784fdeeed8ff9f.gif",
    "80c8b5e077a2b58aec45013484d5ba7f.gif",
    "85b7643f3912a2f8f7f47b6014ca5968.gif",
    "7114e79c293278d5c826407aea4734f9.gif",
    "8b118732fae6c4b738c400d9d1687257.gif",
    "9424d652df29656384cfd6ca18684b56.gif",
    "4edc1b2c3a7513d7df2968e92bd75d09.gif",
    "5e4ff9f247f84ce5a09ddfe9d435dc67..gif"
]


def run_targeted_batch_processor():
    """Run the targeted batch processor for specific file list"""

    print("TARGETED BATCH PROCESSOR - SPECIFIC FILE LIST")
    print("FAST TRANSPARENT WORKFLOW - 100% TRANSPARENT BACKGROUNDS")
    print("WF7 (Histogram) + WF2 (Anti-Fractal) - Ultra Speed Processing")
    print("=" * 80)

    # Paths
    input_dir = Path("C:/Users/Public/ComfyUI-master/input")
    output_dir = Path("C:/Users/Public/ComfyUI-master/output")

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    # Filter existing files
    existing_files = []
    for filename in TARGETED_FILES:
        file_path = input_dir / filename
        if file_path.exists():
            existing_files.append(filename)
        else:
            print(f"NOT FOUND: {filename}")

    print(f"Found {len(existing_files)} files to process")
    print(f"Processing with FAST TRANSPARENT WORKFLOW")
    print("=" * 80)

    # Process each file
    success_count = 0
    error_count = 0
    timeout_count = 0
    skip_count = 0
    total_time = 0

    start_total = time.time()

    for i, filename in enumerate(existing_files, 1):
        print(f"\n[{i}/{len(existing_files)}] FAST TRANSPARENT -> {filename}")

        input_file = input_dir / filename
        output_file = output_dir / \
            f"{Path(filename).stem}_fast_transparent.mp4"

        # Skip if already processed
        if output_file.exists():
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            print(f"ALREADY EXISTS: {filename} ({file_size:.1f}MB) - SKIPPING")
            skip_count += 1
            continue

        try:
            # Run fast transparent workflow
            start_time = time.time()

            cmd = [
                sys.executable,
                "workflow_fast_transparent.py",
                "--input", str(input_file),
                "--output", str(output_dir)
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout

            end_time = time.time()
            duration = end_time - start_time
            total_time += duration

            if result.returncode == 0:
                if output_file.exists():
                    file_size = output_file.stat().st_size / (1024 * 1024)  # MB
                    print(
                        f"SUCCESS: {filename} ({file_size:.1f}MB, {duration:.1f}s)")
                    success_count += 1
                else:
                    print(f"OUTPUT NOT FOUND: {filename}")
                    error_count += 1
            else:
                print(f"PROCESS ERROR: {filename}")
                # First 200 chars only
                print(f"   Error: {result.stderr[:200]}...")
                error_count += 1

        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: {filename} (>5min)")
            timeout_count += 1
        except Exception as e:
            print(f"EXCEPTION: {filename}: {e}")
            error_count += 1

        # Progress update every 10 files
        if i % 10 == 0:
            elapsed = time.time() - start_total
            avg_time = elapsed / i
            remaining = (len(existing_files) - i) * avg_time
            print(f"\n--- PROGRESS UPDATE ---")
            print(
                f"Processed: {i}/{len(existing_files)} ({i/len(existing_files)*100:.1f}%)")
            print(
                f"Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}, Timeouts: {timeout_count}")
            print(
                f"Avg Time: {avg_time:.1f}s/file, ETA: {remaining/60:.1f} minutes")
            print("=" * 50)

    end_total = time.time()
    total_duration = end_total - start_total

    # Final summary
    print("\n" + "=" * 80)
    print("TARGETED BATCH PROCESSING COMPLETE")
    print(f"Success: {success_count}/{len(existing_files)} files")
    print(
        f"Skipped: {skip_count}/{len(existing_files)} files (already processed)")
    print(f"Errors: {error_count}/{len(existing_files)} files")
    print(f"Timeouts: {timeout_count}/{len(existing_files)} files")
    print(
        f"Success Rate: {success_count/(len(existing_files)-skip_count)*100:.1f}% (excluding skipped)")
    print(f"Total Time: {total_duration/60:.1f} minutes")
    if len(existing_files) > skip_count:
        print(
            f"Average: {total_duration/(len(existing_files)-skip_count):.1f}s per new file")
    print("All successful files have 100% TRANSPARENT BACKGROUNDS!")
    print("=" * 80)

    return success_count, error_count, timeout_count, skip_count


if __name__ == "__main__":
    try:
        success, errors, timeouts, skipped = run_targeted_batch_processor()

        if success > 0:
            print(
                f"\nTARGETED SUCCESS! {success} files processed successfully!")
            print("Check output folder for fast transparent results!")

        if skipped > 0:
            print(f"\n{skipped} files were already processed previously!")

        if errors > 0 or timeouts > 0:
            print(f"\n{errors + timeouts} files had issues - check logs above")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
