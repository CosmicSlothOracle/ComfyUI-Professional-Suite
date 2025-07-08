#!/usr/bin/env python3
"""
Optimized batch processor for workflows 7-10 on reduced file set
"Wer nichts Großes erreichen kann, der kann kleine Dinge großartig tun!"
"""

import subprocess
import sys
from pathlib import Path
import time

# Reduced input files as specified by user
input_files = [
    "C:/Users/Public/ComfyUI-master/input/DwfOrtv.gif",
    "C:/Users/Public/ComfyUI-master/input/e0b9c377238ff883cf0d8f76e5499a63.gif",
    "C:/Users/Public/ComfyUI-master/input/e109a1a8c8324b38947ff23eded58d99..gif",
    "C:/Users/Public/ComfyUI-master/input/e675dd23126203.5604763779b3e.gif",
    "C:/Users/Public/ComfyUI-master/input/ebb946e99e5ff654fdaf45112ddac4c7.gif",
    "C:/Users/Public/ComfyUI-master/input/eeac95ccf445298bf822afed492d9b8a.gif",
    "C:/Users/Public/ComfyUI-master/input/eleni.gif",
    "C:/Users/Public/ComfyUI-master/input/erdo.gif",
    "C:/Users/Public/ComfyUI-master/input/eugenia-gifmaker-me.gif",
    "C:/Users/Public/ComfyUI-master/input/ffcb41ab727135955c859e88bc286c54.gif",
    "C:/Users/Public/ComfyUI-master/input/final_dance_pingpong_slowed.gif",
    "C:/Users/Public/ComfyUI-master/input/final-attack-effect-3.gif",
    "C:/Users/Public/ComfyUI-master/input/FNNi.gif",
    "C:/Users/Public/ComfyUI-master/input/giphy.gif",
    "C:/Users/Public/ComfyUI-master/input/gwanwoo-tak-1490340467910.gif",
    "C:/Users/Public/ComfyUI-master/input/gym-roshi_2.gif",
    "C:/Users/Public/ComfyUI-master/input/Hr9Q7O.gif",
    "C:/Users/Public/ComfyUI-master/input/Intro_27_512x512.gif",
    "C:/Users/Public/ComfyUI-master/input/IuGx.gif",
    "C:/Users/Public/ComfyUI-master/input/koi_rotate_1.gif",
    "C:/Users/Public/ComfyUI-master/input/l.gif",
    "C:/Users/Public/ComfyUI-master/input/ldance11_sharpened.gif",
    "C:/Users/Public/ComfyUI-master/input/lexan-cool-effect.gif",
    "C:/Users/Public/ComfyUI-master/input/m3xrxPO_gif (800×400).gif",
    "C:/Users/Public/ComfyUI-master/input/magic_growth.gif",
    "C:/Users/Public/ComfyUI-master/input/merkelflip10f_1_1.gif",
    "C:/Users/Public/ComfyUI-master/input/original-929f40cd2227e48d41c7aec5bb6be5e7.gif",
    "C:/Users/Public/ComfyUI-master/input/output-onlinegiftools (5).gif",
    "C:/Users/Public/ComfyUI-master/input/P9vTI0.gif",
    "C:/Users/Public/ComfyUI-master/input/peer_4.gif",
    "C:/Users/Public/ComfyUI-master/input/pirate.gif",
    "C:/Users/Public/ComfyUI-master/input/+.gif",
    "C:/Users/Public/ComfyUI-master/input/00dd8a85ebe872350d8ffda6435903a1.gif",
    "C:/Users/Public/ComfyUI-master/input/0ab95223125965.560476786fbe2.gif",
    "C:/Users/Public/ComfyUI-master/input/0af40433ddb755bfee5a1738717c7028.gif",
    "C:/Users/Public/ComfyUI-master/input/0be6dbb2639d41162f0a518c28994066.gif",
    "C:/Users/Public/ComfyUI-master/input/0e35f5b16b8ba60a10fdd360de075def.gif",
    "C:/Users/Public/ComfyUI-master/input/0ea53a3cdbfdcf14caf1c8cccdb60143.gif",
    "C:/Users/Public/ComfyUI-master/input/1.gif",
    "C:/Users/Public/ComfyUI-master/input/1ccd5fdfa2791a1665dbca3420a37120.gif",
    "C:/Users/Public/ComfyUI-master/input/2a9bdd70dc936f3c482444e529694edf.gif",
    "C:/Users/Public/ComfyUI-master/input/3d8b3736a14a1034e2badb0fd641566f.gif",
    "C:/Users/Public/ComfyUI-master/input/3e5e30ba640660145fec1041550e75f8.gif",
    "C:/Users/Public/ComfyUI-master/input/4edc1b2c3a7513d7df2968e92bd75d09.gif",
    "C:/Users/Public/ComfyUI-master/input/5e4ff9f247f84ce5a09ddfe9d435dc67..gif",
    "C:/Users/Public/ComfyUI-master/input/7eBQ.gif",
    "C:/Users/Public/ComfyUI-master/input/7ee80664e6f86ac416750497557bf6fc.gif",
    "C:/Users/Public/ComfyUI-master/input/8b118732fae6c4b738c400d9d1687257.gif",
    "C:/Users/Public/ComfyUI-master/input/9be30323126227.560476b52e37e.gif",
    "C:/Users/Public/ComfyUI-master/input/9f9735a5c4d8bd0502ec3e64bdda6cfb.gif",
    "C:/Users/Public/ComfyUI-master/input/9f720323126213.56047641e9c83.gif",
    "C:/Users/Public/ComfyUI-master/input/15fe9e01d0aefa5f3da72da8c0dd9d3f.gif",
    "C:/Users/Public/ComfyUI-master/input/21f78d23126231.5604767a04f53.gif",
    "C:/Users/Public/ComfyUI-master/input/022f076b146c9ffdc0805d383b7a2f32.gif",
    "C:/Users/Public/ComfyUI-master/input/23.gif",
    "C:/Users/Public/ComfyUI-master/input/24ae4572aaab593bc1cc04383bc07591.gif",
    "C:/Users/Public/ComfyUI-master/input/26a09634994b96e38d5bdafd16fa9b75.gif",
    "C:/Users/Public/ComfyUI-master/input/30f0cd23127003.560476bf1c589.gif",
    "C:/Users/Public/ComfyUI-master/input/52de83165cfcec1ba2b2b49fe1c9d883.gif",
    "C:/Users/Public/ComfyUI-master/input/62ef326ab85cdd46ea19e268a4ba4dcf.gif",
    "C:/Users/Public/ComfyUI-master/input/64b6d3d0458eaaf8129a59e50327e77c.gif",
    "C:/Users/Public/ComfyUI-master/input/80c8b5e077a2b58aec45013484d5ba7f.gif",
    "C:/Users/Public/ComfyUI-master/input/81c2091067d61552af5bdccf26a7f477.gif",
    "C:/Users/Public/ComfyUI-master/input/82a55a7f9d9a3cab7545146c78146d9d.gif",
    "C:/Users/Public/ComfyUI-master/input/85b7643f3912a2f8f7f47b6014ca5968.gif",
    "C:/Users/Public/ComfyUI-master/input/90db9de284da8af70d784fdeeed8ff9f.gif",
    "C:/Users/Public/ComfyUI-master/input/200w.gif",
    "C:/Users/Public/ComfyUI-master/input/369e9ceeb279785e7a86bed68490af92.gif",
    "C:/Users/Public/ComfyUI-master/input/00370bedfef25129fd8441c864f67bb9.gif",
    "C:/Users/Public/ComfyUI-master/input/517b57598f117c4be4910f9db8e60536.gif",
    "C:/Users/Public/ComfyUI-master/input/517K.gif",
    "C:/Users/Public/ComfyUI-master/input/535f7143cb7c6c78135f9a84b27d71ab.gif",
    "C:/Users/Public/ComfyUI-master/input/559c2e47e018ac820885a753d51c098e.gif",
    "C:/Users/Public/ComfyUI-master/input/642c8bf2af91afd9e44dec00a06914f6.gif",
    "C:/Users/Public/ComfyUI-master/input/761c9e23126197.5604763ee41d9.gif",
    "C:/Users/Public/ComfyUI-master/input/814a64af2b852a4d3a847ff890091a51.gif",
    "C:/Users/Public/ComfyUI-master/input/837b898b4d1eb49036dfce89c30cba59.gif",
    "C:/Users/Public/ComfyUI-master/input/863b7fcc2b7b4c7d98478fe796490634.gif",
    "C:/Users/Public/ComfyUI-master/input/952fa268bac222d795de5a2729ac11d2.gif",
    "C:/Users/Public/ComfyUI-master/input/1938caaca4055d456a9c12ef8648a057.gif",
    "C:/Users/Public/ComfyUI-master/input/2161e0c91f4e326f104ffe30552232ac.gif",
    "C:/Users/Public/ComfyUI-master/input/animated_sprite_corrected (1).gif",
    "C:/Users/Public/ComfyUI-master/input/animation.gif",
    "C:/Users/Public/ComfyUI-master/input/anotherspriteauramovieclip.gif",
    "C:/Users/Public/ComfyUI-master/input/as.gif",
    "C:/Users/Public/ComfyUI-master/input/b8c60c23126211.56047639576d7.gif",
    "C:/Users/Public/ComfyUI-master/input/b33d0666d4b65b2e92bfe804aaf68fa4.gif",
    "C:/Users/Public/ComfyUI-master/input/b588e7067a7676432635295ee5db43f5.gif",
    "C:/Users/Public/ComfyUI-master/input/bc92a54fd52558c950378140d66059e3.gif",
    "C:/Users/Public/ComfyUI-master/input/be28e91b47891c6861207edd5bca8e6c.gif",
    "C:/Users/Public/ComfyUI-master/input/bf1cfd0c3ab46c304bdd71fe7daf0cbb.gif",
    "C:/Users/Public/ComfyUI-master/input/bf28790973c87cf39ab2eda62d9653b3.gif",
    "C:/Users/Public/ComfyUI-master/input/c0a420e57c75f1f5863d48197fd19c3a.gif",
    "C:/Users/Public/ComfyUI-master/input/c5cd86843eaedd2a1ec8511e8c304b30.gif",
    "C:/Users/Public/ComfyUI-master/input/c19d6274e1fd53c5ca46cdafccb4cbc9.gif",
    "C:/Users/Public/ComfyUI-master/input/d1d71ff4514a99bfb0f0e93ef59e3575.gif",
    "C:/Users/Public/ComfyUI-master/input/d2f092a467a547f4eb80e92a58ec798e.gif",
    "C:/Users/Public/ComfyUI-master/input/d60abfb7ec3ba5dea74f4181782c8a37.gif",
    "C:/Users/Public/ComfyUI-master/input/d74kefa-d3488cf5-a9d7-4e9d-9f57-7f9f19c708d8.gif",
    "C:/Users/Public/ComfyUI-master/input/DDD_optimized_with_bg.gif",
    "C:/Users/Public/ComfyUI-master/input/deomnplanet.gif",
    "C:/Users/Public/ComfyUI-master/input/dhds-stimulate-effect - Konvertiert1.mkv-08-4cb41876ee08787fe1d16354e4f9bcd7.gif",
    "C:/Users/Public/ComfyUI-master/input/dhds-Td1h - Konvertiert1-09-e61937b9f2b91b4dcce893fa807a2fe9.gif",
    "C:/Users/Public/ComfyUI-master/input/dhds-tSfCbU - Konvertiert1-10-2d5897b18c4a3f7bb3f2df206190e01f.gif",
    "C:/Users/Public/ComfyUI-master/input/dhds-VfOdi5X - Konvertiert1-11-f2cb3ba67f45e1dc516dfbb31bbe94f7.gif"
]

# Remaining workflow scripts
workflows = [
    "workflow_07_histogram_matching.py",
    "workflow_08_local_contrast.py",
    "workflow_09_gradient_enhancement.py",
    "workflow_10_soft_sharpening.py"
]


def run_workflow(workflow_script, input_file, output_dir):
    """Run a single workflow on a single input file"""
    try:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"⚠️ File not found: {input_file}")
            return False

        # Skip .lnk and .png files
        if input_path.name.endswith(('.lnk', '.png')):
            print(f"⏭️ Skipping: {input_path.name}")
            return True

        print(f"🔄 {workflow_script} → {input_path.name}")

        cmd = [
            sys.executable, workflow_script,
            "--input", str(input_path),
            "--output", output_dir
        ]

        result = subprocess.run(cmd, capture_output=True,
                                text=True, cwd=".", timeout=600)

        if result.returncode == 0:
            print(f"✅ SUCCESS: {input_path.name}")
            return True
        else:
            print(f"❌ ERROR: {input_path.name}")
            if result.stderr:
                print(f"   Error: {result.stderr[:100]}...")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {input_file} (>10min)")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {input_file}: {str(e)[:50]}...")
        return False


def main():
    """Main execution - autonomous processing"""
    print("🚀 AUTONOMOUS BATCH PROCESSING - WORKFLOWS 7-10")
    print("💪 'Wer nichts Großes erreichen kann, der kann kleine Dinge großartig tun!'")
    print(f"📁 Processing {len(input_files)} files")
    print(f"🔧 Using {len(workflows)} workflows (7-10)")

    output_dir = "C:/Users/Public/ComfyUI-master/output"

    # Statistics
    total_tasks = len(input_files) * len(workflows)
    completed_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0

    start_time = time.time()

    # Process each workflow
    for workflow_idx, workflow in enumerate(workflows, 7):
        print(f"\n{'='*70}")
        print(f"🎯 WORKFLOW {workflow_idx}/10: {workflow}")
        print(f"{'='*70}")

        workflow_start = time.time()
        workflow_success = 0
        workflow_failed = 0
        workflow_skipped = 0

        # Process each input file
        for file_idx, input_file in enumerate(input_files, 1):
            print(f"[{file_idx}/{len(input_files)}] ", end="")

            success = run_workflow(workflow, input_file, output_dir)

            if success is True:
                completed_tasks += 1
                workflow_success += 1
            elif success is False:
                failed_tasks += 1
                workflow_failed += 1
            else:
                skipped_tasks += 1
                workflow_skipped += 1

        workflow_time = time.time() - workflow_start
        print(f"\n📊 WORKFLOW {workflow} SUMMARY:")
        print(f"   ✅ Success: {workflow_success}")
        print(f"   ❌ Failed: {workflow_failed}")
        print(f"   ⏭️ Skipped: {workflow_skipped}")
        print(f"   ⏱️ Time: {workflow_time/60:.1f} minutes")
        print(f"   🎯 Success Rate: {(workflow_success/(workflow_success+workflow_failed)*100):.1f}%" if (
            workflow_success+workflow_failed) > 0 else "   🎯 Success Rate: N/A")

    # Final statistics
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"🎉 AUTONOMOUS PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"📈 Total tasks: {total_tasks}")
    print(f"✅ Completed: {completed_tasks}")
    print(f"❌ Failed: {failed_tasks}")
    print(f"⏭️ Skipped: {skipped_tasks}")
    print(f"📊 Overall Success Rate: {(completed_tasks/(total_tasks-skipped_tasks)*100):.1f}%" if (
        total_tasks-skipped_tasks) > 0 else "📊 Overall Success Rate: N/A")
    print(f"⏱️ Total Time: {total_time/60:.1f} minutes")
    print(f"📁 Output Directory: {output_dir}")

    # Final message
    print(f"\n🌟 MISSION ACCOMPLISHED!")
    print(f"💎 'Kleine Dinge großartig gemacht!' - Check your output folder!")


if __name__ == "__main__":
    main()
