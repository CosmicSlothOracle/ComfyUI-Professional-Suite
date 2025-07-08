#!/usr/bin/env python3
"""
Batch processor for running workflows 5-10 on specified GIF files
Based on user feedback that workflow 1 (Global Palette) is the best approach
"""

import subprocess
import sys
from pathlib import Path
import time

# Input files specified by user
input_files = [
    "C:/Users/Public/ComfyUI-master/input/erdo.gif",
    "C:/Users/Public/ComfyUI-master/input/+.gif",
    "C:/Users/Public/ComfyUI-master/input/00dd8a85ebe872350d8ffda6435903a1.gif",
    "C:/Users/Public/ComfyUI-master/input/0ab95223125965.560476786fbe2.gif",
    "C:/Users/Public/ComfyUI-master/input/0af40433ddb755bfee5a1738717c7028.gif",
    "C:/Users/Public/ComfyUI-master/input/0e35f5b16b8ba60a10fdd360de075def.gif",
    "C:/Users/Public/ComfyUI-master/input/1.gif",
    "C:/Users/Public/ComfyUI-master/input/1ccd5fdfa2791a1665dbca3420a37120.gif",
    "C:/Users/Public/ComfyUI-master/input/1OqPxD.gif",
    "C:/Users/Public/ComfyUI-master/input/2a9bdd70dc936f3c482444e529694edf.gif",
    "C:/Users/Public/ComfyUI-master/input/3e5e30ba640660145fec1041550e75f8.gif",
    "C:/Users/Public/ComfyUI-master/input/3f288d7d75e1a46a359b180e45d62c7c.gif",
    "C:/Users/Public/ComfyUI-master/input/5e4ff9f247f84ce5a09ddfe9d435dc67..gif",
    "C:/Users/Public/ComfyUI-master/input/7c20c1d1a17442f7f3362241bf57e6f8.gif",
    "C:/Users/Public/ComfyUI-master/input/7eBQ.gif",
    "C:/Users/Public/ComfyUI-master/input/7ee80664e6f86ac416750497557bf6fc.gif",
    "C:/Users/Public/ComfyUI-master/input/9be30323126227.560476b52e37e.gif",
    "C:/Users/Public/ComfyUI-master/input/9f9735a5c4d8bd0502ec3e64bdda6cfb.gif",
    "C:/Users/Public/ComfyUI-master/input/9f720323126213.56047641e9c83.gif",
    "C:/Users/Public/ComfyUI-master/input/21f78d23126231.5604767a04f53.gif",
    "C:/Users/Public/ComfyUI-master/input/022f076b146c9ffdc0805d383b7a2f32.gif",
    "C:/Users/Public/ComfyUI-master/input/23.gif",
    "C:/Users/Public/ComfyUI-master/input/24ae4572aaab593bc1cc04383bc07591.gif",
    "C:/Users/Public/ComfyUI-master/input/26a09634994b96e38d5bdafd16fa9b75.gif",
    "C:/Users/Public/ComfyUI-master/input/30f0cd23127003.560476bf1c589.gif",
    "C:/Users/Public/ComfyUI-master/input/52de83165cfcec1ba2b2b49fe1c9d883.gif",
    "C:/Users/Public/ComfyUI-master/input/62ef326ab85cdd46ea19e268a4ba4dcf.gif",
    "C:/Users/Public/ComfyUI-master/input/64b6d3d0458eaaf8129a59e50327e77c.gif",
    "C:/Users/Public/ComfyUI-master/input/81c2091067d61552af5bdccf26a7f477.gif",
    "C:/Users/Public/ComfyUI-master/input/82a55a7f9d9a3cab7545146c78146d9d.gif",
    "C:/Users/Public/ComfyUI-master/input/200w.gif",
    "C:/Users/Public/ComfyUI-master/input/369e9ceeb279785e7a86bed68490af92.gif",
    "C:/Users/Public/ComfyUI-master/input/00370bedfef25129fd8441c864f67bb9.gif",
    "C:/Users/Public/ComfyUI-master/input/517b57598f117c4be4910f9db8e60536.gif",
    "C:/Users/Public/ComfyUI-master/input/517K.gif",
    "C:/Users/Public/ComfyUI-master/input/559c2e47e018ac820885a753d51c098e.gif",
    "C:/Users/Public/ComfyUI-master/input/642c8bf2af91afd9e44dec00a06914f6.gif",
    "C:/Users/Public/ComfyUI-master/input/761c9e23126197.5604763ee41d9.gif",
    "C:/Users/Public/ComfyUI-master/input/837b898b4d1eb49036dfce89c30cba59.gif",
    "C:/Users/Public/ComfyUI-master/input/863b7fcc2b7b4c7d98478fe796490634.gif",
    "C:/Users/Public/ComfyUI-master/input/1938caaca4055d456a9c12ef8648a057.gif",
    "C:/Users/Public/ComfyUI-master/input/2161e0c91f4e326f104ffe30552232ac.gif",
    "C:/Users/Public/ComfyUI-master/input/2609a4ee571128f2079373b8d7b0a1a0.gif",
    "C:/Users/Public/ComfyUI-master/input/4616bd23126229.560476b7483ce (1).gif",
    "C:/Users/Public/ComfyUI-master/input/4616bd23126229.560476b7483ce.gif",
    "C:/Users/Public/ComfyUI-master/input/8995d65c9054dddf1036afccc5e13359.gif",
    "C:/Users/Public/ComfyUI-master/input/16583.gif",
    "C:/Users/Public/ComfyUI-master/input/48443c9ff5614de637efc09bcede2f90.gif",
    "C:/Users/Public/ComfyUI-master/input/325800_989f6.gif",
    "C:/Users/Public/ComfyUI-master/input/1644545_21bf9.gif",
    "C:/Users/Public/ComfyUI-master/input/6220502aa1db1db990dc03c15eb134e5.gif",
    "C:/Users/Public/ComfyUI-master/input/a2e9c46e0d9fab0b7de0688aaf49f25f.gif",
    "C:/Users/Public/ComfyUI-master/input/a63cc5bbc877920b126c3ffe3137efce_w200.gif",
    "C:/Users/Public/ComfyUI-master/input/a6108b31b391378d30856edba57172a4.gif",
    "C:/Users/Public/ComfyUI-master/input/aas.gif",
    "C:/Users/Public/ComfyUI-master/input/ab87d0968a07f8d7873e98d55e8a05aa.gif",
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
    "C:/Users/Public/ComfyUI-master/input/dance_animation.gif",
    "C:/Users/Public/ComfyUI-master/input/dance_combined.gif",
    "C:/Users/Public/ComfyUI-master/input/dance_sprite.gif",
    "C:/Users/Public/ComfyUI-master/input/DDD_optimized_with_bg.gif",
    "C:/Users/Public/ComfyUI-master/input/deomnplanet.gif",
    "C:/Users/Public/ComfyUI-master/input/dhds-comica1750462002773 - Konvertiert.mkv-12-57c93a8e2cc1f29009054c31920f4ad8.gif",
    "C:/Users/Public/ComfyUI-master/input/dhds-stimulate-effect - Konvertiert1.mkv-08-4cb41876ee08787fe1d16354e4f9bcd7.gif",
    "C:/Users/Public/ComfyUI-master/input/dhds-Td1h - Konvertiert1-09-e61937b9f2b91b4dcce893fa807a2fe9.gif",
    "C:/Users/Public/ComfyUI-master/input/dhds-tSfCbU - Konvertiert1-10-2d5897b18c4a3f7bb3f2df206190e01f.gif",
    "C:/Users/Public/ComfyUI-master/input/dhds-VfOdi5X - Konvertiert1-11-f2cb3ba67f45e1dc516dfbb31bbe94f7.gif",
    "C:/Users/Public/ComfyUI-master/input/dodo_1.gif",
    "C:/Users/Public/ComfyUI-master/input/dsvgs.gif",
    "C:/Users/Public/ComfyUI-master/input/DwfOrtv.gif",
    "C:/Users/Public/ComfyUI-master/input/e0b9c377238ff883cf0d8f76e5499a63.gif",
    "C:/Users/Public/ComfyUI-master/input/e109a1a8c8324b38947ff23eded58d99..gif",
    "C:/Users/Public/ComfyUI-master/input/e675dd23126203.5604763779b3e.gif",
    "C:/Users/Public/ComfyUI-master/input/ebb946e99e5ff654fdaf45112ddac4c7.gif",
    "C:/Users/Public/ComfyUI-master/input/eeac95ccf445298bf822afed492d9b8a.gif",
    "C:/Users/Public/ComfyUI-master/input/eleni.gif"
]

# Workflow scripts
workflows = [
    "workflow_05_multi_scale.py",
    "workflow_06_edge_preserving.py",
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
            print(f"Warning: Input file does not exist: {input_file}")
            return False

        # Skip .lnk files
        if input_path.name.endswith('.lnk'):
            print(f"Skipping .lnk file: {input_file}")
            return True

        print(f"Running {workflow_script} on {input_path.name}")

        cmd = [
            sys.executable, workflow_script,
            "--input", str(input_path),
            "--output", output_dir
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")

        if result.returncode == 0:
            print(f"✓ Success: {workflow_script} -> {input_path.name}")
            return True
        else:
            print(f"✗ Error: {workflow_script} -> {input_path.name}")
            print(f"  Error output: {result.stderr}")
            return False

    except Exception as e:
        print(f"✗ Exception: {workflow_script} -> {input_file}: {e}")
        return False


def main():
    """Main execution function"""
    print("🚀 Starting batch processing of workflows 5-10")
    print(f"📁 Processing {len(input_files)} input files")
    print(f"🔧 Using {len(workflows)} workflows")

    output_dir = "C:/Users/Public/ComfyUI-master/output"

    # Ensure output directory exists
    Path(output_dir).mkdir(exist_ok=True)

    # Statistics
    total_tasks = len(input_files) * len(workflows)
    completed_tasks = 0
    failed_tasks = 0

    start_time = time.time()

    # Process each workflow
    for workflow_idx, workflow in enumerate(workflows, 1):
        print(f"\n{'='*60}")
        print(f"🔄 WORKFLOW {workflow_idx}/6: {workflow}")
        print(f"{'='*60}")

        workflow_start = time.time()
        workflow_success = 0
        workflow_failed = 0

        # Process each input file
        for file_idx, input_file in enumerate(input_files, 1):
            print(f"\n[{file_idx}/{len(input_files)}] ", end="")

            success = run_workflow(workflow, input_file, output_dir)

            if success:
                completed_tasks += 1
                workflow_success += 1
            else:
                failed_tasks += 1
                workflow_failed += 1

        workflow_time = time.time() - workflow_start
        print(f"\n📊 Workflow {workflow} completed:")
        print(f"   ✓ Success: {workflow_success}/{len(input_files)}")
        print(f"   ✗ Failed: {workflow_failed}/{len(input_files)}")
        print(f"   ⏱️ Time: {workflow_time:.1f}s")

    # Final statistics
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🎉 BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"📈 Total tasks: {total_tasks}")
    print(f"✅ Completed: {completed_tasks}")
    print(f"❌ Failed: {failed_tasks}")
    print(f"📊 Success rate: {(completed_tasks/total_tasks)*100:.1f}%")
    print(f"⏱️ Total time: {total_time:.1f}s")
    print(f"📁 Output directory: {output_dir}")

    # Workflow comparison based on user feedback
    print(f"\n🔍 WORKFLOW ANALYSIS:")
    print(f"Based on your feedback, Workflow 1 (Global Palette) was the best.")
    print(f"These workflows 5-10 all incorporate global palette features from Workflow 1:")
    print(f"  • Workflow 5: Multi-scale upscaling + Global Palette")
    print(f"  • Workflow 6: Edge-preserving smoothing + Global Palette")
    print(f"  • Workflow 7: Histogram matching + Global Palette")
    print(f"  • Workflow 8: Local contrast normalization + Global Palette")
    print(f"  • Workflow 9: Gradient-based enhancement + Global Palette")
    print(f"  • Workflow 10: Soft sharpening + Global Palette")


if __name__ == "__main__":
    main()
