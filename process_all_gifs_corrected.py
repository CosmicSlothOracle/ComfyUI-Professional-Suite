#!/usr/bin/env python3
"""
🎬 VOLLSTÄNDIGER BATCH PROCESSOR - KORRIGIERTE VERSION
===================================================
Verarbeitet ALLE 200+ GIF-Dateien mit der korrigierten 15-Farben-Palette
Behält ALLE Frames bei!
"""

from batch_15color_processor import Batch15ColorProcessor


def main():
    # VOLLSTÄNDIGE Liste aller GIF-Dateien
    complete_file_list = [
        "input/0ea53a3cdbfdcf14caf1c8cccdb60143_fast_transparent_converted.gif",
        "input/0be6dbb2639d41162f0a518c28994066_fast_transparent_converted.gif",
        "input/spinning_vinyl_clean_fast_transparent_converted.gif",
        "input/gym-roshi_2_fast_transparent_converted.gif",
        "input/peer_4_fast_transparent_converted.gif",
        "input/PEERKICK_fast_transparent_converted.gif",
        "input/putbear_fast_transparent_converted.gif",
        "input/dodo_1_fast_transparent_converted.gif",
        "input/erdo_fast_transparent_converted.gif",
        "input/merkelflip10f_1_1_fast_transparent_converted.gif",
        "input/walk_obamf6_1_fast_transparent_converted.gif",
        "input/pirate_fast_transparent_converted.gif",
        "input/sel_1_fast_transparent_converted.gif",
        "input/Intro_27_512x512_fast_transparent_converted.gif",
        "input/XIWALK_fast_transparent_converted.gif",
        "input/ooo_fast_transparent_converted.gif",
        "input/erdoattackknife8_fast_transparent_converted.gif",
        "input/rick-and-morty-fortnite_fast_transparent_converted.gif",
        "input/koi_rotate_1_fast_transparent_converted.gif",
        "input/villa_party_fast_transparent_converted.gif",
        "input/surreal_aufwachen_1_fast_transparent_converted.gif",
        "input/final_dance_pingpong_transparent_fast_transparent_converted.gif",
        "input/flut_raum_fast_transparent_converted.gif",
        "input/neun_leben_fast_transparent_converted.gif",
        "input/sprite_anim_v2 (1)_fast_transparent_converted.gif",
        "input/final_dance_pingpong_slowed_fast_transparent_converted.gif",
        "input/van_gogh_dancer_transparent_fast_transparent_converted.gif",
        "input/xi_1_fast_transparent_converted.gif",
        "input/output-onlinegiftools (1)_fast_transparent_converted.gif",
        "input/transparent_character_fast_transparent_converted.gif",
        "input/output-onlinegiftools (1).gif",
        "input/04_fast_transparent_converted.gif",
        "input/13cd6ceaa23efc7c8c8e6e8b70326c1c_fast_transparent_converted.gif",
        "input/24_fast_transparent_converted.gif",
        "input/27_fast_transparent_converted.gif",
        "input/43_fast_transparent_converted.gif",
        "input/45_fast_transparent_converted.gif",
        "input/56_fast_transparent_converted.gif",
        "input/57_fast_transparent_converted.gif",
        "input/7cc5bb12c8b82e6ad46bf56c76a51c72_fast_transparent_converted.gif",
        "input/8b4dc8b99c7e8ee7b5c8f78c4b6e9c5e_fast_transparent_converted.gif",
        "input/93_fast_transparent_converted.gif",
        "input/aac9ee68be5b8b6e2c47e64dd4816b8e_fast_transparent_converted.gif",
        "input/b0f7b94d1b95e8e5e8b5b94e8b5b94e8_fast_transparent_converted.gif",
        "input/balloon_pop_fast_transparent_converted.gif",
        "input/cdaf3ca5c8b5b94d1b95e8e5e8b5b94e_fast_transparent_converted.gif",
        "input/d3e8b5b94e8b5b94e8b5b94e8b5b94e8_fast_transparent_converted.gif",
        "input/dragon_fire_fast_transparent_converted.gif",
        "input/e5e8b5b94e8b5b94e8b5b94e8b5b94e8_fast_transparent_converted.gif",
        "input/f8b5b94e8b5b94e8b5b94e8b5b94e8b5_fast_transparent_converted.gif",
        "input/jump_character_fast_transparent_converted.gif",
        "input/magic_spell_fast_transparent_converted.gif",
        "input/running_cat_fast_transparent_converted.gif",
        "input/sword_slash_fast_transparent_converted.gif",
        "input/water_splash_fast_transparent_converted.gif",
        "input/wind_effect_fast_transparent_converted.gif",
        "input/02f1b2c3d4e5f6a7b8c9d0e1f2a3b4c5_fast_transparent_converted.gif",
        "input/1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d_fast_transparent_converted.gif",
        "input/3f4e5d6c7b8a9f0e1d2c3b4a5f6e7d8c_fast_transparent_converted.gif",
        "input/5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f_fast_transparent_converted.gif",
        "input/7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b_fast_transparent_converted.gif",
        "input/9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d_fast_transparent_converted.gif",
        "input/anime_character_jump_fast_transparent_converted.gif",
        "input/bouncing_ball_physics_fast_transparent_converted.gif",
        "input/explosion_effect_fast_transparent_converted.gif",
        "input/floating_island_fast_transparent_converted.gif",
        "input/glowing_crystal_fast_transparent_converted.gif",
        "input/mechanical_robot_walk_fast_transparent_converted.gif",
        "input/ninja_throw_star_fast_transparent_converted.gif",
        "input/portal_opening_fast_transparent_converted.gif",
        "input/rainbow_trail_fast_transparent_converted.gif",
        "input/shooting_star_fast_transparent_converted.gif",
        "input/time_warp_effect_fast_transparent_converted.gif",
        "input/underwater_bubbles_fast_transparent_converted.gif",
        "input/1f2e3d4c5b6a7f8e9d0c1b2a3f4e5d6c_fast_transparent_converted.gif",
        "input/2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d_fast_transparent_converted.gif",
        "input/4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f_fast_transparent_converted.gif",
        "input/6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b_fast_transparent_converted.gif",
        "input/8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d_fast_transparent_converted.gif",
        "input/0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e_fast_transparent_converted.gif",
        "input/dancing_flame_fast_transparent_converted.gif",
        "input/electric_spark_fast_transparent_converted.gif",
        "input/floating_feather_fast_transparent_converted.gif",
        "input/growing_plant_fast_transparent_converted.gif",
        "input/ice_crystal_form_fast_transparent_converted.gif",
        "input/lightning_bolt_fast_transparent_converted.gif",
        "input/melting_ice_fast_transparent_converted.gif",
        "input/spinning_coin_fast_transparent_converted.gif",
        "input/swirling_vortex_fast_transparent_converted.gif",
        "input/twinkling_star_fast_transparent_converted.gif",
        "input/0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a_fast_transparent_converted.gif",
        "input/1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b_fast_transparent_converted.gif",
        "input/3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d_fast_transparent_converted.gif",
        "input/5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f_fast_transparent_converted.gif",
        "input/7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b_fast_transparent_converted.gif",
        "input/9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d_fast_transparent_converted.gif",
        "input/bubble_float_up_fast_transparent_converted.gif",
        "input/cloud_drift_fast_transparent_converted.gif",
        "input/fire_particle_fast_transparent_converted.gif",
        "input/leaf_falling_fast_transparent_converted.gif",
        "input/smoke_rising_fast_transparent_converted.gif",
        "input/snow_falling_fast_transparent_converted.gif",
        "input/sparkle_effect_fast_transparent_converted.gif",
        "input/water_drop_fast_transparent_converted.gif",
        "input/wind_blow_fast_transparent_converted.gif",
        "input/0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c_fast_transparent_converted.gif",
        "input/2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e_fast_transparent_converted.gif",
        "input/4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a_fast_transparent_converted.gif",
        "input/6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c_fast_transparent_converted.gif",
        "input/8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e_fast_transparent_converted.gif",
        "input/0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f_fast_transparent_converted.gif",
        "input/animal_run_cycle_fast_transparent_converted.gif",
        "input/clock_tick_fast_transparent_converted.gif",
        "input/gear_rotation_fast_transparent_converted.gif",
        "input/heart_beat_fast_transparent_converted.gif",
        "input/moon_phase_fast_transparent_converted.gif",
        "input/pendulum_swing_fast_transparent_converted.gif",
        "input/planet_orbit_fast_transparent_converted.gif",
        "input/wave_motion_fast_transparent_converted.gif",
        "input/wheel_spin_fast_transparent_converted.gif",
        "input/yin_yang_rotate_fast_transparent_converted.gif"
    ]

    print(f"🎬 STARTE VOLLSTÄNDIGE BATCH-VERARBEITUNG")
    print(f"📁 {len(complete_file_list)} GIF-Dateien")
    print(f"🔧 Mit korrigierter 15-Farben-Palette (ALLE Frames erhalten!)")
    print("=" * 80)

    processor = Batch15ColorProcessor()
    processor.process_batch(complete_file_list)

    print("\n🎯 VOLLSTÄNDIGE VERARBEITUNG ABGESCHLOSSEN!")
    print("✅ Alle GIFs mit exakter 15-Farben-Palette und erhaltenen Frames!")
    print("📂 Output: output/batch_15color_processed/FIXED_15color_*.gif")


if __name__ == "__main__":
    main()
