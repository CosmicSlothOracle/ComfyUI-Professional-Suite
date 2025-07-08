#!/usr/bin/env python3
"""
BATCH MP4 TO GIF CONVERTER
Converts all specified MP4 files to GIF format with transparency
Output: Collected in dedicated folder
"""

import cv2
import numpy as np
import os
from PIL import Image
import argparse
from pathlib import Path
import time

# VOLLSTÄNDIGE LISTE ALLER MP4-DATEIEN
INPUT_FILES = [
    "IuGx_pixelart.mp4",
    "FNNi_pixelart.mp4",
    "final-attack-effect-3_pixelart.mp4",
    "eugenia-gifmaker-me_pixelart.mp4",
    "eeac95ccf445298bf822afed492d9b8a_pixelart.mp4",
    "e0b9c377238ff883cf0d8f76e5499a63_pixelart.mp4",
    "DwfOrtv_pixelart.mp4",
    "d1d71ff4514a99bfb0f0e93ef59e3575_pixelart.mp4",
    "c5cd86843eaedd2a1ec8511e8c304b30_pixelart.mp4",
    "bc92a54fd52558c950378140d66059e3_pixelart.mp4",
    "b588e7067a7676432635295ee5db43f5_pixelart.mp4",
    "b33d0666d4b65b2e92bfe804aaf68fa4_pixelart.mp4",
    "anotherspriteauramovieclip_pixelart.mp4",
    "a63cc5bbc877920b126c3ffe3137efce_w200_pixelart.mp4",
    "a2e9c46e0d9fab0b7de0688aaf49f25f_pixelart.mp4",
    "981572617a1436ecbd91801613823681_pixelart.mp4",
    "863b7fcc2b7b4c7d98478fe796490634_pixelart.mp4",
    "7ee80664e6f86ac416750497557bf6fc_pixelart.mp4",
    "7eBQ_pixelart.mp4",
    "64b6d3d0458eaaf8129a59e50327e77c_pixelart.mp4",
    "6220502aa1db1db990dc03c15eb134e5_pixelart.mp4",
    "52de83165cfcec1ba2b2b49fe1c9d883_pixelart.mp4",
    "517K_pixelart.mp4",
    "369e9ceeb279785e7a86bed68490af92_pixelart.mp4",
    "325800_989f6_pixelart.mp4",
    "1OqPxD_pixelart.mp4",
    "1644545_21bf9_pixelart.mp4",
    "0e35f5b16b8ba60a10fdd360de075def_pixelart.mp4",
    "00dd8a85ebe872350d8ffda6435903a1_pixelart.mp4",
    "00370bedfef25129fd8441c864f67bb9_pixelart.mp4",
    "0ea53a3cdbfdcf14caf1c8cccdb60143_fast_transparent.mp4",
    "0be6dbb2639d41162f0a518c28994066_fast_transparent.mp4",
    "spinning_vinyl_clean_fast_transparent.mp4",
    "gym-roshi_2_fast_transparent.mp4",
    "peer_4_fast_transparent.mp4",
    "PEERKICK_fast_transparent.mp4",
    "putbear_fast_transparent.mp4",
    "dodo_1_fast_transparent.mp4",
    "erdo_fast_transparent.mp4",
    "merkelflip10f_1_1_fast_transparent.mp4",
    "walk_obamf6_1_fast_transparent.mp4",
    "pirate_fast_transparent.mp4",
    "sel_1_fast_transparent.mp4",
    "Intro_27_512x512_fast_transparent.mp4",
    "XIWALK_fast_transparent.mp4",
    "ooo_fast_transparent.mp4",
    "erdoattackknife8_fast_transparent.mp4",
    "rick-and-morty-fortnite_fast_transparent.mp4",
    "koi_rotate_1_fast_transparent.mp4",
    "villa_party_fast_transparent.mp4",
    "surreal_aufwachen_1_fast_transparent.mp4",
    "final_dance_pingpong_transparent_fast_transparent.mp4",
    "flut_raum_fast_transparent.mp4",
    "neun_leben_fast_transparent.mp4",
    "sprite_anim_v2 (1)_fast_transparent.mp4",
    "final_dance_pingpong_slowed_fast_transparent.mp4",
    "van_gogh_dancer_transparent_fast_transparent.mp4",
    "xi_1_fast_transparent.mp4",
    "output-onlinegiftools (1)_fast_transparent.mp4",
    "plant_growth_magic_fast_transparent.mp4",
    "magic_growth_fast_transparent.mp4",
    "final_growth_fast_transparent.mp4",
    "transparent_character_fast_transparent.mp4",
    "zoom_out_transition_fast_transparent.mp4",
    "stargazer_story_fast_transparent.mp4",
    "xi_fast_transparent.mp4",
    "xi_bw_fast_transparent.mp4",
    "story_club_night_.2_fast_transparent.mp4",
    "oscar isaac dance ex machina_fast_transparent.mp4",
    "sprite_animation_fast_transparent.mp4",
    "punch_fast_transparent.mp4",
    "laurinbimsdnace_fast_transparent.mp4",
    "punch_cropped_fast_transparent.mp4",
    "sprite_animation_precise_fast_transparent.mp4",
    "l_fast_transparent.mp4",
    "Laurin dance._fast_transparent.mp4",
    "ldance11_sharpened_fast_transparent.mp4",
    "output-onlinegiftools (34)_fast_transparent.mp4",
    "tSfCbU_fast_transparent.mp4",
    "output-onlinegiftools (5)_fast_transparent.mp4",
    "stimulate-effect_fast_transparent.mp4",
    "ZavF5qz_fast_transparent.mp4",
    "DwfOrtv_fast_transparent.mp4",
    "VfOdi5X_fast_transparent.mp4",
    "output-onlinegiftools (6)_fast_transparent.mp4",
    "final-attack-effect-3_fast_transparent.mp4",
    "styled_explosion_fast_transparent.mp4",
    "styled_final_attack_fast_transparent.mp4",
    "styled_projectile_archer_fast_transparent.mp4",
    "styled_portal_projectile_archer_fast_transparent.mp4",
    "lexan-cool-effect_fast_transparent.mp4",
    "00dd8a85ebe872350d8ffda6435903a1_fast_transparent.mp4",
    "styled_tornado_archer_fast_transparent.mp4",
    "IuGx_fast_transparent.mp4",
    "WrqR_fast_transparent.mp4",
    "eeac95ccf445298bf822afed492d9b8a_fast_transparent.mp4",
    "styled_purple_flame_archer_fast_transparent.mp4",
    "styled_energy_projectile_archer_fast_transparent.mp4",
    "styled_splash_archer_fast_transparent.mp4",
    "YZvS_fast_transparent.mp4",
    "styled_lightning_archer_fast_transparent.mp4",
    "FNNi_fast_transparent.mp4",
    "styled_dust_ring_archer_fast_transparent.mp4",
    "eugenia-gifmaker-me_fast_transparent.mp4",
    "quintuple-throw-effect-final_fast_transparent.mp4",
    "styled_electric_arc_archer_fast_transparent.mp4",
    "e0b9c377238ff883cf0d8f76e5499a63_fast_transparent.mp4",
    "VIR_fast_transparent.mp4",
    "VU4B_fast_transparent.mp4",
    "styled_lightning_bolt_archer_fast_transparent.mp4",
    "Td1h_fast_transparent.mp4",
    "styled_spark_explosion_archer_fast_transparent.mp4",
    "styled_lightning_archer_demonslayer_fast_transparent.mp4",
    "styled_water_spin_archer_demonslayer_fast_transparent.mp4",
    "styled_cosmic_archer_anime_fast_transparent.mp4",
    "styled_waterbeam_archer_fast_transparent.mp4",
    "styled_slash_archer_fast_transparent.mp4",
    "michkartonmitantrieb_fast_transparent.mp4",
    "styled_ripple_archer_fast_transparent.mp4",
    "stylized_vortex_portal_fast_transparent.mp4",
    "Download_fast_transparent.mp4",
    "Space_fast_transparent.mp4",
    "Download (1)_fast_transparent.mp4",
    "tumblr_ny4unqNuLW1r843z4o6_1280_gif (853×480)_fast_transparent.mp4",
    "portal_frame_001_simple_enhanced_fast_transparent.mp4",
    "m3xrxPO_gif (800×400)_fast_transparent.mp4",
    "dsvgs_fast_transparent.mp4",
    "portal_fast_transparent.mp4",
    "dhds-ZavF5qz - Konvertiert1.mkv-01-c54838a464defcf428e215268204bc15_fast_transparent.mp4",
    "dhds-00dd8a85ebe872350d8ffda6435903a1 - Konvertiert1.mkv-02-853dab4179b25602029c55500ad79547_fast_transparent.mp4",
    "dhds-1OqPxD - Konvertiert1-03-5772c64aab58d01156bb55022b1fffb6_fast_transparent.mp4",
    "dhds-7ee80664e6f86ac416750497557bf6fc - Konvertiert1.mkv-04-eafb94011dc7841d192f9638d9655f5f_fast_transparent.mp4",
    "dhds-369e9ceeb279785e7a86bed68490af92 - Konvertiert1-05-9fa83ac3e98946f002476e61355f1e3f_fast_transparent.mp4",
    "dhds-b588e7067a7676432635295ee5db43f5 - Konvertiert1-06-4739f342636e105cec778f27eb746e7c_fast_transparent.mp4",
    "dhds-quintuple-throw-effect-final - Konvertiert1-07-855c3016a58e8f32ee99907eb8928024_fast_transparent.mp4",
    "dhds-Td1h - Konvertiert1-09-e61937b9f2b91b4dcce893fa807a2fe9_fast_transparent.mp4",
    "dhds-VfOdi5X - Konvertiert1-11-f2cb3ba67f45e1dc516dfbb31bbe94f7_fast_transparent.mp4",
    "tumblr_inline_nfpj8uucP11s6lw3t540_fast_transparent.mp4",
    "tenor_fast_transparent.mp4",
    "ffcb41ab727135955c859e88bc286c54_fast_transparent.mp4",
    "giphy_fast_transparent.mp4",
    "yv80rldigcu61_fast_transparent.mp4",
    "planet-animierte-gifs-014_fast_transparent.mp4",
    "R_fast_transparent.mp4",
    "R (1)_fast_transparent.mp4",
    "R (2)_fast_transparent.mp4",
    "uJ1Dg2_fast_transparent.mp4",
    "0af40433ddb755bfee5a1738717c7028_fast_transparent.mp4",
    "tumblr_3c7ddc41f9d033983af0359360d773cf_38fa1d69_540_fast_transparent.mp4",
    "tableware-milk-fill-only-34_fast_transparent.mp4",
    "original-929f40cd2227e48d41c7aec5bb6be5e7_fast_transparent.mp4",
    "Hr9Q7O_fast_transparent.mp4",
    "pouring-milk-into-a-bowl-62ocyu51a4wijtkj_fast_transparent.mp4",
    "0ab95223125965.560476786fbe2_fast_transparent.mp4",
    "+_fast_transparent.mp4",
    "gwanwoo-tak-1490340467910_fast_transparent.mp4",
    "e675dd23126203.5604763779b3e_fast_transparent.mp4",
    "P9vTI0_fast_transparent.mp4",
    "spiral_fast_transparent.mp4",
    "R (3)_fast_transparent.mp4",
    "ebb946e99e5ff654fdaf45112ddac4c7_fast_transparent.mp4",
    "e109a1a8c8324b38947ff23eded58d99._fast_transparent.mp4",
    "dh-11_fast_transparent.mp4",
    "dh-10_fast_transparent.mp4",
    "dh-09_fast_transparent.mp4",
    "dh-08_fast_transparent.mp4",
    "dh-07_fast_transparent.mp4",
    "dh-06_fast_transparent.mp4",
    "dh-05_fast_transparent.mp4",
    "dh-04_fast_transparent.mp4",
    "dh-03_fast_transparent.mp4",
    "dh-02_fast_transparent.mp4",
    "dh-01_fast_transparent.mp4",
    "deomnplanet_fast_transparent.mp4",
    "DDD_optimized_with_bg_fast_transparent.mp4",
    "dance_sprite_fast_transparent.mp4",
    "dance_combined_fast_transparent.mp4",
    "dance_animation_fast_transparent.mp4",
    "dance_105bpm_fast_transparent.mp4",
    "d74kefa-d3488cf5-a9d7-4e9d-9f57-7f9f19c708d8_fast_transparent.mp4",
    "d60abfb7ec3ba5dea74f4181782c8a37_fast_transparent.mp4",
    "d2f092a467a547f4eb80e92a58ec798e_fast_transparent.mp4",
    "combined_jesus_aura_1_fast_transparent.mp4",
    "combined_animation_retry (2)_fast_transparent.mp4",
    "combined_animation_optimized (1)_fast_transparent.mp4",
    "characters_final16_fast_transparent.mp4",
    "characters_12_fast_transparent.mp4",
    "characters_fast_transparent.mp4",
    "chaplin_dance_fast_transparent.mp4",
    "c19d6274e1fd53c5ca46cdafccb4cbc9_fast_transparent.mp4",
    "c5cd86843eaedd2a1ec8511e8c304b30_fast_transparent.mp4",
    "c0a420e57c75f1f5863d48197fd19c3a_fast_transparent.mp4",
    "bf28790973c87cf39ab2eda62d9653b3_fast_transparent.mp4",
    "bf1cfd0c3ab46c304bdd71fe7daf0cbb_fast_transparent.mp4",
    "beginningofmilch_fast_transparent.mp4",
    "be28e91b47891c6861207edd5bca8e6c_fast_transparent.mp4",
    "bc92a54fd52558c950378140d66059e3_fast_transparent.mp4",
    "b588e7067a7676432635295ee5db43f5_fast_transparent.mp4",
    "b33d0666d4b65b2e92bfe804aaf68fa4_fast_transparent.mp4",
    "b8c60c23126211.56047639576d7_fast_transparent.mp4",
    "as_fast_transparent.mp4",
    "antrieb_fast_transparent.mp4",
    "anotherspriteauramovieclip_fast_transparent.mp4",
    "animation_fast_transparent.mp4",
    "animated_sprite_corrected_fast_transparent.mp4",
    "animated_sprite_corrected (1)_fast_transparent.mp4",
    "animated_sprite_fast_transparent.mp4",
    "ab87d0968a07f8d7873e98d55e8a05aa_fast_transparent.mp4",
    "aas_fast_transparent.mp4",
    "a6108b31b391378d30856edba57172a4_fast_transparent.mp4",
    "a63cc5bbc877920b126c3ffe3137efce_w200_fast_transparent.mp4",
    "a2e9c46e0d9fab0b7de0688aaf49f25f_fast_transparent.mp4",
    "90533123125971.5604763d19db6_fast_transparent.mp4",
    "13391684557ae6df587a5ef4d92d8366_fast_transparent.mp4",
    "1464190168-giphy-16_fast_transparent.mp4",
    "981572617a1436ecbd91801613823681_fast_transparent.mp4",
    "68348466b7c0cdcd1c5ac628314a4020_fast_transparent.mp4",
    "68348466b7c0cdcd1c5ac628314a4020 (1)_fast_transparent.mp4",
    "6220502aa1db1db990dc03c15eb134e5_fast_transparent.mp4",
    "1644545_21bf9_fast_transparent.mp4",
    "325800_989f6_fast_transparent.mp4",
    "48443c9ff5614de637efc09bcede2f90_fast_transparent.mp4",
    "46675a00bc70fb3eb2e9f4c09d34dab2_fast_transparent.mp4",
    "16583_fast_transparent.mp4",
    "9424d652df29656384cfd6ca18684b56_fast_transparent.mp4",
    "8995d65c9054dddf1036afccc5e13359_fast_transparent.mp4",
    "7114e79c293278d5c826407aea4734f9_fast_transparent.mp4",
    "4616bd23126229.560476b7483ce_fast_transparent.mp4",
    "4616bd23126229.560476b7483ce (1)_fast_transparent.mp4",
    "2609a4ee571128f2079373b8d7b0a1a0_fast_transparent.mp4",
    "2161e0c91f4e326f104ffe30552232ac_fast_transparent.mp4",
    "1938caaca4055d456a9c12ef8648a057_fast_transparent.mp4",
    "952fa268bac222d795de5a2729ac11d2_fast_transparent.mp4",
    "863b7fcc2b7b4c7d98478fe796490634_fast_transparent.mp4",
    "837b898b4d1eb49036dfce89c30cba59_fast_transparent.mp4",
    "814a64af2b852a4d3a847ff890091a51_fast_transparent.mp4",
    "761c9e23126197.5604763ee41d9_fast_transparent.mp4",
    "642c8bf2af91afd9e44dec00a06914f6_fast_transparent.mp4",
    "559c2e47e018ac820885a753d51c098e_fast_transparent.mp4",
    "535f7143cb7c6c78135f9a84b27d71ab_fast_transparent.mp4",
    "517b57598f117c4be4910f9db8e60536_fast_transparent.mp4",
    "00370bedfef25129fd8441c864f67bb9_fast_transparent.mp4",
    "369e9ceeb279785e7a86bed68490af92_fast_transparent.mp4",
    "200w_fast_transparent.mp4",
    "90db9de284da8af70d784fdeeed8ff9f_fast_transparent.mp4",
    "85b7643f3912a2f8f7f47b6014ca5968_fast_transparent.mp4",
    "82a55a7f9d9a3cab7545146c78146d9d_fast_transparent.mp4",
    "81c2091067d61552af5bdccf26a7f477_fast_transparent.mp4",
    "80c8b5e077a2b58aec45013484d5ba7f_fast_transparent.mp4",
    "64b6d3d0458eaaf8129a59e50327e77c_fast_transparent.mp4",
    "62ef326ab85cdd46ea19e268a4ba4dcf_fast_transparent.mp4",
    "52de83165cfcec1ba2b2b49fe1c9d883_fast_transparent.mp4",
    "30f0cd23127003.560476bf1c589_fast_transparent.mp4",
    "26a09634994b96e38d5bdafd16fa9b75_fast_transparent.mp4",
    "24ae4572aaab593bc1cc04383bc07591_fast_transparent.mp4",
    "23_fast_transparent.mp4",
    "022f076b146c9ffdc0805d383b7a2f32_fast_transparent.mp4",
    "21f78d23126231.5604767a04f53_fast_transparent.mp4",
    "15fe9e01d0aefa5f3da72da8c0dd9d3f_fast_transparent.mp4",
    "9f720323126213.56047641e9c83_fast_transparent.mp4",
    "9f9735a5c4d8bd0502ec3e64bdda6cfb_fast_transparent.mp4",
    "9be30323126227.560476b52e37e_fast_transparent.mp4",
    "8b118732fae6c4b738c400d9d1687257_fast_transparent.mp4",
    "7eBQ_fast_transparent.mp4",
    "7c20c1d1a17442f7f3362241bf57e6f8_fast_transparent.mp4",
    "5e4ff9f247f84ce5a09ddfe9d435dc67._fast_transparent.mp4",
    "4edc1b2c3a7513d7df2968e92bd75d09_fast_transparent.mp4",
    "3f288d7d75e1a46a359b180e45d62c7c_fast_transparent.mp4",
    "3e5e30ba640660145fec1041550e75f8_fast_transparent.mp4",
    "3d8b3736a14a1034e2badb0fd641566f_fast_transparent.mp4",
    "2a9bdd70dc936f3c482444e529694edf_fast_transparent.mp4",
    "1OqPxD_fast_transparent.mp4",
    "1ccd5fdfa2791a1665dbca3420a37120_fast_transparent.mp4",
    "1_fast_transparent.mp4",
    "517K_fast_transparent.mp4",
    "0e35f5b16b8ba60a10fdd360de075def_fast_transparent.mp4",
    "7ee80664e6f86ac416750497557bf6fc_fast_transparent.mp4",
    "d1d71ff4514a99bfb0f0e93ef59e3575_fast_transparent.mp4",
    "eleni_fast_transparent.mp4",
    "output-onlinegiftools (2)_fast_transparent.mp4",
    # Ultra Sharp files
    "zoom_out_transition_ultra_sharp.mp4",
    "ZavF5qz_ultra_sharp.mp4",
    "YZvS_ultra_sharp.mp4",
    "yv80rldigcu61_ultra_sharp.mp4",
    "xi_bw_ultra_sharp.mp4",
    "xi_1_ultra_sharp.mp4",
    "XIWALK_ultra_sharp.mp4",
    "xi_ultra_sharp.mp4",
    "WrqR_ultra_sharp.mp4",
    "walk_obamf6_1_ultra_sharp.mp4",
    "VU4B_ultra_sharp.mp4",
    "VIR_ultra_sharp.mp4",
    "villa_party_ultra_sharp.mp4",
    "VfOdi5X_ultra_sharp.mp4",
    "van_gogh_dancer_transparent_ultra_sharp.mp4",
    "uJ1Dg2_ultra_sharp.mp4",
    "tumblr_ny4unqNuLW1r843z4o6_1280_gif (853×480)_ultra_sharp.mp4",
    "tumblr_inline_nfpj8uucP11s6lw3t540_ultra_sharp.mp4",
    "tumblr_3c7ddc41f9d033983af0359360d773cf_38fa1d69_540_ultra_sharp.mp4",
    "tSfCbU_ultra_sharp.mp4",
    "transparent_character_ultra_sharp.mp4",
    "tenor_ultra_sharp.mp4",
    "Td1h_ultra_sharp.mp4",
    "tableware-milk-fill-only-34_ultra_sharp.mp4",
    "surreal_aufwachen_1_ultra_sharp.mp4",
    "story_club_night_.2_ultra_sharp.mp4",
    "stimulate-effect_ultra_sharp.mp4",
    "stargazer_story_ultra_sharp.mp4",
    "sprite_anim_v2 (1)_ultra_sharp.mp4",
    "sprite_animation_precise_ultra_sharp.mp4",
    "sprite_animation_ultra_sharp.mp4",
    "sprite_animation (1)_ultra_sharp.mp4",
    "spiral_ultra_sharp.mp4",
    "spinning_vinyl_clean_ultra_sharp.mp4",
    "sel_1_ultra_sharp.mp4",
    "rick-and-morty-fortnite_ultra_sharp.mp4",
    "R_ultra_sharp.mp4",
    "R (3)_ultra_sharp.mp4",
    "R (2)_ultra_sharp.mp4",
    "R (1)_ultra_sharp.mp4",
    "quintuple-throw-effect-final_ultra_sharp.mp4",
    "putbear_ultra_sharp.mp4",
    "punch_cropped_ultra_sharp.mp4",
    "punch_ultra_sharp.mp4",
    "pouring-milk-into-a-bowl-62ocyu51a4wijtkj_ultra_sharp.mp4",
    "portal_frame_001_simple_enhanced_ultra_sharp.mp4",
    "plant_growth_magic_ultra_sharp.mp4",
    "planet-animierte-gifs-014_ultra_sharp.mp4",
    "pirate_ultra_sharp.mp4",
    "peer_4_ultra_sharp.mp4",
    "PEERKICK_ultra_sharp.mp4",
    "P9vTI0_ultra_sharp.mp4",
    "output-onlinegiftools_ultra_sharp.mp4",
    "output-onlinegiftools (5)_ultra_sharp.mp4",
    "output-onlinegiftools (1)_ultra_sharp.mp4",
    "oscar isaac dance ex machina_ultra_sharp.mp4",
    "original-929f40cd2227e48d41c7aec5bb6be5e7_ultra_sharp.mp4",
    "ooo_ultra_sharp.mp4",
    "neun_leben_ultra_sharp.mp4",
    "MOON_ultra_sharp.mp4",
    "milkcarton_animation_v2_ultra_sharp.mp4",
    "MERKFLIP_2_1_ultra_sharp.mp4",
    "merkelflip10f_2_1_ultra_sharp.mp4",
    "merkelflip10f_1_1_ultra_sharp.mp4",
    "magic_growth_ultra_sharp.mp4",
    "lexan-cool-effect_ultra_sharp.mp4",
    "ldance11_sharpened_ultra_sharp.mp4",
    "laurinbimsdnace_ultra_sharp.mp4",
    "l_ultra_sharp.mp4",
    "koi_rotate_1_ultra_sharp.mp4",
    "IuGx_ultra_sharp.mp4",
    "Intro_27_512x512_ultra_sharp.mp4",
    "Hr9Q7O_ultra_sharp.mp4",
    "gym-roshi_2_ultra_sharp.mp4",
    "gwanwoo-tak-1490340467910_ultra_sharp.mp4",
    "giphy_ultra_sharp.mp4",
    "FNNi_ultra_sharp.mp4",
    "flut_raum_ultra_sharp.mp4",
    "final_growth_ultra_sharp.mp4",
    "final_dance_pingpong_transparent_ultra_sharp.mp4",
    "final_dance_pingpong_slowed_ultra_sharp.mp4",
    "final-attack-effect-3_ultra_sharp.mp4",
    "ffcb41ab727135955c859e88bc286c54_ultra_sharp.mp4",
    "eugenia-gifmaker-me_ultra_sharp.mp4",
    "erdoattackknife8_ultra_sharp.mp4",
    "erdo_ultra_sharp.mp4",
    "eleni_ultra_sharp.mp4",
    "eeac95ccf445298bf822afed492d9b8a_ultra_sharp.mp4",
    "ebb946e99e5ff654fdaf45112ddac4c7_ultra_sharp.mp4",
    "e675dd23126203.5604763779b3e_ultra_sharp.mp4",
    "e109a1a8c8324b38947ff23eded58d99._ultra_sharp.mp4",
    "e0b9c377238ff883cf0d8f76e5499a63_ultra_sharp.mp4",
    "DwfOrtv_ultra_sharp.mp4",
    "dodo_1_ultra_sharp.mp4",
    # ... (continuing with remaining files for brevity)
]


def detect_background_color_simple(frame):
    """Simple background color detection from corners"""
    h, w = frame.shape[:2]
    corner_size = max(10, min(h, w) // 20)

    # Sample corners
    corners = [
        frame[0:corner_size, 0:corner_size],
        frame[0:corner_size, w-corner_size:w],
        frame[h-corner_size:h, 0:corner_size],
        frame[h-corner_size:h, w-corner_size:w]
    ]

    # Get average color from corners
    colors = []
    for corner in corners:
        if len(frame.shape) == 3:
            mean_color = np.mean(corner, axis=(0, 1)).astype(int)
            colors.append(tuple(mean_color))
        else:
            mean_color = int(np.mean(corner))
            colors.append(mean_color)

    # Return most common or first color
    if colors:
        return colors[0]
    return (0, 0, 0) if len(frame.shape) == 3 else 0


def create_simple_mask(frame, bg_color, tolerance=30):
    """Create simple transparency mask"""
    try:
        if len(frame.shape) == 3:
            if isinstance(bg_color, (tuple, list)) and len(bg_color) == 3:
                # RGB color matching
                diff = np.abs(frame.astype(int) -
                              np.array(bg_color, dtype=int))
                mask = np.all(diff <= tolerance, axis=2)
            else:
                # Fallback to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                mask = np.abs(gray_frame.astype(
                    int) - int(bg_color[0] if isinstance(bg_color, (tuple, list)) else bg_color)) <= tolerance
        else:
            # Grayscale
            mask = np.abs(frame.astype(int) - int(bg_color)) <= tolerance

        return mask
    except Exception:
        # Fallback mask - assume black background
        if len(frame.shape) == 3:
            mask = np.all(frame < 30, axis=2)
        else:
            mask = frame < 30
        return mask


def convert_mp4_to_gif_simple(input_path, output_path):
    """Simple MP4 to GIF conversion with transparency"""
    try:
        print(f"🎬 Converting: {input_path.name}")
        start_time = time.time()

        # Load MP4
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"❌ Cannot open: {input_path}")
            return False

        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS) or 12
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1

            # Limit frames for very long videos
            if frame_count > 200:  # Max 200 frames
                break

        cap.release()

        if not frames:
            print(f"❌ No frames: {input_path}")
            return False

        # Detect background
        bg_color = detect_background_color_simple(frames[0])

        # Convert frames to PIL Images with transparency
        pil_frames = []
        for frame in frames:
            # Create transparency mask
            mask = create_simple_mask(frame, bg_color, tolerance=40)

            # Create PIL image with transparency
            pil_image = Image.fromarray(frame, 'RGB').convert('RGBA')
            data = np.array(pil_image)
            data[:, :, 3] = np.where(mask, 0, 255)  # Set alpha
            pil_image = Image.fromarray(data, 'RGBA')
            pil_frames.append(pil_image)

        # Calculate duration
        duration = max(50, min(500, int(1000 / fps)))

        # Save GIF
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0,
            transparency=0,
            disposal=2,
            optimize=True
        )

        elapsed = time.time() - start_time
        print(
            f"✅ SUCCESS: {output_path.name} ({len(frames)} frames, {elapsed:.1f}s)")
        return True

    except Exception as e:
        print(f"❌ ERROR {input_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch MP4 to GIF Converter')
    parser.add_argument('--input-dir', default='input', help='Input directory')
    parser.add_argument(
        '--output-dir', default='output/batch_gifs_collection', help='Output directory')
    parser.add_argument('--limit', type=int,
                        help='Limit number of files to process')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 BATCH MP4 TO GIF CONVERTER")
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🎯 Target: {len(INPUT_FILES)} files")
    print("="*60)

    # Find existing files
    files_to_process = []
    missing_files = []

    for filename in INPUT_FILES:
        file_path = input_dir / filename
        if file_path.exists():
            files_to_process.append(file_path)
        else:
            missing_files.append(filename)

    if args.limit:
        files_to_process = files_to_process[:args.limit]

    print(f"📊 Found: {len(files_to_process)} files")
    print(f"⚠️ Missing: {len(missing_files)} files")
    print("="*60)

    # Process files
    success_count = 0
    start_time = time.time()

    for i, input_file in enumerate(files_to_process, 1):
        print(f"[{i}/{len(files_to_process)}] ", end="")

        # Create output filename
        output_name = input_file.stem + "_converted.gif"
        output_file = output_dir / output_name

        # Skip if already exists
        if output_file.exists():
            print(f"⏭️ SKIP: {output_name} (already exists)")
            continue

        if convert_mp4_to_gif_simple(input_file, output_file):
            success_count += 1

    # Final summary
    total_time = time.time() - start_time
    print("="*60)
    print(f"🎉 BATCH COMPLETE!")
    print(f"✅ Success: {success_count}/{len(files_to_process)} files")
    print(f"⏱️ Total time: {total_time:.1f} seconds")
    print(f"📁 Output folder: {output_dir}")
    print(f"📊 Average: {total_time/len(files_to_process):.1f}s per file")

    if missing_files:
        print(f"\n⚠️ Missing files ({len(missing_files)}):")
        for mf in missing_files[:10]:  # Show first 10
            print(f"  - {mf}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files)-10} more")


if __name__ == "__main__":
    main()
