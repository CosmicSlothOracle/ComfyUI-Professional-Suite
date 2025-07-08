#!/usr/bin/env python3
"""
ULTIMATE FUSION WORKFLOW: Combines WF1 + WF7 + WF2
Global Palette + Histogram Matching + Anti-Fractal + 100% TRANSPARENT BACKGROUND
MP4 TO GIF CONVERTER
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageSequence
import argparse
from pathlib import Path

# Liste der zu verarbeitenden MP4-Dateien
INPUT_FILES = [
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
    "final_dance_pingpong_transparent_fast_transparent.mp4"
]


def detect_background_color_advanced(frame):
    """Advanced background color detection from multiple corner regions"""
    h, w = frame.shape[:2]

    # Sample from larger corner regions for better accuracy
    corner_size = min(h, w) // 20  # 5% of smallest dimension

    corners = [
        frame[0:corner_size, 0:corner_size],  # Top-left
        frame[0:corner_size, w-corner_size:w],  # Top-right
        frame[h-corner_size:h, 0:corner_size],  # Bottom-left
        frame[h-corner_size:h, w-corner_size:w]  # Bottom-right
    ]

    # Get most common color from all corner pixels
    all_corner_pixels = []
    for corner in corners:
        if len(frame.shape) == 3:
            pixels = corner.reshape(-1, 3)
            for pixel in pixels:
                all_corner_pixels.append(tuple(pixel))
        else:
            pixels = corner.reshape(-1)
            for pixel in pixels:
                all_corner_pixels.append(pixel)

    from collections import Counter
    most_common = Counter(all_corner_pixels).most_common(1)
    return most_common[0][0] if most_common else (255, 255, 255)


def extract_global_palette(frames, n_colors=48):
    """Extract global color palette from all frames (WF1 feature)"""
    all_pixels = []
    for frame in frames:
        if len(frame.shape) == 3:
            pixels = frame.reshape(-1, 3)
        else:
            pixels = frame.reshape(-1, 1)
        all_pixels.append(pixels)

    combined_pixels = np.vstack(all_pixels)

    # Use k-means clustering to find dominant colors
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=min(n_colors, len(
        combined_pixels)), random_state=42, n_init=10)
    kmeans.fit(combined_pixels)

    return kmeans.cluster_centers_.astype(np.uint8)


def apply_global_palette(frame, palette):
    """Apply global palette to frame (WF1 feature)"""
    if len(frame.shape) == 3:
        original_shape = frame.shape
        pixels = frame.reshape(-1, 3)
    else:
        original_shape = frame.shape
        pixels = frame.reshape(-1, 1)

    # Find closest palette color for each pixel
    distances = np.sqrt(
        ((pixels[:, np.newaxis] - palette[np.newaxis, :]) ** 2).sum(axis=2))
    closest_colors = np.argmin(distances, axis=1)

    # Apply palette colors
    new_pixels = palette[closest_colors]
    return new_pixels.reshape(original_shape)


def match_histogram_channel(source, reference):
    """Match histogram for single channel (WF7 feature)"""
    # Calculate histograms
    source_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
    reference_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])

    # Calculate cumulative distribution functions
    source_cdf = source_hist.cumsum()
    reference_cdf = reference_hist.cumsum()

    # Normalize CDFs
    source_cdf = source_cdf / source_cdf[-1]
    reference_cdf = reference_cdf / reference_cdf[-1]

    # Create lookup table
    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        diff = np.abs(reference_cdf - source_cdf[i])
        lookup_table[i] = np.argmin(diff)

    return lookup_table[source]


def match_histogram(source, reference):
    """Match histogram of source image to reference image (WF7 feature)"""
    if len(source.shape) == 3:
        matched = np.zeros_like(source)
        for i in range(3):
            matched[:, :, i] = match_histogram_channel(
                source[:, :, i], reference[:, :, i])
        return matched
    else:
        return match_histogram_channel(source, reference)


def bilateral_filter_anti_fractal(frame):
    """Bilateral filter for anti-fractal processing (WF2 feature)"""
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(frame, 9, 75, 75)
    return filtered


def create_advanced_transparency_mask(frame, bg_color, tolerance=20):
    """Advanced transparency mask with multiple techniques"""
    if len(frame.shape) == 3:
        # Convert numpy types to regular Python types for consistency
        if hasattr(bg_color[0], 'item'):
            bg_color = tuple(int(c.item()) for c in bg_color)

        if isinstance(bg_color, (list, tuple)) and len(bg_color) == 3:
            # Method 1: Direct color matching
            diff = np.abs(frame.astype(int) - np.array(bg_color, dtype=int))
            mask1 = np.all(diff <= tolerance, axis=2)

            # Method 2: HSV-based background detection (simplified)
            try:
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                bg_color_array = np.uint8(
                    [[[bg_color[0], bg_color[1], bg_color[2]]]])
                bg_hsv = cv2.cvtColor(bg_color_array, cv2.COLOR_RGB2HSV)[0][0]

                # Create HSV mask with wider tolerance - ensure proper types
                lower_hsv = np.array([
                    max(0, int(bg_hsv[0]) - 10),
                    max(0, int(bg_hsv[1]) - 50),
                    max(0, int(bg_hsv[2]) - 50)
                ], dtype=np.uint8)
                upper_hsv = np.array([
                    min(179, int(bg_hsv[0]) + 10),
                    255,
                    255
                ], dtype=np.uint8)

                mask2 = cv2.inRange(hsv, lower_hsv, upper_hsv) > 0
                # Combine masks
                mask = mask1 | mask2
            except:
                # Fallback to simple color matching if HSV fails
                mask = mask1
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            mask = np.abs(gray_frame.astype(int) - int(bg_color)) <= tolerance
    else:
        mask = np.abs(frame.astype(int) - int(bg_color)) <= tolerance

    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask.astype(bool)


def ultimate_fusion_upscale(frame, reference_frame, global_palette, scale_factor=2.0):
    """Ultimate fusion: Global Palette + Histogram Matching + Anti-Fractal"""

    # Phase 1: Apply global palette (WF1)
    frame_with_palette = apply_global_palette(frame, global_palette)

    # Phase 2: Bilateral filter for noise reduction (WF2)
    frame_filtered = bilateral_filter_anti_fractal(frame_with_palette)

    # Phase 3: Upscale
    h, w = frame_filtered.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    upscaled = cv2.resize(frame_filtered, (new_w, new_h),
                          interpolation=cv2.INTER_CUBIC)

    # Phase 4: Histogram matching to reference (WF7)
    ref_upscaled = cv2.resize(
        reference_frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    histogram_matched = match_histogram(upscaled, ref_upscaled)

    # Phase 5: Blend histogram matched with upscaled
    result = cv2.addWeighted(histogram_matched, 0.7, upscaled, 0.3, 0)

    # Phase 6: Final edge-safe sharpening (WF2)
    kernel = np.array([[-0.1, -0.15, -0.1],
                      [-0.15,  1.6, -0.15],
                      [-0.1, -0.15, -0.1]])
    sharpened = cv2.filter2D(result, -1, kernel)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def process_mp4_to_gif_ultimate_fusion(input_path, output_path):
    """Ultimate fusion processing MP4 to GIF with 100% transparent background"""
    try:
        print(f"ULTIMATE FUSION MP4->GIF: {input_path}")

        # Load MP4 with OpenCV
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return False

        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        if not frames:
            print(f"No frames found in {input_path}")
            return False

        print(f"Loaded {len(frames)} frames at {fps} FPS")

        # Advanced background detection
        bg_color = detect_background_color_advanced(frames[0])
        print(f"Background color: {bg_color}")

        # Extract global palette (WF1)
        print("Extracting global color palette...")
        global_palette = extract_global_palette(frames, n_colors=48)

        # Use first frame as reference (WF7)
        reference_frame = frames[0]

        # Process frames with ultimate fusion
        processed_frames = []
        for i, frame in enumerate(frames):
            print(f"Processing frame {i+1}/{len(frames)}")

            # Ultimate fusion upscaling
            upscaled_frame = ultimate_fusion_upscale(
                frame, reference_frame, global_palette, scale_factor=2.0)

            # Create advanced transparency mask
            transparency_mask = create_advanced_transparency_mask(
                upscaled_frame, bg_color, tolerance=25)

            # Create PIL Image with transparency
            if len(upscaled_frame.shape) == 3:
                # Create RGBA image
                pil_image = Image.fromarray(upscaled_frame, 'RGB')
                # Convert to RGBA
                pil_image = pil_image.convert('RGBA')

                # Apply transparency
                data = np.array(pil_image)
                data[:, :, 3] = np.where(
                    transparency_mask, 0, 255)  # Set alpha channel
                pil_image = Image.fromarray(data, 'RGBA')
            else:
                # Convert grayscale to RGBA
                rgb_frame = np.stack([upscaled_frame]*3, axis=2)
                pil_image = Image.fromarray(rgb_frame, 'RGB').convert('RGBA')
                data = np.array(pil_image)
                data[:, :, 3] = np.where(transparency_mask, 0, 255)
                pil_image = Image.fromarray(data, 'RGBA')

            processed_frames.append(pil_image)

        # Calculate frame duration for GIF (convert FPS to duration in ms)
        frame_duration = max(50, int(1000 / fps)) if fps > 0 else 100

        # Save as GIF with transparency
        if processed_frames:
            print(
                f"Saving GIF with {len(processed_frames)} frames, duration: {frame_duration}ms")
            processed_frames[0].save(
                output_path,
                save_all=True,
                append_images=processed_frames[1:],
                duration=frame_duration,
                loop=0,  # Infinite loop
                transparency=0,  # Enable transparency
                disposal=2  # Clear frame before drawing next
            )

            print(f"ULTIMATE SUCCESS: {output_path}")
            return True

    except Exception as e:
        print(f"ERROR: {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Ultimate Fusion Workflow MP4 to GIF with 100% Transparent Background')
    parser.add_argument('--input-dir', default='input',
                        help='Input directory (default: input)')
    parser.add_argument('--output-dir', default='output/ultimate_fusion_gif',
                        help='Output directory (default: output/ultimate_fusion_gif)')
    parser.add_argument('--process-all', action='store_true',
                        help='Process all files in input directory instead of predefined list')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.process_all:
        # Process all MP4 files in directory
        mp4_files = list(input_dir.glob("*.mp4"))
        print(f"Found {len(mp4_files)} MP4 files")
        files_to_process = mp4_files
    else:
        # Process predefined list
        files_to_process = []
        for filename in INPUT_FILES:
            file_path = input_dir / filename
            if file_path.exists():
                files_to_process.append(file_path)
            else:
                print(f"Warning: File not found: {file_path}")

        print(f"Processing {len(files_to_process)} files from predefined list")

    success_count = 0
    for input_file in files_to_process:
        if input_file.name.endswith('.lnk'):
            continue

        # Create output filename
        output_file = output_dir / f"{input_file.stem}_ultimate_fusion.gif"

        if process_mp4_to_gif_ultimate_fusion(str(input_file), str(output_file)):
            success_count += 1

    print(
        f"\n✅ COMPLETED: {success_count}/{len(files_to_process)} files processed successfully")
    print(f"📁 Output directory: {output_dir}")


if __name__ == "__main__":
    main()
