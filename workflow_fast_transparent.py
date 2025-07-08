#!/usr/bin/env python3
"""
FAST TRANSPARENT WORKFLOW - No K-means clustering for speed
WF7 (Histogram Matching) + WF2 (Anti-Fractal) + 100% TRANSPARENT BACKGROUND
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageSequence
import argparse
from pathlib import Path


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
    if most_common:
        color = most_common[0][0]
        if isinstance(color, tuple):
            return tuple(int(c) for c in color)
        else:
            return int(color)
    else:
        return (255, 255, 255)


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
        if isinstance(bg_color, (list, tuple)) and len(bg_color) == 3:
            # Method 1: Direct color matching
            diff = np.abs(frame.astype(int) - np.array(bg_color, dtype=int))
            mask1 = np.all(diff <= tolerance, axis=2)

            # Method 2: HSV-based background detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            bg_hsv = cv2.cvtColor(
                np.uint8([[bg_color]]), cv2.COLOR_RGB2HSV)[0][0]

            # Create HSV mask with wider tolerance
            lower_hsv = np.array(
                [max(0, int(bg_hsv[0])-10), max(0, int(bg_hsv[1])-50), max(0, int(bg_hsv[2])-50)], dtype=np.uint8)
            upper_hsv = np.array(
                [min(179, int(bg_hsv[0])+10), 255, 255], dtype=np.uint8)
            mask2 = cv2.inRange(hsv, lower_hsv, upper_hsv) > 0

            # Combine masks
            mask = mask1 | mask2
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


def fast_upscale_process(frame, reference_frame, scale_factor=2.0):
    """Fast processing: Histogram Matching + Anti-Fractal (no K-means)"""

    # Phase 1: Bilateral filter for noise reduction (WF2)
    frame_filtered = bilateral_filter_anti_fractal(frame)

    # Phase 2: Upscale
    h, w = frame_filtered.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    upscaled = cv2.resize(frame_filtered, (new_w, new_h),
                          interpolation=cv2.INTER_CUBIC)

    # Phase 3: Histogram matching to reference (WF7)
    ref_upscaled = cv2.resize(
        reference_frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    histogram_matched = match_histogram(upscaled, ref_upscaled)

    # Phase 4: Blend histogram matched with upscaled
    result = cv2.addWeighted(histogram_matched, 0.7, upscaled, 0.3, 0)

    # Phase 5: Final edge-safe sharpening (WF2)
    kernel = np.array([[-0.1, -0.15, -0.1],
                      [-0.15,  1.6, -0.15],
                      [-0.1, -0.15, -0.1]])
    sharpened = cv2.filter2D(result, -1, kernel)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def process_gif_fast_transparent(input_path, output_path):
    """Fast processing with 100% transparent background"""
    try:
        print(f"FAST TRANSPARENT: {input_path}")

        # Load GIF
        with Image.open(input_path) as img:
            frames = []
            durations = []

            for frame in ImageSequence.Iterator(img):
                frame_array = np.array(frame.convert('RGB'))
                frames.append(frame_array)
                durations.append(frame.info.get('duration', 100))

        if not frames:
            print(f"No frames found in {input_path}")
            return False

        print(f"Loaded {len(frames)} frames")

        # Advanced background detection
        bg_color = detect_background_color_advanced(frames[0])
        print(f"Background color: {bg_color}")

        # Use first frame as reference (WF7)
        reference_frame = frames[0]

        # Process frames with fast method
        processed_frames = []
        for i, frame in enumerate(frames):
            print(f"Processing frame {i+1}/{len(frames)}")

            # Fast upscaling (no K-means)
            upscaled_frame = fast_upscale_process(
                frame, reference_frame, scale_factor=2.0)

            # Create advanced transparency mask
            transparency_mask = create_advanced_transparency_mask(
                upscaled_frame, bg_color, tolerance=25)

            # Apply 100% transparency to background
            if len(upscaled_frame.shape) == 3:
                # Create RGBA with 0% opacity for background
                alpha_channel = np.where(
                    transparency_mask, 0, 255).astype(np.uint8)
                rgba_frame = np.dstack([upscaled_frame, alpha_channel])
            else:
                rgba_frame = np.dstack([np.stack([upscaled_frame]*3, axis=2),
                                        np.where(transparency_mask, 0, 255).astype(np.uint8)])

            processed_frames.append(rgba_frame)

        # Calculate FPS
        avg_duration = sum(durations) / len(durations) if durations else 100
        fps = max(1, min(30, int(1000 / avg_duration)))

        # Save as MP4 with transparency support
        if processed_frames:
            height, width = processed_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for frame in processed_frames:
                # Convert RGBA to BGR for video writer (transparency info preserved in processing)
                bgr_frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)

            out.release()
            print(f"FAST SUCCESS: {output_path}")
            return True

    except Exception as e:
        print(f"ERROR: {input_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Fast Transparent Workflow')
    parser.add_argument('--input', required=True,
                        help='Input GIF file or directory')
    parser.add_argument('--output', required=True, help='Output directory')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    if input_path.is_file():
        # Single file processing
        output_file = output_dir / f"{input_path.stem}_fast_transparent.mp4"
        process_gif_fast_transparent(str(input_path), str(output_file))
    else:
        # Directory processing
        gif_files = list(input_path.glob("*.gif"))
        print(f"Found {len(gif_files)} GIF files")

        for gif_file in gif_files:
            if gif_file.name.endswith('.lnk'):
                continue

            output_file = output_dir / f"{gif_file.stem}_fast_transparent.mp4"
            process_gif_fast_transparent(str(gif_file), str(output_file))


if __name__ == "__main__":
    main()
