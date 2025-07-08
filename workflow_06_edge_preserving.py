#!/usr/bin/env python3
"""
Workflow 6: Edge-Preserving Smoothing with Global Palette
Combines OpenCV's edge-preserving filter with global color consistency
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageSequence
import argparse
from pathlib import Path


def detect_background_color(frame):
    """Detect background color from corners"""
    h, w = frame.shape[:2]
    corners = [
        frame[0, 0], frame[0, w-1],
        frame[h-1, 0], frame[h-1, w-1]
    ]

    if len(frame.shape) == 3:
        corner_colors = [tuple(corner) for corner in corners]
    else:
        corner_colors = corners

    from collections import Counter
    most_common = Counter(corner_colors).most_common(1)
    return most_common[0][0] if most_common else (255, 255, 255)


def extract_global_palette(frames, n_colors=56):
    """Extract global color palette from all frames"""
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
    """Apply global palette to frame"""
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


def edge_preserving_upscale(frame, scale_factor=2.0):
    """Upscale with edge-preserving smoothing"""
    # First upscale
    h, w = frame.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    upscaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Apply edge-preserving filter
    # Using recursive filtering for better edge preservation
    edge_preserved = cv2.edgePreservingFilter(
        upscaled, flags=2, sigma_s=80, sigma_r=0.4)

    # Blend with original upscaled for balance
    result = cv2.addWeighted(edge_preserved, 0.7, upscaled, 0.3, 0)

    # Subtle sharpening for pixel art clarity
    kernel = np.array([[-0.05, -0.1, -0.05],
                      [-0.1,   1.4, -0.1],
                      [-0.05, -0.1, -0.05]])
    sharpened = cv2.filter2D(result, -1, kernel)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def create_transparency_mask(frame, bg_color, tolerance=25):
    """Create transparency mask based on background color"""
    if len(frame.shape) == 3:
        if isinstance(bg_color, (list, tuple)) and len(bg_color) == 3:
            diff = np.abs(frame.astype(int) - np.array(bg_color, dtype=int))
            mask = np.all(diff <= tolerance, axis=2)
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.abs(gray_frame.astype(int) - int(bg_color)) <= tolerance
    else:
        mask = np.abs(frame.astype(int) - int(bg_color)) <= tolerance

    return mask


def process_gif_edge_preserving(input_path, output_path):
    """Process GIF with edge-preserving smoothing and global palette"""
    try:
        print(f"Processing: {input_path}")

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

        # Detect background color from first frame
        bg_color = detect_background_color(frames[0])
        print(f"Detected background color: {bg_color}")

        # Extract global palette
        print("Extracting global color palette...")
        global_palette = extract_global_palette(frames, n_colors=56)

        # Process frames
        processed_frames = []
        for i, frame in enumerate(frames):
            print(f"Processing frame {i+1}/{len(frames)}")

            # Apply global palette first
            frame_with_palette = apply_global_palette(frame, global_palette)

            # Edge-preserving upscaling
            upscaled_frame = edge_preserving_upscale(
                frame_with_palette, scale_factor=2.0)

            # Create transparency mask
            transparency_mask = create_transparency_mask(
                upscaled_frame, bg_color, tolerance=25)

            # Apply transparency
            if len(upscaled_frame.shape) == 3:
                rgba_frame = np.dstack([upscaled_frame, np.where(
                    transparency_mask, 0, 255).astype(np.uint8)])
            else:
                rgba_frame = np.dstack([np.stack([upscaled_frame]*3, axis=2),
                                        np.where(transparency_mask, 0, 255).astype(np.uint8)])

            processed_frames.append(rgba_frame)

        # Calculate average duration
        avg_duration = sum(durations) / len(durations) if durations else 100
        fps = max(1, min(30, int(1000 / avg_duration)))

        # Save as MP4
        if processed_frames:
            height, width = processed_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for frame in processed_frames:
                # Convert RGBA to BGR for video writer
                bgr_frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)

            out.release()
            print(f"Successfully saved: {output_path}")
            return True

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Edge-Preserving Smoothing Workflow')
    parser.add_argument('--input', required=True,
                        help='Input GIF file or directory')
    parser.add_argument('--output', required=True, help='Output directory')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    if input_path.is_file():
        # Single file processing
        output_file = output_dir / \
            f"{input_path.stem}_workflow06_edge_preserving.mp4"
        process_gif_edge_preserving(str(input_path), str(output_file))
    else:
        # Directory processing
        gif_files = list(input_path.glob("*.gif"))
        print(f"Found {len(gif_files)} GIF files")

        for gif_file in gif_files:
            if gif_file.name.endswith('.lnk'):
                continue

            output_file = output_dir / \
                f"{gif_file.stem}_workflow06_edge_preserving.mp4"
            process_gif_edge_preserving(str(gif_file), str(output_file))


if __name__ == "__main__":
    main()
