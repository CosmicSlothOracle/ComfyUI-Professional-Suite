#!/usr/bin/env python3
"""
ComfyUI Custom Node: Intelligent Spritesheet Processor
Automatische Frame-Erkennung durch Connected Component Analysis
"""

import cv2
import numpy as np
from PIL import Image
from collections import Counter
from pathlib import Path
import torch
import folder_paths
import os


class IntelligentSpritesheetProcessor:
    """
    ComfyUI Node für intelligente Spritesheet-Verarbeitung
    Löst das Problem unregelmäßiger Spritesheet-Layouts vollständig!
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "background_tolerance": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                "min_frame_area": ("INT", {
                    "default": 800,
                    "min": 100,
                    "max": 10000,
                    "step": 100
                }),
                "corner_detection_size": ("FLOAT", {
                    "default": 30.0,
                    "min": 10.0,
                    "max": 100.0,
                    "step": 1.0
                }),
                "morphology_kernel_size": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 15,
                    "step": 2
                })
            },
            "optional": {
                "output_gif": ("BOOLEAN", {"default": True}),
                "gif_duration": ("INT", {
                    "default": 500,
                    "min": 100,
                    "max": 2000,
                    "step": 50
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("extracted_frames", "analysis_report", "frame_count")
    FUNCTION = "process_spritesheet"
    CATEGORY = "🎮 Sprite Processing"

    def tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image"""
        # Tensor format: (batch, height, width, channels)
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)  # Remove batch dimension

        # Convert to numpy and scale to 0-255
        numpy_image = tensor.cpu().numpy()
        if numpy_image.max() <= 1.0:
            numpy_image = (numpy_image * 255).astype(np.uint8)

        return Image.fromarray(numpy_image)

    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor"""
        # Convert to RGB if RGBA
        if pil_image.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()
                             [-1])  # Use alpha as mask
            pil_image = background
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Convert to tensor
        numpy_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(numpy_image)

        # Add batch dimension: (height, width, channels) -> (1, height, width, channels)
        return tensor.unsqueeze(0)

    def detect_background_color(self, image_np, corner_size_ratio=30.0):
        """Detect background color from image corners"""
        h, w = image_np.shape[:2]
        corner_size = int(min(h, w) / corner_size_ratio)

        # Extract corners
        corners = [
            image_np[0:corner_size, 0:corner_size],
            image_np[0:corner_size, w-corner_size:w],
            image_np[h-corner_size:h, 0:corner_size],
            image_np[h-corner_size:h, w-corner_size:w]
        ]

        # Collect corner pixels
        corner_pixels = []
        for corner in corners:
            if len(corner.shape) == 3:
                pixels = corner.reshape(-1, 3)
                for pixel in pixels:
                    corner_pixels.append(tuple(pixel))

        # Find most common color
        if corner_pixels:
            bg_color = Counter(corner_pixels).most_common(1)[0][0]
            return np.array(bg_color, dtype=np.uint8)
        else:
            return np.array([255, 255, 255], dtype=np.uint8)  # Default white

    def create_foreground_mask(self, image_np, bg_color, tolerance, kernel_size):
        """Create foreground mask using background color"""
        # Calculate difference from background
        diff = np.abs(image_np.astype(int) - bg_color.astype(int))
        background_mask = np.all(diff <= tolerance, axis=2)
        foreground_mask = ~background_mask

        # Morphological operations for cleanup
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        foreground_mask = cv2.morphologyEx(
            foreground_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel
        )
        foreground_mask = cv2.morphologyEx(
            foreground_mask, cv2.MORPH_OPEN, kernel)

        return foreground_mask

    def extract_components(self, foreground_mask, image_np, min_area, bg_color, tolerance):
        """Extract connected components as individual frames"""
        # Connected Components Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            foreground_mask, connectivity=8
        )

        extracted_frames = []
        frame_info = []

        total_pixels = image_np.shape[0] * image_np.shape[1]

        for i in range(1, num_labels):  # Skip background label 0
            x, y, w, h, area = stats[i]

            # Filter by area
            if area < min_area or area > total_pixels * 0.8:
                continue

            # Filter by aspect ratio
            aspect_ratio = w / h if h > 0 else float('inf')
            if aspect_ratio > 8 or aspect_ratio < 0.125:
                continue

            # Extract frame with padding
            padding = 5
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image_np.shape[1], x + w + padding)
            y_end = min(image_np.shape[0], y + h + padding)

            frame = image_np[y_start:y_end, x_start:x_end]

            if frame.size == 0:
                continue

            # Remove background from frame
            frame_diff = np.abs(frame.astype(int) - bg_color.astype(int))
            frame_bg_mask = np.all(frame_diff <= tolerance, axis=2)

            # Create RGBA version
            frame_rgba = np.dstack(
                [frame, np.full(frame.shape[:2], 255, dtype=np.uint8)])
            frame_rgba[frame_bg_mask, 3] = 0  # Make background transparent

            # Convert to PIL and then back to tensor
            pil_frame = Image.fromarray(frame_rgba, 'RGBA')
            tensor_frame = self.pil_to_tensor(pil_frame)

            extracted_frames.append(tensor_frame)

            frame_info.append({
                'id': len(extracted_frames),
                'bbox': (x, y, w, h),
                'area': int(area),
                'aspect_ratio': float(aspect_ratio),
                'size': f"{w}x{h}"
            })

        return extracted_frames, frame_info

    def create_analysis_report(self, frame_info, bg_color, image_shape):
        """Create detailed analysis report"""
        report_lines = [
            "🎮 INTELLIGENT SPRITESHEET ANALYSIS REPORT",
            "=" * 50,
            f"📐 Original Image Size: {image_shape[1]}x{image_shape[0]}",
            f"🎯 Detected Background Color: RGB{tuple(bg_color)}",
            f"📦 Total Frames Extracted: {len(frame_info)}",
            "",
            "EXTRACTED FRAMES:",
            "-" * 20
        ]

        for frame in frame_info:
            report_lines.append(
                f"Frame {frame['id']:2d}: {frame['size']:>8} | "
                f"Area: {frame['area']:>6} | "
                f"Ratio: {frame['aspect_ratio']:.2f}"
            )

        if frame_info:
            areas = [f['area'] for f in frame_info]
            ratios = [f['aspect_ratio'] for f in frame_info]

            report_lines.extend([
                "",
                "STATISTICS:",
                "-" * 12,
                f"Area Range: {min(areas)} - {max(areas)}",
                f"Aspect Ratio Range: {min(ratios):.2f} - {max(ratios):.2f}",
                f"Average Area: {sum(areas) / len(areas):.0f}"
            ])

        return "\n".join(report_lines)

    def save_gif_animation(self, frames_tensors, output_path, duration):
        """Save frames as GIF animation"""
        if not frames_tensors:
            return

        # Convert tensors to PIL images
        pil_frames = []
        for tensor_frame in frames_tensors:
            pil_frame = self.tensor_to_pil(tensor_frame)
            pil_frames.append(pil_frame)

        # Save as GIF
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0,
            transparency=0,
            disposal=2
        )

    def process_spritesheet(self, image, background_tolerance=25, min_frame_area=800,
                            corner_detection_size=30.0, morphology_kernel_size=3,
                            output_gif=True, gif_duration=500):
        """
        Main processing function for ComfyUI
        """
        try:
            # Convert ComfyUI tensor to PIL Image
            pil_image = self.tensor_to_pil(image)

            # Convert to numpy array for OpenCV processing
            image_np = np.array(pil_image)

            # Detect background color
            bg_color = self.detect_background_color(
                image_np, corner_detection_size)

            # Create foreground mask
            foreground_mask = self.create_foreground_mask(
                image_np, bg_color, background_tolerance, morphology_kernel_size
            )

            # Extract frames
            extracted_frames, frame_info = self.extract_components(
                foreground_mask, image_np, min_frame_area, bg_color, background_tolerance
            )

            # Create analysis report
            report = self.create_analysis_report(
                frame_info, bg_color, image_np.shape)

            # Save GIF if requested
            if output_gif and extracted_frames:
                output_dir = Path(
                    folder_paths.get_output_directory()) / "spritesheet_processing"
                output_dir.mkdir(exist_ok=True)

                gif_path = output_dir / \
                    f"spritesheet_animation_{len(extracted_frames)}frames.gif"
                self.save_gif_animation(
                    extracted_frames, gif_path, gif_duration)

                report += f"\n\n🎬 GIF Animation saved: {gif_path}"

            # Combine all frames into a batch tensor
            if extracted_frames:
                # Stack all frames into a single batch tensor
                combined_tensor = torch.cat(extracted_frames, dim=0)
            else:
                # Return empty tensor if no frames found
                combined_tensor = torch.zeros(
                    (1, 64, 64, 3), dtype=torch.float32)
                report += "\n\n❌ WARNING: No valid frames detected!"

            return (combined_tensor, report, len(extracted_frames))

        except Exception as e:
            error_report = f"❌ ERROR in Spritesheet Processing: {str(e)}"
            empty_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_tensor, error_report, 0)


# ComfyUI Node Registration
NODE_CLASS_MAPPINGS = {
    "IntelligentSpritesheetProcessor": IntelligentSpritesheetProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IntelligentSpritesheetProcessor": "🎮 Intelligent Spritesheet Processor"
}
