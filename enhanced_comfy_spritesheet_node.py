#!/usr/bin/env python3
"""
ENHANCED COMFYUI SPRITESHEET NODE
Optimierte traditionelle Spritesheet-Verarbeitung für ComfyUI
"""

import torch
import cv2
import numpy as np
from PIL import Image
import folder_paths
from pathlib import Path
from typing import Tuple, List


class EnhancedSpritesheetProcessor:
    """Enhanced Traditional Spritesheet Processing Node für ComfyUI"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "adaptive_tolerance": ("BOOLEAN", {"default": True}),
                "tolerance_override": ("FLOAT", {"default": 30.0, "min": 10.0, "max": 100.0, "step": 1.0}),
                "aggressive_extraction": ("BOOLEAN", {"default": True}),
                "min_area_factor": ("FLOAT", {"default": 3000.0, "min": 1000.0, "max": 10000.0, "step": 100.0}),
                "edge_refinement": ("BOOLEAN", {"default": True}),
                "hsv_analysis": ("BOOLEAN", {"default": True}),
                "multi_zone_sampling": ("BOOLEAN", {"default": True}),
                "morphological_cleanup": ("BOOLEAN", {"default": True}),
                "smooth_edges": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("sprite_frames", "processing_info", "frame_stats")
    FUNCTION = "process_spritesheet_enhanced"
    CATEGORY = "🎮 Enhanced Sprites"

    def enhanced_background_detection(self, image: np.ndarray, multi_zone: bool = True,
                                      hsv_analysis: bool = True) -> Tuple[List[np.ndarray], float]:
        """Enhanced Multi-Zone Background Detection"""
        h, w = image.shape[:2]

        if multi_zone:
            # ERWEITERTE SAMPLING-ZONEN
            corner_size = max(20, min(h, w) // 20)
            edge_width = max(15, min(h, w) // 30)

            zones = [
                # Größere Ecken
                image[:corner_size, :corner_size],
                image[:corner_size, -corner_size:],
                image[-corner_size:, :corner_size],
                image[-corner_size:, -corner_size:],

                # Kanten-Mitten
                image[:edge_width, w//4:3*w//4],
                image[-edge_width:, w//4:3*w//4],
                image[h//4:3*h//4, :edge_width],
                image[h//4:3*h//4, -edge_width:],

                # Border-Strips
                image[:edge_width//2, :],
                image[-edge_width//2:, :],
                image[:, :edge_width//2],
                image[:, -edge_width//2:]
            ]
        else:
            # Original corner-only sampling
            corner_size = 20
            zones = [
                image[:corner_size, :corner_size],
                image[:corner_size, -corner_size:],
                image[-corner_size:, :corner_size],
                image[-corner_size:, -corner_size:]
            ]

        # MULTI-COLOR ANALYSIS
        bg_candidates = []
        for zone in zones:
            if zone.size > 0:
                # RGB Analysis
                pixels = zone.reshape(-1, 3).astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS +
                            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                k = min(2, len(np.unique(pixels.reshape(-1))))

                if k > 0:
                    _, _, centers = cv2.kmeans(
                        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    bg_candidates.extend([center.astype(np.uint8)
                                         for center in centers])

                # HSV Analysis
                if hsv_analysis:
                    hsv_zone = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
                    hsv_pixels = hsv_zone.reshape(-1, 3).astype(np.float32)
                    if len(hsv_pixels) > 0:
                        bg_candidates.append(
                            np.median(hsv_pixels, axis=0).astype(np.uint8))

        # FINAL CLUSTERING
        if bg_candidates:
            all_colors = np.array(bg_candidates, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
            k = min(3, len(all_colors))
            _, _, final_centers = cv2.kmeans(
                all_colors, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            final_bg_colors = [center.astype(np.uint8)
                               for center in final_centers]
        else:
            final_bg_colors = [np.array([255, 255, 255], dtype=np.uint8)]

        # ADAPTIVE TOLERANCE
        variances = []
        for zone in zones:
            if zone.size > 0:
                variance = np.var(zone.reshape(-1, 3), axis=0)
                variances.append(np.mean(variance))

        if variances:
            avg_variance = np.mean(variances)
            if avg_variance < 100:
                tolerance = 20.0
            elif avg_variance < 300:
                tolerance = 30.0
            elif avg_variance < 600:
                tolerance = 40.0
            else:
                tolerance = 50.0
        else:
            tolerance = 30.0

        return final_bg_colors, tolerance

    def create_enhanced_mask(self, image: np.ndarray, bg_colors: List[np.ndarray],
                             tolerance: float, hsv_analysis: bool = True,
                             morphological_cleanup: bool = True, edge_refinement: bool = True) -> np.ndarray:
        """Enhanced Multi-Modal Background Masking"""
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=bool)

        # MULTI-COLOR MASKING
        for bg_color in bg_colors:
            # RGB Distance
            rgb_diff = np.abs(image.astype(int) - bg_color.astype(int))
            rgb_mask = np.all(rgb_diff <= tolerance, axis=2)

            # HSV Analysis
            if hsv_analysis:
                hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hsv_bg = cv2.cvtColor(bg_color.reshape(
                    1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
                hsv_diff = np.abs(hsv_image.astype(int) - hsv_bg.astype(int))
                hsv_tolerance = [tolerance//3, tolerance, tolerance]
                hsv_mask = np.all(hsv_diff <= hsv_tolerance, axis=2)
                combined_mask |= (rgb_mask | hsv_mask)
            else:
                combined_mask |= rgb_mask

        # MORPHOLOGICAL CLEANUP
        foreground_mask = ~combined_mask

        if morphological_cleanup:
            kernel_size = max(3, min(h, w) // 150)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            foreground_mask = cv2.morphologyEx(foreground_mask.astype(
                np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
            foreground_mask = cv2.morphologyEx(
                foreground_mask, cv2.MORPH_OPEN, kernel)

        # EDGE REFINEMENT
        if edge_refinement:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            uncertain_pixels = cv2.morphologyEx(
                foreground_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
            edge_boost = edges_dilated & uncertain_pixels
            foreground_mask = foreground_mask.astype(np.uint8)
            foreground_mask[edge_boost] = 1

        return foreground_mask.astype(np.uint8)

    def extract_frames_enhanced(self, image: np.ndarray, foreground_mask: np.ndarray,
                                aggressive: bool = True, min_area_factor: float = 3000.0) -> List[np.ndarray]:
        """Enhanced Aggressive Frame Extraction"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground_mask, connectivity=8)

        extracted_frames = []
        h, w = image.shape[:2]

        if aggressive:
            min_area = max(300, (w * h) // int(min_area_factor))
            max_aspect_ratio = 20.0
            min_aspect_ratio = 0.05
        else:
            min_area = max(500, (w * h) // 2000)
            max_aspect_ratio = 5.0
            min_aspect_ratio = 0.2

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if area < min_area:
                continue

            x, y, w_comp, h_comp = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

            aspect_ratio = w_comp / h_comp if h_comp > 0 else 0
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue

            # Extract with padding
            padding = 8
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image.shape[1], x + w_comp + padding)
            y_end = min(image.shape[0], y + h_comp + padding)

            frame = image[y_start:y_end, x_start:x_end]

            if frame.size > 0:
                extracted_frames.append(frame)

        return extracted_frames

    def process_individual_frame(self, frame: np.ndarray, smooth_edges: bool = True) -> np.ndarray:
        """Process individual frame with local background removal"""
        # Local background detection
        frame_bg_colors, frame_tolerance = self.enhanced_background_detection(
            frame)
        frame_mask = self.create_enhanced_mask(
            frame, frame_bg_colors, frame_tolerance)

        # Convert to RGBA
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgba = np.dstack(
            [frame_rgb, np.full(frame_rgb.shape[:2], 255, dtype=np.uint8)])

        # Set background transparent
        frame_rgba[frame_mask == 0, 3] = 0

        # Smooth edges
        if smooth_edges:
            alpha = frame_rgba[:, :, 3]
            alpha_smooth = cv2.GaussianBlur(alpha, (3, 3), 0.8)
            edges = cv2.morphologyEx(
                alpha, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            frame_rgba[edges > 0, 3] = alpha_smooth[edges > 0]

        return frame_rgba

    def process_spritesheet_enhanced(self, image, adaptive_tolerance=True, tolerance_override=30.0,
                                     aggressive_extraction=True, min_area_factor=3000.0,
                                     edge_refinement=True, hsv_analysis=True,
                                     multi_zone_sampling=True, morphological_cleanup=True,
                                     smooth_edges=True):
        """Enhanced Traditional Spritesheet Processing for ComfyUI"""

        # Convert ComfyUI tensor to numpy
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image_np = image[0].cpu().numpy()
            else:
                image_np = image.cpu().numpy()

            # Convert to uint8 and BGR
            image_np = (image_np * 255).astype(np.uint8)
            if image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        h, w = image_np.shape[:2]

        # Enhanced Background Detection
        bg_colors, detected_tolerance = self.enhanced_background_detection(
            image_np, multi_zone_sampling, hsv_analysis
        )

        # Use adaptive or override tolerance
        final_tolerance = detected_tolerance if adaptive_tolerance else tolerance_override

        # Enhanced Masking
        foreground_mask = self.create_enhanced_mask(
            image_np, bg_colors, final_tolerance, hsv_analysis,
            morphological_cleanup, edge_refinement
        )

        # Enhanced Frame Extraction
        extracted_frames = self.extract_frames_enhanced(
            image_np, foreground_mask, aggressive_extraction, min_area_factor
        )

        # Process individual frames
        processed_frames = []
        for frame in extracted_frames:
            frame_rgba = self.process_individual_frame(frame, smooth_edges)
            processed_frames.append(frame_rgba)

        # Convert back to ComfyUI format
        if processed_frames:
            # Stack frames for ComfyUI
            comfy_frames = []
            for frame_rgba in processed_frames:
                # Convert RGBA to RGB for ComfyUI (transparency info preserved in alpha)
                frame_rgb = frame_rgba[:, :, :3]
                frame_tensor = torch.from_numpy(
                    frame_rgb.astype(np.float32) / 255.0)
                comfy_frames.append(frame_tensor)

            # Stack all frames
            result_tensor = torch.stack(comfy_frames, dim=0)
        else:
            # Return original if no frames found
            result_tensor = image

        # Processing info
        processing_info = f"Enhanced Traditional Processing\n" \
            f"Image Size: {w}x{h}\n" \
            f"Background Colors: {len(bg_colors)}\n" \
            f"Tolerance: {final_tolerance:.1f} ({'adaptive' if adaptive_tolerance else 'manual'})\n" \
            f"Extracted Frames: {len(processed_frames)}\n" \
            f"Settings: Multi-zone={multi_zone_sampling}, HSV={hsv_analysis}, " \
            f"Aggressive={aggressive_extraction}, Edge-refine={edge_refinement}"

        # Frame stats
        frame_stats = f"Frame Details:\n"
        for i, frame in enumerate(processed_frames):
            h_f, w_f = frame.shape[:2]
            transparency_ratio = np.sum(frame[:, :, 3] == 0) / (h_f * w_f)
            frame_stats += f"Frame {i+1}: {w_f}x{h_f}, Transparency: {transparency_ratio:.2%}\n"

        if not processed_frames:
            frame_stats += "No frames extracted - check parameters!"

        return (result_tensor, processing_info, frame_stats)


# ComfyUI Node Registration
NODE_CLASS_MAPPINGS = {
    "EnhancedSpritesheetProcessor": EnhancedSpritesheetProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedSpritesheetProcessor": "🎮 Enhanced Spritesheet Processor"
}
