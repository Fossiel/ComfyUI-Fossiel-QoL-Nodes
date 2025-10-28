# FossielWebPWrangler.py
import os
import torch
from PIL import Image
import numpy as np
import webp

class FossielWebPWrangler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "WebP_Path": ("STRING", {"default": ""}),
                "Load_Mode": ([
                    "All",
                    "From_first_to_Index_1",
                    "From_Index_1_to_last",
                    "Index_1_to_Index_2",
                    "First_frame",
                    "Last_frame",
                    "Index_1_frame",
                ], {"default": "All"}),
                "Index_1": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "Index_2": ("INT", {"default": 0, "min": 0, "max": 10000}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("images", "Alpha_as_mask", "total_batch_count", "split_batch_count", "Frame_Rate")
    FUNCTION = "load_webp_sequence"
    CATEGORY = "Fossiel/QoL"

    def _get_avg_fps(self, webp_path: str) -> float:
        try:
            with open(webp_path, "rb") as f:
                data = webp.WebPData.from_buffer(f.read())
            dec = webp.WebPAnimDecoder.new(data)
            timestamps_ms = []
            frame_count = 0
            for arr, timestamp_ms in dec.frames():
                frame_count += 1
                timestamps_ms.append(timestamp_ms)
            if frame_count <= 1:
                return 0.0
            total_duration_ms = timestamps_ms[-1]
            if total_duration_ms <= 0:
                return 10.0
            return frame_count / (total_duration_ms / 1000.0)
        except Exception:
            return 10.0

    def load_webp_sequence(self, WebP_Path: str, Load_Mode: str, Index_1: int, Index_2: int):
        if not WebP_Path:
            raise ValueError("WebP path is required")
        WebP_Path = os.path.expanduser(WebP_Path)
        if not os.path.exists(WebP_Path):
            raise ValueError(f"WebP file does not exist: {WebP_Path}")

        raw_fps = self._get_avg_fps(WebP_Path)
        rounded_fps = int(raw_fps + 0.5)

        try:
            webp = Image.open(WebP_Path)
        except Exception:
            raise ValueError(f"Cannot open WebP file: {WebP_Path}")

        if not getattr(webp, "is_animated", False) or webp.n_frames <= 1:
            webp.seek(0)
            img = webp.convert("RGBA")
            arr = np.array(img).astype(np.float32) / 255.0
            rgb = arr[..., :3]
            alpha = arr[..., 3]
            if np.allclose(alpha, 1.0) or np.allclose(alpha, 0.0):
                alpha = np.ones_like(alpha)
            return (
                torch.from_numpy(rgb).unsqueeze(0),
                torch.from_numpy(alpha).unsqueeze(0),
                1, 1, 0.0
            )

        total_frames = webp.n_frames

        def extract_frames(start: int, end: int):
            rgb_frames = []
            alpha_frames = []
            with Image.open(WebP_Path) as img:
                for f in range(max(0, start), min(end, total_frames)):
                    img.seek(f)
                    frame = img.convert("RGBA")
                    arr = np.array(frame).astype(np.float32) / 255.0
                    rgb_frames.append(torch.from_numpy(arr[..., :3]))
                    alpha = arr[..., 3]
                    if np.allclose(alpha, 1.0) or np.allclose(alpha, 0.0):
                        alpha = np.ones_like(alpha)
                    alpha_frames.append(torch.from_numpy(alpha))
            return rgb_frames, alpha_frames

        rgb_list, alpha_list = [], []

        if Load_Mode == "All":
            rgb_list, alpha_list = extract_frames(0, total_frames)
        elif Load_Mode == "From_first_to_Index_1":
            if Index_1 < 0 or Index_1 >= total_frames:
                raise IndexError(f"Index_1 {Index_1} out of range [0, {total_frames-1}]")
            rgb_list, alpha_list = extract_frames(0, Index_1 + 1)
        elif Load_Mode == "From_Index_1_to_last":
            if Index_1 < 0 or Index_1 >= total_frames:
                raise IndexError(f"Index_1 {Index_1} out of range")
            rgb_list, alpha_list = extract_frames(Index_1, total_frames)
        elif Load_Mode == "Index_1_to_Index_2":
            s, e = sorted([Index_1, Index_2])
            if s >= total_frames:
                raise IndexError("Both indices out of range")
            e = min(e, total_frames - 1)
            rgb_list, alpha_list = extract_frames(s, e + 1)
        elif Load_Mode == "First_frame":
            rgb_list, alpha_list = extract_frames(0, 1)
        elif Load_Mode == "Last_frame":
            rgb_list, alpha_list = extract_frames(total_frames - 1, total_frames)
        elif Load_Mode == "Index_1_frame":
            if Index_1 < 0 or Index_1 >= total_frames:
                raise IndexError(f"Index_1 {Index_1} out of range")
            rgb_list, alpha_list = extract_frames(Index_1, Index_1 + 1)
        else:
            raise ValueError(f"Unknown Load_Mode: {Load_Mode}")

        if not rgb_list:
            raise RuntimeError("No frames selected")

        batch_rgb = torch.stack(rgb_list)
        batch_alpha = torch.stack(alpha_list)
        split_count = len(rgb_list)

        return (batch_rgb, batch_alpha, total_frames, split_count, float(rounded_fps))
