# FossielSequenceWrangler.py
import os
import torch
from PIL import Image
import numpy as np
from pathlib import Path

class FossielSequenceWrangler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Sequence_Dir": ("STRING", {"default": "", "multiline": False}),
                "Load_Mode": ([
                    "All",
                    "From_first_to_Index_1",
                    "From_Index_1_to_last",
                    "Index_1_to_Index_2",
                    "First_frame",
                    "Last_frame",
                    "Index_1_frame",
                ], {"default": "All"}),
                "Missing_Alpha_Handling": (["Opaque", "Transparent"], {"default": "Opaque"}),
                "Index_1": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "Index_2": ("INT", {"default": 0, "min": 0, "max": 10000}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("images", "Alpha_as_mask", "total_batch_count", "split_batch_count")
    FUNCTION = "load_image_sequence"
    CATEGORY = "Fossiel/QoL"

    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif', '.gif'}

    def _check_for_animated_files(self, directory: str):
        """Reject directory if it contains any animated WebP or GIF"""
        dir_path = Path(directory)
        if not dir.is_dir():
            return

        for file_path in dir.iterdir():
            if not file_path.is_file():
                continue
            suffix = file_path.suffix.lower()
            if suffix in {'.webp', '.gif'}:
                try:
                    with Image.open(file_path) as img:
                        if getattr(img, "is_animated", False) and img.n_frames > 1:
                            raise ValueError(
                                f"Animated {suffix.upper()[1:]} file detected: {file_path.name}\n\n"
                                "FossielSequenceWrangler does NOT support animated WebP/GIF files.\n"
                                "This node loads sequences of individual static images only.\n"
                                "Use FossielWebPWrangler for animated webp, or extract frames first."
                            )
                except Exception:
                    pass  # Skip unreadable files silently during check

    def load_image_sequence(self, Sequence_Dir: str, Load_Mode: str, Index_1: int, Index_2: int,
                            Missing_Alpha_Handling: str = "Opaque"):
        if not Sequence_Dir:
            raise ValueError("Sequence_Dir is required")

        Sequence_Dir = os.path.expanduser(Sequence_Dir.strip())
        if not os.path.isdir(Sequence_Dir):
            raise ValueError(f"Path is not a directory: {Sequence_Dir}")

        # Safety check: no animated files allowed
        self._check_for_animated_files(Sequence_Dir)

        # Collect and sort image files
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(Path(Sequence_Dir).glob(f'*{ext}'))
            files.extend(Path(Sequence_Dir).glob(f'*{ext.upper()}'))
        files = sorted(files, key=lambda x: x.name.lower())

        if not files:
            raise ValueError(f"No supported image files found in: {Sequence_Dir}")

        total_frames = len(files)

        # Predefine fallback alpha based on user choice
        if Missing_Alpha_Handling == "Opaque":
            fallback_alpha_value = 1.0
        else:  # "Transparent"
            fallback_alpha_value = 0.0

        def load_frame(idx: int):
            path = files[idx]
            try:
                with Image.open(path) as img:
                    # Force RGBA – PIL will add alpha channel if missing
                    img = img.convert("RGBA")
                arr = np.array(img).astype(np.float32) / 255.0
                rgb = arr[..., :3]
                alpha = arr[..., 3]

                # Critical: if original image had no alpha (e.g. JPEG), PIL fills with 255
                # We detect that case and replace with user's preferred fallback
                if np.all(alpha == 1.0):  # very likely no real alpha channel
                    # Double-check by re-opening in "RGB" mode – if size matches, it never had alpha
                    try:
                        with Image.open(path) as img_rgb:
                            if img_rgb.mode in ("RGB", "L", "HSV", "P") or len(img_rgb.getbands()) < 4:
                                alpha = np.full_like(alpha, fallback_alpha_value)
                    except:
                        pass  # fallback to detected value

                return torch.from_numpy(rgb), torch.from_numpy(alpha)

            except Exception as e:
                raise RuntimeError(f"Failed to load image {path.name}: {str(e)}")

        def extract_frames(start: int, end: int):
            rgb_frames = []
            alpha_frames = []
            for i in range(max(0, start), min(end, total_frames)):
                rgb, alpha = load_frame(i)
                rgb_frames.append(rgb)
                alpha_frames.append(alpha)
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

        return (batch_rgb, batch_alpha, total_frames, split_count)
