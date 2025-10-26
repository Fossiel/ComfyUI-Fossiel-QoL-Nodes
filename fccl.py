import math
import torch

class FossielCentralControlLite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Manual_Resolution_Width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "Manual_Resolution_Height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "Target_Resolution_Width": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "Target_Resolution_Height": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "Max_Resolution_Width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "Max_Resolution_Height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "Video_Length": ("INT", {"default": 81, "min": 1, "max": 301, "step": 4}),
                "Overlap_Length": ("INT", {"default": 8, "min": 0, "max": 80, "step": 1}),
                "Current_Generation": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = (
        "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "BOOLEAN"
    )
    RETURN_NAMES = (
        "Man_Res_Width", "Man_Res_Height", "Calc_Res_Width", "Calc_Res_Height",
        "Vid_Length", "Overlap", "Gen_Num", "Start_Index", "End_Index", "batch_count",
        "First_Gen"
    )
    FUNCTION = "process"
    CATEGORY = "Fossiel"
    OUTPUT_NODE = True

    def process(self, Manual_Resolution_Width, Manual_Resolution_Height,
                Target_Resolution_Width, Target_Resolution_Height,
                Max_Resolution_Width, Max_Resolution_Height,
                Video_Length, Overlap_Length, Current_Generation, image=None):

        # Passthroughs
        man_w = max(64, Manual_Resolution_Width)  # Clamp to sane min
        man_h = max(64, Manual_Resolution_Height)
        overlap = max(0, Overlap_Length)
        gen_num = max(1, Current_Generation)

        # Snap Video_Length to valid (1 or 4k+1)
        if Video_Length == 1:
            vid_length = 1
        else:
            k = math.ceil((Video_Length - 1) / 4)
            vid_length = 4 * k + 1
        vid_length = min(301, vid_length)  # Cap

        # Handle optional IMAGE: Extract batch_count and override target dims if present
        batch_count = 0
        target_w = max(1, Target_Resolution_Width)
        target_h = max(1, Target_Resolution_Height)
        if image is not None and isinstance(image, torch.Tensor) and image.numel() > 0:
            batch_count = image.shape[0]  # Batch size
            target_h = image.shape[1]     # Height from first batch item
            target_w = image.shape[2]     # Width from first batch item

        # Calc Res: Exact aspect, even dims, max area <= max_pixels (no per-dim bounds)
        max_w = max(64, Max_Resolution_Width)
        max_h = max(64, Max_Resolution_Height)
        max_pixels = max_w * max_h

        # Reduce to lowest terms
        gcd = math.gcd(target_w, target_h)
        reduced_w = target_w // gcd
        reduced_h = target_h // gcd

        # Make base even: multiply by 2 if needed (ensures both even while exact aspect)
        multiplier = 1
        if reduced_w % 2 != 0 or reduced_h % 2 != 0:
            multiplier = 2
        base_w = multiplier * reduced_w
        base_h = multiplier * reduced_h

        # Find largest k such that (k*base_w * k*base_h) <= max_pixels
        area_base = base_w * base_h
        if area_base == 0:
            calc_w, calc_h = 2, 2  # Fallback
        else:
            k_max_area = math.floor(math.sqrt(max_pixels / area_base))
            k = max(1, k_max_area)  # At least base
            calc_w = k * base_w
            calc_h = k * base_h
            # Trim if over pixels (edge case)
            while calc_w * calc_h > max_pixels and k > 1:
                k -= 1
                calc_w = k * base_w
                calc_h = k * base_h

        # Indices (inclusive end): Treat gen=1 as first (offset 0)
        effective_gen = gen_num - 1
        start_idx = effective_gen * (vid_length - overlap)
        end_idx = start_idx + vid_length - 1

        # Clamp indices sane (no negatives, no absurd overshoots)
        start_idx = max(0, start_idx)
        end_idx = max(start_idx, end_idx)

        # First_Gen: True (1) if gen_num == 1, else False (0)
        first_gen = 1 if gen_num == 1 else 0

        return (man_w, man_h, calc_w, calc_h, vid_length, overlap, gen_num, start_idx, end_idx, batch_count, first_gen)
