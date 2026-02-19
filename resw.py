import torch
import math
import numpy as np
from fractions import Fraction
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import resize as tv_resize
from torchvision.transforms import InterpolationMode

class FossielResolutionWrangler:
    """
    ResolutionWrangler - adapted for ComfyUI conventions:
    - Input image is expected RGB (3 channels)
    - Optional mask provides transparency (0=opaque/keep, 1=transparent)
    - Resize_by: "Max Resolution" uses pixel cap; "Ratio" scales cropped size by percentage
    - Always enforces output divisible by Aspect_tolerance
    - Fallback to cropped_aspect * tolerance when cap too low (may slightly exceed cap)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Aspect_method": (["Automatic", "Manual"], {"default": "Automatic"}),
                "Aspect_X": ("INT", {"default": 4, "min": 1, "max": 9999, "step": 1}),
                "Aspect_Y": ("INT", {"default": 3, "min": 1, "max": 9999, "step": 1}),
                "Crop_position": ([
                    "Left and Top", "Left and Y-Center", "Left and Bottom",
                    "X-Center and Top", "Center", "X-Center and Bottom",
                    "Right and Top", "Right and Y-Center", "Right and Bottom"
                ], {"default": "Center"}),
                "Resize_by": (["Max Resolution", "Ratio"], {"default": "Max Resolution"}),
                "Max_Resolution_X": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 8}),
                "Max_Resolution_Y": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 8}),
                "Ratio": ("FLOAT", {"default": 100.00, "min": 0.10, "max": 10000.00, "step": 0.01}),
                "Aspect_tolerance": (["8", "16"], {"default": "8"}),
                "Resizing_method": ([
                    "nearest-exact", "bilinear", "bicubic", "lanczos"
                ], {"default": "lanczos"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = (
        "IMAGE", "IMAGE", "MASK",
        "IMAGE", "IMAGE", "MASK",
        "INT", "INT", "INT", "INT", "INT", "INT"
    )
    RETURN_NAMES = (
        "Aspect_Image",
        "Aspect_RGBA",
        "Aspect_Mask",
        "Resized_Image",
        "Resized_RGBA",
        "Resized_Mask",
        "Aspect_X",
        "Aspect_Y",
        "Aspect_Image_X",
        "Aspect_Image_Y",
        "Resized_Image_X",
        "Resized_Image_Y"
    )

    FUNCTION = "wrangle"
    CATEGORY = "utils"

    def wrangle(self,
                Aspect_method, Aspect_X, Aspect_Y,
                Crop_position, Resize_by, Max_Resolution_X, Max_Resolution_Y,
                Ratio, Aspect_tolerance, Resizing_method,
                image=None, mask=None):

        tolerance = int(Aspect_tolerance)

        no_input = image is None
        if no_input:
            print("[ResolutionWrangler] No image → using 1024×1024 black + Manual mode")
            image = self.create_black_starter(1024)
            Aspect_method = "Manual"

        print(f"[ResolutionWrangler] Image shape: {image.shape}")

        if image.shape[-1] != 3:
            raise ValueError("Input image must be RGB (3 channels).")

        # Prepare mask if connected **and size matches image**
        mask_rgb = None
        if mask is not None:
            # Common dummy mask size in ComfyUI loaders
            if mask.shape[1:3] == (64, 64) and (image.shape[1:3] != (64, 64)):
                print("[ResolutionWrangler] Ignoring 64×64 dummy mask — treating as no mask")
            else:
                if len(mask.shape) == 4:
                    mask = mask.squeeze(-1) if mask.shape[-1] == 1 else mask.mean(dim=-1)
                mask = mask.clamp_(0, 1)
                # Optional: add safety check that spatial dims match (very recommended)
                if mask.shape[1:3] != image.shape[1:3]:
                    print(f"[ResolutionWrangler] Mask size {mask.shape[1:3]} does not match image {image.shape[1:3]} — ignoring mask")
                else:
                    mask_rgb = mask.unsqueeze(-1).repeat(1,1,1,3)

        # Target aspect
        if Aspect_method == "Manual":
            target_num = Aspect_X
            target_den = Aspect_Y
        else:
            target_num, target_den = self.find_closest_ratio(
                image.shape[2], image.shape[1], max_side=24
            )

        # Crop to aspect
        cropped_rgb, aspect_w, aspect_h = self.expand_and_crop(
            image, target_num, target_den, Crop_position, Aspect_method
        )

        if aspect_w != 0 and aspect_h != 0:
            gcd_crop = math.gcd(aspect_w, aspect_h)
            target_num = aspect_w // gcd_crop
            target_den = aspect_h // gcd_crop

        if no_input:
            cropped_rgb = self.add_placeholder_text(cropped_rgb, aspect_w, aspect_h)

        cropped_pixels = aspect_w * aspect_h

        # Determine effective pixel cap
        if Resize_by == "Ratio":
            multiplier = Ratio / 100.0
            effective_pixel_cap = int(cropped_pixels * multiplier)
        else:
            effective_pixel_cap = Max_Resolution_X * Max_Resolution_Y

        print(f"[ResolutionWrangler] Mode: {Resize_by} | Effective cap: {effective_pixel_cap}")

        # Minimal divisible fallback size based on cropped aspect
        min_w = target_num * tolerance
        min_h = target_den * tolerance
        min_pixels = min_w * min_h

        # Downscale if oversized
        base_w = aspect_w
        base_h = aspect_h
        if cropped_pixels > effective_pixel_cap or not (aspect_w % tolerance == 0 and aspect_h % tolerance == 0):
            print(f"[ResolutionWrangler] Oversized ({cropped_pixels} > {effective_pixel_cap}) → downscaling")
            max_units = min(
                base_w // target_num,
                base_h // target_den,
                int((effective_pixel_cap ** 0.5) / max(target_num, target_den, 1))
            )
            while max_units > 0:
                test_w = max_units * target_num
                test_h = max_units * target_den
                if test_w * test_h <= effective_pixel_cap:
                    base_w = test_w
                    base_h = test_h
                    print(f"[ResolutionWrangler] Downscaled to {base_w}×{base_h}")
                    break
                max_units -= 1
            # If downscale resulted in something smaller than min divisible
            if base_w < min_w or base_h < min_h:
                print(f"[ResolutionWrangler] Downscale too aggressive → fallback to minimal divisible {min_w}×{min_h}")
                base_w = min_w
                base_h = min_h

        # Even if no downscale, ensure we start from at least minimal if going to upscale
        base_w = max(base_w, min_w)
        base_h = max(base_h, min_h)

        # Upscale to largest divisible under cap
        final_w, final_h = self.resize_to_divisible(
            base_w, base_h, target_num, target_den, tolerance, effective_pixel_cap
        )

        print(f"[ResolutionWrangler] Final size: {final_w} × {final_h}")

        resized_rgb = self.resize_image(cropped_rgb, final_w, final_h, Resizing_method)

        # Process mask
        if mask_rgb is not None:
            cropped_mask_rgb, _, _ = self.expand_and_crop(
                mask_rgb, target_num, target_den, Crop_position, Aspect_method
            )
            resized_mask_rgb = self.resize_image(cropped_mask_rgb, final_w, final_h, Resizing_method)
            resized_mask = resized_mask_rgb.mean(dim=-1).clamp_(0, 1)
            aspect_mask = cropped_mask_rgb.mean(dim=-1).clamp_(0, 1)
        else:
            aspect_mask = torch.zeros((cropped_rgb.shape[0], aspect_h, aspect_w), dtype=cropped_rgb.dtype, device=cropped_rgb.device)
            resized_mask = torch.zeros((resized_rgb.shape[0], final_h, final_w), dtype=resized_rgb.dtype, device=resized_rgb.device)

        # RGBA: invert mask for alpha
        if mask is not None:
            aspect_alpha = (1.0 - aspect_mask).unsqueeze(-1)
            resized_alpha = (1.0 - resized_mask).unsqueeze(-1)
        else:
            aspect_alpha = torch.ones((cropped_rgb.shape[0], aspect_h, aspect_w, 1), dtype=cropped_rgb.dtype, device=cropped_rgb.device)
            resized_alpha = torch.ones((resized_rgb.shape[0], final_h, final_w, 1), dtype=resized_rgb.dtype, device=resized_rgb.device)

        aspect_rgba = torch.cat([cropped_rgb, aspect_alpha], dim=-1)
        resized_rgba = torch.cat([resized_rgb, resized_alpha], dim=-1)

        # Simplify aspect output
        gcd = math.gcd(target_num, target_den)
        out_aspect_x = target_num // gcd
        out_aspect_y = target_den // gcd

        return (
            cropped_rgb,
            aspect_rgba,
            aspect_mask,
            resized_rgb,
            resized_rgba,
            resized_mask,
            out_aspect_x,
            out_aspect_y,
            aspect_w,
            aspect_h,
            final_w,
            final_h
        )

    # ────────────────────────────────────────────────
    #   Unchanged helper methods (create_black_starter, add_placeholder_text,
    #   find_closest_ratio, expand_and_crop, resize_image)
    # ────────────────────────────────────────────────

    def create_black_starter(self, size=1024):
        h = w = size
        black = torch.zeros((1, h, w, 3), dtype=torch.float32)
        return black

    def add_placeholder_text(self, tensor, width, height):
        arr = (tensor.squeeze(0).cpu().numpy() * 255).round().astype(np.uint8)
        pil_img = Image.fromarray(arr)
        draw = ImageDraw.Draw(pil_img)
        text = "No image is connected to\nResolutionWrangler"
        try:
            font = ImageFont.truetype("arial.ttf", size=240)
        except Exception:
            try:
                font = ImageFont.truetype("arialbd.ttf", size=240)
            except Exception:
                font = ImageFont.load_default()
        lines = text.split('\n')
        max_text_width = width - 20
        font_size = 240
        while font_size > 80:
            try:
                test_font = ImageFont.truetype("arial.ttf", font_size)
            except:
                test_font = font
            line_widths = [draw.textlength(line, font=test_font) for line in lines]
            if max(line_widths) <= max_text_width:
                font = test_font
                break
            font_size -= 20
        if font_size < 80:
            font_size = 80
            try:
                font = ImageFont.truetype("arial.ttf", 80)
            except:
                pass
        line_heights = [font.getbbox("Ay")[3] for _ in lines]
        line_spacing = font_size // 4
        total_text_height = sum(line_heights) + line_spacing * (len(lines) - 1)
        y = (height - total_text_height) // 2
        for i, line in enumerate(lines):
            line_w = draw.textlength(line, font=font)
            x = (width - line_w) // 2
            draw.text((x, y), line, fill=(255, 255, 255), font=font)
            y += line_heights[i] + line_spacing
        new_arr = np.array(pil_img).astype(np.float32) / 255.0
        new_tensor = torch.from_numpy(new_arr).unsqueeze(0)
        return new_tensor

    def find_closest_ratio(self, width, height, max_side=24):
        if width == 0 or height == 0:
            return 1, 1
        ratio = width / height
        landscape = width > height
        best_error = float('inf')
        best_loss = float('inf')
        best_a, best_b = 1, 1
        def continued_fraction(r):
            a = int(r)
            yield a
            frac = r - a
            seen = set()
            while frac > 1e-9 and len(seen) < 30:
                if frac in seen: break
                seen.add(frac)
                r = 1 / frac
                a = int(r)
                yield a
                frac = r - a
        cf = list(continued_fraction(ratio))
        convergents = []
        p0, q0, p1, q1 = 0, 1, 1, 0
        for a in cf:
            p = a * p1 + p0
            q = a * q1 + q0
            convergents.append((p, q))
            p0, q0, p1, q1 = p1, q1, p, q
        for a, b in convergents:
            if max(a, b) > max_side: continue
            if landscape and a <= b: continue
            if not landscape and b <= a: continue
            g = math.gcd(a, b)
            a_r, b_r = a // g, b // g
            if a_r == 0 or b_r == 0: continue
            error = abs(Fraction(a_r, b_r) - Fraction(width, height))
            if landscape:
                loss = abs(width - height * Fraction(a_r, b_r))
            else:
                loss = abs(height - width * Fraction(b_r, a_r))
            if error < best_error or (abs(error - best_error) < 1e-10 and loss < best_loss):
                best_error = error
                best_loss = loss
                best_a, best_b = a_r, b_r
        if best_a == 1 and best_b == 1:
            if landscape:
                best_a, best_b = 4, 3
            else:
                if height > width * 2.5:
                    best_a, best_b = 2, 3
                elif height > width * 1.5:
                    best_a, best_b = 3, 4
                else:
                    best_a, best_b = 3, 4
        return best_a, best_b

    def expand_and_crop(self, image, target_num, target_den, position, aspect_method):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        orig_h, orig_w = image.shape[1], image.shape[2]
        if aspect_method == "Automatic" and orig_w == orig_h:
            return image, orig_w, orig_h
        orig_ratio = orig_w / orig_h
        targ_ratio = target_num / target_den
        if orig_ratio > targ_ratio:
            fit_units = orig_h // target_den
        else:
            fit_units = orig_w // target_num
        new_w = fit_units * target_num
        new_h = fit_units * target_den
        excess_w = orig_w - new_w
        excess_h = orig_h - new_h
        crop_left = crop_right = crop_top = crop_bottom = 0
        if excess_w > 0:
            if "Left" in position:
                crop_right = excess_w
            elif "Right" in position:
                crop_left = excess_w
            else:
                crop_left = excess_w // 2
                crop_right = excess_w - crop_left
        if excess_h > 0:
            if "Top" in position:
                crop_bottom = excess_h
            elif "Bottom" in position:
                crop_top = excess_h
            else:
                crop_top = excess_h // 2
                crop_bottom = excess_h - crop_top
        cropped = image[
            :, crop_top : orig_h - crop_bottom,
            crop_left : orig_w - crop_right,
            :
        ]
        cropped_h = cropped.shape[1]
        cropped_w = cropped.shape[2]
        if cropped_w != new_w or cropped_h != new_h:
            raise ValueError(f"Crop mismatch: expected {new_w}×{new_h}, got {cropped_w}×{cropped_h}")
        return cropped, cropped_w, cropped_h

    def resize_to_divisible(self, base_w, base_h, aspect_num, aspect_den, tolerance, pixel_cap):
        w = base_w
        h = base_h
        max_w, max_h = w, h
        max_pixels = w * h
        while True:
            next_w = w + aspect_num
            next_h = h + aspect_den
            if next_w * next_h > pixel_cap:
                break
            w, h = next_w, next_h
            if w % tolerance == 0 and h % tolerance == 0:
                current_pixels = w * h
                if current_pixels > max_pixels:
                    max_w, max_h = w, h
                    max_pixels = current_pixels
        return max_w, max_h

    def resize_image(self, image, target_w, target_h, method="lanczos"):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        channels = image.shape[-1]
        if method == "lanczos":
            arr = (image[0].cpu().numpy() * 255).round().astype(np.uint8)
            if channels == 1:
                if arr.shape[-1] == 1:
                    arr = arr.squeeze(-1)
                pil_img = Image.fromarray(arr, mode='L')
            elif channels == 3:
                pil_img = Image.fromarray(arr)
            elif channels == 4:
                pil_img = Image.fromarray(arr, mode='RGBA')
            else:
                raise ValueError(f"Unexpected channels {channels}")
            resized_pil = pil_img.resize((target_w, target_h), Image.LANCZOS)
            new_arr = np.array(resized_pil).astype(np.float32) / 255.0
            if channels == 1:
                new_arr = new_arr[..., np.newaxis]
            return torch.from_numpy(new_arr).unsqueeze(0)
        mode_map = {
            "nearest-exact": InterpolationMode.NEAREST,
            "bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC,
        }
        interpolation = mode_map.get(method, InterpolationMode.BICUBIC)
        antialias = interpolation in [InterpolationMode.BILINEAR, InterpolationMode.BICUBIC]
        img_c_h_w = image[0].permute(2, 0, 1)
        resized = tv_resize(img_c_h_w, (target_h, target_w), interpolation=interpolation, antialias=antialias)
        return resized.permute(1, 2, 0).unsqueeze(0)
