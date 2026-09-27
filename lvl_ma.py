import numpy as np
from PIL import Image, ImageEnhance
import torch
import cv2

class FossielLevelMatcherAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "Match": (
                    [
                        "All",
                        "Brightness Only",
                        "Contrast Only",
                        "Saturation Only",
                        "Brightness & Contrast",
                        "Brightness & Saturation",
                        "Contrast & Saturation",
                    ],
                    {"default": "All"}
                ),
                "Brightness_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "Contrast_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "Saturation_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 2.0, "step": 0.001}),
                "Saturation_algorithm": (["Global", "Midtone-Weighted"], {"default": "Midtone-Weighted"}),
                
                "Use_Influence_Map": ("BOOLEAN", {"default": True}),
                "Invert_Influence_Map": ("BOOLEAN", {"default": False}),
                "Influence_Map_Strength": ("FLOAT", {"default": 1.000, "min": 0.000, "max": 2.000, "step": 0.001}),
            },
            "optional": {
                "Brightness_Ref": ("IMAGE",),
                "Contrast_Ref": ("IMAGE",),
                "Saturation_Ref": ("IMAGE",),
                "Influence_Map": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_levels"
    CATEGORY = "Fossiel"

    def adjust_levels(self, image, Match,
                      Brightness_offset, Contrast_offset, Saturation_offset,
                      Saturation_algorithm,
                      Use_Influence_Map, Invert_Influence_Map, Influence_Map_Strength,
                      Brightness_Ref=None, Contrast_Ref=None, Saturation_Ref=None,
                      Influence_Map=None):

        image_np = image.cpu().numpy()
        batch_size = image_np.shape[0]

        do_brightness = "Brightness" in Match or Match == "All"
        do_contrast = "Contrast" in Match or Match == "All"
        do_saturation = "Saturation" in Match or Match == "All"

        input_first = image_np[0]

        brightness_factor = 1.0
        contrast_factor = 1.0
        saturation_factor = 1.0

        # ==================== ORIGINAL FACTOR CALCULATION (UNTOUCHED) ====================
        # Brightness
        if do_brightness:
            if Brightness_Ref is not None:
                ref = Brightness_Ref.cpu().numpy()[0]
                ref_mean = np.mean(ref)
                input_mean = np.mean(input_first)
                base = ref_mean / input_mean if input_mean > 0 else 1.0
                brightness_factor = base * (1.0 + Brightness_offset)
            else:
                brightness_factor = 1.0 + Brightness_offset

        # Contrast
        if do_contrast:
            if Contrast_Ref is not None:
                ref = Contrast_Ref.cpu().numpy()[0]
                ref_std = np.std(ref)
                input_std = np.std(input_first)
                base = ref_std / input_std if input_std > 0 else 1.0
                contrast_factor = base * (1.0 + Contrast_offset)
            else:
                contrast_factor = 1.0 + Contrast_offset

        # Saturation
        if do_saturation:
            if Saturation_Ref is not None:
                ref = Saturation_Ref.cpu().numpy()[0]
                ref_uint8 = (np.clip(ref, 0, 1) * 255).astype(np.uint8)
                ref_hsv = cv2.cvtColor(ref_uint8, cv2.COLOR_RGB2HSV)
                ref_sat = ref_hsv[:, :, 1]
            else:
                ref_sat = None

            input_uint8 = (np.clip(input_first, 0, 1) * 255).astype(np.uint8)
            input_hsv = cv2.cvtColor(input_uint8, cv2.COLOR_RGB2HSV)
            input_sat = input_hsv[:, :, 1]

            if Saturation_algorithm == "Midtone-Weighted":
                input_v = input_hsv[:, :, 2].astype(np.float32) / 255.0
                mask = (input_v > 0.15) & (input_v < 0.85)
                masked_input = input_sat[mask] if np.any(mask) else input_sat
                input_sat_std = np.std(masked_input)

                if ref_sat is not None:
                    ref_v = ref_hsv[:, :, 2].astype(np.float32) / 255.0
                    ref_mask = (ref_v > 0.15) & (ref_v < 0.85)
                    masked_ref = ref_sat[ref_mask] if np.any(ref_mask) else ref_sat
                    ref_sat_std = np.std(masked_ref)
                else:
                    ref_sat_std = input_sat_std
            else:  # Global
                input_sat_std = np.std(input_sat)
                ref_sat_std = np.std(ref_sat) if ref_sat is not None else input_sat_std

            base = ref_sat_std / input_sat_std if input_sat_std > 0 else 1.0
            saturation_factor = base * (1.0 + Saturation_offset)
        else:
            saturation_factor = 1.0 + Saturation_offset

        # Safety clamps
        brightness_factor = max(0.01, brightness_factor)
        contrast_factor = max(0.01, contrast_factor)
        saturation_factor = max(0.01, saturation_factor)

        # ==================== INFLUENCE MAP HANDLING ====================
        use_influence = (Use_Influence_Map and Influence_Map is not None)

        if use_influence:
            # Convert to grayscale
            inf_np = Influence_Map.cpu().numpy()[0]
            if inf_np.shape[-1] == 3:
                inf_gray = cv2.cvtColor((inf_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
            elif inf_np.shape[-1] == 4:
                inf_gray = cv2.cvtColor((inf_np[:,:,:3] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
            else:
                inf_gray = np.mean(inf_np, axis=-1) if len(inf_np.shape) == 3 else inf_np.squeeze()

            # Stretch resize (no aspect ratio preservation)
            target_h, target_w = image_np.shape[1], image_np.shape[2]
            inf_resized = cv2.resize(inf_gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Invert logic (counter-intuitive)
            if not Invert_Influence_Map:
                inf_resized = 1.0 - inf_resized

            # === Influence_Map_Strength as brightness on the map ===
            if Influence_Map_Strength <= 1.0:
                influence = inf_resized * Influence_Map_Strength
            else:
                # Push towards white
                influence = inf_resized + (1.0 - inf_resized) * (Influence_Map_Strength - 1.0)

            influence = np.clip(influence, 0.0, 1.0)
            influence = influence[..., None]  # For broadcasting
        else:
            influence = None

        # ==================== APPLY ADJUSTMENTS ====================
        output_images = []
        for i in range(batch_size):
            img = image_np[i]
            pil_img = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
            current = pil_img

            if do_brightness:
                current = ImageEnhance.Brightness(current).enhance(brightness_factor)
            if do_contrast:
                current = ImageEnhance.Contrast(current).enhance(contrast_factor)
            if do_saturation:
                current = ImageEnhance.Color(current).enhance(saturation_factor)

            adjusted_np = np.array(current) / 255.0

            # Apply influence
            if use_influence:
                orig_np = np.clip(img, 0.0, 1.0)
                adjusted_np = orig_np * (1.0 - influence) + adjusted_np * influence

            if img.shape[-1] == 1:  # Preserve grayscale
                adjusted_np = adjusted_np[:, :, None]

            output_images.append(adjusted_np)

        output_tensor = torch.from_numpy(np.stack(output_images)).float()
        return (output_tensor,)
