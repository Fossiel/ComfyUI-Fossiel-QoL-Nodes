import numpy as np
from PIL import Image, ImageEnhance
import torch

class FossielLevelMatchmaker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "Match": (["Brightness & Contrast", "Brightness Only", "Contrast Only"], {"default": "Brightness & Contrast"}),
                "Brightness_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "Contrast_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_contrast_brightness"
    CATEGORY = "Fossiel"

    def adjust_contrast_brightness(self, image, Match, Brightness_offset, Contrast_offset, reference_image=None):
        # Convert ComfyUI image tensor to numpy (batch, height, width, channels)
        image_np = image.cpu().numpy()
        batch_size = image_np.shape[0]

        # Default factors when no reference is provided
        Default_brightness = 1.00
        Default_contrast = 1.00

        brightness_factor = Default_brightness
        contrast_factor = Default_contrast

        do_brightness = "Brightness" in Match
        do_contrast = "Contrast" in Match

        if reference_image is not None and (do_brightness or do_contrast):
            # Use first image from reference batch
            ref_np = reference_image.cpu().numpy()[0]

            if do_brightness:
                # Brightness matching (mean)
                ref_brightness = np.mean(ref_np)
                input_brightness = np.mean(image_np[0])
                base_brightness_factor = ref_brightness / input_brightness if input_brightness > 0 else 1.0
                brightness_factor = base_brightness_factor * (1.0 + Brightness_offset)
            else:
                brightness_factor = Default_brightness * (1.0 + Brightness_offset)

            if do_contrast:
                # Contrast matching (std)
                ref_contrast = np.std(ref_np)
                input_contrast = np.std(image_np[0])
                base_contrast_factor = ref_contrast / input_contrast if input_contrast > 0 else 1.0
                contrast_factor = base_contrast_factor * (1.0 + Contrast_offset)
            else:
                contrast_factor = Default_contrast * (1.0 + Contrast_offset)

        else:
            # No reference: just apply targets with offsets
            if do_brightness:
                brightness_factor = Default_brightness * (1.0 + Brightness_offset)
            if do_contrast:
                contrast_factor = Default_contrast * (1.0 + Contrast_offset)

        # Safety clamps
        brightness_factor = max(0.01, brightness_factor)
        contrast_factor = max(0.01, contrast_factor)

        # Process each image in the batch
        output_images = []
        for i in range(batch_size):
            img = image_np[i]
            img_pil = Image.fromarray((img * 255).astype(np.uint8))

            current = img_pil

            # Apply brightness first (if enabled)
            if do_brightness:
                enhancer = ImageEnhance.Brightness(current)
                current = enhancer.enhance(brightness_factor)

            # Then apply contrast (if enabled)
            if do_contrast:
                enhancer = ImageEnhance.Contrast(current)
                current = enhancer.enhance(contrast_factor)

            adjusted_np = np.array(current) / 255.0
            if img.shape[-1] == 1:  # Preserve grayscale if input was grayscale
                adjusted_np = adjusted_np[:, :, None]

            output_images.append(adjusted_np)

        output_tensor = torch.from_numpy(np.stack(output_images)).float()
        return (output_tensor,)
