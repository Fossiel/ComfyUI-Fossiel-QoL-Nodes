import numpy as np
import cv2
import torch

class FossielVideoDeflicker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # ComfyUI image batch (B, H, W, C) tensor in 0-1 float
                "window_size": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "mode": (["mean", "median"], {"default": "mean"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "deflicker_batch"
    CATEGORY = "Fossiel"

    def deflicker_batch(self, images, window_size, mode, strength):
        # Convert ComfyUI batch tensor to numpy: (B, H, W, C) float32 0-1
        batch = images.cpu().numpy()
        num_frames = batch.shape[0]

        if num_frames == 0:
            return (images,)

        # Convert to uint8 for OpenCV (0-255)
        frames_uint8 = (batch * 255).astype(np.uint8)

        brightness_history = []
        output_frames = []

        for i in range(num_frames):
            frame = frames_uint8[i]

            # Measure brightness on grayscale version
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # ComfyUI is RGB order
            if mode == "mean":
                current_brightness = np.mean(gray)
            else:
                current_brightness = np.median(gray)

            # Update rolling history
            brightness_history.append(current_brightness)
            if len(brightness_history) > window_size:
                brightness_history.pop(0)

            # Compute reference brightness
            if len(brightness_history) < 2:
                ref_brightness = current_brightness
            else:
                if mode == "mean":
                    ref_brightness = np.mean(brightness_history)
                else:
                    ref_brightness = np.median(brightness_history)

            # Correction factor with strength control
            if current_brightness > 0:
                factor = (ref_brightness / current_brightness) ** strength
            else:
                factor = 1.0

            # Apply multiplicative correction
            adjusted = cv2.convertScaleAbs(frame, alpha=factor, beta=0)

            # Back to float 0-1 and preserve channel order
            adjusted_float = adjusted.astype(np.float32) / 255.0
            output_frames.append(adjusted_float)

        # Stack back into tensor
        output_batch = np.stack(output_frames)
        output_tensor = torch.from_numpy(output_batch)

        return (output_tensor,)
