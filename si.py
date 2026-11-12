import torch
import os
import folder_paths
from PIL import Image
import numpy as np
from comfy.cli_args import args

class FossielSolidifyImage:
    """
    Saves the input image(s) to temp folder and immediately loads them back.
    This forces the image through PIL/PNG pipeline (same as saving + reloading).
    Supports single and batched images.
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        os.makedirs(self.output_dir, exist_ok=True)
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "solidify"
    CATEGORY = "Fossiel/Utils"
    
    def solidify(self, images):
        # images shape: (batch, height, width, channels)
        batch_size = images.shape[0]
        output_images = []
        
        for i in range(batch_size):
            # Convert tensor to PIL Image
            img_tensor = images[i]
            img_np = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
            
            # Save to temp folder
            temp_filename = f"fossiel_solidify_{i:06d}.png"
            temp_path = os.path.join(self.output_dir, temp_filename)
            pil_image.save(temp_path, format='PNG', compress_level=1)
            
            # Load back immediately
            loaded_pil = Image.open(temp_path).convert("RGB")
            loaded_np = np.array(loaded_pil).astype(np.float32) / 255.0
            loaded_tensor = torch.from_numpy(loaded_np)
            
            output_images.append(loaded_tensor)
        
        # Stack back into batch
        final_images = torch.stack(output_images)
        
        return (final_images,)
