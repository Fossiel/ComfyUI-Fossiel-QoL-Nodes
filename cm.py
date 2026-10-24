# Taken from kijai/ComfyUI-KJNodes (nodes/image_nodes.py — ColorMatch).
# Modified: image_target is the top port so that when this node is bypassed,
# ComfyUI passes the target image through instead of the reference.

import os
import logging
import torch
from concurrent.futures import ThreadPoolExecutor


class FossielColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_target": ("IMAGE",),
                "image_ref": ("IMAGE",),
                "method": (
                    ["mkl", "hm", "reinhard", "mvgd", "hm-mvgd-hm", "hm-mkl-hm"],
                    {"default": "mkl"},
                ),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "multithread": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = "Fossiel/image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch"
    DESCRIPTION = """
color-matcher enables color transfer across images which comes in handy for automatic
color-grading of photographs, paintings and film sequences as well as light-field
and stopmotion corrections.

The methods behind the mappings are based on the approach from Reinhard et al.,
the Monge-Kantorovich Linearization (MKL) as proposed by Pitie et al. and our analytical solution
to a Multi-Variate Gaussian Distribution (MVGD) transfer in conjunction with classical histogram
matching. As shown below our HM-MVGD-HM compound outperforms existing methods.
https://github.com/hahnec/color-matcher/
"""

    def colormatch(self, image_target, image_ref, method, strength=1.0, multithread=True):
        # Skip unnecessary processing
        if strength == 0:
            return (image_target,)

        try:
            from color_matcher import ColorMatcher
        except ImportError as e:
            raise ImportError(
                "Can't import color-matcher, did you install requirements.txt? "
                "Manual install: pip install color-matcher"
            ) from e

        image_ref = image_ref.cpu()
        image_target = image_target.cpu()
        batch_size = image_target.size(0)

        images_target = image_target.squeeze()
        images_ref = image_ref.squeeze()

        image_ref_np = images_ref.numpy()
        images_target_np = images_target.numpy()

        def process(i):
            cm = ColorMatcher()
            image_target_np_i = images_target_np if batch_size == 1 else images_target[i].numpy()
            image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
            try:
                # Avoid potential blur when only the fully color-matched image is used
                image_result = cm.transfer(src=image_target_np_i, ref=image_ref_np_i, method=method)
                if strength != 1:
                    image_result = image_target_np_i + strength * (image_result - image_target_np_i)
                return torch.from_numpy(image_result)
            except Exception as e:
                logging.warning(f"Thread {i} error: {e}")
                return torch.from_numpy(image_target_np_i)  # fallback

        if multithread and batch_size > 1:
            max_threads = min(os.cpu_count() or 1, batch_size)
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                out = list(executor.map(process, range(batch_size)))
        else:
            out = [process(i) for i in range(batch_size)]

        out = torch.stack(out, dim=0).to(torch.float32)
        out.clamp_(0, 1)
        return (out,)
