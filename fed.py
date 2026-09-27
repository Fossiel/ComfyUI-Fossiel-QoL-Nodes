import comfy_kitchen

class FossielForceEager:
    """
    Passthrough node that forces pure-PyTorch paths for Qwen 3.5 / TextGenerate
    by disabling the fused comfy-kitchen CUDA kernels that require a newer driver.

    Wire any value through this node before TextGenerate so it is guaranteed
    to run first. Does not alter the data and does not affect normal Qwen Image 2.1
    image generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": ("*",),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("anything",)
    FUNCTION = "force_eager"
    CATEGORY = "Fossiel/QwenHelpers"
    DESCRIPTION = (
        "Forces Qwen 3.5 TextGenerate (prompt expander) onto pure-PyTorch paths "
        "by disabling fused comfy-kitchen CUDA kernels. Use as a passthrough "
        "before TextGenerate when the NVIDIA driver is too old. "
        "Does not affect normal Qwen Image 2.1 image generation."
    )

    def force_eager(self, anything):
        if not getattr(comfy_kitchen, "_fossiel_force_eager_applied", False):

            def _never(*args, **kwargs):
                return False

            def _disabled(*args, **kwargs):
                raise RuntimeError(
                    "FossielForceEager: fused comfy-kitchen CUDA kernel disabled "
                    "(old driver). The pure-PyTorch fallback should have been used."
                )

            # 1. Primary availability checks used by the model code
            comfy_kitchen.gated_delta_decode_is_available = _never
            if hasattr(comfy_kitchen, "flash_attention_decode_is_available"):
                comfy_kitchen.flash_attention_decode_is_available = _never

            # 2. Lower-level module patches
            try:
                import comfy_kitchen.gated_delta as gd
                gd.is_available = _never
                gd.deltanet_conv_step = _disabled
                gd.gated_delta_decode_fused = _disabled
            except Exception:
                pass

            try:
                import comfy_kitchen.flash_attention as fa
                if hasattr(fa, "is_available"):
                    fa.is_available = _never
                if hasattr(fa, "flash_attention_decode"):
                    fa.flash_attention_decode = _disabled
            except Exception:
                pass

            # 3. Disable the CUDA backend in the registry so dispatch cannot pick it
            try:
                if hasattr(comfy_kitchen, "registry") and hasattr(comfy_kitchen.registry, "disable"):
                    comfy_kitchen.registry.disable("cuda")
                elif hasattr(comfy_kitchen, "disable_backend"):
                    comfy_kitchen.disable_backend("cuda")
            except Exception:
                pass

            comfy_kitchen._fossiel_force_eager_applied = True
            print("[FossielForceEager] Fused comfy-kitchen CUDA kernels disabled for this session.")

        return (anything,)
