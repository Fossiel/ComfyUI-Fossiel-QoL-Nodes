import comfy.latent_formats

class FossielSensorSwitchImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image_true": ("IMAGE",),
                "image_false": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "switch"
    CATEGORY = "Fossiel/QoL"

    def switch(self, switch, image_true=None, image_false=None):
        if image_true is not None and image_false is None:
            return (image_true,)
        if image_false is not None and image_true is None:
            return (image_false,)
        if image_true is not None and image_false is not None:
            return (image_true if switch else image_false,)
        return (None,)

class FossielSensorSwitchClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "clip_true": ("CLIP",),
                "clip_false": ("CLIP",),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "switch"
    CATEGORY = "Fossiel/QoL"

    def switch(self, switch, clip_true=None, clip_false=None):
        if clip_true is not None and clip_false is None:
            return (clip_true,)
        if clip_false is not None and clip_true is None:
            return (clip_false,)
        if clip_true is not None and clip_false is not None:
            return (clip_true if switch else clip_false,)
        return (None,)

class FossielSensorSwitchConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "conditioning_true": ("CONDITIONING",),
                "conditioning_false": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "switch"
    CATEGORY = "Fossiel/QoL"

    def switch(self, switch, conditioning_true=None, conditioning_false=None):
        if conditioning_true is not None and conditioning_false is None:
            return (conditioning_true,)
        if conditioning_false is not None and conditioning_true is None:
            return (conditioning_false,)
        if conditioning_true is not None and conditioning_false is not None:
            return (conditioning_true if switch else conditioning_false,)
        return (None,)

# ------------------------------------------------------------
# FossielSensorSwitchLatent – universal version (normal + WAN)
# ------------------------------------------------------------
class FossielSensorSwitchLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "latent_true": ("LATENT",),
                "latent_false": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "switch"
    CATEGORY = "Fossiel/QoL"

    def _to_raw(self, latent_dict):
        """
        Convert a WAN latent dict to the raw tensor that KSampler expects.
        Normal latents are returned unchanged.
        """
        samples = latent_dict["samples"]

        # WAN nodes store the format object under this key
        fmt = latent_dict.get("latent_format")
        if fmt is not None and isinstance(fmt, (comfy.latent_formats.Wan21,
                                               comfy.latent_formats.Wan22)):
            # `process_in` undoes the scaling that `process_out` applied
            samples = fmt.process_in(samples)

        # Return a *new* dict – never mutate the original
        return {"samples": samples}

    def switch(self, switch, latent_true=None, latent_false=None):
        # ------------------------------------------------------------------
        # 1. Early-out cases (exactly the same as your original node)
        # ------------------------------------------------------------------
        if latent_true is not None and latent_false is None:
            return (self._to_raw(latent_true),)
        if latent_false is not None and latent_true is None:
            return (self._to_raw(latent_false),)
        if latent_true is None and latent_false is None:
            return ({"samples": None},)   # or raise – you decide

        # ------------------------------------------------------------------
        # 2. Both inputs present → pick one and normalise it
        # ------------------------------------------------------------------
        chosen = latent_true if switch else latent_false
        return (self._to_raw(chosen),)

class FossielSensorSwitchMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mask_true": ("MASK",),
                "mask_false": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "switch"
    CATEGORY = "Fossiel/QoL"

    def switch(self, switch, mask_true=None, mask_false=None):
        if mask_true is not None and mask_false is None:
            return (mask_true,)
        if mask_false is not None and mask_true is None:
            return (mask_false,)
        if mask_true is not None and mask_false is not None:
            return (mask_true if switch else mask_false,)
        return (None,)

class FossielSensorSwitchModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model_true": ("MODEL",),
                "model_false": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "switch"
    CATEGORY = "Fossiel/QoL"

    def switch(self, switch, model_true=None, model_false=None):
        if model_true is not None and model_false is None:
            return (model_true,)
        if model_false is not None and model_true is None:
            return (model_false,)
        if model_true is not None and model_false is not None:
            return (model_true if switch else model_false,)
        return (None,)

class FossielSensorSwitchVAE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vae_true": ("VAE",),
                "vae_false": ("VAE",),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "switch"
    CATEGORY = "Fossiel/QoL"

    def switch(self, switch, vae_true=None, vae_false=None):
        if vae_true is not None and vae_false is None:
            return (vae_true,)
        if vae_false is not None and vae_true is None:
            return (vae_false,)
        if vae_true is not None and vae_false is not None:
            return (vae_true if switch else vae_false,)
        return (None,)

class FossielSensorKSamplerSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch_model": ("BOOLEAN", {"default": True}),
                "switch_positive": ("BOOLEAN", {"default": True}),
                "switch_negative": ("BOOLEAN", {"default": True}),
                "switch_latent": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model_true": ("MODEL",),
                "model_false": ("MODEL",),
                "positive_true": ("CONDITIONING",),
                "positive_false": ("CONDITIONING",),
                "negative_true": ("CONDITIONING",),
                "negative_false": ("CONDITIONING",),
                "latent_true": ("LATENT",),
                "latent_false": ("LATENT",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("model", "positive", "negative", "latent")
    FUNCTION = "switch"
    CATEGORY = "Fossiel/QoL"

    def _to_raw_latent(self, latent_dict):
        """Convert WAN latent to raw format expected by KSampler; normal latents unchanged."""
        if latent_dict is None:
            return None
        samples = latent_dict.get("samples")
        if samples is None:
            return latent_dict

        fmt = latent_dict.get("latent_format")
        if fmt is not None and isinstance(fmt, (comfy.latent_formats.Wan21, comfy.latent_formats.Wan22)):
            samples = fmt.process_in(samples)

        # Return new dict to avoid mutation
        return {"samples": samples}

    def switch(self,
               switch_model,
               switch_positive, switch_negative, switch_latent,
               model_true=None, model_false=None,
               positive_true=None, positive_false=None,
               negative_true=None, negative_false=None,
               latent_true=None, latent_false=None):

        # === MODEL SWITCH ===
        if model_true is not None and model_false is None:
            model_out = model_true
        elif model_false is not None and model_true is None:
            model_out = model_false
        elif model_true is not None and model_false is not None:
            model_out = model_true if switch_model else model_false
        else:
            model_out = None  # or raise? up to you

        # === POSITIVE ===
        if positive_true is not None and positive_false is None:
            positive_out = positive_true
        elif positive_false is not None and positive_true is None:
            positive_out = positive_false
        elif positive_true is not None and positive_false is not None:
            positive_out = positive_true if switch_positive else positive_false
        else:
            positive_out = None

        # === NEGATIVE ===
        if negative_true is not None and negative_false is None:
            negative_out = negative_true
        elif negative_false is not None and negative_true is None:
            negative_out = negative_false
        elif negative_true is not None and negative_false is not None:
            negative_out = negative_true if switch_negative else negative_false
        else:
            negative_out = None

        # === LATENT (with WAN fix) ===
        if latent_true is not None and latent_false is None:
            latent_out = self._to_raw_latent(latent_true)
        elif latent_false is not None and latent_true is None:
            latent_out = self._to_raw_latent(latent_false)
        elif latent_true is not None and latent_false is not None:
            chosen = latent_true if switch_latent else latent_false
            latent_out = self._to_raw_latent(chosen)
        else:
            latent_out = None

        return (model_out, positive_out, negative_out, latent_out)
