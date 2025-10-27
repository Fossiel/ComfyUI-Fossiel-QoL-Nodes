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

    def switch(self, switch, latent_true=None, latent_false=None):
        if latent_true is not None and latent_false is None:
            return (latent_true,)
        if latent_false is not None and latent_true is None:
            return (latent_false,)
        if latent_true is not None and latent_false is not None:
            return (latent_true if switch else latent_false,)
        return (None,)

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
                "switch_positive": ("BOOLEAN", {"default": True}),
                "switch_negative": ("BOOLEAN", {"default": True}),
                "switch_latent": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "positive_true": ("CONDITIONING",),
                "positive_false": ("CONDITIONING",),
                "negative_true": ("CONDITIONING",),
                "negative_false": ("CONDITIONING",),
                "latent_true": ("LATENT",),
                "latent_false": ("LATENT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "switch"
    CATEGORY = "Fossiel/QoL"

    def switch(self, switch_positive, switch_negative, switch_latent, positive_true=None, positive_false=None, negative_true=None, negative_false=None, latent_true=None, latent_false=None):
        # Handle positive prompt
        if positive_true is not None and positive_false is None:
            positive_out = positive_true
        elif positive_false is not None and positive_true is None:
            positive_out = positive_false
        elif positive_true is not None and positive_false is not None:
            positive_out = positive_true if switch_positive else positive_false
        else:
            positive_out = None

        # Handle negative prompt
        if negative_true is not None and negative_false is None:
            negative_out = negative_true
        elif negative_false is not None and negative_true is None:
            negative_out = negative_false
        elif negative_true is not None and negative_false is not None:
            negative_out = negative_true if switch_negative else negative_false
        else:
            negative_out = None

        # Handle latent
        if latent_true is not None and latent_false is None:
            latent_out = latent_true
        elif latent_false is not None and latent_true is None:
            latent_out = latent_false
        elif latent_true is not None and latent_false is not None:
            latent_out = latent_true if switch_latent else latent_false
        else:
            latent_out = None

        return (positive_out, negative_out, latent_out)
