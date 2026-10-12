import re

class FossielTextGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "default": "",
                }),
                "max_length": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 32768,
                }),
                "sampling_mode": (["on", "off"], {
                    "default": "on",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.01,
                    "max": 2.0,
                    "step": 0.000001,
                }),
                "top_k": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 1000,
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.05,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
                "presence_penalty": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                }),
            },
            "optional": {
                "image0": ("IMAGE", {
                    "tooltip": "Reference image slot 0. Use 'image0' in prompt.",
                }),
                "image1": ("IMAGE", {
                    "tooltip": "Reference image slot 1. Use 'image1' in prompt.",
                }),
                "image2": ("IMAGE", {
                    "tooltip": "Reference image slot 2. Use 'image2' in prompt.",
                }),
                "image3": ("IMAGE", {
                    "tooltip": "Reference image slot 3. Use 'image3' in prompt.",
                }),
                "image4": ("IMAGE", {
                    "tooltip": "Reference image slot 4. Use 'image4' in prompt.",
                }),
                "image5": ("IMAGE", {
                    "tooltip": "Reference image slot 5. Use 'image5' in prompt.",
                }),
                "image6": ("IMAGE", {
                    "tooltip": "Reference image slot 6. Use 'image6' in prompt.",
                }),
                "image7": ("IMAGE", {
                    "tooltip": "Reference image slot 7. Use 'image7' in prompt.",
                }),
                "video": ("IMAGE", {
                    "tooltip": "Video frames as image batch. Assumed to be 24 FPS; subsampled to 1 FPS internally.",
                }),
                "audio": ("AUDIO",),
                "thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Operate in thinking mode if the model supports it.",
                }),
                "use_default_template": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use the built-in system prompt/template if the model has one.",
                }),
                "mtp": (["auto", "off", "2", "3", "4", "5"], {
                    "default": "auto",
                    "tooltip": "Speculative decoding with the checkpoint's multi-token-prediction head. No effect without MTP weights.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate"
    CATEGORY = "Fossiel/text"

    def generate(
        self,
        clip,
        prompt,
        max_length,
        sampling_mode,
        temperature=0.7,
        top_k=64,
        top_p=0.95,
        min_p=0.05,
        repetition_penalty=1.05,
        seed=0,
        presence_penalty=0.0,
        image0=None,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        video=None,
        audio=None,
        thinking=False,
        use_default_template=True,
        mtp="auto",
    ):
        mtp_arg = False if mtp == "off" else (True if mtp == "auto" else int(mtp))

        ref_images = [
            img for img in (
                image0, image1, image2, image3,
                image4, image5, image6, image7,
            )
            if img is not None
        ]

        tokenize_kwargs = {
            "skip_template": not use_default_template,
            "min_length": 1,
            "thinking": thinking,
            "video": video,
            "audio": audio,
        }

        if len(ref_images) > 1:
            tokenize_kwargs["images"] = [img for img in ref_images]
            tokenize_kwargs["image"] = None
        elif len(ref_images) == 1:
            tokenize_kwargs["image"] = ref_images[0]
        else:
            tokenize_kwargs["image"] = None

        try:
            tokens = clip.tokenize(prompt, **tokenize_kwargs)
        except TypeError:
            tokenize_kwargs.pop("images", None)
            if ref_images:
                tokenize_kwargs["image"] = ref_images[0]
            tokens = clip.tokenize(prompt, **tokenize_kwargs)

        do_sample = sampling_mode == "on"

        generated_ids = clip.generate(
            tokens,
            do_sample=do_sample,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed if do_sample else None,
            mtp=mtp_arg,
        )

        generated_text = clip.decode(generated_ids)
        return (generated_text,)
