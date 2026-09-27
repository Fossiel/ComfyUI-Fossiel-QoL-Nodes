import os
import json
import torch
import folder_paths
import comfy.utils
from comfy.cli_args import args
import hashlib
import safetensors.torch

class FossielSaveLatent:
    SEARCH_ALIASES = ["export latent"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "output_path": ("STRING", {"default": "", "multiline": False}),
                "filename": ("STRING", {"default": "", "multiline": False}),
                "overwrite": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Fossiel"

    def save(self, samples, output_path="", filename="", overwrite=False, prompt=None, extra_pnginfo=None):
        output_path = output_path.strip()
        filename = filename.strip()

        if not output_path:
            raise ValueError("FossielSaveLatent: output_path is empty. Please provide a directory path.")

        if os.path.splitext(output_path)[1]:
            raise ValueError(
                f"FossielSaveLatent: output_path appears to be a file, not a directory:\n{output_path}\n"
                "Please provide only a directory path."
            )

        if not filename:
            raise ValueError("FossielSaveLatent: filename is empty. Please provide a filename.")

        if "/" in filename or "\\" in filename:
            raise ValueError(
                f"FossielSaveLatent: filename must contain only the filename, not a path.\n"
                f"Received: {filename}"
            )

        if not filename.lower().endswith(".latent"):
            filename += ".latent"

        full_path = os.path.join(output_path, filename)

        if os.path.exists(full_path) and not overwrite:
            raise ValueError(
                f"FossielSaveLatent: File already exists:\n{full_path}\n"
                "Either set 'overwrite' to True or choose a different filename."
            )

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        prompt_info = ""
        if prompt is not None:
            prompt_info = json.dumps(prompt)

        metadata = None
        if not args.disable_metadata:
            metadata = {"prompt": prompt_info}
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        output = {}
        output["latent_tensor"] = samples["samples"].contiguous()
        output["latent_format_version_0"] = torch.tensor([])

        comfy.utils.save_torch_file(output, full_path, metadata=metadata)

        results = [{
            "filename": filename,
            "subfolder": "",
            "type": "output"
        }]

        return {"ui": {"latents": results}, "result": (samples,)}


class FossielLoadLatent:
    SEARCH_ALIASES = ["import latent", "open latent"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "full_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    CATEGORY = "Fossiel"
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"

    def load(self, full_path):
        if not full_path or full_path.strip() == "":
            raise ValueError("FossielLoadLatent: full_path is empty. Please provide a complete path and filename (e.g. /path/to/file.latent).")

        full_path = full_path.strip()

        if not os.path.isfile(full_path):
            raise ValueError(f"FossielLoadLatent: File not found: {full_path}")

        latent = safetensors.torch.load_file(full_path, device="cpu")
        multiplier = 1.0
        if "latent_format_version_0" not in latent:
            multiplier = 1.0 / 0.18215
        samples = {"samples": latent["latent_tensor"].float() * multiplier}
        return (samples,)

    @classmethod
    def IS_CHANGED(s, full_path):
        if not full_path or full_path.strip() == "":
            return ""
        full_path = full_path.strip()
        if not os.path.isfile(full_path):
            return ""
        m = hashlib.sha256()
        with open(full_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, full_path):
        if not full_path or full_path.strip() == "":
            return "full_path is empty. Please provide a complete path and filename (e.g. /path/to/file.latent)."
        full_path = full_path.strip()
        if not os.path.isfile(full_path):
            return f"File not found: {full_path}"
        return True


class FossielLatentListener:
    """
    Early-validation guard node.
    Performs the same path/filename checks as FossielSaveLatent (without overwrite).
    If everything is fine it simply passes the input through unchanged.
    Place this before the KSampler (or any early node) to abort before generation starts.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "anything": ("*",),          # any type pass-through
                "output_path": ("STRING", {"default": "", "multiline": False}),
                "filename": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("anything",)
    FUNCTION = "listen"
    CATEGORY = "Fossiel"
    DESCRIPTION = "Validates output path and filename before generation. Passes input through if OK."

    def listen(self, anything, output_path="", filename=""):
        output_path = output_path.strip()
        filename = filename.strip()

        # 1. Empty output_path
        if not output_path:
            raise ValueError(
                "FossielLatentListener: output_path is empty.\n"
                "Please provide a directory path."
            )

        # 2. output_path looks like a file
        if os.path.splitext(output_path)[1]:
            raise ValueError(
                f"FossielLatentListener: output_path appears to be a file, not a directory:\n"
                f"{output_path}\n"
                "Please provide only a directory path."
            )

        # 3. Empty filename
        if not filename:
            raise ValueError(
                "FossielLatentListener: filename is empty.\n"
                "Please provide a filename."
            )

        # 4. filename contains directory separators
        if "/" in filename or "\\" in filename:
            raise ValueError(
                f"FossielLatentListener: filename must contain only the filename, not a path.\n"
                f"Received: {filename}"
            )

        # 5. Auto-add .latent if missing
        if not filename.lower().endswith(".latent"):
            filename += ".latent"

        full_path = os.path.join(output_path, filename)

        # 6. File already exists → error (no overwrite option on purpose)
        if os.path.exists(full_path):
            raise ValueError(
                f"FossielLatentListener: File already exists:\n{full_path}\n"
                "Change the filename or delete the existing file before running."
            )

        # Everything OK → pass the input through unchanged
        return (anything,)
