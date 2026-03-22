import os
import torch
import re
import folder_paths
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoModelForCausalLM
)
#from .imagefunc import log, tensor2pil


smollm2_repo_local = {
    "SmolLM2-135M-Instruct": "smol/SmolLM2-135M-Instruct",
    "SmolLM2-360M-Instruct": "smol/SmolLM2-360M-Instruct",
    "SmolLM2-1.7B-Instruct": "smol/SmolLM2-1.7B-Instruct",
}

smolvlm_all_local = {
    "SmolVLM-Instruct": "smol/SmolVLM-Instruct",
    "SmolVLM2-256M-Video-Instruct": "smol/SmolVLM2-256M-Video-Instruct",
    "SmolVLM2-500M-Video-Instruct": "smol/SmolVLM2-500M-Video-Instruct",
    "SmolVLM2-2.2B-Instruct": "smol/SmolVLM2-2.2B-Instruct",
}


def log(message:str, message_type:str='info'):
    name = '_Fossiel_QoL'

    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    else:
        message = '\033[1;33m' + message + '\033[m'
    print(f"# Fossiel_QoL_Nodes: {name} -> {message}")

try:
    from cv2.ximgproc import guidedFilter
except ImportError as e:
    # print(e)
    log(f"Cannot import name 'guidedFilter' from 'cv2.ximgproc'"
        f"\nA few nodes cannot works properly, while most nodes are not affected. Please REINSTALL package 'opencv-contrib-python'."
        f"\nFor detail refer to \033[4mhttps://github.com/chflame163/ComfyUI_LayerStyle/issues/5\033[0m")

def pil2tensor(image:Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def split_lm2_content(text: str) -> str:
    tag_str = "<|im_start|>assistant"
    if tag_str in text:
        ret_str = text.split(tag_str)[-1].strip()
        return ret_str.replace("<|im_end|>", "")
    else:
        return text


def split_vlm_content(text: str) -> str:
    tag_str = "Assistant:"
    if tag_str in text:
        ret_str = text.split(tag_str)[-1].strip()
        return ret_str
    if text.startswith("Assistant:"):
        return text.split("Assistant:", 1)[-1].strip()
    return text.strip()


class Fossiel_Load_SmolLM2_Model:
    def __init__(self):
        self.NODE_NAME = 'LoadSmolLM2Model'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        smollm2_model_list = list(smollm2_repo_local.keys())
        device_list = ['cuda', 'cpu']
        dtype_list = ["bf16", "fp32"]
        return {"required":
            {
                "model": (smollm2_model_list, {"default": None}),
                "dtype": (dtype_list, {"default": "bf16"}),
                "device": (device_list, {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("SmolLM2_MODEL",)
    RETURN_NAMES = ("smolLM2_model", )
    FUNCTION = "load_smollm2_model"
    CATEGORY = 'Fossiel_QoL_Nodes'

    def load_smollm2_model(self, model, dtype, device):
        subfolder = smollm2_repo_local[model]
        model_path = os.path.join(folder_paths.models_dir, subfolder)

        print(f"[DEBUG LoadSmolLM2] Trying path: {model_path}")
        print(f"[DEBUG] Exists? {os.path.isdir(model_path)}")
        if os.path.isdir(model_path):
            print(f"[DEBUG] Files: {os.listdir(model_path)}")

        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
        ).to(device)

        return ({"tokenizer": tokenizer, "model": model_obj, "dtype": torch_dtype, "device": device},)


class Fossiel_Load_SmolVLM_Model:
    def __init__(self):
        self.NODE_NAME = 'LoadSmolVLMModel'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(smolvlm_all_local.keys())
        device_list = ['cuda', 'cpu']
        dtype_list = ["bf16", "fp32"]
        return {"required":
            {
                "model": (model_list, {"default": "SmolVLM2-500M-Video-Instruct"}),
                "dtype": (dtype_list, {"default": "bf16"}),
                "device": (device_list, {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("SmolVLM_MODEL",)
    RETURN_NAMES = ("smolVLM_model", )
    FUNCTION = "load_smolvlm_model"
    CATEGORY = 'Fossiel_QoL_Nodes'

    def load_smolvlm_model(self, model, dtype, device):
        subfolder = smolvlm_all_local[model]
        model_path = os.path.join(folder_paths.models_dir, subfolder)

        print(f"[DEBUG LoadSmolVLM] Trying path: {model_path}")
        print(f"[DEBUG] Exists? {os.path.isdir(model_path)}")
        if os.path.isdir(model_path):
            print(f"[DEBUG] Files: {sorted(os.listdir(model_path))}")

        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32

        processor = AutoProcessor.from_pretrained(model_path)

        is_smolvlm2 = "SmolVLM2" in model

        if is_smolvlm2:
            model_class = AutoModelForImageTextToText
            family = "smolvlm2"
        else:
            model_class = AutoModelForVision2Seq
            family = "smolvlm"

        try:
            import flash_attn
            use_flash_attention = device == "cuda" and torch_dtype == torch.bfloat16
            attn_impl = "flash_attention_2" if use_flash_attention else "eager"
            print(f"[DEBUG] Using attn: {attn_impl}")
        except ImportError as e:
            print(e, ", use 'eager' instead.")
            attn_impl = "eager"

        model_obj = model_class.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            _attn_implementation=attn_impl,
        ).to(device)

        return ({
            "processor": processor,
            "model": model_obj,
            "dtype": torch_dtype,
            "device": device,
            "family": family,
            "model_name": model,
        },)


class Fossiel_SmolLM2:
    def __init__(self):
        self.NODE_NAME = 'SmolLM2'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
                    "smolLM2_model": ("SmolLM2_MODEL",),
                    "max_new_tokens": ("INT", {"default": 480, "min": 1, "max": 4096}),
                    "do_sample": ("BOOLEAN", {"default": True}),
                    "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "step": 0.1}),
                    "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.1}),
                    "system_prompt": ("STRING", {"default": "You are an expert prompt engineer for AI image generation. Your task is to take a very short or basic idea and expand it into a rich, detailed, high-quality prompt optimized for modern image and video generation models. Avoid writing a story. Avoid naming characters.", "multiline": True}),
                    "user_prompt": ("STRING", {"default": "Basic prompt.", "multiline": True}),
                },
            }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text", )
    FUNCTION = "smollm2"
    CATEGORY = 'Fossiel_QoL_Nodes'

    def smollm2(self, smolLM2_model, max_new_tokens, do_sample, temperature, top_p, system_prompt, user_prompt):
        tokenizer = smolLM2_model["tokenizer"]
        model = smolLM2_model["model"]
        device = smolLM2_model["device"]
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=do_sample)
        ret_text = tokenizer.decode(outputs[0])
        ret_text = split_lm2_content(ret_text)
        log(f"{self.NODE_NAME} response is: {ret_text}")
        return (ret_text,)


class Fossiel_SmolVLM_Classic:
    def __init__(self):
        self.NODE_NAME = 'SmolVLM_Classic'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
                    "image": ("IMAGE",),
                    "smolVLM_model": ("SmolVLM_MODEL",),
                    "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 4096}),
                    "user_prompt": ("STRING", {"default": "describe this image", "multiline": True}),
                },
            }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text", )
    FUNCTION = "smolvlm_classic"
    OUTPUT_IS_LIST = (True,)
    CATEGORY = 'Fossiel_QoL_Nodes'

    def smolvlm_classic(self, image, smolVLM_model, max_new_tokens, user_prompt):
        if smolVLM_model["family"] != "smolvlm":
            raise ValueError(f"Wrong model family ({smolVLM_model['family']}). Use Fossiel_SmolVLM2 node for SmolVLM2.")

        processor = smolVLM_model["processor"]
        model = smolVLM_model["model"]
        device = smolVLM_model["device"]
        torch_dtype = smolVLM_model["dtype"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            },
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

        ret_text = []
        for i in image:
            img = tensor2pil(i).convert("RGB")
            inputs = processor(text=prompt, images=[img], return_tensors="pt")
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch_dtype)
            inputs = {k: v.to(device=device) for k, v in inputs.items()}

            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            result = split_vlm_content(generated_texts[0])
            ret_text.append(result)
            log(f"{self.NODE_NAME} response is: {result}")

        return (ret_text,)


class Fossiel_SmolVLM2:
    def __init__(self):
        self.NODE_NAME = 'SmolVLM2'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
                    "smolVLM_model": ("SmolVLM_MODEL",),
                    "max_new_tokens": ("INT", {"default": 384, "min": 1, "max": 4096}),
                    "user_prompt": ("STRING", {"default": "Describe this image in detail including descriptions of the poses and facial features and facial expressions of characters.", "multiline": True}),
                },
                "optional": {
                    "image": ("IMAGE",),
                }
            }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text", )
    FUNCTION = "smolvlm2"
    OUTPUT_IS_LIST = (True,)
    CATEGORY = 'Fossiel_QoL_Nodes'

    def smolvlm2(self, smolVLM_model, max_new_tokens, user_prompt, image=None):
        if smolVLM_model["family"] != "smolvlm2":
            raise ValueError(f"Wrong model family ({smolVLM_model.get('family', 'unknown')}). Use Fossiel_SmolVLM_Classic for original SmolVLM.")

        if image is None or len(image) == 0:
            raise ValueError("Image input is required for Fossiel_SmolVLM2 node.")

        processor = smolVLM_model["processor"]
        model = smolVLM_model["model"]
        device = smolVLM_model["device"]
        torch_dtype = smolVLM_model["dtype"]

        is_video_like = len(image) > 1
        log(f"{self.NODE_NAME} treating input as {'video-like (multi-frame)' if is_video_like else 'single image'}")

        content = []

        for _ in range(len(image)):
            content.append({"type": "image"})

        content.append({"type": "text", "text": user_prompt})

        messages = [
            {
                "role": "user",
                "content": content
            },
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

        ret_text = []

        pil_images = [tensor2pil(img_tensor).convert("RGB") for img_tensor in image]

        inputs = processor(text=prompt, images=pil_images, return_tensors="pt")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch_dtype)
        inputs = {k: v.to(device=device) for k, v in inputs.items()}

        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        result = split_vlm_content(generated_texts[0])
        ret_text.append(result)
        log(f"{self.NODE_NAME} response is: {result}")

        return (ret_text,)
