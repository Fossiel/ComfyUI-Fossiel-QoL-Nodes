import math
import torch
import comfy.samplers

class FossielCentralControl_v2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Manual_Resolution_Width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "Manual_Resolution_Height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "Aspect_Tolerance": (["2", "4", "8", "16", "32", "64"], {"default": "4", "tooltip": "Divisor for snapping resolution to multiples (lower = closer aspect, higher = VAE compatibility"}),
                "Target_Resolution_Width": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "Target_Resolution_Height": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "Max_Resolution_Width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "Max_Resolution_Height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "Video_Length": ("INT", {"default": 81, "min": 1, "max": 301, "step": 4}),
                "Overlap_Length": ("INT", {"default": 8, "min": 0, "max": 80, "step": 1}),
                "Frame_Rate": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 200.0, "step": 0.01}),
                "steps": ("INT", {"default": 4, "min": 2, "max": 100, "step": 2}),  # UI step=2 for evens only
                "Dual_sampler_split_method": (["Repeat_last_step", "Next_step"], {"default": "Repeat_last_step"}),
                "Step_split_ratio": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "Sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "The sampling algorithm to use for generation"}),
                "Scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple", "tooltip": "The scheduler algorithm to control sampling step distribution"}),
                "Naming_Suffix": (["None", "Current Gen"], {"default": "None"}),
                "Project_Name": ("STRING", {"default": "", "multiline": False}),
                "Scene_Name": ("STRING", {"default": "", "multiline": False}),
                "Delimiter": ("STRING", {"default": "_", "multiline": False}),
                "Current_Generation": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    RETURN_TYPES = (
        "INT", "INT", "INT", "INT", "INT", "FLOAT", "INT", "FLOAT", "INT", "FLOAT", "INT", "FLOAT", "FLOAT", "INT", "INT",
        "STRING", "INT", "INT", "INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "BOOLEAN"
    )
    RETURN_NAMES = (
        "Man_Res_Width", "Man_Res_Height", "Calc_Res_Width", "Calc_Res_Height",
        "Vid_Length", "Vid_Duration", "Overlap", "Overlap_Duration", "Start_Index", "Start_Time", "End_Index", "End_Time",
        "Frame_Rate", "Gen_Num", "batch_count", "File_Name", "steps", "KSampler_1_End_Step", "KSampler_2_Start_Step",
        "cfg", "Sampler", "Scheduler", "First_Gen_Batch_Switch"
    )
    FUNCTION = "process"
    CATEGORY = "Fossiel/QoL"
    OUTPUT_NODE = True

    def process(self, Manual_Resolution_Width, Manual_Resolution_Height, Aspect_Tolerance,
                Target_Resolution_Width, Target_Resolution_Height,
                Max_Resolution_Width, Max_Resolution_Height,
                Video_Length, Overlap_Length, Frame_Rate, 
                steps, Dual_sampler_split_method, Step_split_ratio, cfg, Sampler, Scheduler,
                Naming_Suffix, Project_Name, Scene_Name, Delimiter, Current_Generation,
                image=None):

        # Passthroughs
        man_w = max(64, Manual_Resolution_Width)  # Clamp to sane min
        man_h = max(64, Manual_Resolution_Height)
        overlap = max(0, Overlap_Length)
        gen_num = max(1, Current_Generation)
        aspect_tolerance = int(Aspect_Tolerance)  # Convert COMBO string to int

        # Snap Video_Length to valid (1 or 4k+1)
        if Video_Length == 1:
            vid_length = 1
        else:
            k = math.ceil((Video_Length - 1) / 4)
            vid_length = 4 * k + 1
        vid_length = min(301, vid_length)  # Cap

        # Snap steps to next even multiple of 2 (ceiling, backup for manual entry)
        processed_steps = steps if steps % 2 == 0 else math.ceil(steps / 2) * 2

        # Handle optional IMAGE: Extract batch_count and override target dims if present
        batch_count = 0
        target_w = max(1, Target_Resolution_Width)
        target_h = max(1, Target_Resolution_Height)
        if image is not None and isinstance(image, torch.Tensor) and image.numel() > 0:
            batch_count = image.shape[0]  # Batch size
            target_h = image.shape[1]     # Height from first batch item
            target_w = image.shape[2]     # Width from first batch item

        # Calc Res: Approximate aspect, mod Aspect_Tolerance dims, max area <= max_pixels
        max_w = max(64, Max_Resolution_Width)
        max_h = max(64, Max_Resolution_Height)
        max_pixels = max_w * max_h

        # Calculate target aspect ratio
        if target_h == 0:
            print("Warning: Target height is zero. Falling back to 64x64.")
            calc_w, calc_h = 64, 64
        else:
            aspect_ratio = target_w / target_h

            # Start with approximate width based on max_pixels and aspect ratio
            calc_w = round(math.sqrt(max_pixels * aspect_ratio) / aspect_tolerance) * aspect_tolerance
            calc_h = round(calc_w / aspect_ratio / aspect_tolerance) * aspect_tolerance

            # Adjust to be <= max_pixels
            current_pixels = calc_w * calc_h
            max_iterations = 10  # Prevent infinite loops
            i = 0
            while current_pixels > max_pixels and i < max_iterations:
                calc_w -= aspect_tolerance
                calc_h = round(calc_w / aspect_ratio / aspect_tolerance) * aspect_tolerance
                current_pixels = calc_w * calc_h
                i += 1
            # Ensure strictly <= max_pixels
            while current_pixels > max_pixels and i < max_iterations:
                calc_w -= aspect_tolerance
                calc_h = round(calc_w / aspect_ratio / aspect_tolerance) * aspect_tolerance
                current_pixels = calc_w * calc_h
                i += 1

            # Clamp to min/max dims (64 to 4096)
            calc_w = max(64, min(4096, calc_w))
            calc_h = max(64, min(4096, calc_h))

            # Final check: If still invalid, fallback
            if calc_w < 64 or calc_h < 64 or calc_w * calc_h > max_pixels:
                print(f"Warning: Could not satisfy target {target_w}x{target_h} under {max_pixels} pixels. Falling back to 64x64.")
                calc_w, calc_h = 64, 64

        # Indices (inclusive end): Treat gen=1 as first (offset 0)
        effective_gen = gen_num - 1
        start_idx = effective_gen * (vid_length - overlap)
        end_idx = start_idx + vid_length - 1

        # Clamp indices sane (no negatives, no absurd overshoots)
        start_idx = max(0, start_idx)
        end_idx = max(start_idx, end_idx)

        # First_Gen_Batch_Switch: True (1) if gen_num == 1, else False (0)
        first_gen_batch_switch = 1 if gen_num == 1 else 0

        # === NAMING LOGIC (NEW & IMPROVED) ===
        parts = []
        if Project_Name:
            parts.append(Project_Name)
        if Scene_Name:
            parts.append(Scene_Name)
        if Naming_Suffix == "Current Gen" and parts:  # Only add Gen if there's a base name
            padded_gen = f"{gen_num:03d}"
            parts.append(f"Gen{padded_gen}")

        # Join with delimiter only if needed
        if len(parts) > 1:
            File_Name = Delimiter.join(parts)
        elif len(parts) == 1:
            File_Name = parts[0]
        else:
            File_Name = ""  # Both empty → empty string

        # Step splitter with ratio
        KSampler_1_End_Step = round(processed_steps * Step_split_ratio)

        # Cap KSampler_1_End_Step based on Dual_sampler_split_method
        max_end_step = processed_steps - 1 if Dual_sampler_split_method == "Repeat_last_step" else processed_steps - 2
        KSampler_1_End_Step = min(KSampler_1_End_Step, max_end_step)
        KSampler_1_End_Step = max(1, KSampler_1_End_Step)  # Ensure atest 1 step

        # Set KSampler_2_Start_Step based on Dual_sampler_split_method
        if Dual_sampler_split_method == "Repeat_last_step":
            KSampler_2_Start_Step = KSampler_1_End_Step
        else:  # Next_step
            KSampler_2_Start_Step = KSampler_1_End_Step + 1

        # Round Frame_Rate to 6 decimals
        Frame_Rate = round(Frame_Rate, 6)

        # Calculate durations (in seconds, 6 decimal places)
        Vid_Duration = round((1.000000 / Frame_Rate) * vid_length, 6)
        Overlap_Duration = round((1.000000 / Frame_Rate) * overlap, 6)

        # Calculate Start & End Times
        Start_Time = round((1.000000 / Frame_Rate) * start_idx, 6)
        End_Time = round(Start_Time + Vid_Duration, 6)

        return (
            man_w, man_h, calc_w, calc_h, vid_length, Vid_Duration, overlap, Overlap_Duration, start_idx, Start_Time, end_idx, End_Time,
            Frame_Rate, gen_num, batch_count, File_Name, processed_steps, KSampler_1_End_Step, KSampler_2_Start_Step,
            cfg, Sampler, Scheduler, first_gen_batch_switch
        )
