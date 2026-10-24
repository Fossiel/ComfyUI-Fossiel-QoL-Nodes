import math
import comfy.samplers


class FossielCentralControl_v3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Resolution_Width_1": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "Resolution_Height_1": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "Resolution_Width_2": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "Resolution_Height_2": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "Snap_Video_Length": (["No Snap", "LTX", "MiniMax", "Wan"], {"default": "No Snap"}),
                "Video_Length": ("INT", {"default": 81, "min": 1, "max": 301, "step": 1}),
                "Overlap_Length": ("INT", {"default": 8, "min": 0, "max": 80, "step": 1}),
                "Frame_Rate": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 200.0, "step": 0.01}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1}),
                "Dual_sampler_split_method": (["Repeat_last_step", "Next_step"], {"default": "Repeat_last_step"}),
                "Step_split_ratio": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 1.00, "min": 0.01, "max": 20.00, "step": 0.01}),
                "Sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "The sampling algorithm to use for generation"}),
                "Scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple", "tooltip": "The scheduler algorithm to control sampling step distribution"}),
                "Naming_Suffix": (["None", "Current Gen"], {"default": "None"}),
                "Path_1": ("STRING", {"default": "", "multiline": False}),
                "Path_2": ("STRING", {"default": "", "multiline": False}),
                "Project_Name": ("STRING", {"default": "", "multiline": False}),
                "Scene_Name": ("STRING", {"default": "", "multiline": False}),
                "Delimiter": ("STRING", {"default": "_", "multiline": False}),
                "Current_Generation": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = (
        "INT", "INT", "INT", "INT", "INT", "FLOAT", "INT", "FLOAT", "INT", "FLOAT", "INT", "FLOAT",
        "FLOAT", "INT", "INT", "STRING", "STRING", "STRING", "INT", "INT", "INT", "FLOAT",
        comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "BOOLEAN",
    )
    RETURN_NAMES = (
        "Resolution_Width_1", "Resolution_Height_1", "Resolution_Width_2", "Resolution_Height_2",
        "Vid_Length", "Vid_Duration", "Overlap", "Overlap_Duration", "Start_Index", "Start_Time",
        "End_Index", "End_Time", "Frame_Rate", "Gen_Num", "Batch_Count", "Path_1", "Path_2", "File_Name",
        "steps", "KSampler_1_End_Step", "KSampler_2_Start_Step", "cfg", "Sampler", "Scheduler",
        "First_Gen_Batch_Switch",
    )
    FUNCTION = "process"
    CATEGORY = "Fossiel/QoL"
    OUTPUT_NODE = True

    def snap_video_length(self, video_length, snap_mode):
        """Snap frame count upward to the selected model grid (same rules as FossielPixelBudget)."""
        n = int(video_length)

        if snap_mode == "No Snap":
            return max(1, min(301, n))

        if snap_mode == "Wan":
            # 4k + 1 → 1, 5, 9, 13, ...
            if n < 1:
                n = 1
            if (n - 1) % 4 != 0:
                n = ((n - 1) // 4 + 1) * 4 + 1
            return min(301, n)

        if snap_mode == "LTX":
            # 8k + 1 → 1, 9, 17, 25, ..., 81, 97, 121, ...
            if n < 1:
                n = 1
            if (n - 1) % 8 != 0:
                n = ((n - 1) // 8 + 1) * 8 + 1
            return min(301, n)

        # MiniMax: 17k + 5 → 5, 22, 39, 56, ..., 124, ...
        if n < 5:
            n = 5
        elif (n - 5) % 17 != 0:
            k = math.ceil((n - 5) / 17.0)
            n = 17 * k + 5
        return min(301, n)

    def process(
        self,
        Resolution_Width_1, Resolution_Height_1,
        Resolution_Width_2, Resolution_Height_2,
        Snap_Video_Length, Video_Length, Overlap_Length, Frame_Rate,
        steps, Dual_sampler_split_method, Step_split_ratio, cfg, Sampler, Scheduler,
        Naming_Suffix, Path_1, Path_2, Project_Name, Scene_Name, Delimiter, Current_Generation,
    ):
        # Resolution passthroughs
        res_w1 = max(64, Resolution_Width_1)
        res_h1 = max(64, Resolution_Height_1)
        res_w2 = max(64, Resolution_Width_2)
        res_h2 = max(64, Resolution_Height_2)

        overlap = max(0, Overlap_Length)
        gen_num = max(1, Current_Generation)

        # Snap Video_Length per format (affects duration, indices, times)
        vid_length = self.snap_video_length(Video_Length, Snap_Video_Length)

        # Snap steps to next even multiple of 2 (ceiling, backup for manual entry)
        processed_steps = steps if steps % 2 == 0 else math.ceil(steps / 2) * 2

        # Image input removed — batch count is unused but kept for output compatibility
        Batch_Count = 0

        # Indices (inclusive end): Treat gen=1 as first (offset 0)
        effective_gen = gen_num - 1
        start_idx = effective_gen * (vid_length - overlap)
        end_idx = start_idx + vid_length - 1
        start_idx = max(0, start_idx)
        end_idx = max(start_idx, end_idx)

        first_gen_batch_switch = 1 if gen_num == 1 else 0

        # Naming
        parts = []
        if Project_Name:
            parts.append(Project_Name)
        if Scene_Name:
            parts.append(Scene_Name)
        if Naming_Suffix == "Current Gen" and parts:
            padded_gen = f"{gen_num:03d}"
            parts.append(f"Gen{padded_gen}")
        if len(parts) > 1:
            File_Name = Delimiter.join(parts)
        elif len(parts) == 1:
            File_Name = parts[0]
        else:
            File_Name = ""

        # Step splitter with ratio
        KSampler_1_End_Step = round(processed_steps * Step_split_ratio)
        max_end_step = processed_steps - 1 if Dual_sampler_split_method == "Repeat_last_step" else processed_steps - 2
        KSampler_1_End_Step = min(KSampler_1_End_Step, max_end_step)
        KSampler_1_End_Step = max(1, KSampler_1_End_Step)
        if Dual_sampler_split_method == "Repeat_last_step":
            KSampler_2_Start_Step = KSampler_1_End_Step
        else:
            KSampler_2_Start_Step = KSampler_1_End_Step + 1

        Frame_Rate = round(Frame_Rate, 6)
        Vid_Duration = round((1.000000 / Frame_Rate) * vid_length, 6)
        Overlap_Duration = round((1.000000 / Frame_Rate) * overlap, 6)
        Start_Time = round((1.000000 / Frame_Rate) * start_idx, 6)
        End_Time = round(Start_Time + Vid_Duration, 6)

        return (
            res_w1, res_h1, res_w2, res_h2,
            vid_length, Vid_Duration, overlap, Overlap_Duration,
            start_idx, Start_Time, end_idx, End_Time,
            Frame_Rate, gen_num, Batch_Count,
            Path_1, Path_2, File_Name,
            processed_steps, KSampler_1_End_Step, KSampler_2_Start_Step,
            cfg, Sampler, Scheduler, first_gen_batch_switch,
        )
