import math


class FossielPixelBudget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Pixel_budget": ("INT", {
                    "default": 27100000,
                    "min": 1,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "step": 1,
                }),
                "Frame_count": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "step": 1,
                }),
                "Round": (["Up", "Down"], {"default": "Up"}),
                "Force_Video_Compatibility": ("BOOLEAN", {"default": True}),
                "Video_format": (["Wan", "LTX", "Minimax"], {"default": "Wan"}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("Max_Resolution", "Frame_Count")
    FUNCTION = "calculate"
    CATEGORY = "utils"

    def snap_frame_count(self, frame_count, video_format):
        """Snap frame_count upward to the format grid. Never shortens below a valid target."""
        n = int(frame_count)

        if video_format == "Wan":
            # 4k + 1  →  1, 5, 9, 13, ...
            if n < 1:
                return 1
            if (n - 1) % 4 == 0:
                return n
            return ((n - 1) // 4 + 1) * 4 + 1

        if video_format == "LTX":
            # 8k + 1  →  1, 9, 17, 25, ..., 81, 97, 121, ...
            if n < 1:
                return 1
            if (n - 1) % 8 == 0:
                return n
            return ((n - 1) // 8 + 1) * 8 + 1

        # Minimax: 17k + 5  →  5, 22, 39, 56, ..., 124, ...
        if n < 5:
            return 5
        if (n - 5) % 17 == 0:
            return n
        k = math.ceil((n - 5) / 17.0)
        return 17 * k + 5

    def calculate(self, Pixel_budget, Frame_count, Round, Force_Video_Compatibility, Video_format):
        pixel_budget = int(Pixel_budget)
        frame_count = int(Frame_count)

        if Force_Video_Compatibility:
            frame_count = self.snap_frame_count(frame_count, Video_format)

        if pixel_budget < frame_count:
            raise ValueError(
                f"Impossible operation: The pixel budget of ({pixel_budget}) is smaller than "
                f"the frame count of ({frame_count}). Please either increase the Pixel_budget "
                f"or reduce the Frame_count and try again."
            )

        side = (pixel_budget / float(frame_count)) ** 0.5
        if Round == "Up":
            max_resolution = int(math.ceil(side))
        else:
            max_resolution = int(math.floor(side))
            if max_resolution < 1:
                max_resolution = 1

        return (max_resolution, frame_count)
