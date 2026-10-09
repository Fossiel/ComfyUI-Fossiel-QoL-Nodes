import math

class FossielPixelBudget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Pixel_budget": ("INT", {
                    "default": 1024 * 1024,
                    "min": 1,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                }),
                "Frame_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                }),
                "Round": (["Up", "Down"], {
                    "default": "Up",
                }),
                "Force_Video_Compatibility": ("BOOLEAN", {
                    "default": True,
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("Max_Resolution", "Frame_Count")
    FUNCTION = "calculate"
    CATEGORY = "Fossiel"

    def calculate(self, Pixel_budget, Frame_count, Round, Force_Video_Compatibility):
        # First determine the final Frame_Count (with optional snap)
        output_frame_count = Frame_count
        if Force_Video_Compatibility:
            if output_frame_count % 4 != 1:
                output_frame_count = ((output_frame_count // 4) + 1) * 4 + 1

        # Error check against the Frame_Count that will actually be used
        if Pixel_budget < output_frame_count:
            raise Exception(
                f"Impossible operation: The pixel budget of ({Pixel_budget}) is smaller than the frame count of ({output_frame_count}). "
                f"Please either increase the Pixel_budget or reduce the Frame_count and try again."
            )

        # Resolution is now calculated from the (possibly snapped) Frame_Count
        raw = (Pixel_budget / output_frame_count) ** 0.5

        if Round == "Up":
            max_resolution = math.ceil(raw)
        else:
            max_resolution = math.floor(raw)

        return (max_resolution, output_frame_count)
