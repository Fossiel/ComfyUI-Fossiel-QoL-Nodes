import torch
# "AnyType" class from https://github.com/pythongosssss/ComfyUI-Custom-Scripts
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
def min_(tensor_list):
    # return the element-wise min of the tensor list.
    x = torch.stack(tensor_list)
    mn = x.min(axis=0)[0]
    return torch.clamp(mn, min=0)
def max_(tensor_list):
    # return the element-wise max of the tensor list.
    x = torch.stack(tensor_list)
    mx = x.max(axis=0)[0]
    return torch.clamp(mx, max=1)
def expand_mask(mask, expand, tapered_corners):
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in mask:
        output = m.numpy()
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)
    return torch.stack(out, dim=0)
def parse_string_to_list(s):
    elements = s.split(',')
    result = []
    def parse_number(s):
        try:
            if '.' in s:
                return float(s)
            else:
                return int(s)
        except ValueError:
            return 0
    def decimal_places(s):
        if '.' in s:
            return len(s.split('.')[1])
        return 0
    for element in elements:
        element = element.strip()
        if '...' in element:
            start, rest = element.split('...')
            end, step = rest.split('+')
            decimals = decimal_places(step)
            start = parse_number(start)
            end = parse_number(end)
            step = parse_number(step)
            current = start
            if (start > end and step > 0) or (start < end and step < 0):
                step = -step
            while current <= end:
                result.append(round(current, decimals))
                current += step
        else:
            result.append(round(parse_number(element), decimal_places(element)))
    return result
any = AnyType("*")

class FossielTimeFormatter:
    """
    Converts between frames and time with multiple formatting options.
    """
   
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "Frames": (any, { "default": 0 }),
                "Time_sec": (any, { "default": 0.00000 }),
            },
            "required": {
                "Frames_per_sec": ("INT", {"default": 16, "min": 1, "max": 2000, "step": 1}),
                "Time_format": ([
                    "Frames",
                    "Time (sec)",
                    "HH:MM:SS:FF",
                    "HH:MM:SS:msec",
                    "HH:MM:SS",
                    "Total Seconds",
                    "Total Frames"
                ], {"default": "HH:MM:SS:FF"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("Formatted_Time_(str)", "Filename_Friendly")
    FUNCTION = "format_time"
    CATEGORY = "utils"

    def format_time(self, Frames_per_sec, Time_format,
                    Frames=None, Time_sec=None):
        # Error if both inputs are provided
        if Frames is not None and Time_sec is not None:
            raise ValueError("Time Formatter Error: Both the frame and time input ports are receiving data. "
                           "Please use only the one or the other.")

        fps = float(Frames_per_sec)

        # Determine input values
        if Frames is not None:
            frames = int(Frames)
            seconds = frames / fps
        elif Time_sec is not None:
            seconds = float(Time_sec)
            frames = int(round(seconds * fps))
        else:
            # No input connected
            frames = 0
            seconds = 0.0

        # Pass-through when input type matches selected format
        if Time_format == "Frames" and Frames is not None:
            formatted = str(frames)
        elif Time_format == "Time (sec)" and Time_sec is not None:
            formatted = f"{seconds:.3f}"
        else:
            # Format according to selected output
            if Time_format == "Frames":
                formatted = str(frames)
            elif Time_format == "Time (sec)":
                formatted = f"{seconds:.3f}"
            elif Time_format == "HH:MM:SS:FF":
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                frame_part = frames % Frames_per_sec
                formatted = f"{hours:02d}:{minutes:02d}:{secs:02d}:{frame_part:02d}"
            elif Time_format == "HH:MM:SS:msec":
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                milliseconds = int(round((seconds % 1) * 1000))
                formatted = f"{hours:02d}:{minutes:02d}:{secs:02d}:{milliseconds:03d}"
            elif Time_format == "HH:MM:SS":
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                formatted = f"{hours:02d}:{minutes:02d}:{secs:02d}"
            elif Time_format == "Total Seconds":
                formatted = f"{seconds:.3f}"
            elif Time_format == "Total Frames":
                formatted = str(frames)
            else:
                formatted = str(frames)

        # Create Filename_Friendly version (replace : with - for time formats)
        if Time_format in ["HH:MM:SS:FF", "HH:MM:SS:msec", "HH:MM:SS"]:
            filename_friendly = formatted.replace(":", "-")
        else:
            filename_friendly = formatted

        return (formatted, filename_friendly)
