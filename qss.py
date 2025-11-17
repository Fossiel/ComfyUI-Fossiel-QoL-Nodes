from typing import Dict, Tuple

class FossielQwenSizeStabilizer:
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "Aspect_Ratio": (
                    [
                        "1:1",
                        "2:1",
                        "3:2",
                        "4:3",
                        "16:9 (Under)",
                        "16:9 (Over)",
                        "16:10",
                        "21:9",
                    ],
                    {
                        "default": "1:1"
                    }
                ),
                "Orientation": (
                    ["Landscape", "Portrait"],
                    {
                        "default": "Landscape"
                    }
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("Width", "Height")
    FUNCTION = "get_resolution"
    CATEGORY = "Fossiel/Utils"

    # Hardcoded resolution table — your exact numbers, Qwen-optimized
    RESOLUTIONS = {
        "1:1":           {"L": (1024, 1024), "P": (1024, 1024)},
        "2:1":           {"L": (1148,  724), "P": ( 724, 1148)},
        "3:2":           {"L": (1254,  836), "P": ( 836, 1254)},
        "4:3":           {"L": (1184,  888), "P": ( 888, 1184)},
        "16:9 (Under)":  {"L": (1344,  756), "P": ( 756, 1344)},
        "16:9 (Over)":   {"L": (1376,  774), "P": ( 774, 1376)},
        "16:10":         {"L": (1296,  810), "P": ( 810, 1296)},
        "21:9":          {"L": (1554,  666), "P": ( 666, 1554)},
    }

    def get_resolution(self, Aspect_Ratio: str, Orientation: str) -> Tuple[int, int]:
        key = "P" if Orientation == "Portrait" else "L"
        w, h = self.RESOLUTIONS[Aspect_Ratio][key]
        return (w, h)
