from .fccl import FossielCentralControlLite
from .fcc_v2 import FossielCentralControl_v2

NODE_CLASS_MAPPINGS = {
    "FossielCentralControlLite": FossielCentralControlLite,
    "FossielCentralControl_v2": FossielCentralControl_v2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FossielCentralControlLite": "Fossiel Central Control Lite",
    "FossielCentralControl_v2": "Fossiel Central Control v2",
}
