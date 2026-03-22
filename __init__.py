from .deflicker import FossielVideoDeflicker
from .dpks import FossielDenoisePrecisionKSampler
from .fccl import FossielCentralControlLite
from .fcc_v2 import FossielCentralControl_v2
from .lvl_m import FossielLevelMatcher
from .resw import FossielResolutionWrangler
from .reswxp import FossielResolutionWranglerXP
from .seqw import FossielSequenceWrangler
from .webpw import FossielWebPWrangler
from .senswitch import (
    FossielSensorSwitchImage,
    FossielSensorSwitchClip,
    FossielSensorSwitchConditioning,
    FossielSensorSwitchLatent,
    FossielSensorSwitchMask,
    FossielSensorSwitchModel,
    FossielSensorSwitchVAE,
    FossielSensorKSamplerSwitch
)
from .svlm2 import (
    Fossiel_Load_SmolLM2_Model,
    Fossiel_Load_SmolVLM_Model,
    Fossiel_SmolLM2,
    Fossiel_SmolVLM_Classic,
    Fossiel_SmolVLM2,
)

NODE_CLASS_MAPPINGS = {
    "FossielCentralControlLite": FossielCentralControlLite,
    "FossielCentralControl_v2": FossielCentralControl_v2,
    "FossielDenoisePrecisionKSampler": FossielDenoisePrecisionKSampler,
    "FossielLevelMatcher": FossielLevelMatcher,
    "FossielResolutionWrangler": FossielResolutionWrangler,
    "FossielResolutionWranglerXP": FossielResolutionWranglerXP,
    "Fossiel_Sensor_Switch_Image": FossielSensorSwitchImage,
    "Fossiel_Sensor_Switch_Clip": FossielSensorSwitchClip,
    "Fossiel_Sensor_Switch_Conditioning": FossielSensorSwitchConditioning,
    "Fossiel_Sensor_Switch_Latent": FossielSensorSwitchLatent,
    "Fossiel_Sensor_Switch_Mask": FossielSensorSwitchMask,
    "Fossiel_Sensor_Switch_Model": FossielSensorSwitchModel,
    "Fossiel_Sensor_Switch_VAE": FossielSensorSwitchVAE,
    "Fossiel_Sensor_KSampler_Switch": FossielSensorKSamplerSwitch,
    "FossielSequenceWrangler": FossielSequenceWrangler,
    "Fossiel_LoadSmolLM2Model": Fossiel_Load_SmolLM2_Model,
    "Fossiel_LoadSmolVLMModel": Fossiel_Load_SmolVLM_Model,
    "Fossiel_SmolLM2": Fossiel_SmolLM2,
    "Fossiel_SmolVLM_Classic": Fossiel_SmolVLM_Classic,
    "Fossiel_SmolVLM2": Fossiel_SmolVLM2,
    "FossielVideoDeflicker": FossielVideoDeflicker,
    "FossielWebPWrangler": FossielWebPWrangler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FossielCentralControlLite": "Fossiel Central Control Lite",
    "FossielCentralControl_v2": "Fossiel Central Control v2",
    "FossielDenoisePrecisionKSampler": "Denoise Precision KSampler",
    "FossielLevelMatcher": "Image Level Matcher",
    "FossielResolutionWrangler": "Resolution Wrangler",
    "FossielResolutionWranglerXP": "Resolution Wrangler (Express)",
    "Fossiel_Sensor_Switch_Image": "Sensor Switch Image",
    "Fossiel_Sensor_Switch_Clip": "Sensor Switch Clip",
    "Fossiel_Sensor_Switch_Conditioning": "Sensor Switch Conditioning",
    "Fossiel_Sensor_Switch_Latent": "Sensor Switch Latent",
    "Fossiel_Sensor_Switch_Mask": "Sensor Switch Mask",
    "Fossiel_Sensor_Switch_Model": "Sensor Switch Model",
    "Fossiel_Sensor_Switch_VAE": "Sensor Switch VAE",
    "Fossiel_Sensor_KSampler_Switch": "Sensor KSampler Switch",
    "Fossiel_LoadSmolLM2Model": "Load SmolLM2 Model (Fossiel_QoL)",
    "Fossiel_LoadSmolVLMModel": "Load SmolVLM / SmolVLM2 Model (Fossiel_QoL)",
    "Fossiel_SmolLM2": "SmolLM2 (Fossiel_QoL)",
    "Fossiel_SmolVLM_Classic": "SmolVLM (Gen1) (Fossiel_QoL)",
    "Fossiel_SmolVLM2": "SmolVLM2 (Fossiel_QoL)",
    "FossielSequenceWrangler": "Sequence Wrangler",
    "FossielVideoDeflicker": "Video De-flicker",
    "FossielWebPWrangler": "WebP Wrangler"
}
