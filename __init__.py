from .fccl import FossielCentralControlLite
from .fcc_v2 import FossielCentralControl_v2
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

NODE_CLASS_MAPPINGS = {
    "FossielCentralControlLite": FossielCentralControlLite,
    "FossielCentralControl_v2": FossielCentralControl_v2,
    "Fossiel_Sensor_Switch_Image": FossielSensorSwitchImage,
    "Fossiel_Sensor_Switch_Clip": FossielSensorSwitchClip,
    "Fossiel_Sensor_Switch_Conditioning": FossielSensorSwitchConditioning,
    "Fossiel_Sensor_Switch_Latent": FossielSensorSwitchLatent,
    "Fossiel_Sensor_Switch_Mask": FossielSensorSwitchMask,
    "Fossiel_Sensor_Switch_Model": FossielSensorSwitchModel,
    "Fossiel_Sensor_Switch_VAE": FossielSensorSwitchVAE,
    "Fossiel_Sensor_KSampler_Switch": FossielSensorKSamplerSwitch,
    "FossielWebPWrangler": FossielWebPWrangler

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FossielCentralControlLite": "Fossiel Central Control Lite",
    "FossielCentralControl_v2": "Fossiel Central Control v2",
    "Fossiel_Sensor_Switch_Image": "Sensor Switch Image",
    "Fossiel_Sensor_Switch_Clip": "Sensor Switch Clip",
    "Fossiel_Sensor_Switch_Conditioning": "Sensor Switch Conditioning",
    "Fossiel_Sensor_Switch_Latent": "Sensor Switch Latent",
    "Fossiel_Sensor_Switch_Mask": "Sensor Switch Mask",
    "Fossiel_Sensor_Switch_Model": "Sensor Switch Model",
    "Fossiel_Sensor_Switch_VAE": "Sensor Switch VAE",
    "Fossiel_Sensor_KSampler_Switch": "Sensor KSampler Switch",
    "FossielWebPWrangler": "WebP Wrangler"
}
