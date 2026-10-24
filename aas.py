import torch

class FossielAddSilence:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "Silence_duration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 60.0,
                    "step": 0.001,
                    "display": "number"
                }),
                "Add_silence_at": (["Start", "End", "Both"], {"default": "End"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "add_silence"
    CATEGORY = "Fossiel"

    def add_silence(self, audio, Silence_duration, Add_silence_at):
        waveform = audio["waveform"]          # [batch, channels, samples]
        sample_rate = audio["sample_rate"]

        if Silence_duration <= 0:
            return (audio,)

        silence_samples = int(round(Silence_duration * sample_rate))
        if silence_samples <= 0:
            return (audio,)

        # Create silence with same batch size, channels, and dtype/device
        silence = torch.zeros(
            (waveform.shape[0], waveform.shape[1], silence_samples),
            dtype=waveform.dtype,
            device=waveform.device
        )

        if Add_silence_at == "Start":
            new_waveform = torch.cat([silence, waveform], dim=2)
        elif Add_silence_at == "End":
            new_waveform = torch.cat([waveform, silence], dim=2)
        else:  # Both
            new_waveform = torch.cat([silence, waveform, silence], dim=2)

        return ({"waveform": new_waveform, "sample_rate": sample_rate},)
