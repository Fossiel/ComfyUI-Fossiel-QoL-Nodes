import torch
import node_helpers

class FossielInpaintConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "pixels": ("IMAGE", ),
                "mask": ("MASK", ),
                "noise_mask": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add a noise mask to the latent so sampling will only happen within the mask. Might improve results or completely break things depending on the model."
                }),
            },
            "optional": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "Fossiel"

    def encode(self, positive, negative, pixels, vae, mask, noise_mask=True, latent=None):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
            size=(pixels.shape[1], pixels.shape[2]),
            mode="bilinear"
        )

        orig_pixels = pixels
        pixels = orig_pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

        m = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            pixels[:, :, :, i] -= 0.5
            pixels[:, :, :, i] *= m
            pixels[:, :, :, i] += 0.5
        concat_latent = vae.encode(pixels)

        # Use provided latent if available, otherwise encode the original image
        if latent is not None:
            orig_latent = latent["samples"]
        else:
            orig_latent = vae.encode(orig_pixels)

        out_latent = {}
        out_latent["samples"] = orig_latent
        if noise_mask:
            out_latent["noise_mask"] = mask

        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(
                conditioning,
                {
                    "concat_latent_image": concat_latent,
                    "concat_mask": mask
                }
            )
            out.append(c)

        return (out[0], out[1], out_latent)
