import torch
import comfy.sample
import comfy.samplers
import comfy.utils
import node_helpers
from nodes import latent_preview
import time
import logging
from comfy.utils import ProgressBar

class FossielUnifiedInpaintSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),

                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 100.00, "min": 0.01, "max": 100.00, "step": 0.01, "round": 0.01}),
                
                "noise_mask": ("BOOLEAN", {"default": True}),
                "use_differential_diffusion": ("BOOLEAN", {"default": True}),
                "differential_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # VAE Tiling parameters
                "enable_vae_tiling": ("BOOLEAN", {"default": True}),
                "horizontal_tiles": ("INT", {"default": 2, "min": 1, "max": 64}),
                "vertical_tiles": ("INT", {"default": 2, "min": 1, "max": 64}),
                "overlap": ("INT", {"default": 4, "min": 1, "max": 16}),
                "last_frame_fix": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "sample"
    CATEGORY = "inpaint"
    DESCRIPTION = "Advanced Inpaint Sampler with optional tiled VAE decode"

    def sample(self, model, positive, negative, vae, image, mask,
               seed, steps, cfg, sampler_name, scheduler, denoise,
               noise_mask=True, use_differential_diffusion=True, differential_strength=1.0,
               enable_vae_tiling=False, horizontal_tiles=2, vertical_tiles=2, overlap=4, last_frame_fix=False,
               latent=None):

        denoise_frac = denoise / 100.0

        # ====================== Inpaint Conditioning ======================
        orig_image = image
        batch, height, width, channels = image.shape

        x = (width // 8) * 8
        y = (height // 8) * 8

        mask_resized = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), 
            size=(height, width), 
            mode="bilinear"
        )

        pixels = orig_image.clone()
        if width != x or height != y:
            x_offset = (width % 8) // 2
            y_offset = (height % 8) // 2
            pixels = pixels[:, y_offset:y+y_offset, x_offset:x+x_offset, :]
            mask_resized = mask_resized[:, :, y_offset:y+y_offset, x_offset:x+x_offset]

        m = (1.0 - mask_resized.round()).squeeze(1)
        masked_pixels = pixels.clone()
        for i in range(3):
            masked_pixels[:, :, :, i] -= 0.5
            masked_pixels[:, :, :, i] *= m
            masked_pixels[:, :, :, i] += 0.5

        concat_latent = vae.encode(masked_pixels)

        # Use provided latent if available, otherwise encode the original image
        if latent is not None:
            orig_latent = latent["samples"]
        else:
            orig_latent = vae.encode(orig_image)

        latent_dict = {"samples": orig_latent}
        if noise_mask:
            latent_dict["noise_mask"] = mask_resized.squeeze(1)

        positive = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent,
            "concat_mask": mask_resized.squeeze(1)
        })
        negative = node_helpers.conditioning_set_values(negative, {
            "concat_latent_image": concat_latent,
            "concat_mask": mask_resized.squeeze(1)
        })

        # ====================== Differential Diffusion ======================
        if use_differential_diffusion:
            model = model.clone()
            def diff_forward(sigma, denoise_mask, extra_options):
                return DifferentialDiffusion.forward(sigma, denoise_mask, extra_options, differential_strength)
            model.set_model_denoise_mask_function(diff_forward)

        # ====================== Sampling ======================
        noise = comfy.sample.prepare_noise(latent_dict["samples"], seed, None)

        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler, positive, negative,
            latent_image=latent_dict["samples"],
            denoise=denoise_frac,
            noise_mask=latent_dict.get("noise_mask", None),
            callback=latent_preview.prepare_callback(model, steps),
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=seed
        )

        # Prepare the output latent (denoised latent, no VAE decode)
        output_latent = {"samples": samples}

        # ====================== VAE Decode ======================
        if not enable_vae_tiling:
            # Normal ComfyUI decode
            if samples.is_nested:
                samples = samples.unbind()[0]
            images = vae.decode(samples)
        else:
            # Use the special tiled VAE decoder
            images = self._tiled_vae_decode(vae, samples, horizontal_tiles, vertical_tiles, overlap, last_frame_fix)

        # Final cleanup
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        images = images.clamp(0.0, 1.0).float()

        return (images, output_latent)

    def _tiled_vae_decode(
        self,
        vae,
        samples,
        horizontal_tiles,
        vertical_tiles,
        overlap,
        last_frame_fix,
    ):

        if samples.ndim == 5: # video latent
            if last_frame_fix:
                # Repeat the last frame along dimension 2 (frames)
                # samples: [batch, channels, frames, height, width]
                last_frame = samples[
                    :, :, -1:, :, :
                ]  # shape: [batch, channels, 1, height, width]
                samples = torch.cat([samples, last_frame], dim=2)

            batch, channels, frames, height, width = samples.shape
            time_scale_factor, width_scale_factor, height_scale_factor = (
                vae.downscale_index_formula
            )
            image_frames = 1 + (frames - 1) * time_scale_factor
        else: # 4, image latent
            batch, channels, height, width = samples.shape
            image_frames = 1
            time_scale_factor = 1
            if vae.downscale_index_formula is not None:
                width_scale_factor, height_scale_factor = (
                    vae.downscale_index_formula
                )
            else:
                width_scale_factor = vae.downscale_ratio
                height_scale_factor = vae.downscale_ratio

        # Calculate output image dimensions
        output_height = height * height_scale_factor
        output_width = width * width_scale_factor

        # Initialize output tensor and weight tensor
        # VAE decode returns images in format [batch, height, width, channels]
        output = None
        weights = None

        num_tiles = vertical_tiles * horizontal_tiles
        pbar = ProgressBar(num_tiles)

        time_init = time.perf_counter()
        # vertical units yet to process, divided more or less evenly between tiles
        vert_remain = height + (vertical_tiles-1) * overlap
        v_start = 0
        for v in range(vertical_tiles):
            base_tile_height = vert_remain // (vertical_tiles - v)
            if vert_remain % base_tile_height:
                base_tile_height += 1 # first tiles larger if uneven; rather OOM early than late
            horz_remain = width + (horizontal_tiles-1) * overlap
            h_start = 0
            for h in range(horizontal_tiles):
                base_tile_width = horz_remain // (horizontal_tiles - h)
                if horz_remain % base_tile_width:
                    base_tile_width += 1

                # Adjust end positions for edge tiles
                h_end = (
                    min(h_start + base_tile_width, width)
                    if h < horizontal_tiles - 1
                    else width
                )
                v_end = (
                    min(v_start + base_tile_height, height)
                    if v < vertical_tiles - 1
                    else height
                )

                # Calculate actual tile dimensions
                tile_height = v_end - v_start
                tile_width = h_end - h_start

                logging.info(f"Processing VAE decode tile at row {v}, col {h}:  Position: ({h_start*width_scale_factor}:{h_end*width_scale_factor}, {v_start*height_scale_factor}:{v_end*height_scale_factor}), Size: {tile_width*width_scale_factor}x{tile_height*height_scale_factor}")
                time_before = time.perf_counter()

                # Extract tile
                if samples.ndim == 5:
                    tile = samples[:, :, :, v_start:v_end, h_start:h_end]
                else:
                    tile = samples[:, :, v_start:v_end, h_start:h_end]

                # Create tile latents dict
                tile_latents = {"samples": tile}

                # Decode the tile
                decoded_tile = vae.decode(tile_latents["samples"])

                # Initialize output tensors on first tile
                if output is None:
                    output = torch.zeros(
                        (
                            batch,
                            image_frames,
                            output_height,
                            output_width,
                            decoded_tile.shape[-1],
                        ),
                        device=decoded_tile.device,
                        dtype=decoded_tile.dtype,
                    )
                    weights = torch.zeros(
                        (batch, image_frames, output_height, output_width, 1),
                        device=decoded_tile.device,
                        dtype=decoded_tile.dtype,
                    )

                # Calculate output tile boundaries
                out_h_start = v_start * height_scale_factor
                out_h_end = v_end * height_scale_factor
                out_w_start = h_start * width_scale_factor
                out_w_end = h_end * width_scale_factor

                # Create weight mask for this tile
                tile_out_height = out_h_end - out_h_start
                tile_out_width = out_w_end - out_w_start
                tile_weights = torch.ones(
                    (batch, image_frames, tile_out_height, tile_out_width, 1),
                    device=decoded_tile.device,
                    dtype=decoded_tile.dtype,
                )

                # Calculate overlap regions in output space
                overlap_out_h = overlap * height_scale_factor
                overlap_out_w = overlap * width_scale_factor

                # Apply horizontal blending weights
                if h > 0:  # Left overlap
                    h_blend = torch.linspace(
                        0, 1, overlap_out_w, device=decoded_tile.device
                    )
                    tile_weights[:, :, :, :overlap_out_w, :] *= h_blend.view(
                        1, 1, 1, -1, 1
                    )
                if h < horizontal_tiles - 1:  # Right overlap
                    h_blend = torch.linspace(
                        1, 0, overlap_out_w, device=decoded_tile.device
                    )
                    tile_weights[:, :, :, -overlap_out_w:, :] *= h_blend.view(
                        1, 1, 1, -1, 1
                    )

                # Apply vertical blending weights
                if v > 0:  # Top overlap
                    v_blend = torch.linspace(
                        0, 1, overlap_out_h, device=decoded_tile.device
                    )
                    tile_weights[:, :, :overlap_out_h, :, :] *= v_blend.view(
                        1, 1, -1, 1, 1
                    )
                if v < vertical_tiles - 1:  # Bottom overlap
                    v_blend = torch.linspace(
                        1, 0, overlap_out_h, device=decoded_tile.device
                    )
                    tile_weights[:, :, -overlap_out_h:, :, :] *= v_blend.view(
                        1, 1, -1, 1, 1
                    )

                # Add weighted tile to output
                output[:, :, out_h_start:out_h_end, out_w_start:out_w_end, :] += (
                    decoded_tile * tile_weights
                )

                # Add weights to weight tensor
                weights[
                    :, :, out_h_start:out_h_end, out_w_start:out_w_end, :
                ] += tile_weights
                time_elapsed = time.perf_counter()-time_before
                if vertical_tiles * horizontal_tiles != 1:
                    logging.info(f"({v*horizontal_tiles+h+1}/{num_tiles}) time: {time_elapsed:.2f} seconds") #.format(time_elapsed))

                horz_remain -= base_tile_width
                h_start = h_end - overlap
                pbar.update(1)
            vert_remain -= base_tile_height
            v_start = v_end - overlap

        # Normalize by weights
        output = output / (weights + 1e-8)

        # Reshape output to match expected format [batch * frames, height, width, channels]
        output = output.view(
            batch * image_frames, output_height, output_width, output.shape[-1]
        )

        if last_frame_fix and samples.ndim == 5:
            output = output[:-time_scale_factor, :, :]

        time_total = time.perf_counter()-time_init
        logging.info("VAE total decode time: {:.2f} seconds".format(time_total))

        return output


class DifferentialDiffusion:
    @classmethod
    def forward(cls, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict, strength: float = 1.0):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        threshold = (current_ts - ts_to) / (ts_from - ts_to)
        binary_mask = (denoise_mask >= threshold).to(denoise_mask.dtype)

        if strength < 1.0:
            return strength * binary_mask + (1.0 - strength) * denoise_mask
        return binary_mask
