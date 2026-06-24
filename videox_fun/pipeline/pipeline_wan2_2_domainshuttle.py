import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

from ..utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ..models import AutoencoderKLWan, AutoTokenizer, DomainShuttleWanTransformer3DModel, WanT5EncoderModel
from .pipeline_wan2_2 import Wan2_2Pipeline, WanPipelineOutput, retrieve_timesteps


class Wan2_2DomainShuttlePipeline(Wan2_2Pipeline):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: DomainShuttleWanTransformer3DModel,
        transformer_2: DomainShuttleWanTransformer3DModel = None,
        scheduler: FlowMatchEulerDiscreteScheduler = None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            transformer_2=transformer_2,
            scheduler=scheduler,
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)

    @staticmethod
    def _reference_num_from_transformer(transformer):
        transformer_owner = transformer.module if hasattr(transformer, "module") else transformer
        return int(getattr(transformer_owner.config, "reference_num", getattr(transformer_owner, "reference_num", 5)))

    def prepare_reference_latents(
        self, reference_images, batch_size, height, width, dtype, device, do_classifier_free_guidance, reference_num
    ):
        video_length = reference_images.shape[2]
        if video_length <= 0:
            raise ValueError("At least one reference image is required for DomainShuttle r2v inference.")
        reference_images = self.image_processor.preprocess(
            rearrange(reference_images, "b c f h w -> (b f) c h w"), height=height, width=width
        )
        reference_images = reference_images.to(dtype=torch.float32)
        reference_images = rearrange(reference_images, "(b f) c h w -> b c f h w", f=video_length)

        reference_latents = []
        for i in range(video_length):
            image = reference_images[:, :, i : i + 1].to(device=device, dtype=dtype)
            encoded = []
            for j in range(image.shape[0]):
                encoded.append(self.vae.encode(image[j : j + 1])[0].mode())
            reference_latents.append(torch.cat(encoded, dim=0))
        reference_latents = torch.cat(reference_latents, dim=2)
        if reference_latents.shape[2] > reference_num:
            reference_latents = reference_latents[:, :, :reference_num]
        elif reference_latents.shape[2] < reference_num:
            pad_shape = list(reference_latents.shape)
            pad_shape[2] = reference_num - reference_latents.shape[2]
            reference_latents = torch.cat([reference_latents, reference_latents.new_zeros(pad_shape)], dim=2)
        if do_classifier_free_guidance:
            reference_latents = torch.cat([reference_latents, reference_latents], dim=0)
        if reference_latents.shape[0] != batch_size:
            if reference_latents.shape[0] == 1:
                reference_latents = reference_latents.expand(batch_size, -1, -1, -1, -1)
            else:
                raise ValueError(f"Reference batch {reference_latents.shape[0]} does not match latent batch {batch_size}.")
        return reference_latents.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 97,
        num_inference_steps: int = 40,
        timesteps: Optional[List[int]] = None,
        guidance_scale: Union[float, Tuple[float, float], List[float]] = (4.0, 3.0),
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        reference_images: Optional[torch.FloatTensor] = None,
        domain_code: Optional[torch.LongTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        boundary: float = 0.875,
        comfyui_progressbar: bool = False,
        shift: int = 12,
    ) -> Union[WanPipelineOutput, Tuple]:
        if reference_images is None:
            raise ValueError("`reference_images` is required for DomainShuttle r2v inference.")
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype
        do_classifier_free_guidance = (max(guidance_scale) if isinstance(guidance_scale, (list, tuple)) else guidance_scale) > 1.0

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        in_prompt_embeds = negative_prompt_embeds + prompt_embeds if do_classifier_free_guidance else prompt_embeds

        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, mu=1)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(self.scheduler, device=device, sigmas=sampling_sigmas)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        transformer_config_owner = self.transformer.module if hasattr(self.transformer, "module") else self.transformer
        latent_channels = getattr(transformer_config_owner.config, "in_channels", getattr(transformer_config_owner.config, "in_dim", 16))
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )
        reference_latents = self.prepare_reference_latents(
            reference_images,
            latents.shape[0] * (2 if do_classifier_free_guidance else 1),
            height,
            width,
            weight_dtype,
            device,
            do_classifier_free_guidance,
            self._reference_num_from_transformer(self.transformer),
        )
        if domain_code is not None and do_classifier_free_guidance:
            domain_code = torch.as_tensor(domain_code, dtype=torch.long, device=device)
            if domain_code.ndim == 1:
                domain_code = domain_code.unsqueeze(0)
            domain_code = torch.cat([domain_code, domain_code], dim=0)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        target_shape = (
            self.vae.latent_channels,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            width // self.vae.spatial_compression_ratio,
            height // self.vae.spatial_compression_ratio,
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (transformer_config_owner.config.patch_size[1] * transformer_config_owner.config.patch_size[2])
            * target_shape[1]
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps
        if self.transformer_2 is not None:
            self.transformer_2.num_inference_steps = num_inference_steps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.transformer.current_steps = i
                if self.transformer_2 is not None:
                    self.transformer_2.current_steps = i
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                timestep = t.expand(latent_model_input.shape[0])

                if self.transformer_2 is not None:
                    local_transformer = self.transformer_2 if t >= boundary * self.scheduler.config.num_train_timesteps else self.transformer
                else:
                    local_transformer = self.transformer

                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                    noise_pred = local_transformer(
                        x=latent_model_input,
                        context=in_prompt_embeds,
                        t=timestep,
                        seq_len=seq_len,
                        reference=reference_latents,
                        domain_code=domain_code,
                    )

                if do_classifier_free_guidance:
                    if self.transformer_2 is not None and isinstance(self.guidance_scale, (list, tuple)):
                        transformer_2_config_owner = self.transformer_2.module if hasattr(self.transformer_2, "module") else self.transformer_2
                        sample_guide_scale = self.guidance_scale[1] if t >= transformer_2_config_owner.config.boundary * self.scheduler.config.num_train_timesteps else self.guidance_scale[0]
                    else:
                        sample_guide_scale = self.guidance_scale
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + sample_guide_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "numpy":
            video = self.decode_latents(latents)
        elif output_type != "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()
        if not return_dict:
            video = torch.from_numpy(video)
        return WanPipelineOutput(videos=video)
