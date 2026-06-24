import argparse
import json
import os
import sys
from functools import partial

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import AutoencoderKLWan, AutoencoderKLWan3_8, DomainShuttleWanTransformer3DModel, WanT5EncoderModel
from videox_fun.pipeline.pipeline_wan2_2_domainshuttle import Wan2_2DomainShuttlePipeline
from videox_fun.utils.fp8_optimization import replace_parameters_by_name
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.utils import filter_kwargs, get_image_latent, save_videos_grid


DOMAIN_CODE = {
    "Human": 0,
    "Man": 0,
    "Woman": 0,
    "object": 1,
    "Object": 1,
    "Fantasy Domain": 2,
    "Background": 3,
    "others": 4,
}


def parse_args():
    parser = argparse.ArgumentParser(description="DomainShuttle Wan2.2 r2v batch inference on CUDA.")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--domain_model_name", default="models/Diffusion_Transformer/Wan2.2-DomainShuttle-A14B")
    parser.add_argument("--config_path", default="config/wan2.2/wan_civitai_t2v.yaml")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--video_length", type=int, default=97)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, nargs="+", default=[4.0, 3.0])
    parser.add_argument("--shift", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ulysses_degree", type=int, default=8)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--fsdp_dit", action="store_true", default=True)
    parser.add_argument("--fsdp_text_encoder", action="store_true", default=True)
    parser.add_argument("--memory_mode", default="model_full_load", choices=["model_full_load", "sequential_cpu_offload"])
    parser.add_argument("--max_reference_num", type=int, default=5)
    parser.add_argument(
        "--negative_prompt",
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    )
    return parser.parse_args()


def resolve_path(path, root=None):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(root, path) if root else path


def image_fields(item, max_reference_num):
    paths = []
    if item.get("seg_meta"):
        seg_meta = item["seg_meta"][0] if isinstance(item["seg_meta"], list) else item["seg_meta"]
        seg_root = item.get("seg_root")
        for path in seg_meta.get("seg_file", []):
            paths.append(resolve_path(path.lstrip(os.sep), seg_root) if seg_root and not os.path.isabs(path) else path)
    if not paths and item.get("image_path"):
        paths.append(item["image_path"])
    if item.get("face_path"):
        paths.append(item["face_path"])
    if not paths:
        raise ValueError(f"No reference image path found in item: {item}")
    return paths[:max_reference_num]


def domain_fields(item, max_reference_num, reference_count):
    domain = item.get("domain_code")
    if not domain and item.get("seg_meta"):
        seg_meta = item["seg_meta"][0] if isinstance(item["seg_meta"], list) else item["seg_meta"]
        domain = seg_meta.get("domain_code")
    if not domain:
        return [0] * reference_count
    values = []
    for value in domain[:max_reference_num]:
        if isinstance(value, int):
            values.append(value)
        else:
            values.append(DOMAIN_CODE.get(value, 4))
    return values[:max_reference_num]


def load_jsonl(path, max_reference_num):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            images = image_fields(item, max_reference_num)
            items.append(
                {
                    "prompt": item.get("cap", item.get("prompts", item.get("prompt", ""))),
                    "images": images,
                    "domain_code": domain_fields(item, max_reference_num, len(images)),
                }
            )
    return items


def save_result(sample, output_dir, index, video_length, fps):
    os.makedirs(output_dir, exist_ok=True)
    if video_length == 1:
        image = sample[0, :, 0].transpose(0, 1).transpose(1, 2)
        image = Image.fromarray((image * 255).numpy().astype(np.uint8))
        image.save(os.path.join(output_dir, f"video_{index:05d}.png"))
    else:
        save_videos_grid(sample, os.path.join(output_dir, f"video_{index:05d}.mp4"), fps=fps)


def main():
    args = parse_args()
    config = OmegaConf.load(args.config_path)
    device = set_multi_gpus_devices(args.ulysses_degree, args.ring_degree)
    weight_dtype = torch.bfloat16
    boundary = config["transformer_additional_kwargs"].get("boundary", 0.875)

    transformer = DomainShuttleWanTransformer3DModel.from_pretrained(
        os.path.join(args.domain_model_name, "low_noise_model"),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    transformer_2 = DomainShuttleWanTransformer3DModel.from_pretrained(
        os.path.join(args.domain_model_name, "high_noise_model"),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    chosen_vae = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
    vae = chosen_vae.from_pretrained(
        os.path.join(args.domain_model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.domain_model_name, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.domain_model_name, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    scheduler_cls = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }["Flow_Unipc"]
    scheduler = scheduler_cls(**filter_kwargs(scheduler_cls, OmegaConf.to_container(config["scheduler_kwargs"])))

    pipeline = Wan2_2DomainShuttlePipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

    if args.ulysses_degree > 1 or args.ring_degree > 1:
        if args.fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)
            pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
            print("Add FSDP DIT")
        if args.fsdp_text_encoder:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.text_encoder = shard_fn(pipeline.text_encoder)
            print("Add FSDP TEXT ENCODER")

    if args.memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation", "modulation_ref"], device=device)
        replace_parameters_by_name(transformer_2, ["modulation", "modulation_ref"], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        transformer_2.freqs = transformer_2.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    items = load_jsonl(args.input_json, args.max_reference_num)
    os.makedirs(args.output_dir, exist_ok=True)

    for index, item in enumerate(items):
        reference_images = [get_image_latent(path, sample_size=[args.height, args.width], padding=True) for path in item["images"]]
        reference_images = torch.cat(reference_images, dim=2)
        domain_code = torch.tensor([item["domain_code"]], dtype=torch.long)

        with torch.no_grad():
            video_length = (
                int((args.video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
                if args.video_length != 1
                else 1
            )
            sample = pipeline(
                item["prompt"],
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=video_length,
                generator=generator,
                guidance_scale=args.guidance_scale if len(args.guidance_scale) > 1 else args.guidance_scale[0],
                num_inference_steps=args.num_inference_steps,
                reference_images=reference_images,
                domain_code=domain_code,
                boundary=boundary,
                shift=args.shift,
            ).videos

        if args.ulysses_degree * args.ring_degree > 1:
            import torch.distributed as dist

            if dist.get_rank() == 0:
                save_result(sample, args.output_dir, index, video_length, args.fps)
        else:
            save_result(sample, args.output_dir, index, video_length, args.fps)


if __name__ == "__main__":
    main()
