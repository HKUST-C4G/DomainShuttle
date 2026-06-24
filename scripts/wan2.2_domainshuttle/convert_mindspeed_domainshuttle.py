import argparse
import json
import os
import shutil
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file


DEFAULT_HIGH = (
    "referecnce_codebase/MindSpeed-MM-wan2.2/output_checkpoint/"
    "wan2.2_r2v_high_try_480_0120_dual_branch_w_domain_stage2_adaln_proj/"
    "5000_dcp_to_torch/release/mp_rank_00/model_optim_rng.pt"
)
DEFAULT_LOW = (
    "referecnce_codebase/MindSpeed-MM-wan2.2/output_checkpoint/"
    "wan2.2_r2v_low_try_480_0120_dual_branch_w_domain_stage2_adaln_proj/"
    "5000_dcp_to_torch/release/mp_rank_00/model_optim_rng.pt"
)
DEFAULT_OUTPUT = "models/Diffusion_Transformer/Wan2.2-DomainShuttle-A14B"
DEFAULT_BASE_ASSETS = "/mnt/bn/bes-mllm-shared/chennan/Code/Open-OmniVCus/DiffSynth-Studio/models/Wan2.2-VACE-Fun-A14B"


class PickleStub:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


def _ensure_module(name):
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    sys.modules[name] = module
    if "." in name:
        parent = _ensure_module(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], module)
    return module


def install_pickle_stubs():
    modules = [
        "megatron",
        "megatron.core",
        "megatron.core.transformer",
        "megatron.core.transformer.enums",
        "megatron.core.enums",
        "megatron.core.rerun_state_machine",
        "mindspeed_mm",
        "mindspeed_mm.configs",
        "mindspeed_mm.configs.config",
    ]
    for module in modules:
        _ensure_module(module)

    classes = {
        "megatron.core.transformer.enums": ["AttnBackend"],
        "megatron.core.enums": ["ModelType"],
        "megatron.core.rerun_state_machine": ["RerunMode", "RerunState", "RerunDiagnostic"],
        "mindspeed_mm.configs.config": ["MMConfig", "ConfigReader"],
    }
    for module_name, class_names in classes.items():
        module = sys.modules[module_name]
        for class_name in class_names:
            setattr(module, class_name, type(class_name, (PickleStub,), {"__module__": module_name}))


def map_key(key):
    if key.endswith("._extra_state") or key == "domain_projection.weight":
        return None

    replacements = [
        ("text_embedding.linear_1.", "text_embedding.0."),
        ("text_embedding.linear_2.", "text_embedding.2."),
        (".self_attn.proj_q_ref.", ".self_attn.q_ref."),
        (".self_attn.proj_k_ref.", ".self_attn.k_ref."),
        (".self_attn.proj_v_ref.", ".self_attn.v_ref."),
        (".self_attn.proj_q.", ".self_attn.q."),
        (".self_attn.proj_k.", ".self_attn.k."),
        (".self_attn.proj_v.", ".self_attn.v."),
        (".self_attn.proj_out.", ".self_attn.o."),
        (".self_attn.q_norm.", ".self_attn.norm_q."),
        (".self_attn.k_norm.", ".self_attn.norm_k."),
        (".cross_attn.proj_q.", ".cross_attn.q."),
        (".cross_attn.proj_k.", ".cross_attn.k."),
        (".cross_attn.proj_v.", ".cross_attn.v."),
        (".cross_attn.proj_out.", ".cross_attn.o."),
        (".cross_attn.q_norm.", ".cross_attn.norm_q."),
        (".cross_attn.k_norm.", ".cross_attn.norm_k."),
    ]
    for old, new in replacements:
        key = key.replace(old, new)
    return key


def load_model_state(path):
    install_pickle_stubs()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    return state


def convert_one(source, output_dir):
    source = Path(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = load_model_state(source)

    converted = {}
    skipped = []
    for key, value in state.items():
        mapped = map_key(key)
        if mapped is None:
            skipped.append(key)
            continue
        if torch.is_tensor(value):
            converted[mapped] = value.contiguous()
        else:
            skipped.append(key)

    save_file(converted, output_dir / "diffusion_pytorch_model.safetensors")
    config = {
        "_class_name": "DomainShuttleWanTransformer3DModel",
        "_diffusers_version": "0.30.0",
        "model_type": "r2v",
        "patch_size": [1, 2, 2],
        "text_len": 512,
        "in_dim": 16,
        "dim": 5120,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 16,
        "num_heads": 40,
        "num_layers": 40,
        "window_size": [-1, -1],
        "qk_norm": True,
        "cross_attn_norm": True,
        "eps": 1e-6,
        "in_channels": 16,
        "reference_num": 5,
        "num_domains": 5,
        "boundary": 0.875,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Converted {source} -> {output_dir}")
    print(f"Saved {len(converted)} tensors; skipped {len(skipped)} non-model/unsupported entries.")


def copy_base_assets(source_dir, output_dir):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    assets = [
        ("Wan2.1_VAE.pth", False),
        ("models_t5_umt5-xxl-enc-bf16.pth", False),
        ("configuration.json", False),
        ("google/umt5-xxl", True),
    ]
    for relative_path, is_dir in assets:
        src = source_dir / relative_path
        dst = output_dir / relative_path
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if is_dir:
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"Copied base asset {src} -> {dst}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert MindSpeed-MM DomainShuttle r2v checkpoints to safetensors.")
    parser.add_argument("--high", default=DEFAULT_HIGH)
    parser.add_argument("--low", default=DEFAULT_LOW)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--base_assets_source", default=DEFAULT_BASE_ASSETS)
    return parser.parse_args()


def main():
    args = parse_args()
    convert_one(args.low, os.path.join(args.output, "low_noise_model"))
    convert_one(args.high, os.path.join(args.output, "high_noise_model"))
    copy_base_assets(args.base_assets_source, args.output)


if __name__ == "__main__":
    main()
