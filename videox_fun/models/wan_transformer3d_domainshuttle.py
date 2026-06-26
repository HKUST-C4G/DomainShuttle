# Copyright 2024-2025 The Alibaba Wan Team Authors.
# DomainShuttle r2v inference support adapted from the local MindSpeed-MM
# reference implementation, with Megatron/NPU dependencies removed.

import math
from typing import Optional

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.utils import is_torch_version

from ..utils import cfg_skip
from .attention_utils import attention
from .wan_transformer3d import (
    Head,
    WanLayerNorm,
    WanRMSNorm,
    WanT2VCrossAttention,
    WanTransformer3DModel,
    rope_params,
    sinusoidal_embedding_1d,
)


@amp.autocast(enabled=False)
@torch.compiler.disable()
def domainshuttle_rope_apply(x, grid_sizes, freqs, reference_frames):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        ref_frames = int(reference_frames[i].item() if torch.is_tensor(reference_frames) else reference_frames)
        noise_len = f * h * w
        ref_len = ref_frames * h * w
        seq_len = noise_len + ref_len

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2))

        freqs_vid = torch.cat(
            [
                freqs[0][1 : f + 1].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(noise_len, 1, -1)

        if ref_frames > 0:
            f_ref = freqs[0][:1].view(1, 1, 1, -1).expand(ref_frames, h, w, -1)
            h_ref, w_ref = [], []
            for ref_idx in range(1, ref_frames + 1):
                h_start = ref_idx * h
                w_start = ref_idx * w
                h_ref.append(freqs[1][h_start : h_start + h].view(1, h, 1, -1).expand(1, h, w, -1))
                w_ref.append(freqs[2][w_start : w_start + w].view(1, 1, w, -1).expand(1, h, w, -1))
            freqs_ref = torch.cat([f_ref, torch.cat(h_ref, dim=0), torch.cat(w_ref, dim=0)], dim=-1)
            freqs_i = torch.cat([freqs_vid, freqs_ref.reshape(ref_len, 1, -1)], dim=0)
        else:
            freqs_i = freqs_vid

        x_i = torch.view_as_real(x_i * freqs_i.to(x_i.device)).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)

    return torch.stack(output).to(x.dtype)


def domainshuttle_rope_apply_qk(q, k, grid_sizes, freqs, reference_frames):
    q = domainshuttle_rope_apply(q, grid_sizes, freqs, reference_frames)
    k = domainshuttle_rope_apply(k, grid_sizes, freqs, reference_frames)
    return q, k


class DomainShuttleWanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.q_ref = nn.Linear(dim, dim)
        self.k_ref = nn.Linear(dim, dim)
        self.v_ref = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, reference_token_num, reference_frames, dtype=torch.bfloat16):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        noise_tokens = s - reference_token_num
        x_noise, x_ref = x[:, :noise_tokens], x[:, noise_tokens:]

        q = self.norm_q(self.q(x_noise.to(dtype)))
        k = self.norm_k(self.k(x_noise.to(dtype)))
        v = self.v(x_noise.to(dtype))
        q_ref = self.norm_q(self.q_ref(x_ref.to(dtype)))
        k_ref = self.norm_k(self.k_ref(x_ref.to(dtype)))
        v_ref = self.v_ref(x_ref.to(dtype))

        q = torch.cat([q, q_ref], dim=1).view(b, s, n, d)
        k = torch.cat([k, k_ref], dim=1).view(b, s, n, d)
        v = torch.cat([v, v_ref], dim=1).view(b, s, n, d)
        q, k = domainshuttle_rope_apply_qk(q, k, grid_sizes, freqs, reference_frames)

        x = attention(q.to(dtype), k.to(dtype), v.to(dtype), k_lens=seq_lens, window_size=self.window_size)
        return self.o(x.flatten(2))


class DomainShuttleWanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = DomainShuttleWanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanT2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.modulation_ref = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    @staticmethod
    def _expand_ref_mod(value, reference_token_num):
        b, r, h = value.shape
        tokens_per_ref = reference_token_num // r
        return value.unsqueeze(2).expand(b, r, tokens_per_ref, h).reshape(b, reference_token_num, h)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        reference_token_num,
        reference_frames,
        domain_embedding,
        dtype=torch.bfloat16,
    ):
        e_noise = (self.modulation + e).chunk(6, dim=1)
        e_ref = (self.modulation_ref + e).chunk(6, dim=1)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = e_noise
        shift_msa_ref, scale_msa_ref, gate_msa_ref, shift_mlp_ref, scale_mlp_ref, gate_mlp_ref = e_ref
        if domain_embedding is not None:
            shift_msa_ref = shift_msa_ref + domain_embedding
            scale_msa_ref = scale_msa_ref + domain_embedding
            shift_mlp_ref = shift_mlp_ref + domain_embedding
            scale_mlp_ref = scale_mlp_ref + domain_embedding

        noise_tokens = x.size(1) - reference_token_num
        x_noise, x_ref = x[:, :noise_tokens], x[:, noise_tokens:]
        msa_noise = self.norm1(x_noise) * (1 + scale_msa) + shift_msa
        msa_ref = self.norm1(x_ref) * (1 + self._expand_ref_mod(scale_msa_ref, reference_token_num)) + self._expand_ref_mod(shift_msa_ref, reference_token_num)
        msa_in = torch.cat([msa_noise, msa_ref], dim=1).to(dtype)

        y = self.self_attn(msa_in, seq_lens, grid_sizes, freqs, reference_token_num, reference_frames, dtype=dtype)
        x = x + y * gate_msa
        x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype)

        x_noise, x_ref = x[:, :noise_tokens], x[:, noise_tokens:]
        mlp_noise = self.norm2(x_noise) * (1 + scale_mlp) + shift_mlp
        mlp_ref = self.norm2(x_ref) * (1 + self._expand_ref_mod(scale_mlp_ref, reference_token_num)) + self._expand_ref_mod(shift_mlp_ref, reference_token_num)
        mlp_out = self.ffn(torch.cat([mlp_noise, mlp_ref], dim=1).to(dtype))
        mlp_noise_out, mlp_ref_out = mlp_out[:, :noise_tokens], mlp_out[:, noise_tokens:]
        x = x + torch.cat([gate_mlp * mlp_noise_out, self._expand_ref_mod(gate_mlp_ref, reference_token_num) * mlp_ref_out], dim=1)
        return x


class DomainShuttleWanTransformer3DModel(WanTransformer3DModel):
    @register_to_config
    def __init__(
        self,
        model_type="r2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=5120,
        ffn_dim=13824,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=40,
        num_layers=40,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        reference_num=5,
        num_domains=5,
        in_channels=16,
        boundary=0.875,
        **kwargs,
    ):
        super().__init__(
            model_type="t2v",
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            cross_attn_type="t2v_cross_attn",
        )
        self.model_type = model_type
        self.reference_num = reference_num
        self.patch_embedding_ref = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.domain_embedding = nn.Embedding(num_domains, dim)
        self.dom_to_film_gamma = nn.Linear(dim, dim, bias=False)
        self.dom_to_film_beta = nn.Linear(dim, dim, bias=False)
        self.blocks = nn.ModuleList(
            [
                DomainShuttleWanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
                for _ in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps)

    def _domain_embedding(self, domain_code, batch_size, reference_frames, device, dtype):
        if domain_code is None:
            return None, None, None
        if not torch.is_tensor(domain_code):
            domain_code = torch.tensor(domain_code, dtype=torch.long, device=device)
        domain_code = domain_code.to(device=device, dtype=torch.long)
        if domain_code.ndim == 1:
            domain_code = domain_code.unsqueeze(0)
        if domain_code.size(0) == 1 and batch_size > 1:
            domain_code = domain_code.expand(batch_size, -1)
        domain_code = domain_code[:, :reference_frames]

        domain_embedding = self.domain_embedding(domain_code).to(dtype) if domain_code.size(1) > 0 else None
        if domain_embedding is None:
            domain_embedding = torch.zeros(batch_size, 0, self.dim, dtype=dtype, device=device)
        if domain_embedding.size(1) < reference_frames:
            pad = torch.zeros(
                batch_size,
                reference_frames - domain_embedding.size(1),
                domain_embedding.size(2),
                dtype=domain_embedding.dtype,
                device=domain_embedding.device,
            )
            domain_embedding = torch.cat([domain_embedding, pad], dim=1)
        gamma = self.dom_to_film_gamma(domain_embedding)
        beta = self.dom_to_film_beta(domain_embedding)
        return domain_embedding, gamma, beta

    @cfg_skip()
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        reference=None,
        domain_code=None,
        subject_ref=None,
        cond_flag=True,
        **kwargs,
    ):
        reference = reference if reference is not None else subject_ref
        if reference is None:
            raise ValueError("DomainShuttle r2v inference requires reference/subject_ref latents.")

        device = self.patch_embedding.weight.device
        dtype = x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        noise = self.patch_embedding(x.to(dtype))
        ref = self.patch_embedding_ref(reference.to(device=device, dtype=dtype))
        if ref.shape[2] > self.reference_num:
            ref = ref[:, :, : self.reference_num]
        elif ref.shape[2] < self.reference_num:
            pad_shape = list(ref.shape)
            pad_shape[2] = self.reference_num - ref.shape[2]
            ref = torch.cat([ref, ref.new_zeros(pad_shape)], dim=2)
        batch_size, hidden_size, reference_frames, ref_h, ref_w = ref.shape
        domain_embedding, gamma, beta = self._domain_embedding(domain_code, batch_size, reference_frames, ref.device, ref.dtype)
        if gamma is not None:
            ref = ref * (1 + gamma.permute(0, 2, 1).view(batch_size, hidden_size, reference_frames, 1, 1))
            ref = ref + beta.permute(0, 2, 1).view(batch_size, hidden_size, reference_frames, 1, 1)

        noise_grid_sizes = torch.tensor([noise.shape[2:]], dtype=torch.long, device=device).repeat(batch_size, 1)
        x = torch.cat([noise, ref], dim=2).flatten(2).transpose(1, 2)
        reference_token_num = reference_frames * ref_h * ref_w
        seq_lens = torch.full((batch_size,), x.size(1), dtype=torch.long, device=device)
        seq_len = max(int(seq_len), int(x.size(1)))

        x = torch.cat([x, x.new_zeros(batch_size, seq_len - x.size(1), x.size(2))], dim=1)

        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )
        reference_frames_tensor = torch.full((batch_size,), reference_frames, dtype=torch.long, device=device)

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    e0,
                    seq_lens,
                    noise_grid_sizes,
                    self.freqs,
                    context,
                    None,
                    reference_token_num,
                    reference_frames_tensor,
                    domain_embedding,
                    dtype,
                    **ckpt_kwargs,
                )
            else:
                x = block(
                    x,
                    e0,
                    seq_lens,
                    noise_grid_sizes,
                    self.freqs,
                    context,
                    None,
                    reference_token_num,
                    reference_frames_tensor,
                    domain_embedding,
                    dtype,
                )

        x = self.head(x, e)
        noise_seq_len = int(seq_lens.max().item()) - reference_token_num
        x = x[:, :noise_seq_len]
        return torch.stack(self.unpatchify(x, noise_grid_sizes))
