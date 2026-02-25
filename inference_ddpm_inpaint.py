import os
import argparse
import copy
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# =========================================================
# Config
# =========================================================

@dataclass
class Config:
    device: torch.device
    image_size: int = 32
    channels: int = 1
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "cosine"
    dim: int = 48
    dim_mults: Tuple[int, ...] = (1, 2, 4)


# =========================================================
# Normalization
# =========================================================

def normalize_img(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def denormalize_img(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


# =========================================================
# Diffusion schedules
# =========================================================

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999).float()


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


class Diffusion:
    def __init__(self, config: Config):
        if config.schedule == "cosine":
            betas = cosine_beta_schedule(config.timesteps)
        else:
            betas = linear_beta_schedule(config.timesteps, config.beta_start, config.beta_end)

        self.betas = betas.to(config.device)
        self.alphas = (1.0 - self.betas).to(config.device)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(config.device)


def ddpm_step(x_t, pred_noise, t, diffusion: Diffusion):
    beta_t = diffusion.betas[t]
    alpha_t = diffusion.alphas[t]
    alpha_bar_t = diffusion.alpha_bars[t]

    mean = (1.0 / torch.sqrt(alpha_t)) * (
        x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
    )

    if t > 0:
        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(beta_t)
        return mean + sigma * noise
    return mean


# =========================================================
# Model components (identical to training)
# =========================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden = heads * dim_head
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden, dim, 1), LayerNorm2d(dim))

    def forward(self, x):
        b, _, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)

        q = rearrange(q, "b (h d) x y -> b h d (x y)", h=self.heads)
        k = rearrange(k, "b (h d) x y -> b h d (x y)", h=self.heads)
        v = rearrange(v, "b (h d) x y -> b h d (x y)", h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h d (x y) -> b (h d) x y", x=h, y=w)

        return self.to_out(out)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
        self.block1 = Block(dim, dim_out, groups)
        self.block2 = Block(dim_out, dim_out, groups)
        self.res = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, t_emb):
        scale_shift = self.mlp(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.block1(x)
        h = h * (scale_shift[0] + 1) + scale_shift[1]
        h = self.block2(h)
        return h + self.res(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.op = nn.Conv2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.op = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.op(x)


class ConditionalUNet(nn.Module):
    def __init__(self, dim=48, dim_mults=(1, 2, 4), in_channels=3, out_channels=1):
        super().__init__()

        dims = [dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.init_conv = nn.Conv2d(in_channels, dim, 7, padding=3)

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList()
        for dim_in, dim_out in in_out:
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        self.ups = nn.ModuleList()
        for dim_in, dim_out in reversed(in_out):
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in),
                    ]
                )
            )

        self.final_res = ResnetBlock(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x_noisy, t, cond):
        x = torch.cat([x_noisy, cond], dim=1)
        x = self.init_conv(x)
        residual = x
        t_emb = self.time_mlp(t)

        skips = []
        for b1, b2, attn, down in self.downs:
            x = b1(x, t_emb)
            x = b2(x, t_emb)
            x = attn(x)
            skips.append(x)
            x = down(x)

        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        for b1, b2, attn, up in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)
            x = b1(x, t_emb)
            x = b2(x, t_emb)
            x = attn(x)
            x = up(x)

        x = torch.cat([x, residual], dim=1)
        x = self.final_res(x, t_emb)
        return self.final_conv(x)


# =========================================================
# Inference
# =========================================================

def run_inference(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config(
        device=device,
        timesteps=args.timesteps,
        schedule=args.schedule,
        dim=args.dim,
        dim_mults=tuple(int(x) for x in args.dim_mults.split(",")),
    )

    model = ConditionalUNet(
        dim=config.dim,
        dim_mults=config.dim_mults,
        in_channels=3,
        out_channels=1,
    ).to(device)

    ema_model = copy.deepcopy(model).eval()

    ckpt = torch.load(args.checkpoint, map_location=device)
    ema_model.load_state_dict(ckpt["ema_model"])
    ema_model.eval()

    diffusion = Diffusion(config)

    corrupted_np = np.load(args.input)

    if corrupted_np.ndim == 2:
        corrupted_np = corrupted_np[None, ..., None]
    elif corrupted_np.ndim == 3:
        corrupted_np = corrupted_np[..., None]

    corrupted = corrupted_np[args.index].astype(np.float32)
    mask = (corrupted != 0).astype(np.float32)

    corrupted_t = torch.from_numpy(corrupted).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).to(device)

    corrupted_t = normalize_img(corrupted_t)

    cond = torch.cat([corrupted_t, mask_t], dim=1)

    x = torch.randn((1, 1, config.image_size, config.image_size), device=device)

    with torch.no_grad():
        for i in range(config.timesteps - 1, -1, -1):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            pred_noise = ema_model(x, t, cond)
            x = ddpm_step(x, pred_noise, i, diffusion)

            # inpainting constraint
            x = mask_t * corrupted_t + (1.0 - mask_t) * x

    final = denormalize_img(x).squeeze().cpu().numpy()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, final)

    print("Saved:", args.output)


# =========================================================
# CLI
# =========================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="sample_outputs/final.npy")
    p.add_argument("--index", type=int, default=0)

    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--schedule", default="cosine")
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--dim-mults", default="1,2,4")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
    
    
    
    
    
'''
Example run:

python inference_ddpm_inpaint.py \
  --checkpoint checkpoints/ddpm_inpainting.pt \
  --input cifar10_grayscale_32x32_corrupted.npy \
  --output sample_outputs/final.npy
  
'''
    
    
    
    
    
