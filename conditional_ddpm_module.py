import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
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
# Schedules
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


# =========================================================
# Model blocks
# =========================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = heads * dim_head
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm2d(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift=None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
        self.block1 = Block(dim, dim_out, groups)
        self.block2 = Block(dim_out, dim_out, groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale_shift = self.mlp(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.block1(x, scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Downsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.op = nn.Conv2d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


# =========================================================
# Conditional UNet
# =========================================================

class ConditionalUNet(nn.Module):
    def __init__(
        self,
        dim: int = 48,
        dim_mults: Tuple[int, ...] = (1, 2, 4),
        in_channels: int = 3,
        out_channels: int = 1,
        groups: int = 8,
    ):
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
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i == len(in_out) - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim, groups=groups),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim, groups=groups),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=groups)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=groups)

        self.ups = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = i == len(in_out) - 1
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim, groups=groups),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=groups),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_res = ResnetBlock(dim * 2, dim, time_emb_dim=time_dim, groups=groups)
        self.final_conv = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
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
# Diffusion
# =========================================================

class Diffusion:
    def __init__(self, config: Config):
        self.config = config

        if config.schedule == "cosine":
            betas = cosine_beta_schedule(config.timesteps)
        else:
            betas = linear_beta_schedule(config.timesteps, config.beta_start, config.beta_end)

        self.betas = betas.to(config.device)
        self.alphas = (1.0 - self.betas).to(config.device)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(config.device)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)

        a = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)

        return a * x0 + b * noise, noise


def ddpm_step(x_t, pred_noise, t, diffusion: Diffusion):
    beta_t = diffusion.betas[t]
    alpha_t = diffusion.alphas[t]
    alpha_bar_t = diffusion.alpha_bars[t]

    mean = (1.0 / torch.sqrt(alpha_t)) * (
        x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
    )

    if t > 0:
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(beta_t) * noise

    return mean
