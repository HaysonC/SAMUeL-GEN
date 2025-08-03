""" 
# model2/modules.py

Module for the soft alignment attention mechanism. As well as the UNet architecture with FiLM conditioning and RoPE.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# weight initialization helper
def init_weights(module):
    """Initialize weights for Conv1d, Linear, and normalization layers."""
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        if hasattr(module, 'weight'):
            nn.init.ones_(module.weight)
        if hasattr(module, 'bias'):
            nn.init.zeros_(module.bias)

# helper for rotary half rotation
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class AttentionMixWeight(nn.Module):
    """
    A simple weight between global and local attention. With more global attention at the beginning and more local attention at the end.

    context_mix = sqrt(a_g(t)) * global + sqrt(1 - a_g(t)) * local
    """
    def __init__(self, timesteps=1000):
        super().__init__()
        # Fixed linear schedule: more global attention at early timesteps
        self.timesteps = timesteps
    def forward(self, t):
        """
        t: tensor of shape (batch,) of integer timesteps from 0 to timesteps-1
        Returns fixed a_g and a_l weights for mixing global and local attention.
        """
        t = t.float()
        # normalize t to [0,1]
        t_norm = t / float(self.timesteps - 1)
        # a_g decays from 1 to 0; a_l grows from 0 to 1
        a_g = (1.0 - t_norm).view(-1, 1, 1)
        a_l = t_norm.view(-1, 1, 1)
        return a_g, a_l


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for the diffusion time step.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        # t: (B,) ints
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb

class TimeSampler(nn.Module):
    """
    Up or down samples the time dimension of the input features. Should be along FiLM.
    """
    def __init__(self, scale_factor):
        super().__init__()
        self.scale = scale_factor
    def forward(self, x):
        # x: (B, C, T)
        return F.interpolate(x, scale_factor=self.scale, mode='linear', align_corners=False)

class RoPE(nn.Module):
    """
    Rotary Position Embedding for the soft attention mechanism.
    """
    def __init__(self, dim, seq_len):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(seq_len).float()
        sinusoid = torch.einsum('n,d->nd', pos, inv_freq)
        self.register_buffer('cosine', torch.cos(sinusoid).unsqueeze(0))
        self.register_buffer('sine', torch.sin(sinusoid).unsqueeze(0))
    def forward(self, x):
        # x: (B, heads, T, dim)
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = torch.stack([x1 * self.cosine - x2 * self.sine,
                                 x1 * self.sine + x2 * self.cosine], dim=-1)
        return x_rotated.flatten(-2)

class SoftAlignLocalAttention(nn.Module):
    """
    Soft alignment module for aligning audio features. Consider the neighboring (n=17, 8 left, 8 right) time steps

    Uses an nn.Sequential to create the Query and Key weights for the attention mechanism.
    """
    def __init__(self, dim, heads=8, window_size=16):
        super().__init__()
        self.heads = heads
        self.window = window_size
        self.scale = (dim // heads) ** -0.5
        self.to_q = nn.Conv1d(dim, dim, 1)
        self.to_k = nn.Conv1d(dim, dim, 1)
        self.to_v = nn.Conv1d(dim, dim, 1)
        self.to_out = nn.Conv1d(dim, dim, 1)
    
    def forward(self, x, cond):
        # x, cond: (B, C, T) where C=channels, T=temporal
        B, C, T = x.shape
        # project
        q_proj = self.to_q(x)
        k_proj = self.to_k(cond)
        v_proj = self.to_v(cond)
        # pad k and v for sliding window to preserve sequence length
        pad_left = self.window // 2
        pad_right = self.window - pad_left - 1
        k_proj = F.pad(k_proj, (pad_left, pad_right))
        v_proj = F.pad(v_proj, (pad_left, pad_right))
        # Unfold k and v: (B, C, T_unf, window)
        k_unf = k_proj.unfold(2, self.window, 1)
        v_unf = v_proj.unfold(2, self.window, 1)
        T_unf = k_unf.size(2)  # should equal original T
        head_dim = C // self.heads
        if C % self.heads != 0:
            raise ValueError(f"Channel count {C} not divisible by heads {self.heads}")
        # reshape q, k, v for multi-head
        # q: (B, C, T) -> (B, heads, head_dim, T)
        q = q_proj.reshape(B, self.heads, head_dim, T)
        # k,v: (B, C, T_unf, window) -> (B, heads, head_dim, T_unf, window)
        k = k_unf.reshape(B, self.heads, head_dim, T_unf, self.window)
        v = v_unf.reshape(B, self.heads, head_dim, T_unf, self.window)
        # compute attention
        q = q.unsqueeze(-1)  # (B, heads, head_dim, T_unf, 1)
        dots = (q * k).sum(2) * self.scale  # (B, heads, T_unf, window)
        attn = torch.softmax(dots, dim=-1)
        out = (attn.unsqueeze(2) * v).sum(-1)  # (B, heads, head_dim, T_unf)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, T)
        # project back to original dimension
        return self.to_out(out)

class SoftAlignGlobalAttention(nn.Module):
    """
    Soft alignment module for global attention across the entire sequence.

    Similarity to SoftAlignLocalAttention, but operates on the entire sequence. And uses a different nn.Sequential to create the Query and Key weights for the attention mechanism.
    """
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        # Rotary embedding module, instantiate on first forward
        self.rope = None
    def forward(self, x, cond):
        # x, cond: (B, T, C)
        B, T, C = x.shape
        h = self.heads
        # project to heads
        q = self.to_q(x).view(B, T, h, C//h).permute(0,2,1,3)  # (B,heads,T,head_dim)
        k = self.to_k(cond).view(B, T, h, C//h).permute(0,2,1,3)
        v = self.to_v(cond).view(B, T, h, C//h).permute(0,2,1,3)
        # apply RoPE on Q/K
        head_dim = C // h
        # Initialize RoPE once to avoid repeated buffer allocations
        if self.rope is None or self.rope.cosine.shape[-1] < T: # type: ignore
            self.rope = RoPE(head_dim, T).to(x.device)
        q = self.rope(q)
        k = self.rope(k)
        # attention
        dots = torch.matmul(q, k.transpose(-2,-1)) * self.scale  # (B,heads,T,T)
        attn = torch.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)  # (B,heads,T,head_dim)
        out = out.permute(0,2,1,3).contiguous().view(B, T, C)
        return self.to_out(out)

class SoftAlignAttentionMixed(nn.Module):
    """
    Abstraction for soft alignment attention mechanism. Should use in the attention weight module. And return a mixed context.

    Although the attention weight is calculated with RoPE, the context is not RoPEed.
    """
    def __init__(self, dim, heads=8, window_size=16, timesteps=1000):
        super().__init__()
        self.local = SoftAlignLocalAttention(dim, heads, window_size)
        self.global_ = SoftAlignGlobalAttention(dim, heads)
        self.mixer = AttentionMixWeight(timesteps)
    def forward(self, x, cond, t, cfg_scale=None, null_cond=None):
        # x: (B,C,T), cond: (B,C,T)
        ctx_local = self.local(x, cond)
        ctx_global = self.global_(x.permute(0,2,1), cond.permute(0,2,1))
        ctx_global = ctx_global.permute(0,2,1)
        a_g, a_l = self.mixer(t)
        context = torch.sqrt(a_g) * ctx_global + torch.sqrt(a_l) * ctx_local
        # Classifier-Free Guidance (CFG)
        if cfg_scale is not None and null_cond is not None:
            # Compute context for null_cond
            ctx_local_null = self.local(x, null_cond)
            ctx_global_null = self.global_(x.permute(0,2,1), null_cond.permute(0,2,1))
            ctx_global_null = ctx_global_null.permute(0,2,1)
            context_null = torch.sqrt(a_g) * ctx_global_null + torch.sqrt(a_l) * ctx_local_null
            # Interpolate
            context = context_null + cfg_scale * (context - context_null)
        return context

class BottleNeckCrossAttention(nn.Module):
    """
    Computes a global attention between the bottle neck and the conditioning signal.

    For the attention, we first project the bottle neck and the conditioning signal to the same dimension on the channel axis, then compute the attention using the SoftAlignAttentionMixed module.

    An addtional RoPE is applied to the conditioning signal and the bottle neck before the attention computation, while the RoPE for the bottle neck is specially designed to handle the temporal dimension.
    
    Again, only the attention weight is computed from the RoPEed signals, while the final output context is not RoPEed.
    """
    def __init__(self, dim, heads=8):
        super().__init__()
        self.proj_x = nn.Conv1d(dim, dim, 1)
        self.proj_cond = nn.Conv1d(dim, dim, 1)
        self.attn = SoftAlignAttentionMixed(dim, heads)
    def forward(self, x, cond, t):
        # x, cond: (B,C,Tb) and (B,C,Tc)
        x_p = self.proj_x(x)
        c_p = self.proj_cond(cond)
        return self.attn(x_p, c_p, t)

class FiLM(nn.Module):
    """
    Feature-wise linear modulation.

    Injects the mixed context into the input features. Also injects the time embedding into the input features.

    Enhanced to use both time embedding and context for more effective modulation.
    """
    def __init__(self, in_channels, cond_dim):
        super().__init__()
        # project time embedding to generate FiLM parameters
        self.time_proj = nn.Linear(cond_dim, in_channels * 2)
        # project context to generate FiLM parameters
        self.context_proj = nn.Conv1d(in_channels, in_channels * 2, kernel_size=1)
    
    def forward(self, x, context, t_emb, cfg_scale=None, null_context=None):
        # x: (B,C,T), context: (B,C,T)
        B,C,T = x.shape
        # use time embedding for FiLM
        time_gamma_beta = self.time_proj(t_emb).view(B, 2, C)
        # use context for FiLM (average over temporal dimension)
        ctx_emb = self.context_proj(context).mean(-1)  # (B, 2C)
        ctx_gamma_beta = ctx_emb.view(B, 2, C)
        # fuse time + context modulation
        gamma_beta = time_gamma_beta + ctx_gamma_beta
        gamma, beta = gamma_beta[:, 0], gamma_beta[:, 1]
        return gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)

class ResNetBlock(nn.Module):
    """
    A ResNet block for audio processing. It uses FiLM to inject the mixed context. The resnet uses the mixed soft align attention mechanism.

    GroupNorm + SiLU (Swish) activation

    Activation often SiLU or Swish (instead of ReLU).

    This combo is very common in diffusion UNet ResNet blocks (e.g., in the original DDPM and Stable Diffusion implementations).
    
    """
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.film = FiLM(channels, cond_dim)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x, context, t_emb):
        h = self.act(self.norm(x))
        h = self.conv1(h)
        h = self.film(h, context, t_emb)
        h = self.act(self.norm(h))
        h = self.conv2(h)
        return x + h

class Conv1dBlock(nn.Module):
    """
    A 1D convolutional block for latent processing. It down samples the temproal axis

    The feature/channel axis (64) typically increases in depth
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.norm = nn.GroupNorm(1, out_ch)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class SoftAlignUNet(nn.Module):
    """
    A U-Net architecture with soft alignment for audio processing. It uses the mixed soft align attention mechanism.

    The architecture is similar to the original UNet, but with the addition of the soft alignment attention mechanism. We also use the FiLM module to inject the mixed context and the time embedding into the input features.

    Even at the attention bottleneck layer, use the same t_emb to modulate attention output or FiLM layers, especially if context from cond is time-invariant.
    """
    def __init__(self, base_ch=64, cond_dim=64, timesteps=1000):
        super().__init__()
        # down blocks for input
        self.downs = nn.ModuleList([
            ResNetBlock(base_ch * (2**i), cond_dim) for i in range(3)
        ])
        # parallel down blocks for conditioning context
        self.cond_downs = nn.ModuleList([
            Conv1dBlock(base_ch * (2**i), base_ch * (2**(i+1))) for i in range(3)
        ])
        # upsample blocks: invert downsampling on h (512->256->128->64)
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(base_ch * 8, base_ch * 4, kernel_size=4, stride=2, padding=1),  # 512->256
            nn.ConvTranspose1d(base_ch * 8, base_ch * 2, kernel_size=4, stride=2, padding=1),  # 512->128
            nn.ConvTranspose1d(base_ch * 4, base_ch * 1, kernel_size=4, stride=2, padding=1),  # 256->64
        ])
        # upsample context in parallel, matching feature channels at each stage
        self.cond_ups = nn.ModuleList([
            nn.ConvTranspose1d(base_ch * 8, base_ch * 8, kernel_size=4, stride=2, padding=1),  # 512->512
            nn.ConvTranspose1d(base_ch * 8, base_ch * 4, kernel_size=4, stride=2, padding=1),  # 512->256
            nn.ConvTranspose1d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1),  # 256->128
        ])
        # ResNet blocks after concatenation: use cond_dim matching context channels at each stage
        # ResNet blocks after concatenation: always use global cond_dim (time_emb size)
        self.up_resnets = nn.ModuleList([
            ResNetBlock(base_ch * 8, cond_dim),  # 256+256=512 -> cond_dim=time_emb dim
            ResNetBlock(base_ch * 4, cond_dim),  # 128+128=256
            ResNetBlock(base_ch * 2, cond_dim),  # 64+64=128
        ])
        # bottleneck cross-attention between x and downsampled cond
        self.mid_attn = BottleNeckCrossAttention(base_ch * 8)
        self.mid_resnet = ResNetBlock(base_ch * 8, cond_dim)
        # time embedding for FiLM inside UNet
        self.time_emb = SinusoidalTimeEmbedding(cond_dim)
        # final 1x1 conv to restore base_ch channels
        self.final_conv = nn.Conv1d(base_ch * 2, base_ch, kernel_size=1)
        self.apply(init_weights)
        # Zero-init final_conv for stable initial predictions
        nn.init.zeros_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

    def forward(self, x, context, t):
        # x: (B, C, T), context: (B, C, T), t: (B,)
        skips = []
        h = x
        hc = context
        t_emb = self.time_emb(t)
        # Down path: ResNetBlock then Conv1dBlock downsampling
        for i, down in enumerate(self.downs):
            # ResNet processing at current resolution and channels
            h = down(h, hc, t_emb)
            skips.append(h)
            # downsample both h and context for next stage
            h = self.cond_downs[i](h)
            hc = self.cond_downs[i](hc)
        # Bottleneck cross-attention and FiLM
        h = self.mid_attn(h, hc, t)
        h = self.mid_resnet(h, hc, t_emb)
        # Up path: upsample, concat skip, ResNet
        # Up path: upsample h and hc, concat h with skip, then ResNet
        for i, (up, up_resnet, skip) in enumerate(zip(self.ups, self.up_resnets, reversed(skips))):
            h = up(h)
            hc = self.cond_ups[i](hc)
            h = torch.cat([h, skip], dim=1)
            h = up_resnet(h, hc, t_emb)
        # restore to base channels
        h = self.final_conv(h)
        return h

