r"""
# model2/model.py

---

## Latent Diffusion Model (with v-objective + Soft Alignment + FiLM)

---

### Objective

* **Latent diffusion model** operating on latent shape `(B, 64, 1024)`
* Predict **v**, not ε or x₀ (→ more stable training)
* Use **sinusoidal `t_emb`** (timestep embedding) for diffusion guidance

---

### Embeddings & Inputs

* **Separate `Conv1D` encoders** for:

  * `x_input` (the noisy latent sample)
  * `cond` (conditioning input — e.g., text/audio/etc.)

* **RoPE** is applied to `x_encoded` and `cond_encoded` **on the 1024 axis (time)** to enable rotationally-aware attention

* `t_emb` is sinusoidally embedded → passed through MLP → injected throughout network via **FiLM**

---

### Soft Alignment Attention (Dual Mode)

#### Local Attention

* For each window (size 16), compute **soft attention**:

  * Energy function: `e = Q · Kᵀ / sqrt(d)`
  * `Q = x_encoded_raw`, `K = cond_raw` (no RoPE for locality)
* Applies within **sliding windows of 16** (per x token)
* Output: `context_local = softmax(e_local) @ cond_raw`

#### Global Attention

* Apply RoPE to both `x_encoded` and `cond_encoded`
* Compute **full global attention** between them

  * `context_global = softmax(e_global) @ cond_raw`

#### Mixing

* Combine local and global context with a **noise-schedule-based weighting**:

  $$
  a_g(t) = \sqrt{\bar{\alpha}_t}, \quad a_l(t) = 1 - a_g(t)
  $$

  $$
  \text{context} = a_g(t) \cdot \text{context_global} + a_l(t) \cdot \text{context_local}
  $$

---

### Downstream Pipeline

#### Input Prep

* **`x_input_raw` + `t_emb`** → `Conv1D` → feature map
* Apply **FiLM conditioning** using `context` from above

  * FiLM:

    $$
    \text{FiLM}(x) = \gamma(x_{\text{cond}}, t) \cdot x + \beta(x_{\text{cond}}, t)
    $$
  * Both `γ, β` derived from joint MLP of `t_emb` and `context`

---

### Core Network: UNet over Latents

* **UNet-style Conv1D blocks** with:

  * Swish activations
  * Normalization (LayerNorm or GroupNorm)
  * Dropout (in attention and convs)
  * Downsample along the **time dimension (1024 → ...)**
  * Channel expansion as you go deeper

#### Bottleneck

* **Cross-attention** between the latent bottleneck and `context` (from cond)
* Bottleneck has fewer time steps (e.g., 128 or 64)
* **Use projected positional embeddings** (or RoPE) to align different time scales between `T_b` (bottleneck) and `T_cond`

  * Use interpolated or sinusoidal emb for `T_b` and `T_cond`

---

### Decoder

* Conv1D upsampling (UNet-style)
* FiLM injection continues
* Time embedding reused
* Conditioning modulates decoder throughout

---

### Final Output

* Predict `v` (velocity) — the Denoising Diffusion Implicit Model output:

  $$
  \hat{v}_\theta(x_t, t, c)
  $$

---

### Additional Techniques

* **Residual connections** in conv blocks
* Optional **GRN (Gated Residual Networks)** inside FiLM layers
* Optional **EMA** over weights
* Final prediction uses **(projected) `Conv1D → 64 × 1024` latent**

---

## Summary of Components

| Component           | Strategy                                                      |
| ------------------- | ------------------------------------------------------------- |
| Embedding           | `sin(t)` + MLP → FiLM or additive                             |
| Positional Encoding | RoPE on `x_encoded`, `cond_encoded` (for global attention)    |
| Alignment           | Soft attention (global + local) with schedule-based mixing    |
| Context Injection   | FiLM layers across all stages                                 |
| Core Network        | UNet with Swish, dropout, residuals, normalization            |
| Attention           | Local+Global soft align (encoder), bottleneck cross-attention |
| Decoder             | FiLM + Conv1D upsampling                                      |
| Output              | Predict `v`                                                   |

r"""

"""
Instructions:

1. Implement the SinusoidalTimeEmbedding module.
2. Integrate RoPE into the encoder and decoder.
3. Develop SoftAlignLocalAttention and SoftAlignGlobalAttention modules.
4. Create the SoftAlignAttentionMixed module for combining local and global context.
5. Implement the BottleNeckCrossAttention module.
6. Integrate FiLM layers throughout the model.
7. Ensure compatibility with the v-objective for training.
8. Put parameters in configs.py for easy tuning.

"""

from modules import * 
from config import *


class DiffusionModel(nn.Module):
    """
    Main diffusion model that combines all components.
    
    This model integrates the SinusoidalTimeEmbedding, RoPE, SoftAlignLocalAttention,
    SoftAlignGlobalAttention, SoftAlignAttentionMixed, BottleNeckCrossAttention, and FiLM modules.
    
    It is designed to handle both local and global attention mechanisms with soft alignment,
    and applies feature-wise linear modulation for conditioning.
    """
    def __init__(self):
        super().__init__()
        # Model dimensions
        shapes = MODEL_UNET_SHAPES
        ch_in = shapes[0][0]
        # time embedding dimension set to bottleneck channels
        time_dim = shapes[3][0]
        # context dimension same as initial channels
        context_dim = ch_in
        # time embeddings
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        # encoders for x and condition
        self.x_encoder = nn.Conv1d(ch_in, context_dim, kernel_size=3, padding=1)
        self.cond_encoder = nn.Conv1d(ch_in, context_dim, kernel_size=3, padding=1)
        # soft alignment attention for initial context (local+global mix)
        self.align = SoftAlignAttentionMixed(context_dim, window_size=MODEL_WINDOW_SIZE)
        # initial FiLM conditioning: uses context C and time_emb dim
        self.initial_film = FiLM(in_channels=context_dim, cond_dim=time_dim)
        # UNet core
        self.unet = SoftAlignUNet(base_ch=context_dim, cond_dim=time_dim)
        # output projection to predict v
        self.output_conv = nn.Conv1d(context_dim, ch_in, kernel_size=1)
        # initialize weights
        self.apply(init_weights)
    
    def forward(self, x, cond, t, cfg_scale=None, null_cond=None):
        """
        Forward pass for the diffusion model.
        Supports classifier-free guidance (CFG) by passing cfg_scale and null_cond.
        x, cond: (B, C, T), t: (B,) or scalar
        cfg_scale: float or None
        null_cond: (B, C, T) or None
        """
        # time embedding
        t_emb = self.time_emb(t)
        t_emb = self.time_mlp(t_emb)
        # encode inputs
        x_enc = self.x_encoder(x)
        # If cond is None, use zeros for unconditional
        if cond is None:
            cond = torch.zeros_like(x)
        cond_enc = self.cond_encoder(cond)
        # attention context (CFG-aware)
        context = self.align(x_enc, cond_enc, t, cfg_scale=cfg_scale, null_cond=null_cond) if cfg_scale is not None and null_cond is not None else self.align(x_enc, cond_enc, t)
        # initial FiLM modulation (CFG-aware)
        x_mod = self.initial_film(x_enc, context, t_emb, cfg_scale=cfg_scale, null_context=null_cond) if cfg_scale is not None and null_cond is not None else self.initial_film(x_enc, context, t_emb)
        # UNet processing over latent with context and timestep
        h = self.unet(x_mod, context, t)
        # predict v
        v = self.output_conv(h)

        return v
