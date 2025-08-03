```
Input Noisy Latent x_t (B, 64, 1024)
        │
        ├─── Conv1D (x_encoder) ──────┐
        │                            │
Cond Input c (B, 64, 1024)           │
        │                            │
        ├─── Conv1D (cond_encoder)───┤
                                     │
                           Apply RoPE on x_encoded and cond_encoded (on 1024 axis)
                                     │
         ┌───────────────────────────┴──────────────────────────────┐
         │                                                          │
   Local Soft Attention (window=16, no RoPE on cond_raw)        Global Soft Attention (full with RoPE)
         │                                                          │
         └─────────────── Mix context via noise schedule weights ──┘
                                     │
                         Combined Context (context_mix) (B, C, T)
                                     │
         ┌───────────────────────────────────────────────────────────┐
         │                                                           │
         │              FiLM γ, β computed from                      │
         │             (context_mix + timestep embedding)            │
         │                                                           │
         ▼                                                           ▼
Input x_raw + t_emb (time embedding via MLP)            Conditioning FiLM parameters (γ, β)
         │                                                           │
         └───────────── FiLM Injection into Input Conv1D ────────────┘
                                     │
                              UNet Downsampling (Conv1D + Swish + Norm + Dropout)
                                     │
                          (Repeat FiLM injection at each block)
                                     │
                            Bottleneck (lowest temporal resolution)
                                     │
         ┌───────────────────────────────────────────────────────────┐
         │              Cross-Attention between bottleneck           │
         │             features and context_mix (with projection)    │
         └───────────────────────────────────────────────────────────┘
                                     │
                            UNet Upsampling (Conv1D + Swish + Norm + Dropout)
                                     │
                          (FiLM injection continues here too)
                                     │
                           Final Conv1D Layer → Predict v (B, 64, 1024)

```
