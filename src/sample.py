import torch
from diffusion import Diffusion
from tqdm import tqdm
from utils import extract

@torch.no_grad()
def p_sample_v(model, diffusion: Diffusion, x, cond=None, guidance_scale=1.0, return_frames=False, return_stats=False, debug=False, clamp_range=(-10, 10)):
    """Sampling loop using v-objective predictions. Optionally collects frames for animation."""
    frames = [] if return_frames else None
    # initialize stats lists
    stats_means = []
    stats_stds = []
    assert clamp_range[0] < clamp_range[1], "clamp_range must be a valid range (min < max)"
    for i in tqdm(range(diffusion.timesteps - 1, -1, -1), desc="v-objective sampling", total=diffusion.timesteps):
        t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
        v_pred = model(x, cond, t)
        if guidance_scale != 1.0 and cond is not None:
            # For unconditional, pass zeros as cond
            v_uncond = model(x, torch.zeros_like(cond), t)
            v_pred = v_uncond + guidance_scale * (v_pred - v_uncond)
        alpha_t = extract(diffusion.sqrt_alphas_cumprod, t, x.shape)
        beta_t = extract(diffusion.sqrt_one_minus_alphas_cumprod, t, x.shape)
        alpha_t = torch.clamp(alpha_t, min=1e-6)
        z0_pred = (x - beta_t * v_pred) / alpha_t
        z0_pred = torch.clamp(z0_pred, *clamp_range)
        mean, var, log_var = diffusion.q_posterior_mean_variance(z0_pred, x, t)
        log_var = log_var.clamp(min=-10, max=1)
        noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
        x = mean + noise * torch.exp(0.5 * log_var)
        x = torch.clamp(x, *clamp_range)
        # record stats
        if return_stats:
            stats_means.append(x.mean().item())
            stats_stds.append(x.std().item())
        if debug and (i % 50 == 0 or i == diffusion.timesteps - 1 or i == 0):
            tqdm.write(f"Step {i}: v_pred mean {v_pred.mean().item():.4f}, std {v_pred.std().item():.4f}, min {v_pred.min().item():.4f}, max {v_pred.max().item():.4f}")
            tqdm.write(f"Step {i}: z0_pred mean {z0_pred.mean().item():.4f}, std {z0_pred.std().item():.4f}, min {z0_pred.min().item():.4f}, max {z0_pred.max().item():.4f}")
            tqdm.write(f"Step {i}: x mean {x.mean().item():.4f}, std {x.std().item():.4f}, min {x.min().item():.4f}, max {x.max().item():.4f}")
            tqdm.write(f"Step {i}: alpha_t mean {alpha_t.mean().item():.4f}, beta_t mean {beta_t.mean().item():.4f}")
        if frames is not None:
            frames.append(x[0].detach().cpu().numpy())
    # return with optional stats
    if return_stats:
        if return_frames:
            return x, frames, stats_means, stats_stds
        return x, stats_means, stats_stds
    # original return
    if return_frames:
        return x, frames
    return x
