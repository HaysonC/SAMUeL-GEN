# diffusion.py

import torch
import torch.nn as nn
import math
from tqdm import tqdm
from utils import extract


class Diffusion(nn.Module):
    def __init__(self, timesteps=800, s=0.008):
        super().__init__()
        self.timesteps = timesteps
        self.s = s
        
        betas = self._cosine_beta_schedule(timesteps, s)
        self.register_buffer('betas', betas)
        
        alphas = 1. - betas
        self.register_buffer('alphas', alphas)
        
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))

        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)

        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)
        
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)

        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)

        print(f"sqrt_alphas_cumprod: first={sqrt_alphas_cumprod[0].item():.4f}, last={sqrt_alphas_cumprod[-1].item():.4f}")
        print(f"sqrt_one_minus_alphas_cumprod: first={sqrt_one_minus_alphas_cumprod[0].item():.4f}, last={sqrt_one_minus_alphas_cumprod[-1].item():.4f}")

        snr = alphas_cumprod / (1 - alphas_cumprod)
        self.register_buffer('snr', snr)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Alias for q_posterior for compatibility.
        """
        return self.q_posterior(x_start, x_t, t)

    def p_mean_variance(self, model, x, t, clip_denoised: bool, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        
        # Model now predicts noise residual
        pred_noise = model(x, t, **model_kwargs)
        # Recover x_start from noise prediction
        x_start = self.predict_start_from_noise(x, t, pred_noise)

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, model, x, t, clip_denoised=True, model_kwargs=None):
        model_mean, _, model_log_variance = self.p_mean_variance(model, x=x, t=t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, 1, 1)
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device, model_kwargs=None):
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((shape[0],), i, device=device, dtype=torch.long), model_kwargs=model_kwargs)
        return img

    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1", target_z_0=None, diversity_weight=0.1):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_z_0 = denoise_model(x_t, t)

        if target_z_0 is None:
            target_z_0 = x_start

        if loss_type == 'l1':
            loss = (target_z_0 - predicted_z_0).abs().mean()
        elif loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(target_z_0, predicted_z_0)
        elif loss_type == "snr_mse":
            snr = extract(self.snr, t, x_start.shape).to(x_start.device)
            mse = torch.nn.functional.mse_loss(target_z_0, predicted_z_0, reduction='none')
            loss = (snr * mse).mean()
        else:
            raise NotImplementedError()

        # Diversity loss
        batch_variance = torch.var(predicted_z_0.flatten(1), dim=0).mean()
        diversity_loss = -torch.log(batch_variance + 1e-6)

        total_loss = loss + diversity_weight * diversity_loss
        
        return total_loss, diversity_loss

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672 (Improved DDPM).
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def get_posterior_log_variance(self, t):
        """
        Calculates the log of the posterior variance for a given timestep.
        """
        return extract(self.posterior_log_variance_clipped, t, (1,1,1))

    def get_snr(self, t):
        """
        Get SNR for a given timestep t.
        """
        return extract(self.snr, t, t.shape)

    def sample_timesteps(self, n, device):  
        """
        Sample timesteps for a batch of size n.
        """
        
        t = torch.randint(0, self.timesteps, (n,), device=device, dtype=torch.long)
        
        return t

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (add noise) at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
