import torch
from pathlib import Path
import math
from torch.optim.lr_scheduler import _LRScheduler

from diffusers import StableAudioPipeline
import torchaudio
from pathlib import Path

class LinearWarmupCosineAnnealing(_LRScheduler):
    """
    Custom LR scheduler that combines a linear warmup phase with a cosine annealing phase.
    This is useful for stabilizing training in the initial stages and then smoothly
    decaying the learning rate.
    """
    def __init__(self, optimizer, warmup_steps, max_steps, eta_min: float = 0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            lr_scale = self.last_epoch / self.warmup_steps
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        
        # Cosine annealing phase
        progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        cos_out = math.cos(math.pi * progress)
        
        return [self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + cos_out)
                for base_lr in self.base_lrs]


class EMA:
    """
    Exponential Moving Average for model parameters.
    """
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ==== v-objective helpers ====
def get_v_target(diffusion, z0, epsilon, t):
    """Compute target v = alpha_t * epsilon + beta_t * z0"""
    alphas_cumprod = diffusion.alphas_cumprod
    alphas_t = torch.index_select(alphas_cumprod, 0, t).to(z0.device)
    one_minus = 1.0 - alphas_cumprod
    var_t = torch.index_select(one_minus, 0, t).to(z0.device)
    a = torch.sqrt(alphas_t + 1e-7).view(-1, 1, 1)
    b = torch.sqrt(var_t + 1e-7).view(-1, 1, 1)
    return a * epsilon + b * z0

def extract(a, t, x_shape):
    # Helper to extract coefficients for batch
    out = a.gather(-1, t)
    while len(out.shape) < len(x_shape):
        out = out.unsqueeze(-1)
    return out


def save_sampling_animation(frames, save_path: str | Path = "sampling_animation.mp4", interval=50, cmap="magma"):
    """Create and save a matplotlib animation (MP4) from a list of spectrogram frames."""
    import matplotlib.pyplot as plt
    from matplotlib import animation
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(frames[0], aspect='auto', origin='lower', cmap=cmap)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    def update(frame):
        im.set_data(frame)
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    # Always save as mp4
    anim.save(save_path, writer='ffmpeg')
    plt.close(fig)
    print(f"Animation saved to {save_path}")

def decode(output: torch.Tensor, device="cpu", savePath: str | Path = Path("reconstructed.wav")):
    REPO = "stabilityai/stable-audio-open-1.0"
    dtype = torch.float32
    pipe = StableAudioPipeline.from_pretrained(REPO, torch_dtype=dtype)
    autoencoder = pipe.vae.to(device)
    print("Autoencoder loaded")
    with torch.no_grad():
        output = output.to(device)
        if output.shape != (1, 64, 1024):
            output = output.transpose(1, 2)
        audio = autoencoder.decode(output).sample
    print(f"Audio shape: {audio.shape}")
    sample_rate = 44100  # 44.1 kHz
    torchaudio.save(str(savePath), audio.squeeze(0).cpu().clamp(-1,1), sample_rate=sample_rate)
    return audio