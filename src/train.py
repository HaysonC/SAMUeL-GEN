"""
# model2/train.py

# Training script for the latent diffusion model with soft alignment attention and FiLM conditioning.
"""

from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from config import DATA_KAGGLE_PATH

from config import (
    TRAINING_TIMESTEPS as TIMESTEPS,
    TRAINING_LR as LR,
    TRAINING_EPOCHS as EPOCHS,
    TRAINING_BATCH_SIZE as BATCH_SIZE,
    TRAINING_PATIENCE as PATIENCE,
    TRAINING_COND_DROP_PROB as COND_DROP_PROB,
    TRAINING_CFG_SCALE as CFG_SCALE,
    TRAINING_EMA_DECAY as EMA_DECAY,
    LOSS_WEIGHT_MODE
)

from config import SYSTEM_DEVICE as DEVICE

from model import DiffusionModel
from diffusion import Diffusion
from data_utils import prepare_data
from utils import get_v_target, EMA

import kagglehub

# Device
device = torch.device(DEVICE)
print(f"Using device: {device}")
# Load and prepare data
# prepare_data should return accompaniment and vocals tensors, shape (N, C, T)
data_path = kagglehub.dataset_download(DATA_KAGGLE_PATH)

accomp, vocals, _, _ = prepare_data(data_path)
assert accomp.shape == vocals.shape, "Accomp and vocals must have same shape"

# Swap roles: vocals as input, accomp as target
dataset = TensorDataset(vocals, accomp)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# Initialize model and diffusion
diffusion = Diffusion(timesteps=TIMESTEPS).to(device)
model = DiffusionModel().to(device)

# Param Count
print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")


optimizer = AdamW(model.parameters(), lr=LR)
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

# EMA for model
ema = EMA(model, decay=EMA_DECAY)
ema.register()


def snr_weighted_mse_loss(pred, target, snr, mode='raw', gamma=2.0, max_snr_val=10.0):
    """
    Calculates the SNR-weighted MSE loss with various weighting modes.
    """
    mse = (pred - target) ** 2
    
    if mode == 'raw':
        weight = torch.clamp(snr, max=max_snr_val)
    elif mode == 'normalized':
        weight = snr / (snr.mean() + 1e-5)
    elif mode == 'log':
        weight = torch.log(snr + 1e-5)
    elif mode == 'inverse':
        # As in Imagen or recent DiT papers
        weight = torch.minimum(torch.tensor(gamma, device=snr.device) / (snr + 1e-5), torch.tensor(max_snr_val, device=snr.device))
    else:
        raise ValueError(f"Unknown loss weight mode: {mode}")

    weighted_mse = mse * weight
    return weighted_mse.mean()

# Training loop

best_val_loss = float('inf')
best_epoch = -1
epochs_no_improve = 0
ckpt_dir = Path('checkpoints')
ckpt_dir.mkdir(exist_ok=True)
train_losses = []
val_losses = []

train_mses = []
val_mses = []



for epoch in range(1, EPOCHS+1):
    model.train()
    total_train_loss = 0.0
    total_train_mse = 0.0
    train_bar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{EPOCHS}")
    for batch_idx, (x_voc, x_accomp) in enumerate(train_bar):
        x_voc = x_voc.to(device)      # vocals as input
        x_accomp = x_accomp.to(device) # accomp as target
        t = diffusion.sample_timesteps(x_accomp.size(0), device)
        noise = torch.randn_like(x_accomp)
        x_t = diffusion.q_sample(x_start=x_accomp, t=t, noise=noise)
        # Classifier-Free Guidance (CFG) setup

        use_cfg = torch.rand(1).item() < COND_DROP_PROB
        if use_cfg:
            # Drop condition: use zeros as null_cond
            null_cond = torch.zeros_like(x_voc)
            v_pred = model(x_t, x_voc, t, cfg_scale=CFG_SCALE, null_cond=null_cond)
        else:
            null_cond = None
            v_pred = model(x_t, x_voc, t)
        v_target = get_v_target(diffusion, x_accomp, noise, t)
        snr = diffusion.get_snr(t).to(device).view(-1, 1, 1)
        loss = snr_weighted_mse_loss(v_pred, v_target, snr, mode=LOSS_WEIGHT_MODE)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update()
        total_train_loss += loss.item() * x_accomp.size(0)
        batch_mse = ((v_pred - v_target) ** 2).mean().item()
        total_train_mse += batch_mse * x_accomp.size(0)
        train_bar.set_postfix({"batch": batch_idx, "loss": loss.item(), "MSE": ((v_pred - v_target) ** 2).mean().item()})
        if batch_idx % 180 == 0:
            tqdm.write(f"[DEBUG] Train Epoch {epoch} Batch {batch_idx}: v_target std={v_target.std().item():.4f}, v_pred std={v_pred.std().item():.4f}")
            tqdm.write(f"[DEBUG] SNR stats: mean={snr.mean().item():.4f}, std={snr.std().item():.4f}, min={snr.min().item():.4f}, max={snr.max().item():.4f}")
    avg_train_loss = total_train_loss / len(train_ds)
    avg_train_mse = total_train_mse / len(train_ds)
    train_losses.append(avg_train_loss)
    train_mses.append(avg_train_mse)

    # Validation
    model.eval()
    total_val_loss = 0.0
    total_val_mse = 0.0
    val_bar = tqdm(val_loader, desc=f"Val Epoch {epoch}/{EPOCHS}")
    with torch.no_grad():
        for batch_idx, (x_voc, x_accomp) in enumerate(val_bar):
            x_voc = x_voc.to(device)
            x_accomp = x_accomp.to(device)
            t = torch.randint(0, TIMESTEPS, (x_accomp.size(0),), device=device)
            noise = torch.randn_like(x_accomp)
            x_t = diffusion.q_sample(x_start=x_accomp, t=t, noise=noise)
            # For validation, always use CFG for evaluation robustness
            null_cond = torch.zeros_like(x_voc)
            cfg_scale = 2.0
            v_pred = model(x_t, x_voc, t, cfg_scale=cfg_scale, null_cond=null_cond)
            v_target = get_v_target(diffusion, x_accomp, noise, t)
            snr = diffusion.get_snr(t).to(device).view(-1, 1, 1)
            batch_loss = snr_weighted_mse_loss(v_pred, v_target, snr, mode=LOSS_WEIGHT_MODE).item() * x_accomp.size(0)
            batch_mse = ((v_pred - v_target) ** 2).mean().item()
            total_val_loss += batch_loss
            total_val_mse += batch_mse * x_accomp.size(0)
            val_bar.set_postfix({"batch": batch_idx, "loss": batch_loss / x_accomp.size(0), "MSE": batch_mse})
            if batch_idx % 5 == 0:
                tqdm.write(f"[DEBUG] Val Epoch {epoch} Batch {batch_idx}: batch_loss={batch_loss / x_accomp.size(0):.4f}, raw MSE={batch_mse:.4f}")
            if batch_idx % 10 == 0:
                tqdm.write(f"[DEBUG] Val Epoch {epoch} Batch {batch_idx}: v_target std={v_target.std().item():.4f}, v_pred std={v_pred.std().item():.4f}")
    avg_val_loss = total_val_loss / len(val_ds)
    avg_val_mse = total_val_mse / len(val_ds)
    val_losses.append(avg_val_loss)
    val_mses.append(avg_val_mse)
    print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val MSE={avg_val_mse:.4f}")

    # Save latest checkpoint
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, ckpt_dir / 'model2_latest.pth')

    # Save best checkpoint
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        epochs_no_improve = 0
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, ckpt_dir / 'model2_best.pth')
        print(f"[INFO] Best model saved at epoch {epoch} with val loss {best_val_loss:.4f}")
    else:
        epochs_no_improve += 1

    scheduler.step()

    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping at epoch {epoch} due to no improvement in val loss for {PATIENCE} epochs.")
        break

print(f"Training complete. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
print("Train Losses:", train_losses)
print("Val Losses:", val_losses)

# Plot losses (log scale)

import matplotlib.pyplot as plt
# Plot Losses
plt.figure(figsize=(8,4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training and Validation Losses (Log Scale)')
plt.legend()
plt.tight_layout()
# Enhanced labeling: xticks at each epoch, grid and minor ticks
epochs = list(range(1, len(train_losses) + 1))
plt.xticks(epochs)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.savefig('loss_curve.png')
plt.show()

# Plot MSEs
plt.figure(figsize=(8,4))
plt.plot(train_mses, label='Train MSE')
plt.plot(val_mses, label='Val MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.yscale('log')
plt.title('Training and Validation MSEs (Log Scale)')
plt.legend()
plt.tight_layout()
# Enhanced labeling: xticks at each epoch, grid and minor ticks
epochs = list(range(1, len(train_mses) + 1))
plt.xticks(epochs)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.savefig('mse_curve.png')
plt.show()

# Experiment summary
print("\n=== Experiment Summary ===")
print(f"Total epochs: {len(train_losses)}")
print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
print(f"Final train loss: {train_losses[-1]:.4f}")
print(f"Final val loss: {val_losses[-1]:.4f}")
print("Loss curve saved as loss_curve.png")
print("MSE curve saved as mse_curve.png")
print("Train MSEs:", train_mses)
print("Val MSEs:", val_mses)
