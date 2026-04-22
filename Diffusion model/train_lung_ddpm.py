import os
import math
from pathlib import Path
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


#Config
DATA_DIR = "dataset"
SAVE_DIR = "outputs"
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 120
LR = 1e-4
TIMESTEPS = 1000
BASE_CHANNELS = 64
NUM_WORKERS = 2
SAVE_EVERY = 5
SAMPLE_COUNT = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESUME = True
RESUME_PATH = "outputs/ddpm_epoch_100.pth"

os.makedirs(SAVE_DIR, exist_ok=True)

def png_uint16_to_hu(png_u16):
    x4095 = (png_u16.astype(np.float32) / 65535.0) * 4095.0
    return x4095 - 1024.0  
    
def apply_window_hu(img_hu, center = -600, width = 1500):
    lo = center - width / 2.0
    hi = center + width / 2.0
    x = np.clip(img_hu, lo, hi)
    x = (x - lo) / (hi - lo)
    return x.astype(np.float32)

def preprocess_roi_from_uint16png(img, size = 128):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #resize only if needed
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    hu = png_uint16_to_hu(img)
    x = apply_window_hu(hu, center=-600, width=1500) 
    return x 

#Dataset
class LungROIDataset(Dataset):
    def __init__(self, root_dir, image_size=128):
        self.root_dir = Path(root_dir)
        self.files = sorted([
            p for p in self.root_dir.iterdir()
            if p.suffix.lower() in [".png"]
        ])
        self.image_size = image_size

        if len(self.files) == 0:
            raise ValueError(f"No PNG images found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.files[idx]), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Failed to read image: {self.files[idx]}")

        #preprocess to [0,1]
        img = preprocess_roi_from_uint16png(img, size=self.image_size)

        #convert to tensor and normalize to [-1,1]
        img = torch.from_numpy(img).float().unsqueeze(0)   # [1,H,W]
        img = (img * 2.0) - 1.0

        return img 

#Time embedding
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=time.device) * -emb_scale)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


#Blocks
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_ch)
            )

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t=None):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)

        if self.time_mlp is not None and t is not None:
            time_emb = self.time_mlp(t)
            h = h + time_emb[:, :, None, None]

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.act(h)

        return h + self.residual(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        x = self.block(x, t)
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.block = ConvBlock(out_ch + skip_ch, out_ch, time_emb_dim)

    def forward(self, x, skip, t):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, t)
        return x

#UNet
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, time_emb_dim=256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = ConvBlock(in_channels, base_channels, time_emb_dim)

        self.down1 = DownBlock(base_channels, base_channels * 2, time_emb_dim)   # 128 -> 64
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)  # 64 -> 32
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, time_emb_dim)  # 32 -> 16

        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 8, time_emb_dim)

        self.up1 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 4, time_emb_dim)  # 16 -> 32
        self.up2 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, time_emb_dim)  # 32 -> 64
        self.up3 = UpBlock(base_channels * 2, base_channels * 2, base_channels, time_emb_dim)      # 64 -> 128

        self.final_conv = nn.Sequential(
            ConvBlock(base_channels + base_channels, base_channels, time_emb_dim),
            nn.Conv2d(base_channels, out_channels, 1)
        )

    def forward(self, x, t):
        t = self.time_mlp(t)

        x0 = self.init_conv(x, t)                  # [B, C, 128, 128]

        x1, s1 = self.down1(x0, t)                 # s1: [B, 2C, 128,128], x1: [B,2C,64,64]
        x2, s2 = self.down2(x1, t)                 # s2: [B, 4C, 64,64], x2: [B,4C,32,32]
        x3, s3 = self.down3(x2, t)                 # s3: [B, 8C, 32,32], x3: [B,8C,16,16]

        x = self.bottleneck(x3, t)

        x = self.up1(x, s3, t)                     # [B, 4C, 32,32]
        x = self.up2(x, s2, t)                     # [B, 2C, 64,64]
        x = self.up3(x, s1, t)                     # [B, C, 128,128]

        x = torch.cat([x, x0], dim=1)
        x = self.final_conv[0](x, t)
        x = self.final_conv[1](x)
        return x


#Diffusion
class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device="cuda"):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_hat = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hat", alpha_hat)
        self.register_buffer("sqrt_alpha_hat", torch.sqrt(alpha_hat))
        self.register_buffer("sqrt_one_minus_alpha_hat", torch.sqrt(1.0 - alpha_hat))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1.0 - torch.cat([torch.tensor([1.0], device=device), alpha_hat[:-1]])) / (1.0 - alpha_hat)
        self.register_buffer("posterior_variance", posterior_variance)

    def sample_timesteps(self, batch_size):
        return torch.randint(low=0, high=self.timesteps, size=(batch_size,), device=self.device)

    def noise_images(self, x, t):
        noise = torch.randn_like(x)
        sqrt_alpha_hat_t = self.sqrt_alpha_hat[t][:, None, None, None]
        sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t][:, None, None, None]
        x_t = sqrt_alpha_hat_t * x + sqrt_one_minus_alpha_hat_t * noise
        return x_t, noise

    def forward(self, x):
        t = self.sample_timesteps(x.shape[0])
        x_t, noise = self.noise_images(x, t)
        predicted_noise = self.model(x_t, t.float())
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, n, image_size=128, channels=1):
        self.model.eval()
        x = torch.randn((n, channels, image_size, image_size), device=self.device)

        for i in reversed(range(1, self.timesteps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.float32)

            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_hat = self.alpha_hat[i]
            sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[i]
            sqrt_recip_alpha = self.sqrt_recip_alphas[i]

            predicted_noise = self.model(x, t)

            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = sqrt_recip_alpha * (
                x - ((1 - alpha) / sqrt_one_minus_alpha_hat) * predicted_noise
            ) + torch.sqrt(beta) * noise

        x = x.clamp(-1, 1)
        x = (x + 1) / 2
        return x


#Utils
def save_samples(images, path, nrow=4):
    utils.save_image(images, path, nrow=nrow)


def main():
    print(f"Using device: {DEVICE}")

    dataset = LungROIDataset(DATA_DIR, image_size=128)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False
    )

    model = SimpleUNet(
        in_channels=1,
        out_channels=1,
        base_channels=BASE_CHANNELS,
        time_emb_dim=256
    ).to(DEVICE)

    diffusion = DDPM(
        model=model,
        timesteps=TIMESTEPS,
        beta_start=1e-4,
        beta_end=2e-2,
        device=DEVICE
    ).to(DEVICE)

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=LR)

    start_epoch = 0

    #Resume from checkpoint
    if RESUME and os.path.exists(RESUME_PATH):
        print(f"Resuming from checkpoint: {RESUME_PATH}")
        checkpoint = torch.load(RESUME_PATH, map_location=DEVICE)

        diffusion.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]

        print(f"Resumed from epoch {start_epoch}")
    elif RESUME:
        print(f"Checkpoint not found: {RESUME_PATH}")
        print("Starting from scratch.")

    #Training loop
    for epoch in range(start_epoch, EPOCHS):
        diffusion.train()
        epoch_loss = 0.0

        for batch in loader:
            batch = batch.to(DEVICE)

            optimizer.zero_grad()
            loss = diffusion(batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")

        if (epoch + 1) % SAVE_EVERY == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"ddpm_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": diffusion.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)

            samples = diffusion.sample(SAMPLE_COUNT, image_size=IMG_SIZE, channels=1)
            sample_path = os.path.join(SAVE_DIR, f"samples_epoch_{epoch+1}.png")
            save_samples(samples, sample_path, nrow=4)

            print(f"Saved checkpoint: {ckpt_path}")
            print(f"Saved samples:    {sample_path}")

    final_ckpt = os.path.join(SAVE_DIR, f"ddpm_final_epoch_{EPOCHS}.pth")
    torch.save({
        "epoch": EPOCHS,
        "model_state_dict": diffusion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_ckpt)

    print(f"Training complete. Final model saved to {final_ckpt}")

if __name__ == "__main__":
    main()