import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils


#Config

CKPT_PATH = "outputs/ddpm_final.pth"
OUT_DIR = "synthetic_dataset"
IMG_SIZE = 128
TOTAL_IMAGES = 1604
BATCH_SIZE = 16
TIMESTEPS = 1000
BASE_CHANNELS = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        self.down1 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, time_emb_dim)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 8, time_emb_dim)

        self.up1 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 4, time_emb_dim)
        self.up2 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, time_emb_dim)
        self.up3 = UpBlock(base_channels * 2, base_channels * 2, base_channels, time_emb_dim)

        self.final_conv = nn.Sequential(
            ConvBlock(base_channels + base_channels, base_channels, time_emb_dim),
            nn.Conv2d(base_channels, out_channels, 1)
        )

    def forward(self, x, t):
        t = self.time_mlp(t)

        x0 = self.init_conv(x, t)
        x1, s1 = self.down1(x0, t)
        x2, s2 = self.down2(x1, t)
        x3, s3 = self.down3(x2, t)

        x = self.bottleneck(x3, t)
        x = self.up1(x, s3, t)
        x = self.up2(x, s2, t)
        x = self.up3(x, s1, t)

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

    @torch.no_grad()
    def sample(self, n, image_size=128, channels=1):
        self.model.eval()
        x = torch.randn((n, channels, image_size, image_size), device=self.device)

        for i in reversed(range(1, self.timesteps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.float32)

            beta = self.betas[i]
            alpha = self.alphas[i]
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

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

    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    diffusion.load_state_dict(checkpoint["model_state_dict"])

    saved = 0
    while saved < TOTAL_IMAGES:
        current_batch = min(BATCH_SIZE, TOTAL_IMAGES - saved)
        samples = diffusion.sample(current_batch, image_size=IMG_SIZE, channels=1)

        for i in range(current_batch):
            out_path = os.path.join(OUT_DIR, f"diffusion_img_{saved:04d}.png")
            utils.save_image(samples[i], out_path)
            saved += 1

        print(f"Saved {saved}/{TOTAL_IMAGES}")

    print(f"Saved generated images to {OUT_DIR}")



if __name__ == "__main__":
    main()