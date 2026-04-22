import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(in_ch, affine=True)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        residual = self.skip(residual)

        out = self.norm1(x)
        out = F.relu(out, inplace=True)
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.conv1(out)

        out = self.norm2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)

        return out + residual


class ResBlockDown(nn.Module):
    def __init__(self, in_ch, out_ch, first_block=False):
        super().__init__()
        self.first_block = first_block

        if not first_block:
            self.norm1 = nn.InstanceNorm2d(in_ch, affine=True)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

        self.avgpool = nn.AvgPool2d(2)

    def forward(self, x):
        residual = self.avgpool(self.skip(x))

        out = x
        if not self.first_block:
            out = self.norm1(out)
            out = F.leaky_relu(out, 0.2, inplace=True)

        out = self.conv1(out)
        out = self.norm2(out)
        out = F.leaky_relu(out, 0.2, inplace=True)
        out = self.conv2(out)
        out = self.avgpool(out)

        return out + residual


class ResGenerator128(nn.Module):
    def __init__(self, z_dim=128, base_channels=64):
        super().__init__()
        self.z_dim = z_dim
        self.base = base_channels

        self.fc = nn.Linear(z_dim, 8 * self.base * 4 * 4)

        self.block1 = ResBlockUp(8 * self.base, 8 * self.base)  # 4 -> 8
        self.block2 = ResBlockUp(8 * self.base, 4 * self.base)  # 8 -> 16
        self.block3 = ResBlockUp(4 * self.base, 2 * self.base)  # 16 -> 32
        self.block4 = ResBlockUp(2 * self.base, 1 * self.base)  # 32 -> 64
        self.block5 = ResBlockUp(1 * self.base, self.base // 2)  # 64 -> 128

        self.norm = nn.InstanceNorm2d(self.base // 2, affine=True)
        self.to_img = nn.Conv2d(self.base // 2, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        z = z.view(z.size(0), self.z_dim)
        out = self.fc(z)
        out = out.view(z.size(0), 8 * self.base, 4, 4)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        out = self.norm(out)
        out = F.relu(out, inplace=True)
        out = self.to_img(out)
        return torch.tanh(out)


class ResCritic128(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.base = base_channels

        self.block1 = ResBlockDown(1, self.base, first_block=True)         # 128 -> 64
        self.block2 = ResBlockDown(self.base, 2 * self.base)               # 64 -> 32
        self.block3 = ResBlockDown(2 * self.base, 4 * self.base)           # 32 -> 16
        self.block4 = ResBlockDown(4 * self.base, 8 * self.base)           # 16 -> 8
        self.block5 = ResBlockDown(8 * self.base, 8 * self.base)           # 8 -> 4

        self.final_conv = nn.Conv2d(8 * self.base, 8 * self.base, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(8 * self.base * 4 * 4, 1)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        out = F.leaky_relu(self.final_conv(out), 0.2, inplace=True)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out.view(-1)