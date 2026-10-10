"""UNet baseline, kept identical to the FlareSat/FlareSatFormer v1 architecture for continuity."""
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels, base_filters=32, dropout=0.1, out_channels=1):
        super().__init__()
        f = base_filters
        self.c1, self.c2, self.c3, self.c4 = ConvBlock(in_channels, f), ConvBlock(f, 2 * f), ConvBlock(2 * f, 4 * f), ConvBlock(4 * f, 8 * f)
        self.c5 = ConvBlock(8 * f, 16 * f)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(dropout)
        self.u6, self.c6 = nn.ConvTranspose2d(16 * f, 8 * f, 2, stride=2), ConvBlock(16 * f, 8 * f)
        self.u7, self.c7 = nn.ConvTranspose2d(8 * f, 4 * f, 2, stride=2), ConvBlock(8 * f, 4 * f)
        self.u8, self.c8 = nn.ConvTranspose2d(4 * f, 2 * f, 2, stride=2), ConvBlock(4 * f, 2 * f)
        self.u9, self.c9 = nn.ConvTranspose2d(2 * f, f, 2, stride=2), ConvBlock(2 * f, f)
        self.head = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(self.drop(self.pool(c1)))
        c3 = self.c3(self.drop(self.pool(c2)))
        c4 = self.c4(self.drop(self.pool(c3)))
        c5 = self.c5(self.drop(self.pool(c4)))
        c6 = self.c6(self.drop(torch.cat([self.u6(c5), c4], dim=1)))
        c7 = self.c7(self.drop(torch.cat([self.u7(c6), c3], dim=1)))
        c8 = self.c8(self.drop(torch.cat([self.u8(c7), c2], dim=1)))
        c9 = self.c9(self.drop(torch.cat([self.u9(c8), c1], dim=1)))
        return self.head(c9)

    def encoder_parameters(self):
        return []
