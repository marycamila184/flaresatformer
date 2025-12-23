from models.utils.binary_focal_loss import BinaryFocalLoss
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, batch_norm=True):
        super().__init__()
        padding = kernel_size // 2

        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        layers += [
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_filters: int = 32,
        dropout: float = 0.1,
        out_channels: int = 1,
    ):
        super().__init__()

        # Downsampling
        self.c1 = ConvBlock(in_channels, base_filters)
        self.p1 = nn.MaxPool2d(2)
        self.d1 = nn.Dropout(dropout)

        self.c2 = ConvBlock(base_filters, base_filters * 2)
        self.p2 = nn.MaxPool2d(2)
        self.d2 = nn.Dropout(dropout)

        self.c3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.p3 = nn.MaxPool2d(2)
        self.d3 = nn.Dropout(dropout)

        self.c4 = ConvBlock(base_filters * 4, base_filters * 8)
        self.p4 = nn.MaxPool2d(2)
        self.d4 = nn.Dropout(dropout)

        self.c5 = ConvBlock(base_filters * 8, base_filters * 16)

        # Upsampling
        self.u6 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.c6 = ConvBlock(base_filters * 16, base_filters * 8)
        self.d6 = nn.Dropout(dropout)

        self.u7 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.c7 = ConvBlock(base_filters * 8, base_filters * 4)
        self.d7 = nn.Dropout(dropout)

        self.u8 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.c8 = ConvBlock(base_filters * 4, base_filters * 2)
        self.d8 = nn.Dropout(dropout)

        self.u9 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.c9 = ConvBlock(base_filters * 2, base_filters)
        self.d9 = nn.Dropout(dropout)

        # Output (logits)
        self.out = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(self.d1(self.p1(c1)))
        c3 = self.c3(self.d2(self.p2(c2)))
        c4 = self.c4(self.d3(self.p3(c3)))
        c5 = self.c5(self.d4(self.p4(c4)))

        u6 = self.u6(c5)
        c6 = self.c6(self.d6(torch.cat([u6, c4], dim=1)))

        u7 = self.u7(c6)
        c7 = self.c7(self.d7(torch.cat([u7, c3], dim=1)))

        u8 = self.u8(c7)
        c8 = self.c8(self.d8(torch.cat([u8, c2], dim=1)))

        u9 = self.u9(c8)
        c9 = self.c9(self.d9(torch.cat([u9, c1], dim=1)))

        return self.out(c9)  # logits

