import torch
import torch.nn as nn

# ---------------------
# CNN Block
# ---------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, batch_norm=True):
        super().__init__()
        padding = kernel_size // 2
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        layers += [nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# ---------------------
# Transformer Block (ViT-style)
# ---------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: B x N x C
        x_norm = self.norm1(x)
        x_attn, _ = self.attn(x_norm.transpose(0,1), x_norm.transpose(0,1), x_norm.transpose(0,1))
        x = x + x_attn.transpose(0,1)
        x = x + self.mlp(self.norm2(x))
        return x

# ---------------------
# TransUNet
# ---------------------
class TransUNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 base_filters=32,
                 embed_dim=256,
                 num_heads=8,
                 num_layers=4,
                 dropout=0.1,
                 out_channels=1):
        super().__init__()

        # Encoder (CNN)
        self.c1 = ConvBlock(in_channels, base_filters)
        self.p1 = nn.MaxPool2d(2)
        self.d1 = nn.Dropout(dropout)

        self.c2 = ConvBlock(base_filters, base_filters*2)
        self.p2 = nn.MaxPool2d(2)
        self.d2 = nn.Dropout(dropout)

        self.c3 = ConvBlock(base_filters*2, base_filters*4)
        self.p3 = nn.MaxPool2d(2)
        self.d3 = nn.Dropout(dropout)

        self.c4 = ConvBlock(base_filters*4, base_filters*8)
        self.p4 = nn.MaxPool2d(2)
        self.d4 = nn.Dropout(dropout)

        # Bottleneck Conv → transformer embedding
        self.c5 = ConvBlock(base_filters*8, embed_dim)

        # Positional embedding (dynamic, paper-style)
        self.pos_embed = None
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # Decoder
        self.u6 = nn.ConvTranspose2d(embed_dim, base_filters*8, 2, stride=2)
        self.c6 = ConvBlock(base_filters*16, base_filters*8)
        self.d6 = nn.Dropout(dropout)

        self.u7 = nn.ConvTranspose2d(base_filters*8, base_filters*4, 2, stride=2)
        self.c7 = ConvBlock(base_filters*8, base_filters*4)
        self.d7 = nn.Dropout(dropout)

        self.u8 = nn.ConvTranspose2d(base_filters*4, base_filters*2, 2, stride=2)
        self.c8 = ConvBlock(base_filters*4, base_filters*2)
        self.d8 = nn.Dropout(dropout)

        self.u9 = nn.ConvTranspose2d(base_filters*2, base_filters, 2, stride=2)
        self.c9 = ConvBlock(base_filters*2, base_filters)
        self.d9 = nn.Dropout(dropout)

        self.out = nn.Conv2d(base_filters, out_channels, 1)

    def forward(self, x):
        # Encoder
        c1 = self.c1(x)
        c2 = self.c2(self.d1(self.p1(c1)))
        c3 = self.c3(self.d2(self.p2(c2)))
        c4 = self.c4(self.d3(self.p3(c3)))
        c5 = self.c5(self.d4(self.p4(c4)))

        # Flatten and add positional embedding
        B, C, H, W = c5.shape
        x_seq = c5.flatten(2).transpose(1, 2)  # B x N x C
        N = x_seq.size(1)
        if self.pos_embed is None or self.pos_embed.size(1) != N:
            # Create learnable pos embedding for this feature map size
            self.pos_embed = nn.Parameter(torch.zeros(1, N, C, device=c5.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        x_seq = x_seq + self.pos_embed

        # Transformer
        for blk in self.transformer_blocks:
            x_seq = blk(x_seq)

        # Reshape back to CNN feature map
        c5 = x_seq.transpose(1, 2).view(B, C, H, W)
        c5 = self.dropout(c5)

        # Decoder
        u6 = self.u6(c5)
        c6 = self.c6(self.d6(torch.cat([u6, c4], dim=1)))

        u7 = self.u7(c6)
        c7 = self.c7(self.d7(torch.cat([u7, c3], dim=1)))

        u8 = self.u8(c7)
        c8 = self.c8(self.d8(torch.cat([u8, c2], dim=1)))

        u9 = self.u9(c8)
        c9 = self.c9(self.d9(torch.cat([u9, c1], dim=1)))

        return self.out(c9)
