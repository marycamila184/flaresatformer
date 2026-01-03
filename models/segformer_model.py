from models.utils.binary_focal_loss import BinaryFocalLoss
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, device=x.device)
        binary_tensor = random_tensor.floor()
        return x / keep_prob * binary_tensor


class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, H, W):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).transpose(1, 2)
            x_ = self.norm(x_)
            kv = self.kv(x_)
        else:
            kv = self.kv(x)

        kv = kv.reshape(B, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MixFFN(nn.Module):
    def __init__(self, dim, ff_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.dwconv = nn.Conv2d(ff_dim, ff_dim, 3, padding=1, groups=ff_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ff_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, H, W):
        B, N, _ = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.dwconv(x)
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, sr_ratio, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientMultiHeadAttention(dim, num_heads, sr_ratio)
        self.drop_path = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, ff_dim)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(
            in_ch, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class SegFormerB0Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        embed_dims = [32, 64, 160, 256]
        depths = [2, 2, 2, 2]
        num_heads = [1, 2, 5, 8]
        sr_ratios = [8, 4, 2, 1]
        mlp_ratios = [4, 4, 4, 4]

        self.stages = nn.ModuleList()
        cur = 0
        dpr = torch.linspace(0, 0.1, sum(depths)).tolist()

        for i in range(4):
            patch = OverlapPatchEmbed(
                in_channels if i == 0 else embed_dims[i-1],
                embed_dims[i],
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2
            )

            blocks = nn.ModuleList([
                TransformerBlock(
                    embed_dims[i],
                    num_heads[i],
                    embed_dims[i] * mlp_ratios[i],
                    sr_ratios[i],
                    dpr[cur + j]
                )
                for j in range(depths[i])
            ])

            cur += depths[i]
            self.stages.append(nn.ModuleDict({
                "patch": patch,
                "blocks": blocks
            }))

    def forward(self, x):
        features = []
        for stage in self.stages:
            x, H, W = stage["patch"](x)
            for blk in stage["blocks"]:
                x = blk(x, H, W)
            B, N, C = x.shape
            features.append(x.transpose(1, 2).reshape(B, C, H, W))
        return features


class SegFormerB0(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
    ):
        super().__init__()

        self.encoder = SegFormerB0Encoder(in_channels)

        # projection layers for multi-scale features
        self.proj = nn.ModuleList([
            nn.Linear(c, 256) for c in [32, 64, 160, 256]
        ])

        # self.fuse = nn.Conv2d(256 * 4, 256, kernel_size=1)
        self.fuse = nn.Sequential(nn.Linear(256 * 4, 256),nn.GELU())
        self.pred = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        input_h, input_w = x.shape[2:]

        feats = self.encoder(x)  # list of feature maps

        H, W = feats[0].shape[2:]

        outs = []
        for f, proj in zip(feats, self.proj):
            B, C, h, w = f.shape

            f = f.flatten(2).transpose(1, 2)      # (B, HW, C)
            f = proj(f)                           # (B, HW, 256)
            f = f.transpose(1, 2).reshape(B, 256, h, w)

            f = F.interpolate(
                f, size=(H, W), mode="bilinear", align_corners=False
            )
            outs.append(f)

        x = self.fuse(torch.cat(outs, dim=1))
        x = self.pred(x)

        # restore input resolution
        return F.interpolate(
            x, size=(input_h, input_w), mode="bilinear", align_corners=False
        )
