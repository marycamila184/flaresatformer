from models.utils.dice_focal_loss import BinaryFocalLoss
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
    def __init__(self, dim, num_heads, sr_ratio=1, dropout=0.05):
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


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)
    

class MixFFN(nn.Module):
    def __init__(self, dim, ff_dim, dropout=0.1):
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
        assert x.dim() == 4, f"PatchEmbed expected BCHW, got {x.shape}"

        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class SegFormerEncoder(nn.Module):
    def __init__(self, in_channels, model_type='B0'):
        super().__init__()

        # Configurações por tipo de modelo
        configs = {
            'B0': {
                 'embed_dims': [32, 64, 160, 256],
                'depths': [2, 2, 2, 2],
                'num_heads': [1, 2, 5, 8],
                'sr_ratios': [8, 4, 2, 1],
                'mlp_ratios': [4, 4, 4, 4],
            },
            'B1': {
                'embed_dims': [64, 128, 320, 512],
                'depths': [2, 2, 2, 2],
                'num_heads': [1, 2, 5, 8],
                'sr_ratios': [8, 4, 2, 1],
                'mlp_ratios': [4, 4, 4, 4],
            }
        }

        assert model_type in configs, f"model_type deve ser um de {list(configs.keys())}"
        
        config = configs[model_type]
        embed_dims = config['embed_dims']
        depths = config['depths']
        num_heads = config['num_heads']
        sr_ratios = config['sr_ratios']
        mlp_ratios = config['mlp_ratios']

        self.stages = nn.ModuleList()

        dpr = torch.linspace(0, 0.1, sum(depths)).tolist()
        #dpr = torch.linspace(0, 0.05, sum(depths)).tolist()

        cur = 0

        for i in range(4):
            # Overlapping patch embedding
            patch_embed = OverlapPatchEmbed(
                in_ch=in_channels if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                patch_size=4 if i == 0 else 3,
                stride=3 if i == 0 else 2
            )

            # Transformer blocks
            blocks = nn.ModuleList([
                TransformerBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    ff_dim=embed_dims[i] * mlp_ratios[i],
                    sr_ratio=sr_ratios[i],
                    drop_path=dpr[cur + j],
                )
                for j in range(depths[i])
            ])

            cur += depths[i]

            self.stages.append(nn.ModuleDict({
                "patch": patch_embed,
                "blocks": blocks,
            }))

        self.embed_dims = embed_dims

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            features: list of 4 feature maps
        """
        features = []

        for stage in self.stages:
            # Patch embedding -> tokens
            x, H, W = stage["patch"](x)  # (B, HW, C)

            # Transformer blocks
            for blk in stage["blocks"]:
                x = blk(x, H, W)

            # Tokens -> feature map (B, C, H, W)
            B, N, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, H, W)

            features.append(x)

        return features


class SegFormer(nn.Module):
    def __init__(self, in_channels, num_classes=1, model_type='B0', decoder_dim=384, se_reduction=16):
        super().__init__()

        # -------- Encoder --------
        self.encoder = SegFormerEncoder(in_channels, model_type=model_type)
        embed_dims = self.encoder.embed_dims

        # -------- Decoder (MLP head) --------
        self.proj = nn.ModuleList([
            nn.Linear(embed_dims[0], decoder_dim),
            nn.Linear(embed_dims[1], decoder_dim),
            nn.Linear(embed_dims[2], decoder_dim),
            nn.Linear(embed_dims[3], decoder_dim),
        ])

        # Fuse conv layers for gradual upsampling and SE
        self.fuse3 = nn.Sequential(
            nn.Conv2d(decoder_dim * 2, decoder_dim, 3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.se3 = SEBlock(decoder_dim, reduction=se_reduction)

        self.fuse2 = nn.Sequential(
            nn.Conv2d(decoder_dim * 2, decoder_dim, 3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )

        self.se2 = SEBlock(decoder_dim, reduction=se_reduction)

        self.fuse1 = nn.Sequential(
            nn.Conv2d(decoder_dim * 2, decoder_dim, 3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.se1 = SEBlock(decoder_dim, reduction=se_reduction)

        self.pred = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, x):
            B, _, H, W = x.shape
            feats = self.encoder(x)  # list of 4 feature maps: [stage0, stage1, stage2, stage3]

            # Project each stage to decoder_dim
            proj_feats = []
            for f, proj in zip(feats, self.proj):
                b, c, h, w = f.shape
                f = f.flatten(2).transpose(1, 2)  # (B, HW, C)
                f = proj(f)                        # (B, HW, decoder_dim)
                f = f.transpose(1, 2).reshape(b, -1, h, w)
                proj_feats.append(f)

            # Gradual fusion (UNet style)
            x = F.interpolate(proj_feats[3], size=proj_feats[2].shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, proj_feats[2]], dim=1)
            x = self.fuse3(x)
            x = self.se3(x)

            x = F.interpolate(x, size=proj_feats[1].shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, proj_feats[1]], dim=1)
            x = self.fuse2(x)
            x = self.se2(x)

            x = F.interpolate(x, size=proj_feats[0].shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, proj_feats[0]], dim=1)
            x = self.fuse1(x)
            x = self.se1(x)

            x = self.pred(x)

            # Restore original input resolution
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

            return x
