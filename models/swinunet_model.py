import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------
# Swin Transformer Block
# (W-MSA and SW-MSA pair)
# --------------------
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: B x N x C (flattened patch tokens)
        xi = self.norm1(x)
        attn_out, _ = self.attn(xi.transpose(0,1), xi.transpose(0,1), xi.transpose(0,1))
        x = x + attn_out.transpose(0,1)
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------
# Patch Merging (Downsampling)
# --------------------
class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        # Downsample by taking 2x2 patches
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H//2, W//2


# --------------------
# Patch Expanding (Upsampling)
# --------------------
class PatchExpanding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, dim * 2, bias=False)
        self.norm = nn.LayerNorm(dim//2)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.expand(x)
        # reshape to 2x spatial upsample
        x = x.view(B, H, W, C*2)
        x = x.view(B, H, 2, W, 2, C//2).permute(0,1,3,2,4,5).reshape(B, H*2, W*2, C//2)
        x = self.norm(x)
        return x.view(B, -1, x.shape[-1]), H*2, W*2


class SwinUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1,
                 embed_dim=96, depths=[2,2,2,2],
                 num_heads=[3,6,12,24], window_size=7):
        super().__init__()

        # initial patch embedding
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=4, stride=4)

        # encoder stages
        self.encoder_layers = nn.ModuleList()
        self.merge_layers = nn.ModuleList()
        dims = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
        for i in range(4):
            blocks = nn.ModuleList([
                SwinTransformerBlock(dims[i], num_heads[i], window_size)
                for _ in range(depths[i])
            ])
            self.encoder_layers.append(blocks)
            if i < 3:
                self.merge_layers.append(PatchMerging(dims[i]))

        # decoder upsampling and blocks
        self.expand_layers = nn.ModuleList([PatchExpanding(d) for d in reversed(dims)])
        self.decoder_layers = nn.ModuleList()
        for i in reversed(range(4)):
            blocks = nn.ModuleList([
                SwinTransformerBlock(dims[i], num_heads[i], window_size)
                for _ in range(depths[i])
            ])
            self.decoder_layers.append(blocks)

        self.output_proj = nn.Conv2d(embed_dim, out_ch, kernel_size=1)

    def forward(self, x):
        # patch embedding
        B, _, H, W = x.shape
        x = self.patch_embed(x)             # B x C x H/4 x W/4
        H4, W4 = H//4, W//4
        x = x.flatten(2).transpose(1,2)     # B x N x C

        # encoder forward with skip
        skip_feats = []
        curH, curW = H4, W4
        for idx, blocks in enumerate(self.encoder_layers):
            for blk in blocks:
                x = blk(x)
            skip_feats.append((x, curH, curW))
            if idx < len(self.merge_layers):
                x, curH, curW = self.merge_layers[idx](x, curH, curW)

        # decoder upwards
        for idx, blocks in enumerate(self.decoder_layers):
            x, curH, curW = self.expand_layers[idx](x, curH, curW)
            # skip connection fusion
            skip, sH, sW = skip_feats[-(idx+2)]
            x = torch.cat([x, skip], dim=-1)
            for blk in blocks:
                x = blk(x)

        # final reshape
        x = x.transpose(1,2).view(B, -1, H, W)
        return self.output_proj(x)
