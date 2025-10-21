import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def grid_dist_to_idx(dim=5):
    """Creates distance-based grouping indices for local GSA."""
    dist = torch.zeros((dim, dim))
    i0, j0 = (dim - 1)//2, (dim - 1)//2
    for i in range(dim):
        for j in range(dim):
            distance = math.sqrt((i0 - i)**2 + (j0 - j)**2)
            dist[i, j] = distance
    dist = torch.round(1e6 * dist)
    unique = torch.unique(dist.flatten())
    idxs = torch.zeros_like(dist)
    for i, u in enumerate(unique):
        idxs[dist == u] = i
    return idxs.long(), len(unique)


# -----------------------------------------------------------------------------
# Graph Symmetric Attention (Single Layer)
# -----------------------------------------------------------------------------
class GraphSymmetricAttention(nn.Module):
    """
    Lightweight local GSA layer:
    - Groups features by radial symmetry distance.
    - Performs local self-attention constrained by those groups.
    """
    def __init__(self, in_chans=3, embed_dim=64, kernel_size=5, num_heads=4, patch=8):
        super().__init__()
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch = patch  # patch size for reducing token count

        # Distance-based grouping for symmetry
        self.idxs, self.num_groups = grid_dist_to_idx(kernel_size)
        self.weights = nn.Parameter(torch.Tensor(embed_dim, self.num_groups))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        # Linear projections for Q, K, V
        self.q_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def to_patches(self, t):
        """Convert (B, D, H, W) to (B, N, D) by average pooling patches."""
        B, D, H, W = t.shape
        p = self.patch
        t = t.unfold(2, p, p).unfold(3, p, p)  # (B, D, H/p, W/p, p, p)
        t = t.contiguous().view(B, D, -1, p, p).mean(dim=(-1, -2))  # (B, D, N)
        return t.transpose(1, 2)  # (B, N, D)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            tokens: (B, N_patches, embed_dim)
        """
        B, C, H, W = x.shape

        # 1. Compute Q, K, V
        q = self.q_proj(x)  # (B, D, H, W)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Symmetry-shared convolution weights
        W_sym = torch.index_select(self.weights, 1, self.idxs.flatten().to(x.device))
        W_sym = W_sym.view(self.embed_dim, 1, self.kernel_size, self.kernel_size)

        # 3. Depthwise conv (groups = embed_dim)
        y = F.conv2d(v, W_sym, padding=self.kernel_size // 2, groups=self.embed_dim)  # (B, D, H, W)

        # 4. Patchify for efficiency
        y = self.to_patches(y)  # (B, N, D)
        qf = self.to_patches(q)
        kf = self.to_patches(k)

        # 5. Self-attention
        attn = torch.softmax((qf @ kf.transpose(-1, -2)) / math.sqrt(self.embed_dim), dim=-1)
        out = attn @ y
        out = self.norm(out + y)
        return self.proj(out)


# -----------------------------------------------------------------------------
# Mini Vision Transformer
# -----------------------------------------------------------------------------
class MiniVisionTransformer(nn.Module):
    def __init__(self, embed_dim=64, depth=2, num_heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.encoder(x)
        return self.norm(x[:, 0])  # CLS output


# -----------------------------------------------------------------------------
# Mini-SiT (1 GSA + Transformer encoder)
# -----------------------------------------------------------------------------
class MiniSiT_1GSA(nn.Module):
    def __init__(self, img_size=64, in_chans=3, embed_dim=64, num_heads=4, depth=2, patch=8):
        super().__init__()
        self.gsa = GraphSymmetricAttention(
            in_chans=in_chans,
            embed_dim=embed_dim,
            kernel_size=5,
            num_heads=num_heads,
            patch=patch
        )
        self.vit = MiniVisionTransformer(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        self.fc = nn.Linear(embed_dim, 128)  # final latent

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.gsa(x)
        x = self.vit(x)
        return self.fc(x)


# # -----------------------------------------------------------------------------
# # Debugging entrypoint
# # -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     B, C, H, W = 4, 3, 64, 64
#     model = MiniSiT_1GSA(img_size=64, in_chans=3, embed_dim=64, num_heads=4, depth=2, patch=8)
#     x = torch.randn(B, C, H, W)
#     out = model(x)
#     print("✅ Mini-SiT output:", out.shape)  # expect (B, 128)
