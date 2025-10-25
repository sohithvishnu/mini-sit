import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# =========================
# Utilities
# =========================

def grid_dist_to_idx(dim=5):
    """Distance-based grouping indices for local symmetry (radial bins)."""
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

def _radial_index(kernel_size: int):
    """Return (idxs, num_bins) mapping k×k offsets to a radial bin index from center."""
    assert kernel_size % 2 == 1, "Use odd kernel_size (e.g., 5 or 7)."
    dist = torch.zeros(kernel_size, kernel_size)
    c = (kernel_size - 1)//2
    for i in range(kernel_size):
        for j in range(kernel_size):
            dist[i, j] = math.sqrt((i - c)**2 + (j - c)**2)
    bins = torch.unique(torch.round(dist * 1_000))
    idx = torch.zeros_like(dist, dtype=torch.long)
    for bi, b in enumerate(bins):
        idx[torch.round(dist * 1_000) == b] = bi
    return idx.view(-1), len(bins)

class DropPath(nn.Module):
    """Stochastic depth (per sample)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask

# =========================
# Symmetry layers
# =========================

class GraphSymmetricAttention(nn.Module):
    """
    Lightweight local GSA layer:
      - Q,K,V by 1x1 conv into D channels
      - Depthwise conv on V with radial weight sharing (symmetry)
      - Patchify to tokens and do self-attention across patches
    """
    def __init__(self, in_chans=3, embed_dim=128, kernel_size=5, num_heads=8, patch=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.patch = patch

        # Q K V projections
        self.q_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, bias=True)
        self.v_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, bias=True)

        # Distance-based shared weights for depthwise conv on V
        radial_idxs, num_bins = _radial_index(kernel_size)
        self.register_buffer("radial_idxs", radial_idxs)      # (k*k,)
        self.num_bins = num_bins
        # One scalar per (channel, radial-bin)
        self.radial_weights = nn.Parameter(torch.empty(embed_dim, num_bins))
        nn.init.kaiming_uniform_(self.radial_weights, a=math.sqrt(5))

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(embed_dim)

    def _depthwise_kernel(self, device):
        # Map per-bin weights to full depthwise kernel (D,1,k,k)
        w = torch.index_select(self.radial_weights, 1, self.radial_idxs.to(device))
        return w.view(self.embed_dim, 1, self.kernel_size, self.kernel_size)

    def _to_patches_mean(self, t):
        """(B, D, H, W) -> (B, N, D) with average pooling per non-overlapping patch."""
        B, D, H, W = t.shape
        p = self.patch
        # unfold into patches and average over (p,p)
        t = t.unfold(2, p, p).unfold(3, p, p)      # (B, D, H/p, W/p, p, p)
        t = t.contiguous().mean((-1, -2))          # (B, D, H/p, W/p)
        return t.flatten(2).transpose(1, 2)        # (B, N, D)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            tokens: (B, N_patches, embed_dim)
        """
        B, C, H, W = x.shape
        q = self.q_proj(x)   # (B, D, H, W)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Symmetric depthwise conv on V
        Wdw = self._depthwise_kernel(v.device)
        v_sym = F.conv2d(v, Wdw, padding=self.kernel_size//2, groups=self.embed_dim)  # (B, D, H, W)

        # Patchify
        qf = self._to_patches_mean(q)      # (B, N, D)
        kf = self._to_patches_mean(k)
        vf = self._to_patches_mean(v_sym)  # (B, N, D)

        # Multi-head self-attention across patches
        def split_heads(t):   # (B,N,D) -> (B,h,N,d)
            return t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        qh, kh, vh = split_heads(qf), split_heads(kf), split_heads(vf)

        attn = (qh @ kh.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,h,N,N)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ vh                                          # (B,h,N,d)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.embed_dim)  # (B,N,D)

        out = self.proj_drop(self.out_proj(out))
        # Residual on the value path (pre-norm style in token space)
        return self.norm(out + vf)


class FlipSym(nn.Module):
    """
    Optional horizontal flip symmetry over the token grid.
    mode="invariant": average with flipped tokens
    mode="equivariant": learn a gate alpha in [0,1] between original and flipped
    """
    def __init__(self, grid_h: int, grid_w: int, embed_dim: int, mode: str = "invariant"):
        super().__init__()
        assert mode in ("invariant", "equivariant")
        self.gh, self.gw = grid_h, grid_w
        self.mode = mode
        if mode == "equivariant":
            self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, tokens):  # (B, N, D) with N = gh*gw
        B, N, D = tokens.shape
        t = tokens.view(B, self.gh, self.gw, D)
        t_flip = torch.flip(t, dims=[2])  # flip width dimension
        if self.mode == "invariant":
            tout = 0.5 * (t + t_flip)
        else:
            a = torch.sigmoid(self.alpha)
            tout = a * t + (1 - a) * t_flip
        return tout.view(B, N, D)

# =========================
# ViT (deeper, drop path)
# =========================

class ViTBlock(nn.Module):
    def __init__(self, dim=128, heads=8, mlp_ratio=2.0, attn_drop=0.0, proj_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_drop, batch_first=True)
        self.drop_path1 = DropPath(drop_path)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(proj_drop),
            nn.Linear(hidden, dim), nn.Dropout(proj_drop),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        # Self-attention with residual
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.drop_path1(self.proj_drop(y))
        # MLP with residual
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class MiniVisionTransformer(nn.Module):
    def __init__(self, embed_dim=128, depth=6, num_heads=8, mlp_ratio=2.0, attn_drop=0.0, proj_drop=0.0, drop_path=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # simple linear pos-free encoder (SiT avoids pos enc)
        dpr = torch.linspace(0, drop_path, steps=depth).tolist()
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio, attn_drop, proj_drop, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # x: (B, N, D)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 0]  # CLS

# =========================
# End-to-end Mini-SiT++
# =========================

class MiniSiT_1GSA(nn.Module):
    """
    Balanced Mini-SiT:
      - Efficient GSA with radial symmetric depthwise conv on V
      - Optional flip symmetry over patch tokens
      - Deeper ViT encoder (default depth=6, dim=128, heads=8)
    """
    def __init__(
        self,
        img_size=64,
        in_chans=3,
        embed_dim=128,     # ↑ from 64 → 128
        num_heads=8,       # ↑ heads
        depth=6,           # ↑ depth
        patch=8,
        kernel_size=5,
        use_flip_sym=True,         # new
        flip_mode="invariant",     # or "equivariant"
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch = patch
        self.grid_h = img_size // patch
        self.grid_w = img_size // patch

        # One GSA (local symmetry + tokenization)
        self.gsa = GraphSymmetricAttention(
            in_chans=in_chans,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            num_heads=num_heads,
            patch=patch,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )

        # Optional flip symmetry over (grid_h x grid_w) tokens
        self.flip = FlipSym(self.grid_h, self.grid_w, embed_dim, mode=flip_mode) if use_flip_sym else None

        # Deeper ViT encoder
        self.vit = MiniVisionTransformer(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=2.0,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path
        )

        # Final latent head (kept 128 as in your code)
        self.fc = nn.Linear(embed_dim, 128)

    def forward(self, x):  # x: (B, C, H, W)
        tokens = self.gsa(x)                    # (B, N=gh*gw, D)
        if self.flip is not None:
            tokens = self.flip(tokens)          # inject flip-invariance/equivariance
        z = self.vit(tokens)                    # (B, D)
        return self.fc(z)                       # (B, 128)


