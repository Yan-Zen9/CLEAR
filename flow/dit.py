"""
DiT velocity model for latent flow matching (vendored, self-contained).

The architecture and state-dict layout EXACTLY match VinAI LFM's `models/DiT.py`
(itself from facebookresearch/DiT), so the released LFM checkpoints load with
`strict=True`:

    celeba_256/model_475.pth   DiT-L/2, unconditional          (y table: 1 row)
    imagenet/model_875.pt      DiT-B/2, 1000-class + CFG null  (y table: 1001 rows)

Both operate on 4x32x32 SD latents (stabilityai/sd-vae-ft-mse, x0.18215) — the
same latent space as the rest of MedDiME, so no re-encoding is needed.

timm's Attention/Mlp/PatchEmbed are re-implemented here with IDENTICAL parameter
names (qkv/proj, fc1/fc2, proj) — no timm dependency.

TIME CONVENTION (the one thing that differs from the rest of this repo):
LFM trains with t=0 = DATA, t=1 = NOISE and the velocity pointing toward noise;
MedDiME (Meta flow_matching CondOT) uses t=0 = noise, t=1 = data, v = x1 - x0.
`DiTVelocity` bridges the two:  v_med(x, t) = -dit(1 - t, x, y).
Everything outside this file speaks MedDiME convention only.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ============================================================ timm re-implementations
class PatchEmbed(nn.Module):
    """Image-to-patch embedding (timm-compatible: parameter `proj`)."""

    def __init__(self, img_size, patch_size, in_chans, embed_dim, bias=True):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.proj(x)                       # [B, D, H/p, W/p]
        return x.flatten(2).transpose(1, 2)    # [B, T, D]


class Attention(nn.Module):
    """Standard MHSA (timm-compatible: parameters `qkv`, `proj`)."""

    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, T, D))


class Mlp(nn.Module):
    """Transformer MLP (timm-compatible: parameters `fc1`, `fc2`)."""

    def __init__(self, in_features, hidden_features, act_layer):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ============================================================ embedders
class TimestepEmbedder(nn.Module):
    """Embeds scalar (possibly fractional) timesteps into vectors."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period)
                          * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        if t.dim() == 0:
            t = t[None]
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    """Class-label embedding with a CFG null token when dropout_prob > 0."""

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.in_channels = num_classes + use_cfg_embedding
        self.embedding_table = nn.Embedding(self.in_channels, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        return torch.where(drop, self.num_classes, labels)

    def forward(self, labels, train):
        if train and self.dropout_prob > 0:
            labels = self.token_drop(labels)
        return self.embedding_table(labels)


# ============================================================ DiT
class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio),
                       act_layer=lambda: nn.GELU(approximate="tanh"))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
    """Diffusion/flow transformer, LFM-native: forward(t, x, y) with t=0=data, t=1=noise."""

    def __init__(self, img_resolution=32, patch_size=2, in_channels=4, hidden_size=1024,
                 depth=24, num_heads=16, mlp_ratio=4.0, label_dropout=0.0, num_classes=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.enable_gradient_checkpointing = False

        self.x_embedder = PatchEmbed(img_resolution, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, label_dropout)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, hidden_size),
                                      requires_grad=False)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                      int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:                       # adaLN-Zero: identity at init
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, t, x, y=None):
        """t: [N] LFM time (0=data, 1=noise); x: [N,C,H,W]; y: [N] class labels or None
        (None -> the last embedding row: the sole row when unconditional, the CFG null
        token when class-conditional)."""
        if y is None:
            y = torch.full((x.size(0),), self.y_embedder.in_channels - 1,
                           dtype=torch.long, device=x.device)
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y, self.training)
        for block in self.blocks:
            if self.enable_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)
        return self.unpatchify(self.final_layer(x, c))


# ============================================================ sincos pos-embed
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega = 1.0 / 10000 ** (omega / (embed_dim / 2.0))
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ============================================================ construction / loading
# canonical DiT sizes (patch 2); num_heads is NOT recoverable from a state dict,
# so it is looked up by hidden size here.
DIT_CONFIGS = {
    "DiT-S/2": dict(hidden_size=384, depth=12, num_heads=6, patch_size=2),
    "DiT-B/2": dict(hidden_size=768, depth=12, num_heads=12, patch_size=2),
    "DiT-L/2": dict(hidden_size=1024, depth=24, num_heads=16, patch_size=2),
    "DiT-XL/2": dict(hidden_size=1152, depth=28, num_heads=16, patch_size=2),
}
_HEADS_BY_HIDDEN = {384: 6, 768: 12, 1024: 16, 1152: 16}


def normalize_state_dict(ckpt):
    """torch.load result -> flat DiT state dict.

    Accepts LFM's raw (DDP `module.`-prefixed) state dicts as well as this repo's
    trainer checkpoints ({'ema': ...} / {'model': ...}); strips thop keys.
    """
    sd = ckpt
    if isinstance(ckpt, dict) and not any(torch.is_tensor(v) for v in ckpt.values()):
        sd = ckpt["ema"] if "ema" in ckpt and "model" not in ckpt else ckpt.get("model", ckpt)
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    return {k: v for k, v in sd.items() if not k.endswith(("total_ops", "total_params"))}


def infer_dit_arch(sd):
    """State dict -> constructor kwargs. The weight shapes are the source of truth."""
    pw = sd["x_embedder.proj.weight"]                       # [D, C, p, p]
    hidden_size, in_channels, patch_size = pw.shape[0], pw.shape[1], pw.shape[-1]
    depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    tokens = sd["pos_embed"].shape[1]
    img_resolution = int(round(tokens ** 0.5)) * patch_size
    y_rows = sd["y_embedder.embedding_table.weight"].shape[0]
    if y_rows == 1:                                         # unconditional (single trained row)
        num_classes, label_dropout = 1, 0.0
    else:                                                   # conditional + CFG null token
        num_classes, label_dropout = y_rows - 1, 0.1
    return dict(img_resolution=img_resolution, patch_size=patch_size, in_channels=in_channels,
                hidden_size=hidden_size, depth=depth, num_heads=_HEADS_BY_HIDDEN[hidden_size],
                num_classes=num_classes, label_dropout=label_dropout)


def build_dit_from_state_dict(sd):
    """Build a DiT matching `sd` (already normalized) and load it strictly."""
    arch = infer_dit_arch(sd)
    model = DiT(**arch)
    model.load_state_dict(sd, strict=True)
    return model, arch


# ============================================================ MedDiME-facing wrapper
class DiTVelocity(nn.Module):
    """MedDiME-convention velocity field on an LFM DiT:  v(x, t) = -dit(1 - t, x, y).

    forward(x, t) with continuous t in [0,1], t=0 = noise -> t=1 = data, output =
    velocity toward data (CondOT: target x1 - x0) — the convention every solver,
    trainer and the CF sampler in this repo uses.

    class_label: fixed class fed to a CONDITIONAL DiT (e.g. an ImageNet id);
    None -> the null/unconditional token. cfg_scale > 1 (with class_label set)
    applies classifier-free guidance: v = v_null + s*(v_cond - v_null).
    Both are no-ops for the unconditional CelebA-HQ model.
    """

    def __init__(self, dit: DiT, class_label=None, cfg_scale=1.0):
        super().__init__()
        self.dit = dit
        self.class_label = None if class_label is None else int(class_label)
        self.cfg_scale = float(cfg_scale)

    def _y(self, x):
        if self.class_label is None:
            return None
        return torch.full((x.shape[0],), self.class_label, dtype=torch.long, device=x.device)

    def forward(self, x, t):
        t = torch.as_tensor(t, device=x.device).float().reshape(-1)
        if t.numel() == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0])
        s = (1.0 - t).clamp(0.0, 1.0)                       # MedDiME t -> LFM time
        y = self._y(x)
        if y is None or self.cfg_scale == 1.0:
            return -self.dit(s, x, y)
        v_cond = self.dit(s, x, y)
        v_null = self.dit(s, x, None)
        return -(v_null + self.cfg_scale * (v_cond - v_null))
