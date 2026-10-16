"""
Loading + I/O for flow-matching counterfactual generation (CLEAR).

Loaders: velocity DiT, latent/pixel classifiers, VAE, latent test set.
I/O: save/decode latents, save image tensors, write labels, attribution heatmaps.

The velocity model is an LFM-format DiT (flow/dit.py); `load_flow_model` accepts
the released VinAI LFM checkpoints directly and returns a `DiTVelocity` wrapper
that speaks CLEAR convention (t=0 noise -> t=1 data, velocity toward data).
"""

import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

from flow.dit import DiTVelocity, build_dit_from_state_dict, normalize_state_dict
from data.latent import get_latent_loader_by_name
from cf.classifier import get_classifier_bundle  # unified classifier loader (vendored from FiRe)
from cf.config import DEFAULT_VAE_SCALING


def load_flow_model(checkpoint_path: str, device: str = "cuda"):
    """Load the latent velocity model (DiT) from an LFM-format checkpoint.

    Accepts the released VinAI LFM state dicts (raw, DDP `module.`-prefixed —
    e.g. celeba_256 model_475.pth, imagenet model_875.pt) as well as this repo's
    trainer checkpoints ({'model'/'ema'}; EMA preferred). The architecture is
    inferred from the weight shapes (flow.dit.infer_dit_arch) — no env flag.

    Returns (model, info): `model` is a DiTVelocity in CLEAR convention —
    model(x, t), continuous t in [0,1], 0 = noise -> 1 = data.

    Conditional checkpoints (e.g. ImageNet, 1000-class + CFG null token):
    CF_FLOW_CLASS_LABEL fixes the class fed to the DiT (unset -> null token =
    unconditional velocity); CF_FLOW_CFG_SCALE > 1 adds classifier-free guidance.
    Both are no-ops for the unconditional CelebA-HQ checkpoint.
    """
    print("^" * 30 + f"\nLoading flow (velocity) model: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = normalize_state_dict(ckpt)
    dit, arch = build_dit_from_state_dict(sd)                    # strict=True inside

    label_env = os.environ.get("CF_FLOW_CLASS_LABEL", "").strip()
    class_label = int(label_env) if label_env else None
    cfg_scale = float(os.environ.get("CF_FLOW_CFG_SCALE", 1.0))
    model = DiTVelocity(dit, class_label=class_label, cfg_scale=cfg_scale).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)  # inference guidance needs gradients only w.r.t. the latent input
    gstep = ckpt.get("global_step", "N/A") if isinstance(ckpt, dict) and "global_step" in ckpt else "N/A"
    cond = f"class_label={class_label} cfg={cfg_scale}" if arch["num_classes"] > 1 else "unconditional"
    print(f"✅ flow model ready (DiT hidden={arch['hidden_size']} depth={arch['depth']} "
          f"patch={arch['patch_size']} latent={arch['img_resolution']} "
          f"num_classes={arch['num_classes']} | {cond} | global_step={gstep})")
    return model, {"global_step": gstep, "arch": arch}


def load_pixel_classifier_bundle(checkpoint_path: str, device: str = "cuda",
                                 query_label: int = 31, classifier_mode: str = "auto"):
    """Load the PIXEL classifier via the UNIFIED interface (cf.classifier).

    Returns a bundle {model, mean, std, preprocess} — the same convention as
    FiRe. For medical (binary) datasets the checkpoint should carry
    'model_state_dict' with a 2-class head (StandardDenseNetClassifier).
    """
    return get_classifier_bundle(checkpoint_path, torch.device(device),
                                 query_label=query_label, classifier_mode=classifier_mode)


def load_vae(model_name: str = "stabilityai/sd-vae-ft-mse", device: str = "cuda"):
    """Load the VAE (image<->latent). Default: SD1.5 MSE VAE (4-ch, 8x, good at 256)."""
    print("^" * 30 + f"\nLoading VAE: {model_name}")
    vae = AutoencoderKL.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)        # frozen; we still backprop to its INPUT through the graph
    print(f"   latent_channels={vae.config.latent_channels}  scaling_factor={vae.config.scaling_factor}")
    return vae


def load_test_loader(ds_name: str, device: str, batch_size: int = 1):
    """Load the CF latent loader.

    Preferred (CelebA-HQ): flat latent dir + labels/split CSV, filtered by
    CF_SPLIT/CF_ATTR — set CF_LABELS_CSV and CF_LATENT_DIR. Falls back to the
    structured `<DS>_latent-<split>/features|labels` loader otherwise.
    """
    if ds_name == "ImageNet":                                    # class-subdir latents, true-index labels
        from data.latent import get_imagefolder_latent_loader
        from cf.config import IMAGENET_CE_WNID_TO_IDX, IMAGENET_CE_PAIRS
        latent_dir = os.environ.get("CF_LATENT_DIR", "").strip()
        if not latent_dir:
            raise ValueError("CF_DATASET=imagenet requires CF_LATENT_DIR (the *_latent root with wnid subdirs).")
        # CF_SRC_WNID = ONE source class (single direction) -> enables target-class CFG
        # (CF_FLOW_CLASS_LABEL=target). CF_PAIR = both classes of a pair. Else all 6.
        src_wnid = os.environ.get("CF_SRC_WNID", "").strip()
        pair = os.environ.get("CF_PAIR", "").strip().lower()
        if src_wnid:
            if src_wnid not in IMAGENET_CE_WNID_TO_IDX:
                raise ValueError(f"CF_SRC_WNID='{src_wnid}' not in {list(IMAGENET_CE_WNID_TO_IDX)}")
            wnid_map = {src_wnid: IMAGENET_CE_WNID_TO_IDX[src_wnid]}
        elif pair:
            if pair not in IMAGENET_CE_PAIRS:
                raise ValueError(f"CF_PAIR='{pair}' not in {list(IMAGENET_CE_PAIRS)}")
            wnid_map = {w: IMAGENET_CE_WNID_TO_IDX[w] for w in IMAGENET_CE_PAIRS[pair]}
        else:
            wnid_map = IMAGENET_CE_WNID_TO_IDX                    # all 6 (smoke / not per-pair)
        shard_id = int(os.environ.get("CF_SHARD_ID", 0))
        num_shards = int(os.environ.get("CF_NUM_SHARDS", 1))
        loader = get_imagefolder_latent_loader(latent_dir, wnid_map,
                                               batch_size=batch_size, device=device, shuffle=False,
                                               shard_id=shard_id, num_shards=num_shards)
        print(f"✅ ImageNet imagefolder-latent loader: {('src='+src_wnid) if src_wnid else ('pair='+(pair or 'ALL-6'))} "
              f"shard={shard_id}/{num_shards} ({len(loader.dataset)} samples)")
        return loader

    csv_path = os.environ.get("CF_LABELS_CSV", "").strip()
    if csv_path:
        from data.latent import get_flat_csv_loader
        latent_dir = os.environ.get("CF_LATENT_DIR", "").strip()
        if not latent_dir:
            raise ValueError("CF_LABELS_CSV is set but CF_LATENT_DIR is not.")
        split = os.environ.get("CF_SPLIT", "val")
        attr = os.environ.get("CF_ATTR", "Smiling")
        shard_id = int(os.environ.get("CF_SHARD_ID", 0))
        num_shards = int(os.environ.get("CF_NUM_SHARDS", 1))
        loader = get_flat_csv_loader(latent_dir, csv_path, split=split, attr=attr,
                                     batch_size=batch_size, device=device, shuffle=False,
                                     shard_id=shard_id, num_shards=num_shards)
        print(f"✅ flat+CSV loader: split={split} attr={attr} shard={shard_id}/{num_shards} "
              f"({len(loader.dataset)} samples)")
        return loader

    split = "train" if os.environ.get("CF_USE_TRAIN", "").lower() == "true" else "test"
    loader = get_latent_loader_by_name(
        dataset_name=f"{ds_name}_latent-{split}",
        batch_size=batch_size, device=device, shuffle=False,
    )
    print(f"✅ loaded {split} set: {len(loader.dataset)} samples")
    return loader


def load_real_image_batch(image_stems, img_dir, size, device):
    """Load TRUE original images by file stem -> [b,3,size,size] in [-1,1] (RGB).

    Used as the exact, unchanged background for the paste composite, so the output
    never carries the VAE round-trip error in the preserved (background) region.
    """
    from PIL import Image
    # ImageNet paste-back: files are <img_dir>/<wnid>/<wnid>_<n>.JPEG. Set
    # CF_IMG_EXT=.JPEG and CF_IMG_NESTED_WNID=true; the wnid is the stem prefix.
    ext = os.environ.get("CF_IMG_EXT", ".jpg")
    nested = os.environ.get("CF_IMG_NESTED_WNID", "").lower() == "true"
    imgs = []
    for stem in image_stems:
        if nested:
            path = os.path.join(img_dir, stem.split("_")[0], f"{stem}{ext}")
        else:
            path = os.path.join(img_dir, f"{stem}{ext}")
        im = Image.open(path).convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0          # [H,W,3] in [0,1]
        imgs.append(torch.from_numpy(a).permute(2, 0, 1) * 2.0 - 1.0)   # [3,H,W] in [-1,1]
    return torch.stack(imgs).to(device)


# ============================================================ I/O helpers
def create_dir_if_nonexist(path: str):
    os.makedirs(path, exist_ok=True)


def get_save_path_with_version(base_path: str, with_version: bool = False) -> str:
    if not with_version:
        os.makedirs(base_path, exist_ok=True)
        return base_path
    version = 0
    while True:
        path = f"{base_path}_v{version}"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            return path
        version += 1


def save_latent_as_npy(latent: torch.Tensor, save_path: str):
    """Save a [1, C, H, W] latent tensor to .npy."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, latent.detach().cpu().numpy())


def decode_to_tensor(latent: torch.Tensor, vae,
                     scaling_factor: float = DEFAULT_VAE_SCALING):
    """VAE-decode to an image tensor in [-1, 1] (gradients NOT tracked)."""
    with torch.no_grad():
        return vae.decode(latent / scaling_factor).sample


def save_image_tensor(image: torch.Tensor, save_path: str):
    """Save a [1,3,H,W] image tensor in [-1,1] as PNG."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = ((image.clamp(-1, 1) + 1.0) / 2.0 * 255.0)[0].permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(save_path, img[:, :, ::-1].astype(np.uint8))         # RGB -> BGR


