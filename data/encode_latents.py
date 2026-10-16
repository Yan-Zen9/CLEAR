"""
Encode a directory of images to VAE latents (flat, same filenames).

    image <name>.jpg  ->  latent <name>.npy   (shape [4, H/8, W/8], scaled)

Latents are stored ALREADY scaled (z = vae.encode(x).mean * scaling_factor), so
decoding is vae.decode(z / scaling_factor) — matching cf/ and the flow trainer.
Deterministic (uses the posterior mean). Flat output dir; no train/test, no labels.

    python data/encode_latents.py \
        --src /scratch/.../CelebAMask-HQ/CelebA-HQ-img-256 \
        --dst /scratch/.../CelebAMask-HQ/CelebA-HQ-latent \
        --size 256 --batch_size 64
"""

import argparse
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from diffusers import AutoencoderKL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0, help="this worker's index in [0, num_shards)")
    ap.add_argument("--num_shards", type=int, default=1, help="total workers; each takes files[shard::num_shards]")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(args.vae).to(device).eval()
    sf = float(vae.config.scaling_factor)
    tf = T.Compose([T.Resize(args.size), T.CenterCrop(args.size), T.ToTensor()])  # -> [0,1]

    files = sorted(f for f in os.listdir(args.src)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    total = len(files)
    if args.num_shards > 1:                                 # data-parallel: disjoint filename slices
        files = files[args.shard::args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] encoding {len(files)}/{total} images -> {args.dst}  "
          f"(vae={args.vae}, sf={sf}, size={args.size}, bs={args.batch_size}, device={device})")

    batch, names = [], []

    @torch.no_grad()
    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device) * 2.0 - 1.0           # [-1,1]
        z = vae.encode(x).latent_dist.mean * sf                 # [B,4,H/8,W/8], scaled
        z = z.cpu().numpy()
        for n, zi in zip(names, z):
            np.save(os.path.join(args.dst, os.path.splitext(n)[0] + ".npy"), zi)  # [4,h,w]
        batch.clear(); names.clear()

    done = 0
    for i, f in enumerate(files):
        if os.path.exists(os.path.join(args.dst, os.path.splitext(f)[0] + ".npy")):
            done += 1
            continue
        with Image.open(os.path.join(args.src, f)) as im:
            batch.append(tf(im.convert("RGB")))
            names.append(f)
        if len(batch) >= args.batch_size:
            flush()
        if i % 2000 == 0:
            print(f"  {i}/{len(files)}")
    flush()
    n_out = len([f for f in os.listdir(args.dst) if f.endswith(".npy")])
    print(f"done: {n_out} latents in {args.dst} (skipped {done} already-existing)")


if __name__ == "__main__":
    main()
