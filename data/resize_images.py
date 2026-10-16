"""
Resize a directory of images to a square size (parallel).

CelebA-HQ is 1024x1024; resize to 256 before VAE-encoding to latents.

    python data/resize_images.py \
        --src /scratch/.../CelebAMask-HQ/CelebA-HQ-img \
        --dst /scratch/.../CelebAMask-HQ/CelebA-HQ-img-256 \
        --size 256 --workers 16
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool

from PIL import Image


def _resize_one(fname, src, dst, size, quality):
    out = os.path.join(dst, fname)
    if os.path.exists(out):                       # resumable
        return True
    try:
        with Image.open(os.path.join(src, fname)) as im:
            im = im.convert("RGB").resize((size, size), Image.LANCZOS)
            im.save(out, quality=quality)
        return True
    except Exception as e:
        print(f"  ! {fname}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    files = sorted(f for f in os.listdir(args.src)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"resizing {len(files)} images: {args.src} -> {args.dst} @ {args.size}x{args.size}")

    fn = partial(_resize_one, src=args.src, dst=args.dst, size=args.size, quality=args.quality)
    with Pool(args.workers) as pool:
        results = pool.map(fn, files, chunksize=64)

    ok = sum(results)
    print(f"done: {ok} ok, {len(results) - ok} failed -> {args.dst}")


if __name__ == "__main__":
    main()
