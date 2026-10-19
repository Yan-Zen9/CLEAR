# CLEAR: Cooperative Latent Flow for Visual Counterfactual Explanations with Accurate Region Preservation

**Accepted to WACV 2027** (Direct Round 2 submission).

[![arXiv](https://img.shields.io/badge/arXiv-2610.23577-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.23577)
[![Project Page](https://img.shields.io/badge/Project-Page-4285F4?logo=googlechrome&logoColor=white)](https://yan-zen9.github.io/CLEAR/)

**CLEAR** is an efficient latent Flow Matching framework for high-quality visual counterfactual explanations, addressing conflicts between classifier guidance and the generative flow, as well as unintended changes caused by VAE reconstruction. **Cooperative Flow Guidance** decomposes classifier guidance relative to the instantaneous flow velocity, adaptively suppressing conflicting components while preserving useful ones. **Adaptive Support Preservation (ASP)** dynamically localizes editable regions and preserves factual content outside them, while our analysis further reveals how directional conflicts evolve throughout generation. Experiments on CelebA-HQ and ImageNet demonstrate that CLEAR substantially improves counterfactual quality over latent-space baselines while achieving competitive quality with image-space methods at significantly lower computational cost.

## Installation

Python 3.10 or newer is recommended. Install the PyTorch build appropriate for your CUDA or ROCm platform if the pinned build in `requirements.txt` is not suitable.

```bash
git clone https://github.com/Yan-Zen9/CLEAR.git
cd CLEAR

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick start

### 1. Download pretrained weights

**Latent Flow Matching models.** We use the public checkpoints of [LFM](https://github.com/VinAIResearch/LFM); the download links are in the checkpoint table of its README. Place them under `checkpoints/` (the paths used in step 3).

| LFM experiment | Backbone | Dataset | Resolution | File |
| --- | --- | --- | --- | --- |
| `celeb_f8_dit` | DiT-L/2 (unconditional) | CelebA-HQ | 256×256 | `model_475.pth` |
| `imnet_f8_ditb2` | DiT-B/2 (class-conditional, used unconditionally at inference) | ImageNet | 256×256 | `model_875.pth` |

**VAE.** `stabilityai/sd-vae-ft-mse`, downloaded automatically by `diffusers` on first use.

**Classifiers under explanation.** We use the same classifiers as ACE.

- CelebA-HQ: download `checkpoints_decision_densenet.tar.gz` from the [STEEX](https://github.com/valeoai/STEEX/releases) releases and use `celebamaskhq/checkpoint.tar`. Keep the `celebamaskhq/` folder name in the path, since the classifier type is detected from it.
- ImageNet: the torchvision ResNet-50 (`IMAGENET1K_V1`):

```bash
wget -P checkpoints/classifiers/imagenet https://download.pytorch.org/models/resnet50-0676ba61.pth
```

### 2. Prepare the data

CLEAR starts from the VAE latent of the factual image and preserves the region outside the edit with the original pixels, so images are first resized to 256×256 and encoded to latents.

```bash
# CelebA-HQ
python data/resize_images.py --src <CelebAMask-HQ>/CelebA-HQ-img --dst data/celebahq/img256 --size 256
python data/encode_latents.py --src data/celebahq/img256 --dst data/celebahq/latents --size 256

# ImageNet: encode each class folder (<wnid>/<wnid>_<n>.JPEG) separately
python data/encode_latents.py --src data/imagenet/n02389026 --dst data/imagenet_latents/n02389026 --size 256
```

CelebA-HQ additionally needs a label table `data/celebahq/labels.csv` with the columns `hq_idx,split,Smiling,Young`. Data splits follow ACE.

### 3. Inference: generate counterfactuals

CelebA-HQ Smile as an example:

```bash
CF_DATASET=celeba \
CF_DIFFUSION_CKPT=checkpoints/lfm/celeba-hq/model_475.pth \
CF_PIXEL_CLASSIFIER_CKPT=checkpoints/classifiers/celebamaskhq/checkpoint.tar \
CF_CLASSIFIER_QUERY_LABEL=31 \
CF_LATENT_DIR=data/celebahq/latents \
CF_LABELS_CSV=data/celebahq/labels.csv \
CF_ATTR=Smiling \
CF_IMG_DIR=data/celebahq/img256 \
CF_SALIENCY_MODE=gradcam \
CF_MASK_RATIO=0.05 \
CF_OUTPUT_DIR=outputs/celebahq_smile \
python cf/generate.py
```

The other tasks only change the following variables:

| Task | `CF_DATASET` | `CF_CLASSIFIER_QUERY_LABEL` / `CF_ATTR` (or `CF_PAIR`) | `CF_MASK_RATIO` |
| --- | --- | --- | --- |
| CelebA-HQ Smile | `celeba` | `31` / `Smiling` | `0.05` |
| CelebA-HQ Age | `celeba` | `39` / `Young` | `0.12` |
| ImageNet | `imagenet` | `CF_PAIR=sorrel_zebra`, `persian_egyptian` or `cougar_cheetah` | `0.30` |

For ImageNet, also set `CF_LATENT_DIR=data/imagenet_latents`, `CF_IMG_DIR=data/imagenet`, `CF_IMG_EXT=.JPEG` and `CF_IMG_NESTED_WNID=true`, and use the ImageNet LFM checkpoint and the ResNet-50 classifier.

`CF_IMG_DIR` must be set: otherwise the region outside the edit keeps the VAE reconstruction instead of the exact factual pixels. All other hyperparameters default to the paper settings (`t_s=0.6`, `N=20`, `s=8`, `τ=0.5`, dilation radius 4). Results (`x_0.png`, `x_cf.png`, ...) are written to `CF_OUTPUT_DIR`.

### 4. Evaluation

Our evaluation protocol follows ACE: we use the evaluation code of ACE together with the evaluation model weights linked there. On CelebA-HQ we report FID, sFID, FVA, FS, MNAC, CD, COUT and flip rate; on ImageNet, FID, sFID, S3, COUT and flip rate. Efficiency (inference time and peak GPU memory) follows MaskDiME and is recorded per sample in `metrics.csv`.

## Training

CLEAR needs no training: all reported results use the public LFM checkpoints. To apply CLEAR to your own data, you can train your own latent flow model (an unconditional DiT) on latents encoded with `data/encode_latents.py` (see step 2):

```bash
python train/train_flow.py --latent_dir data/celebahq/latents --model DiT-L/2 --batch_size 64 --epochs 500
```

Set `--batch_size` (per GPU) and `--epochs` explicitly. For multiple GPUs, launch with `torchrun`:

```bash
torchrun --standalone --nproc_per_node=8 train/train_flow.py --latent_dir data/celebahq/latents --model DiT-L/2 --batch_size 64 --epochs 500
```

Add `--init_from <LFM checkpoint> --lr 1e-5` to fine-tune a public LFM checkpoint instead of training from scratch, and `--resume_from <ckpt.tar>` to resume a run.

Checkpoints are written to `records/<ds_name>_latent/Logs_Checkpoints/checkpoints/version_<n>/`, where `--ds_name` (default `CelebA`) only names the folder. `ckpt.tar` holds the full training state for resuming, and `ema_step<N>.pt` holds the EMA weights. Pass an `ema_step<N>.pt` file as `CF_DIFFUSION_CKPT` in step 3 to generate counterfactuals with your own model.

## Datasets

CelebA-HQ download and preparation follow ACE / STEEX ([CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)). ImageNet uses the three class pairs of the DiME / ACE protocol: sorrel ↔ zebra, Persian cat ↔ Egyptian cat, and cougar ↔ cheetah.

## BibTeX

```bibtex
@misc{zeng2026clear,
  title         = {{CLEAR}: Cooperative Latent Flow for Visual Counterfactual Explanations with Accurate Region Preservation},
  author        = {Yan Zeng and Changlu Guo and Oskar Kristoffersen and Anders Nymark Christensen and Morten Rieger Hannemose and Anders Bjorholm Dahl},
  year          = {2026},
  eprint        = {2610.23577},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2610.23577}
}
```

## License

CLEAR is released under the [CC BY-NC 4.0](LICENSE) license. Third-party code is credited in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
