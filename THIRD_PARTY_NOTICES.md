# Third-party notices

CLEAR is released under CC BY-NC 4.0 (see [LICENSE](LICENSE)). Parts of this repository are derived from the projects below. Their copyright remains with the respective holders, and their license terms continue to apply to the derived files.

| Upstream project | License | Used in |
| --- | --- | --- |
| [LFM: Flow Matching in Latent Space](https://github.com/VinAIResearch/LFM) (VinAI) | AGPL-3.0 | `flow/dit.py` follows the architecture and state-dict layout of LFM's `models/DiT.py`, so that the released LFM checkpoints load directly. |
| [DiT: Scalable Diffusion Models with Transformers](https://github.com/facebookresearch/DiT) (Meta) | CC BY-NC | `flow/dit.py`, via LFM, which adapts DiT. |
| [flow_matching](https://github.com/facebookresearch/flow_matching) (Meta) | CC BY-NC 4.0 | `flow/path.py`, `flow/scheduler.py` and `flow/solver.py` are a minimal distillation of this library. |
| [Nerfies project page template](https://github.com/nerfies/nerfies.github.io) | CC BY-SA 4.0 | Layout and styles of the project page in `docs/`, which is therefore also licensed under CC BY-SA 4.0. |
| [Bulma](https://github.com/jgthms/bulma) | MIT | `docs/static/css/bulma.min.css`. |
| [Font Awesome Free](https://fontawesome.com/license/free) | Icons CC BY 4.0, fonts SIL OFL 1.1, code MIT | `docs/static/js/fontawesome.all.min.js`. |
| WACV 2027 logo | Copyright of the WACV 2027 organizers | `docs/assets/wacv-2027.png`, recolored from the logo on the [WACV 2027 website](https://wacv.thecvf.com/Conferences/2027) and used only to identify the venue. |

Changes: the time convention was converted (LFM uses `t=0` for data and `t=1` for noise, CLEAR uses `t=0` for noise and `t=1` for data), `timm` layers were re-implemented with identical parameter names to remove the dependency, and only the pieces needed by CLEAR were kept.

## Not redistributed

This repository does not contain any model weights or datasets. The LFM checkpoints, the `stabilityai/sd-vae-ft-mse` VAE, the CelebA-HQ and ImageNet classifiers, and the datasets are downloaded from their original sources and are subject to their own licenses and terms of use.
