"""
Configuration for flow-matching counterfactual generation (MedDiME).

Replaces the DDPM CfConfigLatent. DDPM-only knobs (total_timesteps,
cf_from_timestep, ts_guided, rescale_t/_to, start_tau, inpaint_th,
save_attribution_at, ...) are gone. Flow matching adds `t_start` (image-to-image
start time in (0,1); closer to 1 = less noise) and `num_steps` (ODE steps).

Dataset / paths are env-driven (CF_DATASET, CF_MODEL_VAR, CF_DIFFUSION_CKPT,
CF_PIXEL_CLASSIFIER_CKPT, MEDDIME_RECORDS_ROOT, MEDDIME_WEIGHTS_DIR, ...).
Missing checkpoints WARN instead of asserting, so the module imports anywhere.

Support attribution:
    CF_SALIENCY_MODE=input_gradient  (default; original input-gradient method)
    CF_SALIENCY_MODE=gradcam         (target-logit Grad-CAM, at every step)
    CF_GRADCAM_LAYER=model.layer4    (optional module path; default auto-detection)
The CE guidance gradient and other sampling settings are unchanged by this switch.
"""

import os
from dataclasses import dataclass, asdict, is_dataclass
from pprint import pformat

# ------------------------------------------------------------------ VAE config
# Default VAE = SD1.5 MSE-finetuned: best 4-channel / 8x reconstruction at 256,
# light decoder (cheap per-step decode+backprop). The real scaling factor is
# read from vae.config at runtime; DEFAULT_VAE_SCALING is only a fallback.
DEFAULT_VAE_MODEL = "stabilityai/sd-vae-ft-mse"
DEFAULT_VAE_SCALING = 0.18215

# Path roots (env-overridable; empty -> provide ckpt paths via env at run time)
_RECORDS_ROOT = os.environ.get("MEDDIME_RECORDS_ROOT", "").strip()
_WEIGHTS_DIR = os.environ.get("MEDDIME_WEIGHTS_DIR", "").strip()

# ----------------------------------------------------- ImageNet CE benchmark maps
# The DiME/ACE/FastDiME 6-class subset. Folder (wnid) -> TRUE ImageNet-1k index
# (the resnet50 outputs 1000-way logits, so guidance/flip use these indices, NOT
# the 0..5 ImageFolder order). Three flip pairs: sorrel<->zebra, cheetah<->cougar,
# Egyptian<->Persian.
IMAGENET_CE_WNID_TO_IDX = {
    "n02123394": 283,   # Persian cat
    "n02124075": 285,   # Egyptian cat
    "n02125311": 286,   # cougar
    "n02130308": 293,   # cheetah
    "n02389026": 339,   # sorrel (horse)
    "n02391049": 340,   # zebra
}
IMAGENET_CE_PAIR = {
    283: 285, 285: 283,   # Persian  <-> Egyptian
    286: 293, 293: 286,   # cougar   <-> cheetah
    339: 340, 340: 339,   # sorrel   <-> zebra
}
# CF is run ONE PAIR at a time (metrics = per-pair in-class FID, ACE protocol):
# CF_PAIR restricts the loader to a pair's two wnid dirs (2600 imgs -> one out dir).
IMAGENET_CE_PAIRS = {
    "persian_egyptian": ["n02123394", "n02124075"],
    "cougar_cheetah":   ["n02125311", "n02130308"],
    "sorrel_zebra":     ["n02389026", "n02391049"],
}


@dataclass
class CfConfig:
    """Flow-matching counterfactual generation config."""

    # --- flow matching sampling ---
    num_steps: int            # number of ODE (Euler) steps
    t_start: float            # start time in (0,1); closer to 1 = less noise
    scheduler: str            # 'cond_ot' | 'polynomial' | 'cosine'

    # --- single support mask (paper M^x; saliency de-speckled by mask_presmooth) ---
    model_var: str            # 'DiME' | 'FastDiME-woM' | 'FastDiME'  (only names the fallback out-dir)
    mask_ratio: float               # top-ρ support fraction — THE per-task knob (Eq. 11)
    use_lcc: bool

    # --- guidance (classifier cross-entropy only, paper Eq. 6) ---
    start_scale: float        # max guidance scale s (Eq. 10)

    # --- data / checkpoints ---
    ds_name: str
    diffusion_checkpoint_path: str
    vae_model_name: str

    # --- pixel classifier guidance ---
    classify_space: str = "pixel"          # 'pixel' (decode->classify) | 'latent' (ablation)
    use_mask: bool = True
    pixel_classifier_checkpoint_path: str = ""
    classifier_query_label: int = 31       # CelebA query label (ignored for medical StandardDenseNet)
    classifier_mode: str = "auto"          # 'auto' | 'legacy_celeba' | 'celebahq'
    vae_scaling_factor: float = DEFAULT_VAE_SCALING
    dilation_pixel: int = 4                # dw2-verified base dilation (retry ladder widens it)
    mask_feather: int = 7
    mask_presmooth: int = 5                # blur saliency before top-k (de-speckle); 0 = off

    # --- support attribution (independent of classifier guidance) ---
    saliency_mode: str = "input_gradient"  # 'input_gradient' | 'gradcam'
    gradcam_layer: str = ""               # optional classifier module path

    # --- io / misc ---
    confounder_filter: str = "all"


# ------------------------------------------------------------ env-driven build
model_var = os.environ.get("CF_MODEL_VAR", "FastDiME")
dataset = os.environ.get("CF_DATASET", "celeba")

# dataset-specific defaults: ds_name, guidance scale, default pixel classifier.
# Guidance is ALWAYS the classifier cross-entropy only (paper Eq. 6) — there is no
# preservation (L1/TV/perceptual) loss; decision-irrelevant content is kept by Adaptive
# Support Preservation alone.
if dataset == "celeba":
    ds_name = "CelebA"
    start_scale = float(os.environ.get("CF_SCALE", 8.0))
    # CelebA-HQ 256 classifier to explain: DenseNet121, 3-class head (20=Male,31=Smile,39=Young).
    # 'celebamaskhq' in the path -> auto-detected as 'celebahq' mode (256, 0.5/0.5 norm).
    # (oracle_celebahq/checkpoint.tar is a separate 40-class ResNet50-ft used only for eval.)
    _clf_default = os.path.join(_WEIGHTS_DIR, "celebamaskhq", "checkpoint.tar") if _WEIGHTS_DIR else ""
elif dataset in ["isic_ruler", "isic_melanoma_nevus", "che_pm", "Che_PM"]:
    ds_name = {"isic_ruler": "ISIC_Ruler", "isic_melanoma_nevus": "ISIC_Melanoma_Nevus",
               "che_pm": "Che_PM", "Che_PM": "Che_PM"}[dataset]
    start_scale = float(os.environ.get("CF_SCALE", 8.0))
    _clf_default = ""                                                   # set CF_PIXEL_CLASSIFIER_CKPT
elif dataset == "imagenet":
    ds_name = "ImageNet"
    # ACE/DiME/FastDiME 6-class CE benchmark on a torchvision ResNet-50 (1000-way).
    # Target = paired class index (generate.py, per-wnid).
    start_scale = float(os.environ.get("CF_SCALE", 8.0))
    # ACE classifier = torchvision resnet50 IMAGENET1K_V1 (bare state_dict). Default to
    # the cached hub file; override with CF_PIXEL_CLASSIFIER_CKPT.
    _clf_default = (os.path.join(_WEIGHTS_DIR, "imagenet_classifier", "resnet50-0676ba61.pth")
                    if _WEIGHTS_DIR else "")
else:
    raise NotImplementedError(f"Dataset {dataset} not supported yet")

# flow-matching CF trajectory: N=20 Euler steps from t_start=0.6 -> 1 (paper).
num_steps = int(os.environ.get("CF_NUM_STEPS", 20))
t_start = float(os.environ.get("CF_T_START", 0.6))
scheduler = os.environ.get("CF_SCHEDULER", "cond_ot")

# SINGLE SUPPORT (paper M^x, Eq. 11): ONE top-ρ saliency mask reused for the classifier
# view (Eq. 5), the latent gate M^z = avg_pool(M^x) (Eq. 12-13), and the final factual-pixel
# composite (Eq. 14). mask_ratio is the per-task knob ρ (smile 0.05 / age 0.12 / imagenet 0.30);
# structural constants (dilation, feather, presmooth, steps) stay fixed.
mask_ratio = float(os.environ.get("CF_MASK_RATIO", 0.05))
_use_lcc_env = os.environ.get("CF_USE_LCC", "").lower()
use_lcc = _use_lcc_env != "false" if _use_lcc_env else True

confounder_filter = os.environ.get("CF_CONFOUNDER_FILTER", "all")

# checkpoint paths: env first, else derive from MEDDIME_RECORDS_ROOT/WEIGHTS_DIR if set
_diff_default = (os.path.join(_RECORDS_ROOT, f"{ds_name}_flow", "checkpoints", "ckpt.tar")
                 if _RECORDS_ROOT else "")
diffusion_checkpoint_path = os.environ.get("CF_DIFFUSION_CKPT", _diff_default)
pixel_classifier_checkpoint_path = os.environ.get("CF_PIXEL_CLASSIFIER_CKPT", _clf_default)
vae_model_name = os.environ.get("CF_VAE_NAME", DEFAULT_VAE_MODEL)

classify_space = os.environ.get("CF_CLASSIFY_SPACE", "pixel")
use_mask = os.environ.get("CF_USE_MASK", "true").lower() != "false"
classifier_query_label = int(os.environ.get("CF_CLASSIFIER_QUERY_LABEL", 31))
classifier_mode = os.environ.get("CF_CLASSIFIER_MODE", "auto")

assert model_var in ["DiME", "FastDiME", "FastDiME-woM"], (
    f"model_var '{model_var}' not supported by the flow sampler "
    f"(two-pass FastDiME-2/2+ unavailable in latent)."
)

cfconfig = CfConfig(
    num_steps=num_steps,
    t_start=t_start,
    scheduler=scheduler,
    model_var=model_var,
    mask_ratio=mask_ratio,
    use_lcc=use_lcc,
    start_scale=start_scale,
    ds_name=ds_name,
    diffusion_checkpoint_path=diffusion_checkpoint_path,
    vae_model_name=vae_model_name,
    confounder_filter=confounder_filter,
    classify_space=classify_space,
    use_mask=use_mask,
    pixel_classifier_checkpoint_path=pixel_classifier_checkpoint_path,
    classifier_query_label=classifier_query_label,
    classifier_mode=classifier_mode,
    saliency_mode=os.environ.get("CF_SALIENCY_MODE", "input_gradient").strip().lower(),
    gradcam_layer=os.environ.get("CF_GRADCAM_LAYER", "").strip(),
)

if cfconfig.saliency_mode not in {"input_gradient", "gradcam"}:
    raise ValueError("CF_SALIENCY_MODE must be 'input_gradient' or 'gradcam'.")

for _label, _env, _p in [("diffusion", "CF_DIFFUSION_CKPT", diffusion_checkpoint_path),
                         ("pixel classifier", "CF_PIXEL_CLASSIFIER_CKPT", pixel_classifier_checkpoint_path)]:
    if not _p or not os.path.exists(_p):
        print(f"⚠️  [cf.config] {_label} checkpoint not found: {_p or '(empty)'}\n"
              f"    set {_env} to override.")

print(f"\n{'='*50}\n✅ Flow-matching CF config ready\n{'='*50}")
print(f"dataset={ds_name}  t_start={t_start}  steps={num_steps}  scale={start_scale}  guidance=Lc (classifier CE only)")
print(f"saliency={cfconfig.saliency_mode} gradcam_layer={cfconfig.gradcam_layer or 'auto'}")
print(f"scheduler={scheduler}  mask[single support ratio={mask_ratio} lcc={use_lcc}]")
print(f"NOTE: coop/retry/dilation/presmooth are env-only (see the runner's echo) — NOT in this CfConfig dump.")
print(f"{'='*50}\n")


# ------------------------------------------------------------ print / save
def _to_dict(cfg):
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(vars(cfg))
    return {"config": cfg}


def print_config(cfg):
    print("=" * 50)
    print("CF config:")
    print(pformat(_to_dict(cfg)))
    print("=" * 50)


def save_config(cfg, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(pformat(_to_dict(cfg)))
