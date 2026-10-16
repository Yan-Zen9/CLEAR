"""
CLEAR counterfactual core (latent flow matching): Adaptive Support Preservation +
Cooperative Flow Guidance + the guided sampler, merged into one module.

The classifier is a REAL pixel classifier reached through the frozen VAE decoder, so
guidance backprops latent -> velocity DiT -> decoder -> classifier. Per step t -> t+dt
(t_start -> 1, clean = 1):

    z_var = z_t (requires grad)
    v     = vθ(z_var, t)                          # flow velocity (detached)       (Eq. 3)
    z_hat = z_var + (1-t)·v                       # one-step clean estimate (CondOT)(Eq. 4)
    x_hat = D(z_hat / sf)                         # decode to image, [-1,1]
    x_view= m·x_hat + (1-m)·x0_real               # classifier view (Eq. 5)

    --- ONE classifier backward yields BOTH of these (guided_gradient) ---
    g_cls   = ∂CE(f(x_view), y) / ∂z_var          # classifier force  u_t = -g_cls (Eq. 6)
    sal_pix = |∂CE / ∂x_view|                     # saliency -> THIS step's support

    push  = coop(g_cls, v)                        # Cooperative Flow Guidance (Eq. 7-10)
    z_{t+dt} = m_lat·(z_t + dt·(v - push)) + (1-m_lat)·transport(z_0, t+dt; eps)

GUIDANCE OBJECTIVE: the ONLY guidance loss is the classifier cross-entropy L_cls
(paper Eq. 6). There is NO preservation (L1 / TV / perceptual) term — decision-irrelevant
content is preserved purely by Adaptive Support Preservation (the latent gate + the final
factual-pixel composite), not by a proximity loss.

COOPERATIVE FLOW GUIDANCE (Eq. 7-10): reconcile the classifier force u_t = -g_cls with
the instantaneous flow velocity v_t. Decompose u_t into components parallel / orthogonal
to v_t, attenuate ONLY the conflicting parallel part by β=clip(-cos(u,v)/τ, 0, 1), then
cap the guided push magnitude at ‖v‖ (guide_vrel=1) scaled by s.

Saliency modes: CF_SALIENCY_MODE=gradcam (paper, Eq. 11) or input_gradient.
Grad-CAM uses target-logit feature maps from the factual image initially and from the
current classifier view at each step, via a separate classifier pass. CF_GRADCAM_LAYER
optionally selects a classifier module by dotted name.

SUPPORT (single mask M^x, Eq. 11): pre-smooth -> top-ρ -> largest-connected-component ->
round dilation. The SAME support is used (i) as the classifier view M^x (Eq. 5, optionally
edge-feathered for a seam-free composite), (ii) average-pooled to the latent grid M^z
(Eq. 12) as the soft gate that constrains the guided update (Eq. 13), and (iii) for the
final factual-pixel composite (Eq. 14). Do NOT "dilate before LCC": that bridges speckle
into one huge blob which LCC can no longer prune. The FIRST support comes from the ORIGINAL
image x0_real; every later support comes from that step's classifier view.

Background uses a FIXED eps so at t=1 it recovers the original latent exactly.
use_mask=False -> plain flow. NOTE: on ROCm the sampler is NOT bit-reproducible
(non-deterministic conv backward amplified through 20 guided steps); CF_SEED fixes eps
only. Compare runs by AGGREGATE metrics, never per-sample.
"""

import os
import contextlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from flow import AffineProbPath, build_scheduler
from cf.classifier import classifier_logits
from cf.saliency import target_gradcam


# ============================================================ mixed precision
# bf16 autocast for the heavy forwards (UNet + VAE decode + classifier). Values
# handed back to the ODE integrator (v, x_hat) and the guidance gradient are
# upcast to fp32, so integration/guidance stay fp32-stable while the conv-heavy
# forward/backward run in bf16. Toggle with CF_AMP=false.
_AMP_ENABLED = os.environ.get("CF_AMP", "true").lower() != "false"
_AMP_DTYPE = torch.bfloat16
_GRAD_CHECKPOINT = os.environ.get("CF_GRAD_CHECKPOINT", "false").lower() == "true"
_DETACH_FLOW_GUIDANCE = os.environ.get("CF_DETACH_FLOW", "false").lower() == "true"
# Force the VAE decode to fp32 even under AMP: its bf16 conv-BACKWARD faults on
# ROCm/MIOpen at high resolution (1024px "Memory access fault ... Reason: Unknown"),
# and bf16 buys nothing here — the decode dominates each guided step at any precision
# (~90% of wall-clock), so this is free. UNet + classifier still run bf16 under AMP.
# Disable with CF_DECODE_FP32=false (e.g. a GPU where bf16 decode is fine and faster).
_DECODE_FP32 = os.environ.get("CF_DECODE_FP32", "true").lower() != "false"


def _memory_efficient(function, *args):
    """Optionally recompute frozen inference modules during backward."""
    if _GRAD_CHECKPOINT and torch.is_grad_enabled():
        return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=False)
    return function(*args)


def _amp_ctx():
    if _AMP_ENABLED and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=_AMP_DTYPE)
    return contextlib.nullcontext()


def _decode_fp32_ctx():
    """Nested context that turns autocast OFF for the VAE decode (see _DECODE_FP32).
    A no-op when AMP is already off or CF_DECODE_FP32=false."""
    if _DECODE_FP32 and _AMP_ENABLED and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", enabled=False)
    return contextlib.nullcontext()


# ============================================================ saliency / masks
@torch.enable_grad()
def target_saliency(bundle, x_img, y_target):
    """Single-shot target-class saliency |d(target logit)/dx| -> [B,1,H,W].

    Places the SEED mask (step 0, from the original image); every LATER mask reuses
    guided_gradient's sal_pix (|dCE/dx_view|, the same classifier pass as the gradient).
    This is the signal MaskDiME uses. Speckle is dealt with DOWNSTREAM by
    _masks_from_saliency's pre-smooth (one conv), NOT by noise-averaging: SmoothGrad was
    removed here — as a guidance direction it crashed flip 97%->70%, and as a mask it only
    ever reached this seed call after the D1+D2 refactor. See logs/ablation_smile Batch A/D.
    """
    x = x_img.detach().requires_grad_(True)
    with _amp_ctx():
        logits = classifier_logits(bundle, x)
        target = logits.gather(1, y_target.view(-1, 1).long()).sum()
    grad = torch.autograd.grad(target, x)[0]
    return grad.abs().mean(dim=1, keepdim=True).float()


def _largest_connected_component_batch(mask):
    out = torch.zeros_like(mask)
    for i in range(mask.shape[0]):
        m = (mask[i, 0].cpu().numpy() > 0.5).astype(np.uint8)
        if m.sum() == 0:
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            out[i, 0] = mask[i, 0]
            continue
        largest = max(range(1, n), key=lambda k: stats[k, cv2.CC_STAT_AREA])
        out[i, 0] = torch.from_numpy((labels == largest).astype(np.float32)).to(mask.device)
    return out


_DISK_KERNEL_CACHE = {}


def _disk_kernel(radius, device, dtype):
    """Cached circular (disk) structuring element as a conv kernel [1,1,k,k]."""
    key = (radius, device, dtype)
    k = _DISK_KERNEL_CACHE.get(key)
    if k is None:
        rng = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        ys, xs = torch.meshgrid(rng, rng, indexing="ij")
        disk = ((xs ** 2 + ys ** 2) <= radius * radius).to(dtype)
        k = disk.view(1, 1, 2 * radius + 1, 2 * radius + 1)
        _DISK_KERNEL_CACHE[key] = k
    return k


def dilate_round(mask, radius=5):
    """Binary dilation with a ROUND (disk) structuring element, on GPU.

    Equivalent to the old cv2 MORPH_ELLIPSE dilation (a pixel turns on if any
    disk-neighbor is on), but as a single conv2d — no per-image .cpu()/numpy
    round-trip, so no host sync inside the per-step loop. Batched over [B,1,H,W].
    """
    if radius <= 0:
        return mask
    m = (mask > 0.5).to(mask.dtype)
    k = _disk_kernel(radius, m.device, m.dtype)
    out = F.conv2d(m, k, padding=radius)
    return (out > 0.5).to(mask.dtype)


def gaussian_blur(mask, kernel_size=5, sigma=2.0):
    """Soft edges to avoid decode seams (optional). kernel_size must be odd."""
    if kernel_size <= 1 or sigma <= 0:
        return mask
    pad = kernel_size // 2
    coords = torch.arange(kernel_size, device=mask.device, dtype=mask.dtype) - (kernel_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    g2d = (g[:, None] * g[None, :])[None, None]
    g2d = g2d.expand(mask.shape[1], 1, kernel_size, kernel_size)
    return F.conv2d(mask, g2d, padding=pad, groups=mask.shape[1])


def make_hard_mask(saliency, k_percent=0.03, dilation=2, use_lcc=True):
    """Saliency -> HARD binary mask, [B,1,H,W] in {0,1}.

    Order: top-k threshold -> largest-connected-component (localize to the main
    decision blob, drop scattered outliers) -> SMALL round dilation (margin).
    LCC must come BEFORE dilation: dilating a scattered top-k map first bridges
    the speckle into one huge blob (a 1% saliency balloons to ~12%), after which
    LCC can no longer prune it. So a SPECKLY saliency has to be denoised UPSTREAM
    of top-k (see _masks_from_saliency's pre-smooth) -- never by reordering
    LCC/dilation. Given a clean saliency, topk->LCC lands one tight blob on the
    decision region and the small dilation only adds a thin margin.
    """
    B = saliency.shape[0]
    flat = saliency.view(B, -1)
    thr = torch.quantile(flat, q=1 - k_percent, dim=1, keepdim=True).view(B, 1, 1, 1)
    mask = (saliency >= thr).float()
    if use_lcc:
        mask = _largest_connected_component_batch(mask)
    mask = dilate_round(mask, radius=dilation)
    return mask


def feather_pixel(m_hard, feather=13):
    """Hard mask -> feathered (Gaussian edge) pixel mask for seamless compositing."""
    if not feather or feather <= 1:
        return m_hard
    return gaussian_blur(m_hard, kernel_size=feather, sigma=max(1.0, feather / 4))


def downsample_to_latent(m, latent_hw):
    """Average-pool a PIXEL mask to latent resolution. NOT feathered — the pooling
    itself gives a mildly soft boundary (no 8x8 decode blocks); used for the LARGE
    latent-edit mask, whose hard boundary lands in a quiet region that paste-back
    discards anyway, so no extra Gaussian feather is needed there."""
    h_lat = latent_hw[-2] if not isinstance(latent_hw, int) else latent_hw
    ratio = max(1, m.shape[-2] // h_lat)
    m_lat = F.avg_pool2d(m, kernel_size=ratio, stride=ratio)   # soft (avg, not max)
    if m_lat.shape[-2:] != tuple(latent_hw):
        m_lat = F.interpolate(m_lat, size=tuple(latent_hw), mode="bilinear", align_corners=False)
    return m_lat


# ============================================================ guidance (B)
def decode(vae, z, scaling_factor):
    """VAE-decode a latent to an image in [-1,1], keeping the graph (frozen VAE)."""
    return vae.decode(z / scaling_factor).sample


@torch.enable_grad()
def guided_gradient(model, bundle, z_t, t_b, path, y_target, *,
                    vae, scaling_factor, x0_pixel, pixel_mask,
                    grad_clip_sigma=0.0, x0_real=None):
    """Classifier force g_cls = ∂L_cls/∂z_t, backprop through the one-step clean predictor.

    The ONLY guidance objective is the classifier cross-entropy L_cls = CE(f(x_view), y)
    (paper Eq. 6); there is no preservation (L1 / TV / perceptual) term. ONE classifier
    backward yields BOTH the latent guidance g_cls=∂CE/∂z AND the pixel saliency
    sal_pix=|∂CE/∂x_view| that places the support — so at every step the support and the
    gradient are the SAME classifier evaluation on the SAME classifier view x_view.

    Returns: g_cls [B,C,h,w] (classifier latent gradient, detached), v [B,C,h,w] (velocity,
    detached), prob [B] (target prob, detached), losses dict, x_view [B,3,H,W] (detached
    classifier view), sal_pix [B,1,H,W] (support saliency, detached).
    """
    y_target = y_target.view(-1).long()
    z_var = z_t.detach().requires_grad_(True)

    with _amp_ctx():                                             # bf16 forward
        if _DETACH_FLOW_GUIDANCE:
            with torch.no_grad():
                v = model(z_var, t_b)
        else:
            v = _memory_efficient(model, z_var, t_b)
        z_hat = path.velocity_to_target(v, z_var, t_b)           # x_t + (1-t)v        (Eq. 4)
        with _decode_fp32_ctx():                                 # fp32 decode (ROCm bf16-bwd fault; free)
            z_dec = z_hat.float() if _DECODE_FP32 else z_hat
            x_hat = _memory_efficient(lambda z: decode(vae, z, scaling_factor), z_dec)
        x_hat = x_hat.float() if _DECODE_FP32 else x_hat         # keep downstream math in fp32
        orig = x0_real if x0_real is not None else x0_pixel      # background = TRUE original (== output bg)
        x_view = x_hat
        if pixel_mask is not None and orig is not None:          # classifier view (Eq. 5)
            x_view = pixel_mask * x_hat + (1.0 - pixel_mask) * orig

        logits = _memory_efficient(lambda x: classifier_logits(bundle, x), x_view)
        ce = F.cross_entropy(logits, y_target, reduction="none")  # L_cls, the sole objective (Eq. 6)
        cls_loss = ce.sum()

    prob = torch.softmax(logits.detach().float(), dim=1).gather(1, y_target.view(-1, 1)).squeeze(1)
    # ONE classifier backward -> BOTH the latent guidance (∂CE/∂z) AND the pixel saliency
    # (∂CE/∂x_view) that places the support. So the support and the gradient are, by
    # construction, the SAME classifier evaluation on the SAME classifier view x_view.
    g_cls, sal_grad = torch.autograd.grad(cls_loss, (z_var, x_view))
    g_cls = g_cls.detach()
    sal_pix = sal_grad.abs().mean(dim=1, keepdim=True).detach()   # [B,1,H,W] support saliency
    if grad_clip_sigma and grad_clip_sigma > 0:               # clamp the classifier grad ±kσ:
        flat = g_cls.flatten(1)                               # drops outlier components ->
        mu = flat.mean(dim=1).view(-1, 1, 1, 1)               # cleaner, more localized edits,
        sd = flat.std(dim=1).view(-1, 1, 1, 1)                # keeps the bulk of the gradient.
        g_cls = torch.maximum(torch.minimum(g_cls, mu + grad_clip_sigma * sd), mu - grad_clip_sigma * sd)
    losses = {"Lc": ce.detach().mean()}
    # upcast values handed back to the fp32 integrator (keeps ODE step fp32-stable).
    return g_cls, v.detach().float(), prob, losses, x_view.detach().float(), sal_pix


# ============================================================ cooperative guidance
def cooperative_guidance(g, v, eff_scale, *, project="off", vrel=0.0, conflict_tau=0.5, vmin=0.0):
    """Reconcile the CLASSIFIER gradient g (pass g_cls ONLY, not the joint loss) with the
    generative flow v so the two forces cooperate instead of fighting (adversarial). Returns the
    guidance PUSH to subtract from v — the update is x += dt*(v - push). The actual force is -g;
    conflict = when -g opposes v, i.e. g·v > 0.

      (a) project='conflict'      : HARD — drop g's flow-parallel component whenever g·v > 0.
          The classifier may steer ACROSS the flow (a real edit) but not backward against it.
      (a') project='conflict_soft': SOFT — attenuate that parallel part by β=clip(cos(g,v)/tau,0,1),
          so weak conflict is barely touched (keeps flip/semantics) and strong conflict is removed.
      (b) vrel>0: cap ||push|| <= vrel*max(||v||, vmin) per sample; vmin floors the reference so
          guidance doesn't vanish where ||v|| is naturally small.

    project='off' and vrel=0 reproduce the plain additive guidance eff_scale*g.
    """
    if project in ("conflict", "conflict_soft"):
        gf, vf = g.flatten(1), v.flatten(1)
        dot = (gf * vf).sum(1, keepdim=True)                      # g·v per sample
        vn2 = (vf * vf).sum(1, keepdim=True).clamp_min(1e-12)
        g_par = (dot / vn2).view(-1, 1, 1, 1) * v                 # component of g along v
        if project == "conflict":
            beta = (dot > 0).float().view(-1, 1, 1, 1)           # hard: full removal on conflict
        else:                                                     # soft: scale removal by conflict strength
            gn = gf.norm(dim=1, keepdim=True).clamp_min(1e-12)
            vn_ = vf.norm(dim=1, keepdim=True).clamp_min(1e-12)
            beta = ((dot / (gn * vn_)) / conflict_tau).clamp(0.0, 1.0).view(-1, 1, 1, 1)
        g = g - beta * g_par                                      # remove (part of) the conflicting parallel
    push = eff_scale * g
    if vrel and vrel > 0:
        vn = v.flatten(1).norm(dim=1).clamp_min(max(vmin, 1e-12)).view(-1, 1, 1, 1)
        pn = push.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
        push = push * (vrel * vn / pn).clamp(max=1.0)            # only shrink, never grow
    return push


# ============================================================ sampler
class GuidedFlow:
    """Classifier-guided flow-matching sampler (gradient B, pixel guidance, single
    dual support masks). No early-stop; optional adaptive guidance."""

    def __init__(self, *, num_steps=20, scheduler="cond_ot", device="cuda",
                 classify_space="pixel", scaling_factor=0.18215, vae_decoder=None,
                 use_mask=True, mask_ratio=0.05, dilation_pixel=4,
                 use_lcc=True, mask_feather=7, mask_presmooth=5,
                 grad_clip_sigma=3.0, adaptive_guidance=False, adapt_threshold=0.85,
                 guide_project="conflict_soft", guide_vrel=1.0, guide_tau=0.5, guide_vmin=0.0,
                 labels=None, saliency_mode="input_gradient", gradcam_layer=""):
        self.saliency_mode = str(saliency_mode).strip().lower()
        if self.saliency_mode not in {"input_gradient", "gradcam"}:
            raise ValueError("saliency_mode must be 'input_gradient' or 'gradcam'.")
        self.gradcam_layer = str(gradcam_layer).strip()
        self.grad_clip_sigma = float(grad_clip_sigma)   # ③ clamp guidance grad to ±kσ (0=off)
        self.adaptive_guidance = bool(adaptive_guidance)  # FiRe: eff = base·(n/N)·max(0,(τ-p)/τ)
        self.adapt_threshold = float(adapt_threshold)     # τ: ease guidance to 0 as target prob -> τ
        # cooperative guidance (reconcile classifier g with flow v; off = plain additive)
        self.guide_project = str(guide_project)           # 'off' | 'conflict' | 'conflict_soft' (a)
        self.guide_vrel = float(guide_vrel)               # cap ||push|| <= vrel*max(||v||,vmin) (b); 0=off
        self.guide_tau = float(guide_tau)                 # soft-projection temperature (conflict_soft)
        self.guide_vmin = float(guide_vmin)               # floor for the ||v|| reference in the (b) cap
        self.path = AffineProbPath(build_scheduler(scheduler))
        self.num_steps = int(num_steps)
        self.device = device
        self.classify_space = classify_space
        self.scaling_factor = scaling_factor
        self.vae = vae_decoder
        self.use_mask = use_mask
        # SINGLE SUPPORT (paper M^x, Eq. 11): ONE top-ρ saliency mask, reused for
        #   - the classifier view (Eq. 5, edge-feathered for a seam-free composite),
        #   - the latent gate M^z = avg_pool(M^x) that constrains the guided update (Eq. 12-13),
        #   - the final factual-pixel composite (Eq. 14).
        # mask_ratio is the per-task knob ρ (smile .05 / age .12 / imagenet .30).
        self.mask_ratio = float(mask_ratio)
        self.dil_pix = dilation_pixel
        self.use_lcc = use_lcc
        self.feather = mask_feather
        self.presmooth = int(mask_presmooth)   # >1: blur saliency before top-k (de-speckle); 0=off
        self.labels = labels

    # ----------------------------------------------------------------- helpers
    def _t_batch(self, value, bs):
        return torch.full((bs,), float(value), device=self.device)

    def add_noise(self, z_0, t_start, noise=None):
        """Partially noise the original latent to time t_start (image-to-image start)."""
        eps = torch.randn_like(z_0) if noise is None else noise
        return self.path.sample(x_0=eps, x_1=z_0, t=self._t_batch(t_start, z_0.shape[0])).x_t

    def _transport_background(self, z_0, t_value, eps):
        """Original latent on the path at time t with a FIXED eps (recovers z_0 at t=1)."""
        return self.path.sample(x_0=eps, x_1=z_0, t=self._t_batch(t_value, z_0.shape[0])).x_t

    def _saliency(self, bundle, x_img, y_target):
        """Seed support signal; Grad-CAM also evaluates each subsequent view."""
        if self.saliency_mode == "gradcam":
            with _amp_ctx():
                return target_gradcam(bundle, x_img, y_target, self.gradcam_layer)
        return target_saliency(bundle, x_img, y_target)

    def _masks_from_saliency(self, saliency, latent_hw):
        """ONE saliency -> (m_pixel, m_lat), both derived from the SAME support M^x:
          m_pixel: the top-ρ support (Eq. 11), edge-feathered -> classifier view (Eq. 5)
                   and final factual-pixel composite (Eq. 14).
          m_lat:   the SAME support average-pooled to latent resolution -> soft gate M^z
                   (Eq. 12) constraining the guided update (Eq. 13).

        presmooth>1 blurs the saliency ONCE here, upstream of top-k: an isolated spike
        averages down against its dark neighbours (dies at top-k) while a dense cluster
        reinforces (survives, contiguous), so LCC locks onto the real decision blob
        instead of a noise island. 0 = off (plain single-shot behaviour)."""
        if self.presmooth > 1:
            ks = int(self.presmooth) | 1                   # gaussian_blur needs an odd kernel
            saliency = gaussian_blur(saliency, kernel_size=ks, sigma=max(1.0, ks / 4))
        m_hard = make_hard_mask(saliency, self.mask_ratio, self.dil_pix, self.use_lcc)
        m_pixel = feather_pixel(m_hard, self.feather)      # soft EDGE of the SAME support
        m_lat = downsample_to_latent(m_hard, latent_hw)    # P(M^x) -> soft latent gate
        if self.saliency_mode == "gradcam":
            # A class can have no positive CAM evidence. Do not turn its all-zero
            # map into a full-image mask merely because the top-k threshold is zero.
            has_signal = (saliency.flatten(1).amax(1) > 0).reshape(-1, 1, 1, 1)
            m_pixel = m_pixel * has_signal
            m_lat = m_lat * has_signal
        return m_pixel, m_lat

    # -------------------------------------------------------------- main loop
    def sample_loop(self, model, bundle, x_t, z_0, y_target, x0_pixel,
                    t_start=0.6, scale=1.0, progress=True, noise=None, x0_real=None):
        """Guided integration t_start -> 1 (classifier-only guidance, single support mask).

        Returns z_cf [B,C,h,w], x_cf [B,3,H,W] (pixel-composited if use_mask),
        records dict.
        """
        assert x_t.ndim == 4 and z_0.ndim == 4 and x0_pixel is not None
        y_target = y_target.view(-1).long().to(self.device)
        masking = self.use_mask
        bs = z_0.shape[0]
        latent_hw = z_0.shape[-2:]
        eps = torch.randn_like(z_0) if noise is None else noise   # FIXED background noise

        # INITIAL support from the ORIGINAL image (Eq. 11, target-class saliency on x_F). It builds
        # the first classifier view; every LATER support comes from that step's classifier view via
        # guided_gradient's sal_pix (the SAME classifier pass as the gradient).
        m_pixel = m_lat = None
        if masking:
            sal0 = self._saliency(bundle, x0_real if x0_real is not None else x0_pixel, y_target)
            m_pixel, m_lat = self._masks_from_saliency(sal0, latent_hw)

        ts = torch.linspace(t_start, 1.0, self.num_steps + 1, device=self.device)
        x = x_t
        records = {"prob": [], "max_grad": [], "losses": {}, "m_lat": None, "m_pixel": None}

        it = range(self.num_steps)
        if progress:
            it = tqdm(it, total=self.num_steps, desc="cf (flow) :: ", dynamic_ncols=False)

        for i in it:
            t_cur, t_next = ts[i], ts[i + 1]
            dt = (t_next - t_cur)
            t_b = self._t_batch(t_cur.item(), bs)

            g_cls, v, prob, losses, x_view, sal_pix = guided_gradient(
                model, bundle, x, t_b, self.path, y_target,
                vae=self.vae, scaling_factor=self.scaling_factor,
                x0_pixel=x0_pixel, pixel_mask=m_pixel if masking else None,
                grad_clip_sigma=self.grad_clip_sigma, x0_real=x0_real,
            )

            if self.adaptive_guidance:                            # FiRe-style per-sample scale:
                ramp = (i + 1) / self.num_steps                   # n/N (gentle early, full late)
                adapt = ((self.adapt_threshold - prob) / self.adapt_threshold).clamp(0.0, 1.0)
                eff_scale = (scale * ramp * adapt).view(-1, 1, 1, 1)   # [B,1,1,1]; -> 0 once p >= τ
            else:
                eff_scale = scale
            # Cooperative Flow Guidance on the classifier force (g_cls): reconcile the
            # direction vs v (Eq. 7-9) and cap the push magnitude vs ||v|| (Eq. 10).
            push = cooperative_guidance(g_cls, v, eff_scale, project=self.guide_project,
                                        vrel=self.guide_vrel, conflict_tau=self.guide_tau,
                                        vmin=self.guide_vmin)
            x = x + dt * (v - push)                               # guided Euler step

            if masking:                                           # Adaptive Support Preservation
                # FRESH support from THIS step's classifier view (sal_pix = |∂CE/∂x_view|, the SAME
                # classifier pass that produced g_cls) — so the gate M^z and the gradient driving the
                # edit are consistent (both from x_view_i). m_pixel carries forward as the "previous
                # support" that builds the next classifier view + the final composite.
                # Grad-CAM (paper) evaluates the target logit on the same current classifier view;
                # input_gradient reuses the step's |∂CE/∂x_view| map.
                support_saliency = (self._saliency(bundle, x_view, y_target)
                                    if self.saliency_mode == "gradcam" else sal_pix)
                m_pixel, m_lat = self._masks_from_saliency(support_saliency, latent_hw)
                bg = self._transport_background(z_0, t_next.item(), eps)   # original latent noised to t_next
                x = m_lat * x + (1.0 - m_lat) * bg                # outside support = original flow (Eq. 13)

            records["prob"].append(prob.detach())
            records["max_grad"].append(g_cls.flatten(1).abs().amax(1).detach())
            for k_, val in losses.items():
                records["losses"].setdefault(k_, []).append(val)

        z_cf = x.detach()
        records["m_lat"], records["m_pixel"] = m_lat, m_pixel

        # final image: foreground = decoded CF (m), background = the TRUE original
        # image x0_real (1-m), so the preserved region never carries the VAE
        # round-trip error. Falls back to D(z_0) only if no real image was supplied.
        x_cf = None
        if self.vae is not None:
            with torch.no_grad():
                dec = decode(self.vae, z_cf, self.scaling_factor)
                bg = x0_real if x0_real is not None else x0_pixel
                x_cf = (m_pixel * dec + (1.0 - m_pixel) * bg) if masking else dec
        return z_cf, x_cf, records

    # ------------------------------------------------------------ constructor
    @classmethod
    def from_config(cls, cfg, device="cuda", vae_decoder=None, labels=None):
        return cls(
            num_steps=getattr(cfg, "num_steps", 20),
            scheduler=getattr(cfg, "scheduler", "cond_ot"),
            device=device,
            classify_space=getattr(cfg, "classify_space", "pixel"),
            scaling_factor=getattr(cfg, "vae_scaling_factor", 0.18215),
            vae_decoder=vae_decoder,
            use_mask=getattr(cfg, "use_mask", True),
            # SINGLE SUPPORT ratio ρ (paper M^x). env > cfg. Per task: smile .05 / age .12 /
            # imagenet .30.
            mask_ratio=float(os.environ.get("CF_MASK_RATIO", getattr(cfg, "mask_ratio", 0.05))),
            dilation_pixel=int(os.environ.get("CF_DILATION_PIXEL", getattr(cfg, "dilation_pixel", 4))),
            use_lcc=cfg.use_lcc,
            mask_feather=int(os.environ.get("CF_MASK_FEATHER", getattr(cfg, "mask_feather", 7))),
            # de-speckle the single-shot saliency before top-k; 0 = off
            mask_presmooth=int(os.environ.get("CF_MASK_PRESMOOTH", getattr(cfg, "mask_presmooth", 5))),
            grad_clip_sigma=float(os.environ.get("CF_GRAD_CLIP_SIGMA", getattr(cfg, "grad_clip_sigma", 3.0))),
            adaptive_guidance=os.environ.get("CF_ADAPTIVE_GUIDANCE", "false").lower() == "true",
            adapt_threshold=float(os.environ.get("CF_ADAPT_THRESHOLD", 0.85)),
            # Cooperative Flow Guidance = soft conflict projection (Eq. 7-9) + ||v||-relative cap
            # (Eq. 10), acting on the classifier force g_cls. Ablations: CF_GUIDE_PROJECT=off
            # (Additive Guidance) / CF_GUIDE_PROJECT=conflict (Hard Projection).
            guide_project=os.environ.get("CF_GUIDE_PROJECT", "conflict_soft"),
            guide_vrel=float(os.environ.get("CF_GUIDE_VREL", 1.0)),
            guide_tau=float(os.environ.get("CF_GUIDE_TAU", 0.5)),
            guide_vmin=float(os.environ.get("CF_GUIDE_VMIN", 0.0)),
            labels=labels,
            saliency_mode=os.environ.get("CF_SALIENCY_MODE", getattr(cfg, "saliency_mode", "input_gradient")),
            gradcam_layer=os.environ.get("CF_GRADCAM_LAYER", getattr(cfg, "gradcam_layer", "")),
        )
