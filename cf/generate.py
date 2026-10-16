"""
Flow-matching counterfactual generation (MedDiME) — CLI entry point.

Unified classifier interface (cf.classifier, vendored from FiRe):
the classifier outputs multi-class logits; the target is a class INDEX; guidance
uses cross_entropy; images are normalized via normalize_for_classifier.

Flow (pixel classifier guidance through the VAE decoder):
    z0 (.npy) --decode--> x0_pixel ; add noise -> integrate -> z_cf, x_cf
    validity = argmax classifier(x_cf) == y_target.

    CF_DATASET=isic_melanoma_nevus CF_PIXEL_CLASSIFIER_CKPT=<clf> \
    CF_DIFFUSION_CKPT=<flow.tar> python cf/generate.py
"""

import os
import sys
import csv
import time
import gc
import glob
import json
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cf.config import cfconfig, print_config, save_config
from cf.core import GuidedFlow
from cf.classifier import classifier_logits, classifier_target_prob
from cf.loading import (
    load_flow_model, load_pixel_classifier_bundle, load_vae, load_test_loader,
    save_latent_as_npy, save_image_tensor, decode_to_tensor, load_real_image_batch,
    create_dir_if_nonexist, get_save_path_with_version,
)


def imagenet_y_target(names, device):
    """CE target per sample = the PAIRED ImageNet class of the source (from the
    <wnid>_<n> filename prefix). This replaces the binary 0<->1 flip for ImageNet."""
    from cf.config import IMAGENET_CE_WNID_TO_IDX, IMAGENET_CE_PAIR
    ys = []
    for n in names:
        wnid = os.path.splitext(os.path.basename(n))[0].split("_")[0]
        ys.append(IMAGENET_CE_PAIR[IMAGENET_CE_WNID_TO_IDX[wnid]])
    return torch.tensor(ys, device=device).long()


def batch_cf(flow, model, bundle, vae, z_0, names, save_paths, device, y_target_override=None):
    """Generate and save a BATCH of counterfactuals at once.

    z_0: [b,C,H,W] latents; names/save_paths: length-b lists. The heavy compute
    (decode, guided sampling, classifier) runs batched on the whole chunk; only
    per-sample file I/O loops. Returns batch wall-clock + peak memory + batch_size.

    y_target_override: [b] class indices to flip TOWARD. When None (CelebA/medical),
    the target is the binary flip of the classifier's own prediction. ImageNet passes
    the paired-class index (multi-class, so 999-pred would be nonsense).
    """
    b = z_0.shape[0]
    sf = cfconfig.vae_scaling_factor
    x0_pixel = decode_to_tensor(z_0, vae, sf)                 # D(z_0): [b,3,H,W] in [-1,1]

    # background = TRUE original image (paste composite); set CF_IMG_DIR to enable.
    # Without it the background falls back to the decoded original D(z_0) (baseline).
    x0_real = None
    img_dir = os.environ.get("CF_IMG_DIR", "").strip()
    if img_dir:
        try:
            image_stems = [os.path.splitext(os.path.basename(n))[0] for n in names]
            x0_real = load_real_image_batch(image_stems, img_dir, x0_pixel.shape[-1], device)
        except Exception as e:
            print(f"⚠️ real-image load failed ({e}); background falls back to decoded original")

    with torch.no_grad():
        probs0 = torch.softmax(classifier_logits(bundle, x0_pixel), dim=1)
        pred = probs0.argmax(dim=1)
        n_cls = probs0.shape[1]
        if y_target_override is not None:                    # ImageNet: paired-class index
            y_target = y_target_override.to(device).view(-1).long()
        else:
            y_target = (n_cls - 1 - pred).long()             # binary flip (0<->1)
    prob0_t = probs0.gather(1, y_target.view(-1, 1)).squeeze(1)

    for j in range(b):
        save_latent_as_npy(z_0[j:j + 1], os.path.join(save_paths[j], "z_0.npy"))
        # x_0.png MUST be the TRUE original (paste-back source), NOT D(z_0): the VAE
        # round-trip adds ~3-4 FID of pure reconstruction noise across the whole image,
        # which biases every paired metric (FID/FVA/FS/MNAC/CD/COUT). x_cf's background is
        # already the true original, so real and cf then match everywhere but the edit.
        x0_save = x0_real[j:j + 1] if x0_real is not None else x0_pixel[j:j + 1]
        save_image_tensor(x0_save, os.path.join(save_paths[j], "x_0.png"))

    eps = torch.randn_like(z_0)                               # FIXED noise: start + background share it
    x_t = flow.add_noise(z_0, t_start=cfconfig.t_start, noise=eps)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    z_cf, x_cf, _ = flow.sample_loop(
        model, bundle, x_t=x_t, z_0=z_0, y_target=y_target, x0_pixel=x0_pixel,
        t_start=cfconfig.t_start, scale=cfconfig.start_scale, noise=eps, x0_real=x0_real,
    )
    gen_time = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

    with torch.no_grad():
        logits_cf = classifier_logits(bundle, x_cf)
        prob_cf_t = torch.softmax(logits_cf, dim=1).gather(1, y_target.view(-1, 1)).squeeze(1)
        flipped = (logits_cf.argmax(dim=1) == y_target)

    for j in range(b):
        with open(os.path.join(save_paths[j], "lab_target.txt"), "w") as f:   # FULL overwrite (retry-safe)
            f.write(f"img_name: {names[j]}\n")
            f.write(f"label: {pred[j].item()}\n")
            f.write(f"target: {y_target[j].item()}\n")
            f.write(f"prob_target_x0: {prob0_t[j].item():.4f}\n")
            f.write(f"prob_target_x_cf: {prob_cf_t[j].item():.4f}\n")
            f.write(f"flipped: {bool(flipped[j].item())}\n")
        save_latent_as_npy(z_cf[j:j + 1], os.path.join(save_paths[j], "z_cf.npy"))
        save_image_tensor(x_cf[j:j + 1], os.path.join(save_paths[j], "x_cf.png"))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"generation_time_sec": gen_time, "peak_memory_gb": peak_mem, "batch_size": b}


def main():
    print_config(cfconfig)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    seed = int(os.environ.get("CF_SEED", "0"))       # >0 => deterministic eps (fair A/B)
    if seed:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"seed={seed} (deterministic noise)")

    direct_output = os.environ.get("CF_OUTPUT_DIR", "").strip()
    if direct_output:
        save_res_path = direct_output
    else:
        out_root = (os.environ.get("CF_RECORDS_ROOT") or os.environ.get("MEDDIME_RECORDS_ROOT")
                    or "./cf_records").strip()
        base = os.path.join(out_root, "cf_generating", f"{cfconfig.ds_name}_flow_{cfconfig.model_var}")
        save_res_path = os.path.join(base, os.environ.get("CF_VERSION_NAME", "version_0"))
    create_dir_if_nonexist(save_res_path)
    print(f"results -> {save_res_path}")

    loader = load_test_loader(cfconfig.ds_name, device=device, batch_size=64)
    vae = load_vae(cfconfig.vae_model_name, device=device)
    cfconfig.vae_scaling_factor = float(vae.config.scaling_factor)   # authoritative scaling from the VAE
    model, _ = load_flow_model(cfconfig.diffusion_checkpoint_path, device=device)
    bundle = load_pixel_classifier_bundle(
        cfconfig.pixel_classifier_checkpoint_path, device=device,
        query_label=cfconfig.classifier_query_label, classifier_mode=cfconfig.classifier_mode,
    )
    flow = GuidedFlow.from_config(cfconfig, device=device, vae_decoder=vae)

    save_config(cfconfig, os.path.join(save_res_path, "config_record.txt"))
    with open(os.path.join(save_res_path, "environment_record.json"), "w") as f:
        json.dump({k: v for k, v in sorted(os.environ.items()) if k.startswith("CF_")}, f, indent=2)


    shard_id = int(os.environ.get("CF_SHARD_ID", 0))
    num_shards = int(os.environ.get("CF_NUM_SHARDS", 1))
    metrics_name = f"metrics_shard{shard_id}.csv" if num_shards > 1 else "metrics.csv"
    metrics_path = os.path.join(save_res_path, metrics_name)
    new_file = not os.path.exists(metrics_path)
    mf = open(metrics_path, "a", newline="")
    writer = csv.writer(mf)
    if new_file:
        writer.writerow(["sample_id", "sample_name", "generation_time_sec", "peak_memory_gb"])

    cf_batch_size = int(os.environ.get("CF_BATCH_SIZE", 8))   # images per forward (B x N memory!)
    max_samples = int(os.environ.get("CF_MAX_SAMPLES", 0))    # 0 = all; >0 caps for quick tests
    print(f"cf_batch_size={cf_batch_size}  max_samples={max_samples or 'all'}")
    total, skipped, done = 0, 0, 0
    my_dirs = []                                              # THIS shard's samples (stem, path) — for shard-local retry
    for loader_idx, (latent, lab, names) in enumerate(loader):
        # build the worklist for this loader batch (confounder filter + resume skip)
        work = []
        for i in range(len(lab)):
            if max_samples and total >= max_samples:
                break
            if cfconfig.confounder_filter != "all" and int(cfconfig.confounder_filter) != int(lab[i]):
                continue
            total += 1
            stem = names[i].replace(".npy", "")               # hq id -> shard-safe, resume-safe
            cfg_name = f"t{cfconfig.t_start}_s{cfconfig.num_steps}_{stem}"
            this_path = get_save_path_with_version(os.path.join(save_res_path, cfg_name))
            my_dirs.append((stem, this_path))                 # record every shard sample (retry is shard-local)
            lt = os.path.join(this_path, "lab_target.txt")
            if os.path.exists(lt) and "prob_target_x_cf" in open(lt).read():
                skipped += 1
                continue
            work.append((i, names[i], this_path))

        # process the worklist in sub-batches of cf_batch_size
        for s in range(0, len(work), cf_batch_size):
            chunk = work[s:s + cf_batch_size]
            idxs = [w[0] for w in chunk]
            chunk_names = [w[1] for w in chunk]
            chunk_paths = [w[2] for w in chunk]
            z_0 = latent[idxs].to(device)
            yt = imagenet_y_target(chunk_names, device) if cfconfig.ds_name == "ImageNet" else None
            try:
                perf = batch_cf(flow, model, bundle, vae, z_0, chunk_names, chunk_paths, device,
                                y_target_override=yt)
                per_sample = perf["generation_time_sec"] / max(1, perf["batch_size"])
                for nm in chunk_names:                          # amortized per-sample time
                    done += 1
                    writer.writerow([done, nm, f"{per_sample:.4f}", f"{perf['peak_memory_gb']:.4f}"])
                mf.flush()
            except Exception as e:
                import traceback
                print(f"❌ batch [{chunk_names[0]}...x{len(chunk_names)}] failed: {e}")
                traceback.print_exc()

        if max_samples and total >= max_samples:
            break

    # --- retry phase: re-run UNFLIPPED samples with FRESH noise, ESCALATING per round.
    # First rounds are same-strength (fresh eps, zero FID cost). Later rounds raise
    # scale/tau (FID-free, pushes harder inside the SAME mask, boosts COUT) and only
    # then grow topk/dilation (mild mask growth, capped). Per-round schedules are
    # env comma-lists (index = round, clamped to last); absent -> keep base value.
    # Retry is SHARD-LOCAL (iterates this shard's my_dirs, not a shared glob). ---
    max_retries = int(os.environ.get("CF_MAX_RETRIES", 0))
    latent_dir = os.environ.get("CF_LATENT_DIR", "").strip()

    def _sched(env, cast):
        v = os.environ.get(env, "").strip()
        return [cast(x) for x in v.split(",")] if v else None

    topk_sched = _sched("CF_RETRY_TOPK", float)
    dil_sched = _sched("CF_RETRY_DILATION", int)
    scale_sched = _sched("CF_RETRY_SCALE", float)
    tau_sched = _sched("CF_RETRY_TAU", float)
    tstart_sched = _sched("CF_RETRY_TSTART", float)

    def _pick(sched, rnd, fallback):
        return sched[min(rnd, len(sched) - 1)] if sched else fallback

    for rnd in range(max_retries):
        if not latent_dir:
            print("⚠️ CF_MAX_RETRIES set but CF_LATENT_DIR missing — skipping retry."); break
        unflipped = []
        for stem, d in my_dirs:                               # SHARD-LOCAL (no cross-shard glob)
            lt = os.path.join(d, "lab_target.txt")
            if not os.path.exists(lt):
                continue
            fl = [ln for ln in open(lt).read().splitlines() if ln.startswith("flipped:")]
            if fl and fl[-1].strip() == "flipped: False":     # LAST status only (robust)
                unflipped.append((stem, d))
        if not unflipped:
            print(f"retry {rnd+1}: all flipped — stop."); break
        # apply this round's escalation (live-mutate the sampler + config knobs)
        # retry grows the single support ratio (mask_ratio) so hard samples get a slightly
        # larger editable region; the final factual-pixel composite still preserves everything
        # outside it.
        flow.mask_ratio = _pick(topk_sched, rnd, flow.mask_ratio)
        flow.dil_pix = _pick(dil_sched, rnd, flow.dil_pix)
        flow.adapt_threshold = _pick(tau_sched, rnd, flow.adapt_threshold)
        cfconfig.start_scale = _pick(scale_sched, rnd, cfconfig.start_scale)
        cfconfig.t_start = _pick(tstart_sched, rnd, cfconfig.t_start)
        print(f"=== retry {rnd+1}/{max_retries}: {len(unflipped)} unflipped "
              f"(mask_ratio={flow.mask_ratio} dil={flow.dil_pix} "
              f"scale={cfconfig.start_scale} tau={flow.adapt_threshold} t={cfconfig.t_start}) ===",
              flush=True)
        for s in range(0, len(unflipped), cf_batch_size):
            chunk = unflipped[s:s + cf_batch_size]
            hq_ids = [c[0] for c in chunk]; paths = [c[1] for c in chunk]
            names = [f"{hq}.npy" for hq in hq_ids]
            try:
                def _load_z(hq):                                # flat, else nested <wnid>/<stem>.npy
                    p = os.path.join(latent_dir, f"{hq}.npy")
                    if not os.path.exists(p):
                        p = os.path.join(latent_dir, hq.split("_")[0], f"{hq}.npy")
                    return np.load(p)
                zs = [_load_z(hq) for hq in hq_ids]
                zs = [z[0] if (z.ndim == 4 and z.shape[0] == 1) else z for z in zs]
                z_0 = torch.from_numpy(np.stack(zs)).float().to(device)
                yt = imagenet_y_target(names, device) if cfconfig.ds_name == "ImageNet" else None
                batch_cf(flow, model, bundle, vae, z_0, names, paths, device,
                         y_target_override=yt)                  # fresh eps, overwrites
            except Exception as e:
                import traceback; print(f"❌ retry batch failed: {e}"); traceback.print_exc()

    mf.close()
    print(f"\n✅ done. total={total} skipped={skipped} -> {save_res_path}")


if __name__ == "__main__":
    main()
