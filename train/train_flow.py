"""
Train / fine-tune the latent velocity DiT for flow matching (CLEAR).

Unconditional CondOT flow matching in CLEAR convention (t=0 noise -> t=1 data):
    x_t = t*x1 + (1-t)*x0(noise) ; target v = x1 - x0 ; loss = MSE(v, model(x_t, t))
The model is an LFM-format DiT behind the DiTVelocity wrapper (flow/dit.py), so
checkpoints saved here are plain DiT state dicts — interchangeable with the
released LFM weights and loadable by cf.loading.load_flow_model.

Quality features: weight EMA (decay 0.999), LR warmup, logit-normal t sampling
(SD3-style), bf16 autocast. Sampling/eval should use EMA.

From scratch (single GPU):
    python train/train_flow.py --latent_dir <dir> --ds_name CelebA --model DiT-L/2
Fine-tune from an LFM release:
    python train/train_flow.py --latent_dir <dir> --ds_name CelebA \
        --init_from <.../flow/celeba-hq/model_475.pth> --lr 1e-5
Multi-GPU (SLURM, 1 task per GCD): launched via srun; ranks read RANK/LOCAL_RANK/
WORLD_SIZE (or SLURM_PROCID/NTASKS/LOCALID). --batch_size is PER-GPU.
"""

import argparse
import gc
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from train.config import BaseConfig, TrainingConfig, save_config_train_diffusion
from train.utils import setup_log_directory, DeviceDataLoader
from flow.dit import DiT, DIT_CONFIGS, DiTVelocity, normalize_state_dict, build_dit_from_state_dict
from train.perf_monitor import PerfMonitor, estimate_gflops
from data.latent import LatentFeatureDataset
from flow import CondOTProbPath

DEFAULT_LOG_INTERVAL = 50_000


def setup_ddp():
    """Init distributed from SLURM/torchrun env. Returns (rank, dev_index, world, distributed)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:          # torchrun / exported by sbatch
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif int(os.environ.get("SLURM_NTASKS", 1)) > 1:                 # plain srun
        rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    else:
        return 0, 0, 1, False
    if world <= 1:
        return 0, 0, 1, False
    dev_index = local_rank if torch.cuda.device_count() > 1 else 0   # 1-task-per-GCD -> only cuda:0 visible
    torch.cuda.set_device(dev_index)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    return rank, dev_index, world, True


class Ema:
    """Exponential moving average of model weights (params + float buffers)."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if torch.is_floating_point(s):
                s.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                s.copy_(v)            # int buffers (e.g. num_batches_tracked)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        for k in self.shadow:
            if k in sd:
                self.shadow[k].copy_(sd[k])


def sample_t(bs, mode, device):
    """Sample time t in (0,1). 'logit_normal' (SD3) concentrates near 0.5."""
    if mode == "logit_normal":
        t = torch.sigmoid(torch.randn(bs, device=device))
    else:
        t = torch.rand(bs, device=device)
    return t.clamp(1e-4, 1 - 1e-4)


def train_one_epoch(model, raw_dit, ema, path, loader, optimizer, scheduler,
                    loss_fn, epoch, global_step, perf_monitor, checkpoint_callback,
                    device, num_epochs, t_sampling, is_main):
    loss_sum, loss_n = 0.0, 0                        # plain accumulation (no cross-rank sync)
    model.train()
    iterator = tqdm(loader, dynamic_ncols=True) if is_main else loader
    if is_main:
        iterator.set_description(f"Train(flow) :: Epoch {epoch}/{num_epochs}")
    for batch_data in iterator:
        x0s = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
        x0s = x0s.to(device)

        bs = x0s.shape[0]
        t = sample_t(bs, t_sampling, device)
        eps = torch.randn_like(x0s)
        ps = path.sample(x_0=eps, x_1=x0s, t=t)    # x_t, target velocity dx_t

        with torch.autocast("cuda", dtype=torch.bfloat16):   # bf16: stable, no GradScaler
            pred_v = model(ps.x_t, t)
            loss = loss_fn(ps.dx_t, pred_v)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # guard against loss spikes
        optimizer.step()
        scheduler.step()
        global_step += 1

        loss_value = loss.detach().item()
        loss_sum += loss_value; loss_n += 1
        if is_main:
            ema.update(raw_dit)                     # EMA tracks the (DDP-synced) raw DiT
            perf_monitor.record_step(loss_value)
            if perf_monitor.should_log(global_step):
                checkpoint_callback(global_step, perf_monitor.flush(global_step))
            iterator.set_postfix_str(f"Loss: {loss_value:.6f} | lr: {scheduler.get_last_lr()[0]:.2e} | Step: {global_step}")
    return global_step, loss_sum / max(1, loss_n)


def main():
    parser = argparse.ArgumentParser(description="Train CLEAR latent velocity DiT (flow matching)")
    parser.add_argument("--latent_dir", type=str, required=True, help="latent features dir")
    parser.add_argument("--model", type=str, default="DiT-L/2", choices=list(DIT_CONFIGS),
                        help="architecture when training from scratch (ignored with --init_from)")
    parser.add_argument("--init_from", type=str, default=None,
                        help="LFM-format checkpoint to warm-start from (arch inferred from it)")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None, help="PER-GPU batch size")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=None)
    parser.add_argument("--ds_name", type=str, default="CelebA")
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--t_sampling", choices=["logit_normal", "uniform"], default="logit_normal")
    parser.add_argument("--grad_checkpoint", action="store_true",
                        help="activation checkpointing in the DiT blocks (fits bigger batches)")
    args = parser.parse_args()

    rank, dev_index, world, distributed = setup_ddp()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{dev_index}" if torch.cuda.is_available() else "cpu")

    base_config = BaseConfig()
    base_config.ds_name = args.ds_name
    base_config.DEVICE = device
    training_config = TrainingConfig()
    if args.batch_size is not None:
        training_config.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        training_config.NUM_EPOCHS = args.epochs
    if args.lr is not None:
        training_config.LR = args.lr
    log_interval = args.checkpoint_interval or DEFAULT_LOG_INTERVAL

    base_config.DATASET = f"{base_config.ds_name}_latent"
    base_config.root_log_dir = os.path.join(base_config.record_dir, base_config.DATASET, "Logs_Checkpoints", "Inference")
    base_config.root_checkpoint_dir = os.path.join(base_config.record_dir, base_config.DATASET, "Logs_Checkpoints", "checkpoints")
    base_config.root_prevs_dir = os.path.join(base_config.record_dir, base_config.DATASET, "PreVs")

    # infer latent shape (C, h, w) from the data
    _lat_files = sorted(glob.glob(os.path.join(args.latent_dir, "**", "*.npy"), recursive=True))
    if not _lat_files:
        raise FileNotFoundError(f"No .npy latents under {args.latent_dir}")
    img_shape = tuple(np.load(_lat_files[0]).shape[-3:])

    # build the DiT: from an LFM checkpoint (fine-tune) or from scratch (unconditional)
    if args.init_from:
        sd = normalize_state_dict(torch.load(args.init_from, map_location="cpu"))
        raw_dit, arch = build_dit_from_state_dict(sd)
        assert (arch["in_channels"], arch["img_resolution"]) == (img_shape[0], img_shape[-1]), (
            f"checkpoint latent {arch['in_channels']}x{arch['img_resolution']} != data {img_shape} "
            f"(the LFM 256px weights only fit 32x32x4 latents)")
        model_info = {"init_from": args.init_from, **arch}
    else:
        cfg = DIT_CONFIGS[args.model]
        raw_dit = DiT(img_resolution=img_shape[-1], in_channels=img_shape[0],
                      num_classes=1, label_dropout=0.0, **cfg)
        model_info = {"model": args.model, "img_resolution": img_shape[-1],
                      "in_channels": img_shape[0], **cfg}
    if args.grad_checkpoint:
        raw_dit.enable_gradient_checkpointing = True
    raw_dit = raw_dit.to(device)
    velocity = DiTVelocity(raw_dit)                  # training runs in CLEAR convention
    model = DDP(velocity, device_ids=[dev_index]) if distributed else velocity
    ema = Ema(raw_dit, decay=args.ema_decay) if is_main else None    # EMA in plain-DiT key space

    # Only rank 0 creates the versioned run dir / writes config (avoids a race).
    log_dir = checkpoint_dir = None
    if is_main:
        log_dir, checkpoint_dir, _, _ = setup_log_directory(config=base_config)
        save_config_train_diffusion(training_config, base_config, save_dir=checkpoint_dir,
                                    model_info=model_info)
        print(f"latent shape (from data): {img_shape}  | {len(_lat_files)} latents | world={world} "
              f"| per-gpu bs={training_config.BATCH_SIZE} | {model_info} | t={args.t_sampling}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.LR)
    warmup = max(1, args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup))

    dataset = LatentFeatureDataset(latent_dir=args.latent_dir)
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        loader = DataLoader(dataset, batch_size=training_config.BATCH_SIZE, sampler=sampler,
                            num_workers=2, pin_memory=True, drop_last=True)
    else:
        sampler = None
        loader = DataLoader(dataset, batch_size=training_config.BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=True)
    dataloader = DeviceDataLoader(loader, device)

    loss_fn = nn.MSELoss()
    path = CondOTProbPath()

    perf_monitor = None
    if is_main:
        gflops = estimate_gflops(velocity, img_shape, device)
        perf_monitor = PerfMonitor(log_path=Path(log_dir) / "train_metrics.csv",
                                   log_interval_steps=log_interval, gflops=gflops, mode="latent")

    global_step, start_epoch = 0, 1
    if args.resume_from and Path(args.resume_from).exists():
        ckpt = torch.load(args.resume_from, map_location=device)
        raw_dit.load_state_dict(ckpt["model"])
        if is_main and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["opt"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt.get("global_step", 0)
        start_epoch = (global_step // max(1, len(dataloader))) + 1
        if is_main:
            print(f"🔄 resumed at step {global_step}")

    checkpoint_dir_path = Path(checkpoint_dir) if is_main else None

    def checkpoint_callback(step, metrics):
        # (1) full state for RESUME: single rolling file, overwritten atomically
        full = {"model": raw_dit.state_dict(), "ema": ema.state_dict(),
                "opt": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "global_step": step,
                "metrics": metrics, "mode": "flow"}
        tmp = checkpoint_dir_path / "ckpt.tar.tmp"
        torch.save(full, tmp)
        os.replace(tmp, checkpoint_dir_path / "ckpt.tar")
        # (2) lightweight EMA-only snapshot for FID model selection
        torch.save({"ema": ema.state_dict(), "global_step": step, "metrics": metrics},
                   checkpoint_dir_path / f"ema_step{step}.pt")
        print(f"💾 ckpt.tar(resume) + ema_step{step}.pt | avg_loss={metrics['avg_loss']:.6f}")

    for epoch in range(start_epoch, training_config.NUM_EPOCHS + 1):
        torch.cuda.empty_cache()
        gc.collect()
        if sampler is not None:
            sampler.set_epoch(epoch)               # reshuffle differently each epoch
        global_step, _ = train_one_epoch(
            model, raw_dit, ema, path, dataloader, optimizer, scheduler, loss_fn,
            epoch, global_step, perf_monitor, checkpoint_callback,
            device, training_config.NUM_EPOCHS, args.t_sampling, is_main,
        )

    if distributed:
        dist.barrier()
    if is_main:
        final = perf_monitor.finalize(global_step) or {"avg_loss": float("nan")}
        checkpoint_callback(global_step, final)    # always save final (raw + EMA)
        print("^" * 30 + "Training Complete")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
