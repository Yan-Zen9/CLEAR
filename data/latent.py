"""
Latent-feature datasets for MedDiME (merged).

Consolidates the two previously-duplicated loaders:
  * LatentFeatureDataset  - features only, for velocity-model (flow) training.
  * LatentLabeledDataset  - features + labels, for counterfactual generation.
"""

import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from train.utils import DeviceDataLoader


# --------------------------------------------------------------- features only
class LatentFeatureDataset(Dataset):
    """Pre-extracted latent features (no labels) for diffusion / flow training."""

    def __init__(self, latent_dir):
        self.latent_dir = latent_dir
        self.latent_files = sorted(glob(os.path.join(latent_dir, "**/*.npy"), recursive=True))
        if len(self.latent_files) == 0:
            raise ValueError(f"No .npy files found under: {latent_dir}")
        print(f"✅ LatentFeatureDataset: {len(self.latent_files)} latents from {latent_dir}")

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent = np.load(self.latent_files[idx])
        if latent.ndim == 4 and latent.shape[0] == 1:
            latent = latent[0]
        return torch.from_numpy(latent).float(), 0  # placeholder label


def get_latent_loader_by_dir(latent_dir, batch_size=32, pin_memory=True,
                             shuffle=True, num_workers=4, device="cpu"):
    """DataLoader (device-wrapped) over a flat/nested directory of latent .npy files."""
    dataset = LatentFeatureDataset(latent_dir=latent_dir)
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory,
                        num_workers=num_workers, shuffle=shuffle, drop_last=True)
    return DeviceDataLoader(loader, device), dataset


# ----------------------------------------------------------- features + labels
class LatentLabeledDataset(Dataset):
    """Latent features + labels (flat or nested directory layout)."""

    def __init__(self, features_dir, labels_dir, transform=None):
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.transform = transform

        subdirs = [d for d in os.listdir(features_dir)
                   if os.path.isdir(os.path.join(features_dir, d))]
        if subdirs:
            self.latent_paths, self.label_paths = [], []
            for subdir in sorted(subdirs):
                sub_feat = os.path.join(features_dir, subdir)
                sub_lab = os.path.join(labels_dir, subdir)
                for feat_path in sorted(glob(os.path.join(sub_feat, "**", "*.npy"), recursive=True)):
                    rel = os.path.relpath(feat_path, sub_feat)
                    lab_path = os.path.join(sub_lab, rel)
                    if os.path.exists(lab_path):
                        self.latent_paths.append(feat_path)
                        self.label_paths.append(lab_path)
            self.indices = list(range(len(self.latent_paths)))
        else:
            files = [f for f in os.listdir(features_dir) if f.endswith(".npy")]
            self.indices = sorted(int(f.replace(".npy", "")) for f in files)
            self.latent_paths = None
            self.label_paths = None
        print(f"✅ LatentLabeledDataset: {len(self.indices)} samples from {features_dir}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.latent_paths:
            latent_path = self.latent_paths[idx]
            label_path = self.label_paths[idx]
            img_name = os.path.basename(latent_path)
        else:
            file_idx = self.indices[idx]
            latent_path = os.path.join(self.features_dir, f"{file_idx}.npy")
            label_path = os.path.join(self.labels_dir, f"{file_idx}.npy")
            img_name = f"{file_idx}.npy"

        latent = torch.from_numpy(np.load(latent_path)).float()
        label = torch.from_numpy(np.load(label_path)).long()

        if latent.dim() == 4:
            latent = latent.squeeze(0)
        elif latent.dim() != 3:
            raise ValueError(f"Unexpected latent shape: {latent.shape}")

        label = label.squeeze(0).item() if label.dim() > 0 else label.item()
        if self.transform is not None:
            latent = self.transform(latent)
        return latent, label, img_name


def get_latent_loader_by_name(dataset_name, batch_size=1, device="cuda",
                              shuffle=False, num_workers=0):
    """DataLoader over '<DS>_latent-<split>' under CF_DATA_ROOT / MEDDIME_DATA_ROOT."""
    parts = dataset_name.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid dataset_name '{dataset_name}', expect 'ISIC_Ruler_latent-test'")
    ds_base, split = parts

    data_root_base = (os.environ.get("CF_DATA_ROOT") or os.environ.get("MEDDIME_DATA_ROOT") or "").strip()
    if not data_root_base:
        raise ValueError("Set CF_DATA_ROOT (or MEDDIME_DATA_ROOT) to the latent data root.")
    data_root = os.path.join(data_root_base, ds_base)
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Latent data root not found: {data_root}")

    features_dir = os.path.join(data_root, split, "features")
    labels_dir = os.path.join(data_root, split, "labels")
    for d in (features_dir, labels_dir):
        if not os.path.exists(d):
            raise FileNotFoundError(f"Directory not found: {d}")

    dataset = LatentLabeledDataset(features_dir=features_dir, labels_dir=labels_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=(device == "cuda"), drop_last=False)


# ------------------------------------------- flat latents + label/split table
class FlatCsvLatentDataset(Dataset):
    """Flat latent dir plus a labels/split CSV or TSV.

    Supports CelebA-HQ ``hq_idx,split,<attr>`` and ISIC
    ``image,label,split`` schemas; returns ``(latent, label, stem.npy)``.
    """

    def __init__(self, latent_dir, csv_path, split, attr, shard_id=0, num_shards=1):
        import csv as _csv
        self.latent_dir = latent_dir
        self.items = []                                       # (file stem, label)
        with open(csv_path) as f:
            sample = f.read(4096)
            f.seek(0)
            dialect = _csv.Sniffer().sniff(sample, delimiters=",\t;")
            reader = _csv.DictReader(f, dialect=dialect)
            fields = reader.fieldnames or []
            is_isic = {"image", "label", "split"}.issubset(fields)
            if not is_isic and attr not in fields:
                raise ValueError(f"attr '{attr}' not in table columns {fields}")
            for r in reader:
                if r["split"] == split:
                    if is_isic:
                        stem = os.path.splitext(os.path.basename(r["image"]))[0]
                        label_name = r["label"].strip().lower()
                        label_map = {"nevus": 0, "melanoma": 1}
                        if label_name not in label_map:
                            raise ValueError("Unknown ISIC label: " + r["label"])
                        self.items.append((stem, label_map[label_name]))
                    else:
                        self.items.append((str(r["hq_idx"]), int(r[attr])))
        if not self.items:
            raise ValueError(f"No rows with split='{split}' in {csv_path}")
        if num_shards > 1:                                    # disjoint slice for multi-GPU array
            self.items = self.items[shard_id::num_shards]
        print(f"✅ FlatCsvLatentDataset: {len(self.items)} samples "
              f"(split={split}, attr={attr}, shard={shard_id}/{num_shards}, dir={latent_dir})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        stem, lab = self.items[idx]
        a = np.load(os.path.join(self.latent_dir, f"{stem}.npy"))
        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]
        return torch.from_numpy(a).float(), int(lab), f"{stem}.npy"


def get_flat_csv_loader(latent_dir, csv_path, split="val", attr="Smiling",
                        batch_size=1, device="cuda", shuffle=False, num_workers=0,
                        shard_id=0, num_shards=1):
    """DataLoader over a flat latent dir filtered to one split via a labels CSV.

    shard_id/num_shards partition the split disjointly for multi-GPU SLURM arrays.
    """
    ds = FlatCsvLatentDataset(latent_dir, csv_path, split, attr, shard_id, num_shards)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=(device == "cuda"), drop_last=False)


# ------------------------------------------- class-subdir latents (ImageNet CE)
class LatentImageFolderDataset(Dataset):
    """Latents laid out as ``<root>/<class_subdir>/*.npy`` (ImageFolder style).

    label = ``subdir_to_label[subdir]`` (for ImageNet CE this is the TRUE
    ImageNet-1k index, e.g. n02391049 -> 340, NOT the 0..N folder order).
    Returns ``(latent, label, stem.npy)``; the stem keeps its ``<wnid>_<n>`` name
    so paste-back can recover the class folder from the filename prefix.
    """

    def __init__(self, latent_dir, subdir_to_label, shard_id=0, num_shards=1):
        from itertools import zip_longest
        self.latent_dir = latent_dir
        per_class = []                                           # list per class of (relpath_no_ext, label)
        for sub in sorted(subdir_to_label):
            d = os.path.join(latent_dir, sub)
            if not os.path.isdir(d):
                continue
            per_class.append([(os.path.join(sub, f[:-4]), int(subdir_to_label[sub]))
                              for f in sorted(os.listdir(d)) if f.endswith(".npy")])
        # round-robin interleave across classes -> any prefix spans all classes
        # (good for smoke tests) and each shard stays class-balanced.
        self.items = [x for tup in zip_longest(*per_class) for x in tup if x is not None]
        if not self.items:
            raise ValueError(f"No <subdir>/*.npy latents under: {latent_dir}")
        if num_shards > 1:                                       # disjoint slice for multi-GPU array
            self.items = self.items[shard_id::num_shards]
        print(f"✅ LatentImageFolderDataset: {len(self.items)} samples "
              f"(classes={len(subdir_to_label)}, shard={shard_id}/{num_shards}, dir={latent_dir})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel, lab = self.items[idx]
        a = np.load(os.path.join(self.latent_dir, rel + ".npy"))
        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]
        stem = os.path.basename(rel)
        return torch.from_numpy(a).float(), int(lab), f"{stem}.npy"


def get_imagefolder_latent_loader(latent_dir, subdir_to_label, batch_size=1, device="cuda",
                                  shuffle=False, num_workers=0, shard_id=0, num_shards=1):
    """DataLoader over class-subdir latents; label = subdir_to_label[subdir]."""
    ds = LatentImageFolderDataset(latent_dir, subdir_to_label, shard_id, num_shards)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=(device == "cuda"), drop_last=False)
