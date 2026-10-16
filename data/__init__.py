"""Data loading for CLEAR (latent features)."""

from data.latent import (
    LatentFeatureDataset,
    LatentLabeledDataset,
    get_latent_loader_by_dir,
    get_latent_loader_by_name,
)

__all__ = [
    "LatentFeatureDataset",
    "LatentLabeledDataset",
    "get_latent_loader_by_dir",
    "get_latent_loader_by_name",
]
