"""
Flow-matching schedulers (vendored, self-contained).

Distilled from the Meta `flow_matching` library (Lipman et al., "Flow Matching
Guide and Codebase", arXiv:2412.06264) so that MedDiME has NO dependency on the
external package. Only the pieces MedDiME needs are kept.

A scheduler defines the affine probability path coefficients
    X_t = alpha_t * X_1 + sigma_t * X_0
where X_1 is the data (image/latent at t=1) and X_0 is noise (at t=0), together
with their time-derivatives. See `flow.path.AffineProbPath`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SchedulerOutput:
    """Coefficients of the affine path at time t.

    Attributes:
        alpha_t: data coefficient.
        sigma_t: noise coefficient.
        d_alpha_t: d/dt alpha_t.
        d_sigma_t: d/dt sigma_t.
    """

    alpha_t: Tensor
    sigma_t: Tensor
    d_alpha_t: Tensor
    d_sigma_t: Tensor


class Scheduler(ABC):
    """Base scheduler."""

    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        """Args: t in [0, 1], shape (...)."""
        ...


class CondOTScheduler(Scheduler):
    """Conditional optimal-transport (linear / rectified) schedule.

    alpha_t = t, sigma_t = 1 - t  ->  X_t = t * X_1 + (1 - t) * X_0.
    This is the straight-line path; the default for MedDiME.
    """

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1 - t,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-torch.ones_like(t),
        )


class PolynomialConvexScheduler(Scheduler):
    """Polynomial convex schedule: alpha_t = t**n, sigma_t = 1 - t**n."""

    def __init__(self, n: float) -> None:
        assert n > 0, f"`n` must be positive, got {n}."
        self.n = float(n)

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t ** self.n,
            sigma_t=1 - t ** self.n,
            d_alpha_t=self.n * (t ** (self.n - 1)),
            d_sigma_t=-self.n * (t ** (self.n - 1)),
        )


class CosineScheduler(Scheduler):
    """Cosine schedule: alpha_t = sin(pi/2 t), sigma_t = cos(pi/2 t)."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        pi = torch.pi
        return SchedulerOutput(
            alpha_t=torch.sin(pi / 2 * t),
            sigma_t=torch.cos(pi / 2 * t),
            d_alpha_t=pi / 2 * torch.cos(pi / 2 * t),
            d_sigma_t=-pi / 2 * torch.sin(pi / 2 * t),
        )


SCHEDULERS = {
    "cond_ot": CondOTScheduler,
    "polynomial": PolynomialConvexScheduler,
    "cosine": CosineScheduler,
}


def build_scheduler(name: str = "cond_ot", **kwargs) -> Scheduler:
    name = name.lower()
    if name not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler '{name}'. Available: {list(SCHEDULERS)}")
    return SCHEDULERS[name](**kwargs)
