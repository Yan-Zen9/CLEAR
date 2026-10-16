"""
Affine probability path for flow matching (vendored, self-contained).

Distilled from Meta's `flow_matching` library so CLEAR does not import the
external package. Implements the affine path

    X_t = alpha_t * X_1 + sigma_t * X_0,     X_1 = data,  X_0 = noise,

together with the exact conversions between the three equivalent
parametrizations a velocity model can be read in:

    * velocity  v_t = d_alpha_t * X_1 + d_sigma_t * X_0
    * target    X_1  (the clean data/latent estimate)
    * epsilon   X_0  (the noise)

For CondOT (alpha_t = t, sigma_t = 1 - t):
    v_t        = X_1 - X_0
    X_1        = velocity_to_target(v, x_t, t) = x_t + (1 - t) * v
    X_0        = velocity_to_epsilon(v, x_t, t) = x_t - t * v

`velocity_to_target` is the flow-matching analogue of DDPM's "predicted x0"
(the clean estimate used for classifier guidance).
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from flow.scheduler import Scheduler, CondOTScheduler


def expand_tensor_like(input_tensor: Tensor, expand_to: Tensor) -> Tensor:
    """Reshape a per-sample (batch,) tensor to broadcast against `expand_to`."""
    assert input_tensor.ndim == 1, "Expected a 1D (batch,) tensor."
    assert input_tensor.shape[0] == expand_to.shape[0], "Batch size mismatch."
    dim_diff = expand_to.ndim - input_tensor.ndim
    return input_tensor.view(input_tensor.shape[0], *([1] * dim_diff)).expand_as(expand_to)


@dataclass
class PathSample:
    """A sample from the affine path.

    Attributes:
        x_t:  the noised sample at time t.
        dx_t: the conditional velocity at x_t (the FM regression target).
        x_1:  the data endpoint.
        x_0:  the noise endpoint.
        t:    the time, shape (batch,).
    """

    x_t: Tensor
    dx_t: Tensor
    x_1: Tensor
    x_0: Tensor
    t: Tensor


class AffineProbPath:
    """Affine probability path with a pluggable scheduler.

    Args:
        scheduler: provides alpha_t, sigma_t and their derivatives.
    """

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    # ------------------------------------------------------------------ sample
    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> PathSample:
        """Return X_t = alpha_t X_1 + sigma_t X_0 and velocity dX_t.

        Args:
            x_0: noise endpoint, shape (batch, ...).
            x_1: data endpoint, shape (batch, ...).
            t:   times in [0, 1], shape (batch,).
        """
        assert t.ndim == 1 and t.shape[0] == x_1.shape[0], "t must be (batch,)."
        s = self.scheduler(t)
        alpha_t = expand_tensor_like(s.alpha_t, x_1)
        sigma_t = expand_tensor_like(s.sigma_t, x_1)
        d_alpha_t = expand_tensor_like(s.d_alpha_t, x_1)
        d_sigma_t = expand_tensor_like(s.d_sigma_t, x_1)

        x_t = alpha_t * x_1 + sigma_t * x_0
        dx_t = d_alpha_t * x_1 + d_sigma_t * x_0
        return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)

    # ------------------------------------------------------- representation maps
    def _coeffs(self, t: Tensor):
        s = self.scheduler(t)
        return s.alpha_t, s.d_alpha_t, s.sigma_t, s.d_sigma_t

    def velocity_to_target(self, velocity: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        """velocity -> X_1 (clean data/latent estimate). Analogue of DDPM x0."""
        alpha_t, d_alpha_t, sigma_t, d_sigma_t = self._coeffs(t)
        denom = d_alpha_t * sigma_t - d_sigma_t * alpha_t
        a_t = expand_tensor_like(-d_sigma_t / denom, x_t)
        b_t = expand_tensor_like(sigma_t / denom, x_t)
        return a_t * x_t + b_t * velocity

    def velocity_to_epsilon(self, velocity: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        """velocity -> X_0 (noise)."""
        alpha_t, d_alpha_t, sigma_t, d_sigma_t = self._coeffs(t)
        denom = d_sigma_t * alpha_t - d_alpha_t * sigma_t
        a_t = expand_tensor_like(-d_alpha_t / denom, x_t)
        b_t = expand_tensor_like(alpha_t / denom, x_t)
        return a_t * x_t + b_t * velocity

    def target_to_velocity(self, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        """X_1 -> velocity."""
        alpha_t, d_alpha_t, sigma_t, d_sigma_t = self._coeffs(t)
        a_t = expand_tensor_like(d_sigma_t / sigma_t, x_t)
        b_t = expand_tensor_like(
            (d_alpha_t * sigma_t - d_sigma_t * alpha_t) / sigma_t, x_t
        )
        return a_t * x_t + b_t * x_1

    def epsilon_to_target(self, epsilon: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        """X_0 -> X_1."""
        alpha_t, _, sigma_t, _ = self._coeffs(t)
        a_t = expand_tensor_like(1.0 / alpha_t, x_t)
        b_t = expand_tensor_like(-sigma_t / alpha_t, x_t)
        return a_t * x_t + b_t * epsilon


class CondOTProbPath(AffineProbPath):
    """Affine path with the CondOT (linear) schedule: alpha=t, sigma=1-t."""

    def __init__(self):
        super().__init__(CondOTScheduler())
