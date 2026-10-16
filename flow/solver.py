"""
Minimal fixed-step ODE solvers for flow matching (vendored, self-contained).

Used for *unguided* sampling (e.g. generating samples for FID). The guided +
masked counterfactual loop lives in `cf.core` because masking requires
per-step state replacement that a plain ODE integrator cannot express.

No `torchdiffeq` dependency.
"""

import torch


@torch.no_grad()
def euler_sample(velocity_fn, x_init, num_steps=50, t_start=0.0, t_end=1.0,
                 return_intermediates=False):
    """Integrate dx/dt = v(x, t) from t_start to t_end with forward Euler.

    Args:
        velocity_fn: callable (x, t_scalar_tensor[batch]) -> velocity.
        x_init: initial state (batch, ...) at t_start (noise when t_start=0).
        num_steps: number of Euler steps.
        t_start, t_end: integration interval (default noise->data, 0->1).
        return_intermediates: if True, also return the list of states.
    """
    x = x_init
    bs = x.shape[0]
    ts = torch.linspace(t_start, t_end, num_steps + 1, device=x.device, dtype=x.dtype)
    traj = [x]
    for i in range(num_steps):
        t = ts[i]
        dt = ts[i + 1] - ts[i]
        t_batch = torch.full((bs,), float(t), device=x.device, dtype=x.dtype)
        v = velocity_fn(x, t_batch)
        x = x + dt * v
        if return_intermediates:
            traj.append(x)
    return (x, traj) if return_intermediates else x


@torch.no_grad()
def midpoint_sample(velocity_fn, x_init, num_steps=50, t_start=0.0, t_end=1.0,
                    return_intermediates=False):
    """Integrate with the midpoint (RK2) rule. Same signature as euler_sample."""
    x = x_init
    bs = x.shape[0]
    ts = torch.linspace(t_start, t_end, num_steps + 1, device=x.device, dtype=x.dtype)
    traj = [x]
    for i in range(num_steps):
        t = ts[i]
        dt = ts[i + 1] - ts[i]
        t_batch = torch.full((bs,), float(t), device=x.device, dtype=x.dtype)
        k1 = velocity_fn(x, t_batch)
        t_half = torch.full((bs,), float(t + 0.5 * dt), device=x.device, dtype=x.dtype)
        k2 = velocity_fn(x + 0.5 * dt * k1, t_half)
        x = x + dt * k2
        if return_intermediates:
            traj.append(x)
    return (x, traj) if return_intermediates else x


SOLVERS = {"euler": euler_sample, "midpoint": midpoint_sample}
