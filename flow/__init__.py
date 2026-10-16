"""
Vendored, self-contained flow-matching core for CLEAR.

Flow math is a minimal distillation of Meta's `flow_matching` library
(arXiv:2412.06264); the velocity model is an LFM-format DiT (flow/dit.py) so the
released VinAI LFM checkpoints load directly. NO external flow/timm dependency.

Public API:
    CondOTProbPath / AffineProbPath  - affine path X_t = alpha_t X_1 + sigma_t X_0
    CondOTScheduler / build_scheduler - path coefficient schedules
    euler_sample / midpoint_sample   - fixed-step ODE solvers (unguided sampling)
    DiT / DiTVelocity / build_dit_from_state_dict / normalize_state_dict - velocity model
"""

from flow.scheduler import (
    Scheduler,
    SchedulerOutput,
    CondOTScheduler,
    PolynomialConvexScheduler,
    CosineScheduler,
    build_scheduler,
)
from flow.path import AffineProbPath, CondOTProbPath, PathSample, expand_tensor_like
from flow.solver import euler_sample, midpoint_sample, SOLVERS
from flow.dit import (
    DiT,
    DiTVelocity,
    DIT_CONFIGS,
    build_dit_from_state_dict,
    normalize_state_dict,
    infer_dit_arch,
)

__all__ = [
    "DiT",
    "DiTVelocity",
    "DIT_CONFIGS",
    "build_dit_from_state_dict",
    "normalize_state_dict",
    "infer_dit_arch",
    "Scheduler",
    "SchedulerOutput",
    "CondOTScheduler",
    "PolynomialConvexScheduler",
    "CosineScheduler",
    "build_scheduler",
    "AffineProbPath",
    "CondOTProbPath",
    "PathSample",
    "expand_tensor_like",
    "euler_sample",
    "midpoint_sample",
    "SOLVERS",
]
