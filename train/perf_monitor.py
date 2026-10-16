import csv
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch


def _try_import_flops_tools():
    """
    Try importing fvcore or thop to estimate FLOPs.
    Return the backend name and callable for subsequent use.
    """
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        return ("fvcore", FlopCountAnalysis)
    except Exception:
        pass

    try:
        from thop import profile  # type: ignore

        return ("thop", profile)
    except Exception:
        pass

    return (None, None)


def estimate_gflops(model: torch.nn.Module, input_shape: Tuple[int, ...], device: torch.device) -> Optional[float]:
    """
    Estimate the model's FLOPs from a single forward pass.
    Prefer fvcore, then thop; return None if neither dependency is available.
    """
    backend, handler = _try_import_flops_tools()
    if backend is None:
        print("⚠️ Neither fvcore nor thop is available; skipping GFLOPs estimation.")
        return None

    model = model.to(device)
    model.eval()
    dummy_x = torch.randn((1, *input_shape), device=device)
    dummy_t = torch.zeros((1,), device=device, dtype=torch.long)
    model_inputs = (dummy_x, dummy_t)

    with torch.no_grad():
        if backend == "fvcore":
            flop_analyzer = handler(model, model_inputs)
            total_flops = flop_analyzer.total()  # type: ignore[attr-defined]
        else:  # thop
            total_flops, _ = handler(model, model_inputs, verbose=False)

    gflops = total_flops / 1e9
    print(f"✅ Estimated model GFLOPs: {gflops:.3f}")
    return gflops


class PerfMonitor:
    """
    Training performance monitor:
    - Record elapsed time, GPU memory, and average loss for each interval.
    - Write metrics to CSV for subsequent analysis.
    """

    def __init__(
        self,
        log_path: Path,
        log_interval_steps: int = 10_000,
        gflops: Optional[float] = None,
        mode: str = "latent",
    ) -> None:
        self.log_path = log_path
        self.log_interval_steps = log_interval_steps
        self.gflops = gflops
        self.mode = mode

        self._interval_start_time = time.perf_counter()
        self._interval_step_counter = 0
        self._loss_accumulator = 0.0

        self._init_log_file()
        self._reset_cuda_peak()

    # --------------------------------------------------------------------- #
    # public api
    # --------------------------------------------------------------------- #
    def record_step(self, loss_value: float) -> None:
        """Accumulate the step count and loss after each training step."""
        self._interval_step_counter += 1
        self._loss_accumulator += loss_value

    def should_log(self, global_step: int) -> bool:
        return global_step % self.log_interval_steps == 0 and global_step > 0

    def flush(self, global_step: int) -> Dict[str, Any]:
        """
        When should_log is true, return interval metrics and append them to the log file.
        """
        elapsed = time.perf_counter() - self._interval_start_time
        steps = max(self._interval_step_counter, 1)
        avg_loss = self._loss_accumulator / steps
        steps_per_sec = steps / max(elapsed, 1e-6)

        max_mem = self._get_cuda_peak_memory()

        metrics = {
            "global_step": int(global_step),
            "mode": self.mode,
            "interval_steps": steps,
            "elapsed_sec": elapsed,
            "steps_per_sec": steps_per_sec,
            "avg_loss": avg_loss,
            "max_memory_gb": max_mem,
            "gflops": self.gflops,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        self._append_row(metrics)
        self._reset_interval_state()
        return metrics

    def finalize(self, global_step: int) -> Optional[Dict[str, Any]]:
        """
        At the end of training, flush any partial logging interval.
        """
        if self._interval_step_counter == 0:
            return None
        print("ℹ️ Flushing metrics for the final partial logging interval.")
        return self.flush(global_step)

    # --------------------------------------------------------------------- #
    # internal helpers
    # --------------------------------------------------------------------- #
    def _init_log_file(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with self.log_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "timestamp",
                    "mode",
                    "global_step",
                    "interval_steps",
                    "elapsed_sec",
                    "steps_per_sec",
                    "avg_loss",
                    "max_memory_gb",
                    "gflops",
                ])
                writer.writeheader()

    def _append_row(self, metrics: Dict[str, Any]) -> None:
        with self.log_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp",
                "mode",
                "global_step",
                "interval_steps",
                "elapsed_sec",
                "steps_per_sec",
                "avg_loss",
                "max_memory_gb",
                "gflops",
            ])
            writer.writerow(metrics)

    def _reset_interval_state(self) -> None:
        self._interval_start_time = time.perf_counter()
        self._interval_step_counter = 0
        self._loss_accumulator = 0.0
        self._reset_cuda_peak()

    def _reset_cuda_peak(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _get_cuda_peak_memory(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
        return 0.0


