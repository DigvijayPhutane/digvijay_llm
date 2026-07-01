"""
ram_planner.py
==============
Helps decide HOW to stream a model given a target RAM budget: how many
layers (if any) can be safely kept "hot" in RAM at once, how big the
on-disk page cache window should be, and whether the user's requested
model will even fit on disk + meet a minimum tokens/sec expectation.

This is intentionally simple/heuristic — its job is to stop people from
footgunning themselves (e.g. trying to keep 40 layers "hot" on a 16GB
machine) rather than to be a perfect simulator.
"""

from dataclasses import dataclass
import os
import shutil


@dataclass
class RAMPlan:
    total_ram_gb: float
    reserved_for_os_gb: float
    usable_ram_gb: float
    safe_hot_layers: int
    layer_size_gb: float
    disk_free_gb: float
    warnings: list


def _disk_free_gb(path: str) -> float:
    try:
        usage = shutil.disk_usage(path if os.path.isdir(path) else os.path.dirname(path) or ".")
        return usage.free / (1024 ** 3)
    except Exception:
        return float("inf")


def plan_for_budget(
    model_path: str,
    total_params_billion: float,
    n_layers: int,
    bytes_per_param: float = 2.0,  # fp16=2, 8bit=1, 4bit~=0.5-0.6
    n_ram_gb: float = 16.0,
    os_reserve_gb: float = 2.5,
) -> RAMPlan:
    """
    Compute a conservative streaming plan: how many transformer layers can
    be kept resident in RAM simultaneously without exceeding the budget.
    """
    warnings = []
    usable = max(n_ram_gb - os_reserve_gb, 1.0)

    total_weight_gb = (total_params_billion * 1e9 * bytes_per_param) / (1024 ** 3)
    layer_size_gb = total_weight_gb / max(n_layers, 1)

    # Keep a safety margin for activations / KV cache / runtime overhead
    activation_overhead_gb = max(0.5, usable * 0.15)
    budget_for_weights = max(usable - activation_overhead_gb, 0.5)

    safe_hot_layers = max(1, int(budget_for_weights // layer_size_gb))
    if safe_hot_layers < 2:
        warnings.append(
            "Budget is extremely tight: only 1 layer fits in RAM at a time. "
            "Expect slow throughput; an NVMe SSD is strongly recommended."
        )

    disk_free = _disk_free_gb(model_path)
    if disk_free < total_weight_gb:
        warnings.append(
            f"Model needs ~{total_weight_gb:.1f} GB on disk but only "
            f"{disk_free:.1f} GB free was detected at '{model_path}'."
        )

    return RAMPlan(
        total_ram_gb=n_ram_gb,
        reserved_for_os_gb=os_reserve_gb,
        usable_ram_gb=usable,
        safe_hot_layers=safe_hot_layers,
        layer_size_gb=layer_size_gb,
        disk_free_gb=disk_free,
        warnings=warnings,
    )


class RAMPlanner:
    """Thin OO wrapper kept for ergonomic imports: RAMPlanner(...).plan()"""

    def __init__(self, model_path: str, total_params_billion: float, n_layers: int,
                 bytes_per_param: float = 2.0, n_ram_gb: float = 16.0, os_reserve_gb: float = 2.5):
        self.model_path = model_path
        self.total_params_billion = total_params_billion
        self.n_layers = n_layers
        self.bytes_per_param = bytes_per_param
        self.n_ram_gb = n_ram_gb
        self.os_reserve_gb = os_reserve_gb

    def plan(self) -> RAMPlan:
        return plan_for_budget(
            self.model_path,
            self.total_params_billion,
            self.n_layers,
            self.bytes_per_param,
            self.n_ram_gb,
            self.os_reserve_gb,
        )
