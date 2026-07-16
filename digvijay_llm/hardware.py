"""System inspection and automatic configuration helpers."""

from __future__ import annotations

import logging
import os
import platform
import shutil
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency path
    psutil = None

logger = logging.getLogger("digvijay_llm")


@dataclass
class SystemInfo:
    os_family: str
    os_version: str
    cpu_cores: int
    cpu_model: str
    ram_total_gb: float
    ram_available_gb: float
    storage_total_gb: float
    storage_free_gb: float
    gpu_name: Optional[str] = None
    gpu_vendor: Optional[str] = None
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    gpu_backend: Optional[str] = None
    torch_available: bool = False
    torch_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_ram_info() -> tuple[float, float]:
    if psutil is not None:
        vm = psutil.virtual_memory()
        return vm.total / (1024**3), vm.available / (1024**3)

    try:
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            mem_status = ctypes.create_string_buffer(64)
            if kernel32.GlobalMemoryStatusEx(mem_status):
                total = int(ctypes.c_ulonglong.from_buffer(mem_status, 8).value)  # type: ignore[index]
                avail = int(ctypes.c_ulonglong.from_buffer(mem_status, 16).value)  # type: ignore[index]
                return total / (1024**3), avail / (1024**3)
        else:
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            return total / (1024**3), avail / (1024**3)
    except (AttributeError, ValueError, OSError):
        pass

    return 16.0, 8.0


def detect_system() -> SystemInfo:
    """Collect a compact view of the host machine for runtime tuning."""
    ram_total_gb, ram_available_gb = _get_ram_info()
    disk = shutil.disk_usage(".")

    cpu_model = platform.processor() or platform.machine() or "unknown"
    os_family = "windows" if os.name == "nt" else "macos" if platform.system() == "Darwin" else "linux" if platform.system() else "unknown"
    os_version = platform.platform()

    gpu_name = None
    gpu_vendor = None
    vram_total_gb = 0.0
    vram_free_gb = 0.0
    gpu_backend = None

    try:
        import torch

        torch_available = True
        torch_version = getattr(torch, "__version__", None)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_total_gb = props.total_memory / (1024**3)
            vram_free_gb = (props.total_memory - torch.cuda.memory_reserved(0)) / (1024**3)
            gpu_vendor = "nvidia"
            gpu_backend = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_name = "Apple Silicon GPU"
            gpu_vendor = "apple"
            gpu_backend = "metal"
            vram_total_gb = max(4.0, vm.total / (1024**3) * 0.4)
            vram_free_gb = max(2.0, vm.available / (1024**3) * 0.4)
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            gpu_name = "Intel GPU"
            gpu_vendor = "intel"
            gpu_backend = "xpu"
            vram_total_gb = max(4.0, vm.total / (1024**3) * 0.3)
            vram_free_gb = max(2.0, vm.available / (1024**3) * 0.3)
        else:
            torch_available = True
            torch_version = getattr(torch, "__version__", None)
            if getattr(torch.version, "hip", None) is not None:
                gpu_name = "AMD ROCm GPU"
                gpu_vendor = "amd"
                gpu_backend = "rocm"
                vram_total_gb = max(4.0, vm.total / (1024**3) * 0.35)
                vram_free_gb = max(2.0, vm.available / (1024**3) * 0.35)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.debug("Torch GPU detection failed: %s", exc)
        torch_available = False
        torch_version = None

    return SystemInfo(
        os_family=os_family,
        os_version=os_version,
        cpu_cores=os.cpu_count() or 1,
        cpu_model=cpu_model,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        storage_total_gb=disk.total / (1024**3),
        storage_free_gb=disk.free / (1024**3),
        gpu_name=gpu_name,
        gpu_vendor=gpu_vendor,
        vram_total_gb=vram_total_gb,
        vram_free_gb=vram_free_gb,
        gpu_backend=gpu_backend,
        torch_available=torch_available,
        torch_version=torch_version,
    )


def recommend_config(
    system_info: Optional[SystemInfo] = None,
    model_size_gb: Optional[float] = None,
    backend: Optional[str] = None,
    prefer_gpu: bool = True,
) -> Dict[str, Any]:
    """Recommend an inference configuration from the detected hardware."""
    info = system_info or detect_system()
    model_size_gb = model_size_gb or 20.0
    if backend is None:
        backend = "gguf"

    if info.gpu_backend and prefer_gpu:
        device = info.gpu_backend
        if info.gpu_backend == "cuda":
            device = "cuda"
        elif info.gpu_backend == "metal":
            device = "mps"
        elif info.gpu_backend == "rocm":
            device = "rocm"
        elif info.gpu_backend == "xpu":
            device = "xpu"
        else:
            device = "cpu"
    else:
        device = "cpu"

    if device != "cpu" and info.vram_total_gb > 0:
        gpu_layers = int(min(80, max(0, (info.vram_total_gb / max(model_size_gb / 20.0, 1.0)) * 1.2)))
    else:
        gpu_layers = 0

    if info.ram_total_gb <= 16:
        n_ctx = 2048
        n_batch = 4
        n_ram_gb = max(8.0, min(info.ram_total_gb, 16.0))
    elif info.ram_total_gb <= 32:
        n_ctx = 4096
        n_batch = 8
        n_ram_gb = min(info.ram_total_gb, 32.0)
    else:
        n_ctx = 8192
        n_batch = 16
        n_ram_gb = min(info.ram_total_gb, 64.0)

    n_threads = max(1, min(info.cpu_cores, 16))
    if device == "cpu":
        n_threads = max(1, min(info.cpu_cores, max(2, info.cpu_cores // 2)))

    return {
        "backend": backend,
        "device": device,
        "n_gpu_layers": gpu_layers,
        "n_threads": n_threads,
        "n_ctx": n_ctx,
        "n_batch": n_batch,
        "n_ram_gb": round(n_ram_gb, 2),
        "system_info": info.to_dict(),
    }


def detect_params(
    model_path: Optional[str] = None,
    model_size_gb: Optional[float] = None,
    detect: bool = True,
    backend: Optional[str] = None,
    n_gpu_layers: Optional[int] = None,
    n_threads: Optional[int] = None,
    n_ctx: Optional[int] = None,
    n_batch: Optional[int] = None,
    n_ram_gb: Optional[float] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a configuration dict used by the public API."""
    if not detect:
        return {
            "backend": backend or "gguf",
            "device": device or "cpu",
            "n_gpu_layers": n_gpu_layers or 0,
            "n_threads": n_threads or 1,
            "n_ctx": n_ctx or 2048,
            "n_batch": n_batch or 4,
            "n_ram_gb": n_ram_gb or 16.0,
            "system_info": None,
        }

    inferred_backend = backend or ("gguf" if model_path and model_path.endswith(".gguf") else "safetensors")
    info = detect_system()
    config = recommend_config(system_info=info, model_size_gb=model_size_gb, backend=inferred_backend)

    overrides = {
        "backend": inferred_backend,
        "device": device or config["device"],
        "n_gpu_layers": n_gpu_layers if n_gpu_layers is not None else config["n_gpu_layers"],
        "n_threads": n_threads if n_threads is not None else config["n_threads"],
        "n_ctx": n_ctx if n_ctx is not None else config["n_ctx"],
        "n_batch": n_batch if n_batch is not None else config["n_batch"],
        "n_ram_gb": n_ram_gb if n_ram_gb is not None else config["n_ram_gb"],
        "system_info": info.to_dict(),
    }
    return overrides
