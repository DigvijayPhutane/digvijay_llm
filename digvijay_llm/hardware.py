"""System inspection and automatic configuration helpers."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import struct
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency path
    psutil = None

logger = logging.getLogger("digvijay_llm")

TORCH_INSTALL_HINT = (
    "Install a GPU build of PyTorch from https://pytorch.org/get-started/locally/ "
    "(e.g. pip install torch --index-url https://download.pytorch.org/whl/cu128). "
    "The PyPI wheel pulled in by `pip install digvijay_llm[hf]` is CPU-only on Windows."
)
LLAMA_CPP_INSTALL_HINT = (
    "Reinstall llama-cpp-python with GPU support, e.g. "
    'CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python'
)

# Room left free on the GPU for the KV cache and compute buffers when
# deciding how many GGUF layers to offload.
GPU_HEADROOM_GB = 1.0
# Used when the layer count of a model cannot be read. Smaller is safer:
# guessing too few layers under-offloads, guessing too many runs out of VRAM.
DEFAULT_LAYER_GUESS = 32


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
    # The backend PyTorch can actually use ("cuda", "rocm", "metal", "xpu").
    # None when no GPU was found OR a GPU exists that torch cannot use.
    gpu_backend: Optional[str] = None
    torch_available: bool = False
    torch_version: Optional[str] = None
    # Whether the installed llama-cpp-python was built with GPU offload.
    # None when llama-cpp-python is not installed or cannot tell.
    llama_cpp_gpu_offload: Optional[bool] = None

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


def _empty_gpu_fields() -> Dict[str, Any]:
    return {
        "gpu_name": None,
        "gpu_vendor": None,
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "gpu_backend": None,
    }


def _detect_torch_gpu(ram_total_gb: float, ram_available_gb: float) -> Dict[str, Any]:
    """Return the GPU fields of SystemInfo as seen by PyTorch."""
    result: Dict[str, Any] = {**_empty_gpu_fields(), "torch_available": False, "torch_version": None}
    try:
        import torch
    except ImportError:
        return result
    except Exception as exc:  # e.g. OSError: a CUDA DLL failed to load on Windows
        logger.warning("PyTorch is installed but failed to import (%s); treating it as unavailable", exc)
        return result

    result["torch_available"] = True
    result["torch_version"] = getattr(torch, "__version__", None)

    try:
        if torch.cuda.is_available():
            # ROCm builds of torch expose AMD GPUs through the torch.cuda API.
            is_rocm = getattr(torch.version, "hip", None) is not None
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory
            try:
                free, total = torch.cuda.mem_get_info(0)
            except Exception:
                free = total - torch.cuda.memory_reserved(0)
            result.update(
                gpu_name=torch.cuda.get_device_name(0),
                gpu_vendor="amd" if is_rocm else "nvidia",
                gpu_backend="rocm" if is_rocm else "cuda",
                vram_total_gb=total / (1024**3),
                vram_free_gb=free / (1024**3),
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Apple unified memory: treat a share of system RAM as "VRAM".
            result.update(
                gpu_name="Apple Silicon GPU",
                gpu_vendor="apple",
                gpu_backend="metal",
                vram_total_gb=max(4.0, ram_total_gb * 0.4),
                vram_free_gb=max(2.0, ram_available_gb * 0.4),
            )
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            try:
                total_gb = torch.xpu.get_device_properties(0).total_memory / (1024**3)
                free_gb = total_gb
            except Exception:
                total_gb = max(4.0, ram_total_gb * 0.3)
                free_gb = max(2.0, ram_available_gb * 0.3)
            result.update(
                gpu_name="Intel GPU",
                gpu_vendor="intel",
                gpu_backend="xpu",
                vram_total_gb=total_gb,
                vram_free_gb=free_gb,
            )
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.debug("Torch GPU detection failed: %s", exc)
        # Never report a half-detected GPU.
        result.update(_empty_gpu_fields())

    return result


def _probe_nvidia_smi() -> Optional[Tuple[str, float, float]]:
    """Return (name, total_gb, free_gb) for GPU 0 via nvidia-smi, if present."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = proc.stdout.strip().splitlines()
    if not lines:
        return None
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) != 3:
        return None
    try:
        return parts[0], float(parts[1]) / 1024, float(parts[2]) / 1024
    except ValueError:
        return None


def _llama_cpp_gpu_offload() -> Optional[bool]:
    try:
        import llama_cpp
    except Exception:
        return None
    fn = getattr(llama_cpp, "llama_supports_gpu_offload", None)
    if fn is None:
        return None
    try:
        return bool(fn())
    except Exception:
        return None


def detect_system() -> SystemInfo:
    """Collect a compact view of the host machine for runtime tuning."""
    ram_total_gb, ram_available_gb = _get_ram_info()
    disk = shutil.disk_usage(".")

    cpu_model = platform.processor() or platform.machine() or "unknown"
    os_family = "windows" if os.name == "nt" else "macos" if platform.system() == "Darwin" else "linux" if platform.system() else "unknown"
    os_version = platform.platform()

    gpu = _detect_torch_gpu(ram_total_gb, ram_available_gb)

    if gpu["gpu_backend"] is None:
        # torch is missing, CPU-only, or failed: the GPU may still be usable
        # by llama.cpp, so record it without claiming a torch backend.
        smi = _probe_nvidia_smi()
        if smi is not None:
            name, total_gb, free_gb = smi
            gpu.update(gpu_name=name, gpu_vendor="nvidia", vram_total_gb=total_gb, vram_free_gb=free_gb)
        elif os_family == "macos" and platform.machine() == "arm64":
            gpu.update(
                gpu_name="Apple Silicon GPU",
                gpu_vendor="apple",
                vram_total_gb=max(4.0, ram_total_gb * 0.4),
                vram_free_gb=max(2.0, ram_available_gb * 0.4),
            )

    return SystemInfo(
        os_family=os_family,
        os_version=os_version,
        cpu_cores=os.cpu_count() or 1,
        cpu_model=cpu_model,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        storage_total_gb=disk.total / (1024**3),
        storage_free_gb=disk.free / (1024**3),
        llama_cpp_gpu_offload=_llama_cpp_gpu_offload(),
        **gpu,
    )


# ---------- model inspection ----------

_GGUF_SCALAR_FORMATS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_GGUF_STRING = 8
_GGUF_ARRAY = 9


def _gguf_read_string(f) -> bytes:
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length)


def _gguf_read_value(f, value_type: int):
    if value_type in _GGUF_SCALAR_FORMATS:
        fmt = _GGUF_SCALAR_FORMATS[value_type]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if value_type == _GGUF_STRING:
        return _gguf_read_string(f)
    if value_type == _GGUF_ARRAY:
        elem_type, count = struct.unpack("<IQ", f.read(12))
        if elem_type in _GGUF_SCALAR_FORMATS:
            # Skip fixed-size arrays without decoding them.
            f.seek(struct.calcsize(_GGUF_SCALAR_FORMATS[elem_type]) * count, os.SEEK_CUR)
        else:
            for _ in range(count):
                _gguf_read_value(f, elem_type)
        return None
    raise ValueError(f"unknown GGUF value type {value_type}")


def is_gguf_path(path: Optional[str]) -> bool:
    return bool(path) and path.lower().endswith(".gguf")


def read_gguf_metadata(path: str) -> Dict[str, int]:
    """Read `block_count` and `context_length` from a GGUF header.

    Returns only the keys that were found (empty for non-GGUF files).
    """
    found: Dict[str, int] = {}
    wanted = ("block_count", "context_length")
    arch = None
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return found
            (version,) = struct.unpack("<I", f.read(4))
            if version < 2:
                return found
            _tensor_count, kv_count = struct.unpack("<QQ", f.read(16))
            for _ in range(kv_count):
                key = _gguf_read_string(f).decode("utf-8", "replace")
                (value_type,) = struct.unpack("<I", f.read(4))
                value = _gguf_read_value(f, value_type)
                if key == "general.architecture" and isinstance(value, bytes):
                    arch = value.decode("utf-8", "replace")
                    continue
                prefix, _, name = key.rpartition(".")
                if name in wanted and isinstance(value, int) and (arch is None or prefix == arch):
                    found[name] = int(value)
                    if len(found) == len(wanted):
                        break
    except (OSError, struct.error, ValueError):
        pass
    return found


def read_gguf_layer_count(path: str) -> Optional[int]:
    """Read `<arch>.block_count` from a GGUF header, or None if unavailable."""
    return read_gguf_metadata(path).get("block_count")


def _model_size_gb(path: Optional[str]) -> Optional[float]:
    if not path:
        return None
    try:
        if os.path.isfile(path):
            return os.path.getsize(path) / (1024**3)
        if os.path.isdir(path):
            names = os.listdir(path)
            # HF repos often ship the same weights as both .safetensors and
            # .bin; count one format only, the one the loader will use.
            for ext in (".safetensors", ".bin", ".gguf"):
                total = sum(os.path.getsize(os.path.join(path, n)) for n in names if n.endswith(ext))
                if total:
                    return total / (1024**3)
    except OSError:
        return None
    return None


def _hf_config(path: str) -> Optional[Dict[str, Any]]:
    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    text_config = config.get("text_config")  # multimodal checkpoints nest the LM config
    return text_config if isinstance(text_config, dict) else config


def _first_int(config: Dict[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _model_layer_count(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    if os.path.isfile(path) and is_gguf_path(path):
        return read_gguf_layer_count(path)
    if os.path.isdir(path):
        config = _hf_config(path)
        if config:
            return _first_int(config, "num_hidden_layers", "n_layer", "num_layers")
    return None


def _model_context_length(path: Optional[str]) -> Optional[int]:
    """The context length the model was trained for, if it can be read."""
    if not path:
        return None
    if os.path.isfile(path) and is_gguf_path(path):
        return read_gguf_metadata(path).get("context_length")
    if os.path.isdir(path):
        config = _hf_config(path)
        if config:
            return _first_int(config, "max_position_embeddings", "n_positions", "max_sequence_length", "seq_length")
    return None


def _gguf_gpu_layers(vram_free_gb: float, model_size_gb: float, n_layers: Optional[int]) -> int:
    """Layers to offload: -1 (all) if the model fits, else a proportional share."""
    usable = vram_free_gb - GPU_HEADROOM_GB
    if usable <= 0:
        return 0
    if model_size_gb <= usable:
        return -1
    layers = n_layers or DEFAULT_LAYER_GUESS
    return max(0, int(layers * usable / model_size_gb))


# ---------- recommendations ----------

def recommend_config(
    system_info: Optional[SystemInfo] = None,
    model_size_gb: Optional[float] = None,
    backend: Optional[str] = None,
    prefer_gpu: bool = True,
    n_layers: Optional[int] = None,
    ram_budget_gb: Optional[float] = None,
    model_context_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Recommend an inference configuration from the detected hardware.

    The GGUF backend offloads with llama.cpp, which does not need PyTorch;
    the safetensors backend needs a torch build that can see the GPU.
    `ram_budget_gb` (the caller's n_ram_gb) sizes the context instead of the
    machine's total RAM, and the context never exceeds `model_context_length`.
    """
    info = system_info or detect_system()
    size_known = model_size_gb is not None
    model_size_gb = model_size_gb or 20.0
    if backend is None:
        backend = "gguf"
    warnings: List[str] = []

    device = "cpu"
    gpu_layers = 0

    if backend == "gguf":
        gpu_present = info.gpu_vendor is not None and info.vram_total_gb > 0
        if gpu_present and info.llama_cpp_gpu_offload is False:
            warnings.append(
                f"GPU '{info.gpu_name}' found, but the installed llama-cpp-python was built "
                f"without GPU offload, so the model will run on CPU. {LLAMA_CPP_INSTALL_HINT}"
            )
        elif gpu_present and prefer_gpu:
            device = {"nvidia": "cuda", "amd": "rocm", "apple": "metal", "intel": "xpu"}.get(info.gpu_vendor, "cpu")
            vram_free = info.vram_free_gb or info.vram_total_gb
            if not size_known:
                # Without the real size, never claim the whole model fits.
                gpu_layers = max(0, int(DEFAULT_LAYER_GUESS * (vram_free - GPU_HEADROOM_GB) / model_size_gb))
            else:
                gpu_layers = _gguf_gpu_layers(vram_free, model_size_gb, n_layers)
            if gpu_layers == 0:
                device = "cpu"
    else:
        if info.gpu_backend and prefer_gpu:
            # Torch device names: ROCm GPUs are addressed as "cuda".
            device = {"cuda": "cuda", "rocm": "cuda", "metal": "mps", "xpu": "xpu"}.get(info.gpu_backend, "cpu")
        elif info.gpu_vendor == "nvidia":
            if info.torch_available:
                reason = f"the installed PyTorch {info.torch_version} cannot use it (CPU-only build)"
            else:
                reason = "PyTorch is not installed"
            warnings.append(
                f"NVIDIA GPU '{info.gpu_name}' found, but {reason}, so the safetensors backend "
                f"will run on CPU. {TORCH_INSTALL_HINT}"
            )

    # n_batch is how many prompt tokens llama.cpp evaluates per step; tiny
    # values make prompt processing (and so time-to-first-token) crawl.
    budget = ram_budget_gb if ram_budget_gb else info.ram_total_gb
    if budget <= 8:
        n_ctx, n_batch = 1024, 128
    elif budget <= 16:
        n_ctx, n_batch = 2048, 256
    elif budget <= 32:
        n_ctx, n_batch = 4096, 512
    else:
        n_ctx, n_batch = 8192, 512
    if model_context_length:
        n_ctx = min(n_ctx, model_context_length)
        n_batch = min(n_batch, n_ctx)
    n_ram_gb = min(info.ram_total_gb, 16.0 if info.ram_total_gb <= 16 else 32.0 if info.ram_total_gb <= 32 else 64.0)

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
        "warnings": warnings,
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
    """Return a configuration dict used by the public API.

    Any argument left as None is filled from hardware detection; explicit
    values (including 0) are always kept.
    """
    if not detect:
        return {
            "backend": backend or "gguf",
            "device": device or "cpu",
            "n_gpu_layers": n_gpu_layers or 0,
            "n_threads": n_threads or os.cpu_count() or 1,
            "n_ctx": n_ctx or 2048,
            "n_batch": n_batch or 512,
            "n_ram_gb": n_ram_gb or 16.0,
            "warnings": [],
            "system_info": None,
        }

    inferred_backend = backend or ("gguf" if is_gguf_path(model_path) else "safetensors")
    if model_size_gb is None:
        model_size_gb = _model_size_gb(model_path)
    info = detect_system()
    config = recommend_config(
        system_info=info,
        model_size_gb=model_size_gb,
        backend=inferred_backend,
        n_layers=_model_layer_count(model_path),
        ram_budget_gb=n_ram_gb,
        model_context_length=_model_context_length(model_path),
    )

    # Only warn about the GPU when the caller left the choice to us.
    warnings = config["warnings"] if device is None and n_gpu_layers is None else []
    for message in warnings:
        logger.warning(message)

    overrides = {
        "backend": inferred_backend,
        "device": device or config["device"],
        "n_gpu_layers": n_gpu_layers if n_gpu_layers is not None else config["n_gpu_layers"],
        "n_threads": n_threads if n_threads is not None else config["n_threads"],
        "n_ctx": n_ctx if n_ctx is not None else config["n_ctx"],
        "n_batch": n_batch if n_batch is not None else config["n_batch"],
        "n_ram_gb": n_ram_gb if n_ram_gb is not None else config["n_ram_gb"],
        "warnings": warnings,
        "system_info": info.to_dict(),
    }
    return overrides
