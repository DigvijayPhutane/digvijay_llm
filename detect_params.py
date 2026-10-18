"""
detect_params.py
================
Run this script BEFORE using digvijay_llm.
It detects your system (RAM, GPU, disk) with the same code the library
uses at load time, and prints the parameters it would pick.

Usage:
    python detect_params.py                     # generic model sizes
    python detect_params.py path/to/model.gguf  # a specific model
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from digvijay_llm.hardware import (  # noqa: E402
    GPU_HEADROOM_GB,
    _gguf_gpu_layers,
    detect_params,
    detect_system,
)

# (label, n_layers, Q4_K_M size in GB)
TYPICAL_MODELS = [
    ("7B  model", 32, 4.1),
    ("13B model", 40, 7.9),
    ("34B model", 60, 20.0),
    ("70B model", 80, 38.0),
]


def describe_layers(n_gpu_layers, n_layers):
    if n_gpu_layers == -1:
        return "all layers on GPU"
    if n_gpu_layers == 0:
        return "CPU only"
    return f"{n_gpu_layers} of {n_layers} layers on GPU, rest on CPU"


def main(argv):
    info = detect_system()

    print("\n" + "=" * 60)
    print("   digvijay_llm - System Detector & Parameter Recommender")
    print("=" * 60)
    print("\nYOUR SYSTEM")
    print(f"   RAM        : {info.ram_total_gb:.1f} GB total  |  {info.ram_available_gb:.1f} GB free")
    print(f"   Disk (cwd) : {info.storage_total_gb:.1f} GB total  |  {info.storage_free_gb:.1f} GB free")
    print(f"   CPU cores  : {info.cpu_cores}")
    if info.gpu_name:
        print(f"   GPU        : {info.gpu_name}")
        print(f"   VRAM       : {info.vram_total_gb:.1f} GB total  |  {info.vram_free_gb:.1f} GB free")
        print(f"   PyTorch GPU: {info.gpu_backend or 'not usable'}")
        offload = {True: "yes", False: "no (CPU-only build)", None: "unknown / not installed"}
        print(f"   llama.cpp GPU offload: {offload[info.llama_cpp_gpu_offload]}")
    else:
        print("   GPU        : none detected (CPU-only mode)")

    if len(argv) > 1:
        path = argv[1]
        cfg = detect_params(model_path=path)
        print(f"\nRECOMMENDED FOR {path}")
        for key in ("backend", "device", "n_gpu_layers", "n_threads", "n_ctx", "n_batch", "n_ram_gb"):
            print(f"   {key:<13}: {cfg[key]}")
        for warning in cfg["warnings"]:
            print(f"   WARNING: {warning}")
        return 0

    gpu_usable = info.gpu_vendor is not None and info.llama_cpp_gpu_offload is not False
    vram_free = info.vram_free_gb or info.vram_total_gb
    print("\n" + "-" * 60)
    print("   GGUF (Q4_K_M) ESTIMATES PER MODEL SIZE")
    print(f"   (free VRAM {vram_free:.1f} GB, keeping {GPU_HEADROOM_GB:.0f} GB headroom)")
    print("-" * 60)
    for label, n_layers, size_gb in TYPICAL_MODELS:
        layers = _gguf_gpu_layers(vram_free, size_gb, n_layers) if gpu_usable else 0
        fits_disk = info.storage_free_gb >= size_gb
        print(f"\n  [{label}]  ~{size_gb} GB  -  {describe_layers(layers, n_layers)}")
        if not fits_disk:
            print(f"   needs {size_gb} GB free disk (you have {info.storage_free_gb:.1f} GB here)")

    print("\nThe library picks these automatically; just run:")
    print('   llm = LowRAMLLM.from_gguf("path/to/model.gguf")')
    print("Pass a model path to this script to see its exact settings.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
