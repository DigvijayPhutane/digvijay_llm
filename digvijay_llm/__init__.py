"""
digvijay_llm
============

Run large open-source LLMs (including 70B-class models) on machines with
very limited RAM (e.g. 16 GB) by NEVER loading the full weight set into
memory at once. Instead, weights stay on disk (your "ROM" — SSD/HDD) and
are streamed in just-in-time, layer by layer, then released.

IMPORTANT HONEST NOTE (please read):
-------------------------------------
It is physically impossible to *compute* directly on disk/ROM — the CPU/GPU
can only execute operations on data that is in RAM/VRAM. What IS possible,
and what this library does, is keep the bulk of the model's weights on disk
and only pull small pieces (one layer / one tensor at a time) into RAM right
before they're needed, then drop them again. This keeps peak RAM usage close
to "one layer's worth of weights" instead of "the whole model's worth",
which is what makes running a 70B model on a 16GB machine feasible (at the
cost of disk I/O speed -> slower tokens/sec, especially on HDD; an NVMe SSD
is strongly recommended).

Two backends are provided:

1. GGUFStreamingEngine
   - Wraps llama.cpp (via llama-cpp-python) which already implements
     production-grade memory-mapped (mmap) weight streaming + quantization.
   - Recommended for real, fast usage. Requires a .gguf quantized model file.

2. SafetensorsLayerStreamer
   - A pure PyTorch engine written from scratch in this library that loads
     a HuggingFace-format (.safetensors) checkpoint ONE TRANSFORMER LAYER
     AT A TIME directly from disk, runs the forward pass for that layer,
     then frees it before loading the next layer.
   - Useful when you only have raw HF weights and don't want to convert to
     GGUF, or want full Python-level control over the streaming behavior.

Quick start
-----------
    from digvijay_llm import LowRAMLLM

    llm = LowRAMLLM.from_gguf("models/llama-70b.Q4_K_M.gguf", n_ram_gb=16)
    print(llm.generate("Explain quantum entanglement simply."))

    # or, for raw HF safetensors checkpoints:
    llm = LowRAMLLM.from_safetensors("models/llama-70b-hf/")
    print(llm.generate("Write a haiku about the ocean."))
"""

from .engine_gguf import GGUFStreamingEngine
from .engine_safetensors import SafetensorsLayerStreamer
from .api import LowRAMLLM
from .ram_planner import RAMPlanner, plan_for_budget

__all__ = [
    "LowRAMLLM",
    "GGUFStreamingEngine",
    "SafetensorsLayerStreamer",
    "RAMPlanner",
    "plan_for_budget",
]

__version__ = "0.1.0"
