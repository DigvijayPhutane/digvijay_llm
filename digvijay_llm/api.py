"""
api.py
======
The single class most users will actually import: `LowRAMLLM`.
Picks the right backend (GGUF mmap streaming vs safetensors layer
streaming) based on what's passed in, and exposes one consistent
.generate() / .stream() / .chat() interface regardless of backend.
"""

import os
from typing import Any, Callable, Dict, Iterator, List, Optional

from .hardware import detect_params


class LowRAMLLM:
    """
    Unified entrypoint for digvijay_llm.

    Use the classmethods to construct it:
        LowRAMLLM.from_gguf(path, n_ram_gb=16)
        LowRAMLLM.from_safetensors(dir_path, n_ram_gb=16)
        LowRAMLLM.auto(path_or_dir, n_ram_gb=16)   # detects which backend to use
    """

    def __init__(self, engine, backend: str, config: Optional[Dict[str, Any]] = None):
        self._engine = engine
        self.backend = backend
        self.config = config or {}
        self.device_info = self.config.get("system_info")

    # ---------- constructors ----------

    @classmethod
    def from_gguf(
        cls,
        model_path: str,
        n_ram_gb: float = 16.0,
        n_gpu_layers: int = 0,
        n_threads: Optional[int] = None,
        verbose: bool = False,
        detect: bool = True,
        n_ctx: Optional[int] = None,
        n_batch: Optional[int] = None,
        device: Optional[str] = None,
    ) -> "LowRAMLLM":
        from .engine_gguf import GGUFStreamingEngine

        cfg = detect_params(
            model_path=model_path,
            detect=detect,
            backend="gguf",
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_ram_gb=n_ram_gb,
            device=device,
        )
        if n_ram_gb is None or n_ram_gb == 16.0 and not detect:
            n_ram_gb = cfg["n_ram_gb"]
        if n_gpu_layers == 0 and detect:
            n_gpu_layers = cfg["n_gpu_layers"]
        if n_threads is None:
            n_threads = cfg["n_threads"]
        if n_ctx is None:
            n_ctx = cfg["n_ctx"]
        if n_batch is None:
            n_batch = cfg["n_batch"]
        if device is None:
            device = cfg["device"]

        engine = GGUFStreamingEngine(
            model_path=model_path,
            n_ram_gb=n_ram_gb,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=verbose,
            n_ctx=n_ctx,
            n_batch=n_batch,
            device=device,
        )
        return cls(engine, backend="gguf-mmap", config=cfg)

    @classmethod
    def from_safetensors(
        cls,
        model_dir: str,
        offload_dir: Optional[str] = None,
        n_ram_gb: float = 16.0,
        dtype: str = "float16",
        detect: bool = True,
        device: Optional[str] = None,
        n_ctx: Optional[int] = None,
        n_batch: Optional[int] = None,
    ) -> "LowRAMLLM":
        from .engine_safetensors import SafetensorsLayerStreamer

        cfg = detect_params(
            model_path=model_dir,
            detect=detect,
            backend="safetensors",
            n_ram_gb=n_ram_gb,
            device=device,
            n_ctx=n_ctx,
            n_batch=n_batch,
        )
        if n_ram_gb is None or n_ram_gb == 16.0 and not detect:
            n_ram_gb = cfg["n_ram_gb"]
        if device is None:
            device = cfg["device"]
        if n_ctx is None:
            n_ctx = cfg["n_ctx"]
        if n_batch is None:
            n_batch = cfg["n_batch"]

        engine = SafetensorsLayerStreamer(
            model_dir=model_dir,
            offload_dir=offload_dir,
            n_ram_gb=n_ram_gb,
            dtype=dtype,
            device=device,
            n_ctx=n_ctx,
            n_batch=n_batch,
        )
        return cls(engine, backend="safetensors-layer-stream", config=cfg)

    @classmethod
    def auto(cls, path: str, n_ram_gb: float = 16.0, **kwargs) -> "LowRAMLLM":
        """Detects whether `path` is a .gguf file or an HF checkpoint dir."""
        if os.path.isfile(path) and path.endswith(".gguf"):
            return cls.from_gguf(path, n_ram_gb=n_ram_gb, **kwargs)
        if os.path.isdir(path):
            has_st = any(f.endswith(".safetensors") for f in os.listdir(path))
            if has_st:
                return cls.from_safetensors(path, n_ram_gb=n_ram_gb, **kwargs)
        raise ValueError(
            f"Could not detect model type at '{path}'. Expected a .gguf file "
            "or a directory containing .safetensors shards."
        )

    # ---------- inference ----------

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, **kwargs) -> str:
        return self._engine.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)

    def stream(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, callback: Optional[Callable[[str], None]] = None) -> Iterator[str]:
        if self.backend == "gguf-mmap":
            yield from self._engine.generate(prompt, max_tokens=max_tokens, temperature=temperature, stream=True, callback=callback)
        else:
            yield from self._engine.stream_generate(prompt, max_tokens=max_tokens, temperature=temperature, callback=callback)

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 256, temperature: float = 0.7) -> str:
        if self.backend == "gguf-mmap":
            return self._engine.chat(messages, max_tokens=max_tokens, temperature=temperature)
        # Fallback for safetensors backend: flatten chat messages into a prompt
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        return self._engine.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def benchmark(self, prompt: str, max_tokens: int = 64) -> Dict[str, Any]:
        if hasattr(self._engine, "benchmark"):
            return self._engine.benchmark(prompt, max_tokens=max_tokens)
        return {"backend": self.backend, "device": self.config.get("device", "cpu")}

    def __repr__(self):
        return f"<LowRAMLLM backend={self.backend!r}>"
