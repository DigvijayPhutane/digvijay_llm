"""
api.py
======
The single class most users will actually import: `LowRAMLLM`.
Picks the right backend (GGUF mmap streaming vs safetensors layer
streaming) based on what's passed in, and exposes one consistent
.generate() / .stream() / .chat() interface regardless of backend.
"""

import os
from typing import Optional, Iterator, List, Dict


class LowRAMLLM:
    """
    Unified entrypoint for digvijay_llm.

    Use the classmethods to construct it:
        LowRAMLLM.from_gguf(path, n_ram_gb=16)
        LowRAMLLM.from_safetensors(dir_path, n_ram_gb=16)
        LowRAMLLM.auto(path_or_dir, n_ram_gb=16)   # detects which backend to use
    """

    def __init__(self, engine, backend: str):
        self._engine = engine
        self.backend = backend

    # ---------- constructors ----------

    @classmethod
    def from_gguf(
        cls,
        model_path: str,
        n_ram_gb: float = 16.0,
        n_gpu_layers: int = 0,
        n_threads: Optional[int] = None,
        verbose: bool = False,
    ) -> "LowRAMLLM":
        from .engine_gguf import GGUFStreamingEngine

        engine = GGUFStreamingEngine(
            model_path=model_path,
            n_ram_gb=n_ram_gb,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=verbose,
        )
        return cls(engine, backend="gguf-mmap")

    @classmethod
    def from_safetensors(
        cls,
        model_dir: str,
        offload_dir: Optional[str] = None,
        n_ram_gb: float = 16.0,
        dtype: str = "float16",
    ) -> "LowRAMLLM":
        from .engine_safetensors import SafetensorsLayerStreamer

        engine = SafetensorsLayerStreamer(
            model_dir=model_dir,
            offload_dir=offload_dir,
            n_ram_gb=n_ram_gb,
            dtype=dtype,
        )
        return cls(engine, backend="safetensors-layer-stream")

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

    def stream(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> Iterator[str]:
        if self.backend == "gguf-mmap":
            yield from self._engine.generate(prompt, max_tokens=max_tokens, temperature=temperature, stream=True)
        else:
            yield from self._engine.stream_generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 256, temperature: float = 0.7) -> str:
        if self.backend == "gguf-mmap":
            return self._engine.chat(messages, max_tokens=max_tokens, temperature=temperature)
        # Fallback for safetensors backend: flatten chat messages into a prompt
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        return self._engine.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def __repr__(self):
        return f"<LowRAMLLM backend={self.backend!r}>"
