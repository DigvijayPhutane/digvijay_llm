"""
engine_gguf.py
==============
Production-grade low-RAM backend. Wraps llama-cpp-python, which loads
GGUF files via mmap() — the OS keeps weights on disk and pages them into
RAM lazily as the model touches them, evicting cold pages under memory
pressure. Combined with quantization (Q4_K_M, Q5_K_M, etc.) this is what
lets a 70B model run on a 16GB machine.

This module just gives that capability a clean, documented entrypoint
inside digvijay_llm, with sane low-RAM defaults (no mlock, limited
context, mmap forced on, conservative thread count).
"""

import os
from typing import Optional, Iterator


class GGUFStreamingEngine:
    """
    Streams a quantized .gguf model off disk instead of loading it fully
    into RAM, via memory-mapping (mmap).

    Parameters
    ----------
    model_path : str
        Path to a .gguf quantized model file (e.g. produced by
        llama.cpp's convert + quantize scripts, or downloaded pre-quantized
        from a hub like Hugging Face / TheBloke-style repos).
    n_ram_gb : float
        Approximate RAM budget; used only to pick a sane default context
        size / batch size so you don't accidentally request a KV cache
        bigger than your machine can hold.
    n_gpu_layers : int
        Number of layers to offload to GPU if you have one (0 = pure CPU+
        disk streaming, which is the "low RAM" use case this library is
        built for).
    n_threads : Optional[int]
        CPU threads to use. Defaults to os.cpu_count().
    """

    def __init__(
        self,
        model_path: str,
        n_ram_gb: float = 16.0,
        n_gpu_layers: int = 0,
        n_threads: Optional[int] = None,
        verbose: bool = False,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model not found at: {model_path}")

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "engine_gguf requires the 'llama-cpp-python' package. "
                "Install it with: pip install llama-cpp-python"
            ) from e

        self.model_path = model_path
        self.n_ram_gb = n_ram_gb

        # Conservative context size: bigger context = bigger KV cache = more
        # RAM. We scale it down automatically on tight budgets.
        if n_ram_gb <= 8:
            ctx = 1024
        elif n_ram_gb <= 16:
            ctx = 2048
        elif n_ram_gb <= 32:
            ctx = 4096
        else:
            ctx = 8192

        self._llm = Llama(
            model_path=model_path,
            n_ctx=ctx,
            n_threads=n_threads or os.cpu_count(),
            n_gpu_layers=n_gpu_layers,
            use_mmap=True,   # critical: keeps weights disk-backed ("ROM"), not fully copied into RAM
            use_mlock=False, # critical: do NOT force-pin pages into RAM, let the OS evict freely
            verbose=verbose,
        )
        self._ctx_size = ctx

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ):
        if stream:
            return self._stream(prompt, max_tokens, temperature, top_p)

        out = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return out["choices"][0]["text"]

    def _stream(self, prompt, max_tokens, temperature, top_p) -> Iterator[str]:
        for chunk in self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        ):
            yield chunk["choices"][0]["text"]

    def chat(self, messages, max_tokens: int = 256, temperature: float = 0.7):
        """OpenAI-style chat interface, e.g.
        messages=[{"role": "user", "content": "hi"}]"""
        out = self._llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return out["choices"][0]["message"]["content"]

    @property
    def context_size(self) -> int:
        return self._ctx_size
