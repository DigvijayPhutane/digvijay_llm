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

import logging
import os
import time
from typing import Any, Callable, Dict, Iterator, Optional

from .context import apply_keep, fit_messages, fit_to_context

logger = logging.getLogger("digvijay_llm")


class _CapturedPrompt(Exception):
    def __init__(self, prompt):
        super().__init__("prompt captured")
        self.prompt = prompt


class _PromptCapture:
    """Stands in for a Llama so a chat handler renders its prompt without
    running the model: everything is delegated except create_completion."""

    def __init__(self, llm):
        self._llm = llm

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def create_completion(self, *args, **kwargs):
        raise _CapturedPrompt(kwargs["prompt"] if "prompt" in kwargs else args[0])


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
        built for; -1 = every layer). Requires a GPU-enabled build of
        llama-cpp-python.
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
        n_ctx: Optional[int] = None,
        n_batch: Optional[int] = None,
        device: Optional[str] = None,
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

        n_ram_gb = n_ram_gb or 16.0
        self.model_path = model_path
        self.n_ram_gb = n_ram_gb
        self.device = device or "cpu"
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads or os.cpu_count() or 1
        self.n_batch = n_batch or 512  # llama.cpp's default; tiny batches make prompt processing crawl

        # Conservative context size: bigger context = bigger KV cache = more
        # RAM. We scale it down automatically on tight budgets.
        if n_ctx is not None:
            ctx = n_ctx
        elif n_ram_gb <= 8:
            ctx = 1024
        elif n_ram_gb <= 16:
            ctx = 2048
        elif n_ram_gb <= 32:
            ctx = 4096
        else:
            ctx = 8192
        self.n_batch = min(self.n_batch, ctx)

        try:
            self._llm = Llama(
                model_path=model_path,
                n_ctx=ctx,
                n_threads=self.n_threads,
                n_gpu_layers=n_gpu_layers,
                n_batch=self.n_batch,
                use_mmap=True,   # critical: keeps weights disk-backed ("ROM"), not fully copied into RAM
                use_mlock=False, # critical: do NOT force-pin pages into RAM, let the OS evict freely
                verbose=verbose,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize llama.cpp backend for '{model_path}': {exc}") from exc
        self._ctx_size = ctx

    def _fit_prompt(self, prompt: str, max_tokens: int):
        """Return (prompt, max_tokens) that fit the context window.

        The prompt stays a string when it fits (so llama.cpp tokenizes it as
        usual); a too-long prompt is left-truncated and passed as token ids.
        """
        tokens = self._llm.tokenize(prompt.encode("utf-8"))
        first_is_bos = bool(tokens) and tokens[0] == self._llm.token_bos()
        keep, max_tokens = fit_to_context(len(tokens), max_tokens, self._ctx_size, first_is_bos)
        return (prompt if keep is None else apply_keep(tokens, keep)), max_tokens

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        callback: Optional[Callable[[str], None]] = None,
    ):
        if stream:
            return self._stream(prompt, max_tokens, temperature, top_p, callback)

        prompt_in, max_tokens = self._fit_prompt(prompt, max_tokens)
        out = self._llm(
            prompt_in,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        text = out["choices"][0]["text"]
        if callback is not None:
            callback(text)
        return text

    def _stream(self, prompt, max_tokens, temperature, top_p, callback) -> Iterator[str]:
        prompt_in, max_tokens = self._fit_prompt(prompt, max_tokens)
        for chunk in self._llm(
            prompt_in,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        ):
            text = chunk["choices"][0]["text"]
            if not text:  # llama.cpp emits "" while a multi-byte character is incomplete
                continue
            if callback is not None:
                callback(text)
            yield text

    def _chat_prompt_len(self, messages) -> int:
        """Token length of the prompt llama.cpp will build for `messages`.

        Runs the same chat handler create_chat_completion would pick, against
        a stand-in that captures the rendered prompt instead of generating.
        """
        llm = self._llm
        try:
            # Same lookup order as Llama.create_chat_completion.
            handler = getattr(llm, "chat_handler", None) or getattr(llm, "_chat_handlers", {}).get(llm.chat_format)
            if handler is None:
                from llama_cpp import llama_chat_format

                handler = llama_chat_format.get_chat_completion_handler(llm.chat_format)
            handler(llama=_PromptCapture(llm), messages=messages)
        except _CapturedPrompt as captured:
            prompt = captured.prompt
            if isinstance(prompt, str):
                prompt = llm.tokenize(prompt.encode("utf-8"), special=True)
            return len(prompt)
        except Exception as exc:
            logger.debug("Could not render the chat prompt (%s); estimating its length", exc)
        # Estimate: message text plus a few template tokens per message.
        return sum(len(llm.tokenize(str(m.get("content") or "").encode("utf-8"), add_bos=False)) + 8 for m in messages)

    def chat(self, messages, max_tokens: int = 256, temperature: float = 0.7):
        """OpenAI-style chat interface, e.g.
        messages=[{"role": "user", "content": "hi"}]

        If the conversation doesn't fit n_ctx, the oldest messages are
        dropped (system messages are kept); see context.fit_messages.
        """
        messages, max_tokens = fit_messages(messages, self._chat_prompt_len, max_tokens, self._ctx_size)
        out = self._llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return out["choices"][0]["message"]["content"]

    def benchmark(self, prompt: str, max_tokens: int = 64) -> Dict[str, Any]:
        """Greedy-decode `max_tokens` and time it with real model tokens.

        ttft_s is prompt processing + first token; tokens_per_sec is the
        decode rate over the tokens after the first one.
        """
        prompt_ids = self._llm.tokenize(prompt.encode("utf-8"))
        first_is_bos = bool(prompt_ids) and prompt_ids[0] == self._llm.token_bos()
        keep, max_tokens = fit_to_context(len(prompt_ids), max_tokens, self._ctx_size, first_is_bos)
        prompt_ids = apply_keep(prompt_ids, keep)
        eos = self._llm.token_eos()
        completion_tokens = 0
        first_token_at = None

        start = time.perf_counter()
        generator = self._llm.generate(prompt_ids, temp=0.0, top_p=1.0, reset=True)
        try:
            for token in generator:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                if token == eos:
                    break
                completion_tokens += 1
                if completion_tokens >= max_tokens:
                    break
        finally:
            generator.close()
        end = time.perf_counter()

        ttft = (first_token_at or end) - start
        decode_time = end - (first_token_at or end)
        decode_tokens = max(completion_tokens - 1, 0)
        return {
            "ttft_s": round(ttft, 4),
            "total_s": round(end - start, 4),
            "tokens_per_sec": round(decode_tokens / decode_time, 2) if decode_tokens and decode_time > 0 else 0.0,
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": completion_tokens,
            "backend": "gguf",
            "device": self.device,
            "n_gpu_layers": self.n_gpu_layers,
        }

    @property
    def context_size(self) -> int:
        return self._ctx_size
