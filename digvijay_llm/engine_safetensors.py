"""
engine_safetensors.py
======================
For raw HuggingFace-format checkpoints (.safetensors, no GGUF conversion
done). This engine loads the model with EMPTY weights first, then attaches
disk-offload hooks (built on top of Hugging Face `accelerate`) so that:

  - Only ONE transformer layer's weights are materialized in RAM at a time.
  - Right before a layer's forward() runs, its weights are read from the
    .safetensors shard on disk straight into RAM.
  - Immediately after that layer's forward() finishes, its weights are
    freed from RAM again.

This is the "layer-by-layer disk streaming" approach — RAM peak ~= size of
the largest single layer + activations, instead of the whole model. It is
slower than keeping everything in RAM (disk I/O per layer per token) but it
is what makes a 70B model loadable on a 16 GB machine without GGUF
conversion.

Under the hood this uses `accelerate`'s `disk_offload` / `dispatch_model`
machinery (the same machinery HF itself recommends for low-RAM inference),
wrapped here with sane defaults + the digvijay_llm.generate() interface.
"""

import gc
import logging
import os
import re
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

from .context import fit_messages, fit_to_context

logger = logging.getLogger("digvijay_llm")


class SafetensorsLayerStreamer:
    """
    Parameters
    ----------
    model_dir : str
        Path to a local directory containing a HuggingFace checkpoint:
        config.json + one or more *.safetensors weight shards + tokenizer
        files. (Use `huggingface-cli download <repo> --local-dir ...` to
        fetch one without loading it into RAM first.)
    offload_dir : str
        Scratch directory on disk used as the staging area accelerate
        streams weights through. Needs free space roughly equal to the
        model size. Defaults to "<model_dir>/.digvijay_offload".
    n_ram_gb : float
        Approximate RAM budget, used to decide max_memory passed to
        accelerate so it knows how aggressively to offload to disk.
    dtype : str
        "float16" (default, halves RAM vs float32) or "float32".
    """

    def __init__(
        self,
        model_dir: str,
        offload_dir: Optional[str] = None,
        n_ram_gb: float = 16.0,
        dtype: str = "float16",
        device: str = "cpu",
        n_ctx: Optional[int] = None,
        n_batch: Optional[int] = None,
    ):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"model_dir not found: {model_dir}")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from accelerate import infer_auto_device_map, init_empty_weights
        except ImportError as e:
            raise ImportError(
                "engine_safetensors requires 'torch', 'transformers', and "
                "'accelerate'. Install with: "
                "pip install torch transformers accelerate safetensors"
            ) from e

        self._torch = torch
        self.model_dir = model_dir
        self.offload_dir = offload_dir or os.path.join(model_dir, ".digvijay_offload")
        self.device = device
        self.n_ctx = n_ctx or 2048
        self.n_batch = n_batch or 512
        os.makedirs(self.offload_dir, exist_ok=True)

        torch_dtype = torch.float16 if dtype == "float16" else torch.float32

        # 1. Build the model architecture with NO weights materialized yet
        #    (this costs ~0 RAM regardless of model size).
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_dir)
        with init_empty_weights():
            # transformers >= 4.56 renamed `torch_dtype` to `dtype`.
            import transformers
            dtype_kwarg = "dtype" if self._version_tuple(transformers.__version__) >= (4, 56) else "torch_dtype"
            empty_model = AutoModelForCausalLM.from_config(config, **{dtype_kwarg: torch_dtype})

        # 2. Reserve almost all RAM for ONE layer at a time by capping the
        #    "max_memory" accelerate is allowed to use on this device; the
        #    rest is automatically offloaded to `offload_dir` on disk and
        #    streamed back in layer-by-layer during forward().
        max_memory = self._max_memory(torch, device, n_ram_gb)

        device_map = infer_auto_device_map(
            empty_model,
            max_memory=max_memory,
            no_split_module_classes=self._guess_no_split_classes(empty_model),
            dtype=torch_dtype,
        )

        # 3. Load real weights directly from the safetensors shards on disk
        #    INTO the device_map plan above — anything that doesn't fit the
        #    RAM budget gets a disk-offload hook instead of being loaded.
        from accelerate import load_checkpoint_and_dispatch

        self._model = load_checkpoint_and_dispatch(
            empty_model,
            checkpoint=model_dir,
            device_map=device_map,
            offload_folder=self.offload_dir,
            offload_state_dict=True,   # streams shards in piece by piece while loading too
            dtype=torch_dtype,
        )
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.device_map = device_map
        self._input_device = self._embedding_device(self._model, device_map, device)

    @staticmethod
    def _version_tuple(version: str) -> tuple:
        """'4.56.0.dev0' -> (4, 56); tolerates suffixes like '5.0.0rc1'."""
        match = re.match(r"(\d+)\.(\d+)", version)
        return (int(match.group(1)), int(match.group(2))) if match else (0, 0)

    @staticmethod
    def _embedding_device(model, device_map: Dict[str, Any], device: str = "cuda") -> str:
        """Device where the input embedding runs, so prompts can be sent there.

        accelerate keys device_map by module prefix ("" = whole model) and uses
        integer GPU indexes; weights offloaded to "disk" execute on the CPU.
        """
        try:
            embedding = model.get_input_embeddings()
            name = next(n for n, m in model.named_modules() if m is embedding)
        except Exception:
            return "cpu"
        matches = [k for k in device_map if k == "" or name == k or name.startswith(k + ".")]
        if not matches:
            return "cpu"
        value = device_map[max(matches, key=len)]
        if isinstance(value, int):
            return f"{'xpu' if device.startswith('xpu') else 'cuda'}:{value}"
        if value in ("disk", "meta"):
            return "cpu"
        return str(value)

    @staticmethod
    def _max_memory(torch, device: str, n_ram_gb: float) -> Dict[Any, str]:
        """Build accelerate's max_memory map. accelerate wants integer GPU
        indexes (not "cuda"), "mps", "cpu" or "disk" as keys."""
        cpu_budget_gb = max(1.0, n_ram_gb * 0.5)  # heuristic safety margin
        max_memory: Dict[Any, str] = {"cpu": f"{cpu_budget_gb}GiB"}

        if device.startswith(("cuda", "xpu")):
            index = int(device.split(":", 1)[1]) if ":" in device else 0
            gpu_budget_gb = cpu_budget_gb
            if device.startswith("cuda"):
                try:
                    free_bytes, _total = torch.cuda.mem_get_info(index)
                    gpu_budget_gb = max(0.5, free_bytes / (1024 ** 3) - 1.0)  # keep 1 GB for activations
                except Exception as exc:
                    logger.debug("Could not read free VRAM: %s", exc)
            max_memory[index] = f"{gpu_budget_gb:.2f}GiB"
        elif device == "mps":
            max_memory["mps"] = f"{cpu_budget_gb}GiB"
        elif device != "cpu":
            raise ValueError(
                f"Unsupported device for the safetensors backend: {device!r} "
                "(use 'cpu', 'cuda', 'cuda:N', 'mps' or 'xpu'; ROCm GPUs are 'cuda')"
            )
        return max_memory

    @staticmethod
    def _guess_no_split_classes(model):
        """Find the repeated transformer-block class so accelerate never
        splits a single decoder layer's weights across two offload
        locations (which would break the "one layer in RAM at a time"
        guarantee)."""
        names = set()
        for module in model.modules():
            cls_name = module.__class__.__name__
            if "Layer" in cls_name or "Block" in cls_name:
                names.add(cls_name)
        return list(names) if names else None

    def _encode(self, prompt: str, max_tokens: int, add_special_tokens: bool = True):
        """Tokenize, fit prompt + output into n_ctx (see context.fit_to_context),
        and move the ids to the device the embedding runs on.

        Returns (inputs, max_tokens)."""
        inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
        ids = inputs["input_ids"]
        bos = getattr(self._tokenizer, "bos_token_id", None)
        first_is_bos = ids.shape[1] > 0 and bos is not None and int(ids[0, 0]) == bos
        keep, max_tokens = fit_to_context(ids.shape[1], max_tokens, self.n_ctx, first_is_bos)
        if keep is not None:
            head, tail = keep
            inputs = {k: self._torch.cat([v[:, :head], v[:, v.shape[1] - tail:]], dim=1) for k, v in inputs.items()}
        target = getattr(self, "_input_device", "cpu")
        if target != "cpu":
            inputs = {k: v.to(target) for k, v in inputs.items()}
        return dict(inputs), max_tokens

    def _generation_kwargs(self, max_tokens: int, temperature: float, top_p: float) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(max_new_tokens=max_tokens, pad_token_id=self._tokenizer.eos_token_id)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kwargs.update(do_sample=False)
        return kwargs

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        callback: Optional[Callable[[str], None]] = None,
        add_special_tokens: bool = True,
    ) -> str:
        """Generate a completion. `callback`, if given, receives the full text
        once at the end (use stream_generate for per-token callbacks)."""
        torch = self._torch
        inputs, max_tokens = self._encode(prompt, max_tokens, add_special_tokens)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **self._generation_kwargs(max_tokens, temperature, top_p))
        text = self._tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gc.collect()
        if callback is not None:
            callback(text)
        return text

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        callback: Optional[Callable[[str], None]] = None,
        add_special_tokens: bool = True,
    ) -> Iterator[str]:
        """Token-by-token streaming using transformers' TextIteratorStreamer.

        Errors raised inside the background generation thread are re-raised
        here; breaking out of the loop early stops generation.
        """
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
        import threading

        torch = self._torch
        inputs, max_tokens = self._encode(prompt, max_tokens, add_special_tokens)
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        cancel = threading.Event()

        class _StopOnCancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return torch.full((input_ids.shape[0],), cancel.is_set(), dtype=torch.bool, device=input_ids.device)

        gen_kwargs = dict(
            **inputs,
            **self._generation_kwargs(max_tokens, temperature, top_p),
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([_StopOnCancel()]),
        )
        errors: List[BaseException] = []

        def _run():
            try:
                with torch.no_grad():
                    self._model.generate(**gen_kwargs)
            except BaseException as exc:
                errors.append(exc)
                streamer.end()  # unblock the consumer loop below

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        try:
            for new_text in streamer:
                if not new_text:  # the streamer emits "" while it buffers partial words
                    continue
                if callback is not None:
                    callback(new_text)
                yield new_text
        finally:
            cancel.set()
            thread.join()
            gc.collect()
        if errors:
            raise RuntimeError(f"Generation failed: {errors[0]}") from errors[0]

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Chat using the checkpoint's own chat template when it has one.

        If the conversation doesn't fit n_ctx, the oldest messages are
        dropped (system messages are kept); see context.fit_messages.
        """
        # The rendered template already contains BOS/special tokens.
        templated = bool(getattr(self._tokenizer, "chat_template", None))
        if not templated:
            logger.warning("Tokenizer has no chat template; falling back to a plain 'role: content' prompt")

        def render(msgs) -> str:
            if templated:
                return self._tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            return "\n".join(f"{m['role']}: {m['content']}" for m in msgs) + "\nassistant:"

        def count(msgs) -> int:
            # verbose=False: an over-long candidate is expected here, don't warn about it.
            ids = self._tokenizer(render(msgs), add_special_tokens=not templated, verbose=False)["input_ids"]
            return len(ids)

        messages, max_tokens = fit_messages(messages, count, max_tokens, self.n_ctx)
        return self.generate(
            render(messages), max_tokens=max_tokens, temperature=temperature, add_special_tokens=not templated
        )

    def benchmark(self, prompt: str, max_tokens: int = 64) -> Dict[str, Any]:
        """Greedy-decode and time real tokens (see GGUFStreamingEngine.benchmark)."""
        from transformers.generation.streamers import BaseStreamer

        torch = self._torch
        inputs, max_tokens = self._encode(prompt, max_tokens)

        class _TokenTimer(BaseStreamer):
            def __init__(self):
                self.seen_prompt = False
                self.first_token_at = None
                self.tokens = 0

            def put(self, value):
                if not self.seen_prompt:  # generate() first sends the prompt ids
                    self.seen_prompt = True
                    return
                if self.first_token_at is None:
                    self.first_token_at = time.perf_counter()
                self.tokens += int(value.numel())

            def end(self):
                pass

        timer = _TokenTimer()
        start = time.perf_counter()
        with torch.no_grad():
            self._model.generate(**inputs, **self._generation_kwargs(max_tokens, 0.0, 1.0), streamer=timer)
        end = time.perf_counter()

        first = timer.first_token_at or end
        decode_tokens = max(timer.tokens - 1, 0)
        return {
            "ttft_s": round(first - start, 4),
            "total_s": round(end - start, 4),
            "tokens_per_sec": round(decode_tokens / (end - first), 2) if decode_tokens and end > first else 0.0,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": timer.tokens,
            "backend": "safetensors",
            "device": self.device,
        }
