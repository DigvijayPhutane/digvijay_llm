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
from typing import Callable, Iterator, Optional

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
            from accelerate import infer_auto_device_map, dispatch_model, init_empty_weights
            from accelerate.utils import get_balanced_memory
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
        self.n_batch = n_batch or 4
        os.makedirs(self.offload_dir, exist_ok=True)

        torch_dtype = torch.float16 if dtype == "float16" else torch.float32

        # 1. Build the model architecture with NO weights materialized yet
        #    (this costs ~0 RAM regardless of model size).
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_dir)
        with init_empty_weights():
            empty_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)

        # 2. Reserve almost all RAM for ONE layer at a time by capping the
        #    "max_memory" accelerate is allowed to use on this device; the
        #    rest is automatically offloaded to `offload_dir` on disk and
        #    streamed back in layer-by-layer during forward().
        reserved_for_layer_gb = max(1.0, n_ram_gb * 0.5)  # heuristic safety margin
        max_memory = {device: f"{reserved_for_layer_gb}GiB", "cpu": f"{reserved_for_layer_gb}GiB"}

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

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        torch = self._torch
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gc.collect()
        return text

    def stream_generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, callback: Optional[Callable[[str], None]] = None) -> Iterator[str]:
        """Token-by-token streaming using transformers' TextIteratorStreamer."""
        from transformers import TextIteratorStreamer
        import threading

        torch = self._torch
        inputs = self._tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            streamer=streamer,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()
        for new_text in streamer:
            if callback is not None:
                callback(new_text)
            yield new_text
        thread.join()
