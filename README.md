# digvijay_llm

Run big open-source LLMs — including 70B-class models — on machines with as little as **16 GB of RAM**, by streaming weights from disk ("ROM") instead of loading the full model into RAM.

## ⚠️ Read this first (important honesty note)

A CPU/GPU can only *compute* on data sitting in RAM/VRAM — it is physically impossible to run matrix multiplications directly on a disk. What this library actually does, and what makes low-RAM 70B inference possible, is **disk-backed weight streaming**: weights live on disk and are pulled into RAM only for the layer currently being computed, then released. Peak RAM usage becomes roughly "one layer's worth of weights + activations" instead of "the entire model's weights." The tradeoff is speed — disk I/O is much slower than RAM, so expect lower tokens/sec than a fully-in-RAM setup, especially on HDDs. An NVMe SSD is strongly recommended.

## How it works

`digvijay_llm` ships two backends:

| Backend | File format | Mechanism | Best for |
|---|---|---|---|
| `GGUFStreamingEngine` | `.gguf` (quantized) | Wraps `llama.cpp` via `llama-cpp-python`, using `mmap()` so the OS pages weights in/out of RAM on demand | Production use — fastest, most memory-efficient, supports 4-bit/5-bit quantization |
| `SafetensorsLayerStreamer` | raw HuggingFace `.safetensors` checkpoints | Builds the model with empty weights, then uses `accelerate`'s disk-offload hooks to load **one transformer layer at a time** from disk right before it's needed | When you only have an unconverted HF checkpoint and don't want to quantize/convert to GGUF first |

Both are wrapped behind one simple class, `LowRAMLLM`, so your code doesn't need to care which backend is active.

## Install

```bash
pip install -r requirements.txt
# or, pick only what you need:
pip install digvijay_llm[gguf]   # GGUF backend only (lighter install)
pip install digvijay_llm[hf]     # safetensors backend only
pip install digvijay_llm[all]    # both
```

## Quick start

### Option A — GGUF model (recommended, fastest)

```python
from digvijay_llm import LowRAMLLM

llm = LowRAMLLM.from_gguf("models/llama-3-70b.Q4_K_M.gguf", n_ram_gb=16)
print(llm.generate("Explain quantum entanglement simply."))

# streaming token-by-token
for token in llm.stream("Write a short poem about the sea."):
    print(token, end="", flush=True)

# chat interface
reply = llm.chat([{"role": "user", "content": "Hello!"}])
print(reply)
```

Get a `.gguf` file either by quantizing a model yourself with `llama.cpp`'s `convert_hf_to_gguf.py` + `llama-quantize`, or by downloading an already-quantized `.gguf` from a model hub.

### Option B — raw HuggingFace safetensors checkpoint (no conversion needed)

```python
from digvijay_llm import LowRAMLLM

llm = LowRAMLLM.from_safetensors("models/llama-3-70b-hf/", n_ram_gb=16)
print(llm.generate("Write a haiku about the ocean."))
```

### Option C — auto-detect

```python
from digvijay_llm import LowRAMLLM

llm = LowRAMLLM.auto("models/llama-3-70b.Q4_K_M.gguf", n_ram_gb=16)
# or
llm = LowRAMLLM.auto("models/llama-3-70b-hf/", n_ram_gb=16)
```

## Planning your RAM budget

```python
from digvijay_llm import plan_for_budget

plan = plan_for_budget(
    model_path="models/llama-3-70b-hf/",
    total_params_billion=70,
    n_layers=80,
    bytes_per_param=2.0,  # fp16
    n_ram_gb=16,
)
print(plan)
for w in plan.warnings:
    print("WARNING:", w)
```

This estimates how many transformer layers can safely stay resident in RAM at once given your budget, and warns you if your disk doesn't have enough free space for the model or if your RAM is so tight that throughput will suffer badly.

## Practical tips for 70B-on-16GB

1. **Use a quantized GGUF (Q4_K_M or smaller)** — this is by far the biggest lever. A 70B model at Q4 is roughly ~38–40 GB on disk vs ~140 GB at fp16; mmap streaming then only needs to page small chunks at a time.
2. **Use an NVMe SSD**, not a spinning HDD or network drive — random-access read speed directly determines your tokens/sec.
3. **Keep context length modest** (the KV cache also eats RAM) — `digvijay_llm` auto-shrinks context size on tight budgets, but you can override it.
4. **Close other RAM-heavy apps** while running — the OS needs free RAM to use as page cache for the mmap'd weights.
5. **`n_gpu_layers`** — if you have even a small GPU, offloading a few layers to VRAM (`n_gpu_layers=N`) reduces disk pressure further.

## Project layout

```
digvijay_llm/
├── __init__.py            # public API surface
├── api.py                 # LowRAMLLM unified class
├── engine_gguf.py          # llama.cpp mmap-based backend
├── engine_safetensors.py   # accelerate disk-offload layer-streaming backend
└── ram_planner.py          # RAM budget heuristics
examples/
├── run_gguf_example.py
└── run_safetensors_example.py
tests/
└── test_ram_planner.py
```

## License

MIT
