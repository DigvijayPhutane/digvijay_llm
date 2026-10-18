# digvijay_llm

Run big open-source LLMs — including 70B-class models — on machines with as little as **16 GB of RAM**, by streaming weights from disk instead of loading the full model into RAM.

This version is more production-oriented: it can detect your hardware automatically, choose sensible defaults for backend/device/threading/context, expose the detected configuration through `llm.config`, and support streaming plus basic benchmarking.

```bash
pip install digvijay_llm
```

---

## ⚠️ Read this first (important honesty note)

A CPU/GPU can only *compute* on data sitting in RAM/VRAM — it is physically impossible to run matrix multiplications directly on a disk. What this library actually does, and what makes low-RAM 70B inference possible, is **disk-backed weight streaming**: weights live on disk and are pulled into RAM only for the layer currently being computed, then released.

Peak RAM usage becomes roughly "one layer's worth of weights + activations" instead of "the entire model's weights." The tradeoff is speed — disk I/O is much slower than RAM, so expect lower tokens/sec than a fully-in-RAM setup, especially on HDDs. An NVMe SSD is strongly recommended.

---

## How it works

`digvijay_llm` ships two backends:

| Backend | File format | Mechanism | Best for |
|---|---|---|---|
| `GGUFStreamingEngine` | `.gguf` (quantized) | Wraps `llama.cpp` via `llama-cpp-python`, using `mmap()` so the OS pages weights in/out of RAM on demand | Production use — fastest, most memory-efficient, supports 4-bit/5-bit quantization |
| `SafetensorsLayerStreamer` | raw HuggingFace `.safetensors` checkpoints | Builds the model with empty weights, then uses `accelerate`'s disk-offload hooks to load **one transformer layer at a time** from disk right before it's needed | When you only have an unconverted HF checkpoint and don't want to quantize/convert to GGUF first |

Both are wrapped behind one simple class, `LowRAMLLM`, so your code doesn't need to care which backend is active.

---

## Install

```bash
pip install digvijay_llm

# then install the backend you need:
pip install digvijay_llm[gguf]   # GGUF backend only (recommended, lighter)
pip install digvijay_llm[hf]     # safetensors / HuggingFace backend only
pip install digvijay_llm[all]    # both backends
```

---

## Step 1 — Automatic hardware detection

You no longer need to run a separate script first. `digvijay_llm` can detect your machine automatically when you create a model instance.

```python
from digvijay_llm import LowRAMLLM

llm = LowRAMLLM.from_gguf("models/llama-70b.Q4_K_M.gguf")
print(llm.config)
print(llm.device_info)
```

The library will infer a sensible configuration for:

- backend selection
- GPU vs CPU placement
- `n_gpu_layers`
- `n_threads`
- `n_ctx`: sized from your `n_ram_gb` budget, and never larger than the context the model was trained for
- `n_batch`
- `n_ram_gb`

You can still override any of these manually when needed.

If a prompt doesn't fit the context window, the library keeps its most recent tokens (and the leading BOS token) and logs a warning. If `max_tokens` is larger than the room left, it is capped, the same way llama.cpp does it. Both backends follow the same rule.

`chat()` trims whole messages instead, so the chat template stays intact. When a conversation outgrows the context, the oldest turns are dropped, the system message is always kept, and a warning says how many were dropped. If the newest message is too long on its own, you get a clear error.

To see what will be picked without loading a model, run the standalone report. It uses the same detection code:

```bash
python detect_params.py                      # your hardware + typical model sizes
python detect_params.py models/model.gguf    # exact settings for one model
```

---

## Step 2 — Get a model

Go to [huggingface.co](https://huggingface.co) and search for `llama 70b GGUF`. Download the `Q4_K_M` version. Recommended repos:

- `bartowski/Meta-Llama-3.1-70B-Instruct-GGUF`
- `TheBloke/Llama-2-70B-GGUF`

Place the `.gguf` file anywhere on your machine, e.g. `models/llama-70b.Q4_K_M.gguf`

---

## Step 3 — Write your code

### Option A — GGUF model (recommended, fastest)

```python
from digvijay_llm import LowRAMLLM

llm = LowRAMLLM.from_gguf(
    "models/llama-70b.Q4_K_M.gguf",
    detect=True,      # default: auto-detect and tune the config
    n_ram_gb=16,
    # n_gpu_layers left unset = auto; 0 forces CPU, -1 offloads every layer
    n_threads=8,
)

# Generate text
print(llm.generate("Explain quantum entanglement simply."))

# Stream token by token
for token in llm.stream("Write a short poem about the sea."):
    print(token, end="", flush=True)

# Chat interface
reply = llm.chat([{"role": "user", "content": "Hello!"}])
print(reply)

# Basic benchmark summary
print(llm.benchmark("Summarize the benefits of local inference."))
```

### Option B — Raw HuggingFace safetensors checkpoint (no conversion needed)

```python
from digvijay_llm import LowRAMLLM

llm = LowRAMLLM.from_safetensors("models/llama-3-70b-hf/", detect=True, n_ram_gb=16)
print(llm.generate("Write a haiku about the ocean."))
```

### Option C — Auto-detect format

```python
from digvijay_llm import LowRAMLLM

llm = LowRAMLLM.auto("models/llama-3-70b.Q4_K_M.gguf", n_ram_gb=16)
# or
llm = LowRAMLLM.auto("models/llama-3-70b-hf/", n_ram_gb=16)
```

The same auto-detection logic applies here; the library will inspect the current machine and choose sensible runtime settings.

---

## New in this version

- Built-in hardware detection for CPU, CUDA, ROCm, Metal, and Intel XPU-style backends
- Automatic tuning for `n_gpu_layers`, `n_threads`, `n_ctx`, `n_batch`, and `n_ram_gb`
- `llm.config` and `llm.device_info` for inspecting the detected settings
- Streaming with callback support via `llm.stream(...)`
- `llm.benchmark(...)` counts real model tokens and reports:
  - `ttft_s`: time to first token.
  - `total_s`: total time.
  - `tokens_per_sec`: decode rate after the first token.
  - `prompt_tokens` and `completion_tokens`.
- Clearer runtime error handling for missing optional dependencies

---

## GPU support

digvijay_llm works on **every machine**, whether it has no GPU, a small one or a big one. If you leave `n_gpu_layers` unset, the library reads the model's file size and layer count (from the GGUF header). It then compares them with free VRAM, keeping 1 GB of headroom:

- If the whole model fits, it uses `n_gpu_layers=-1`, which offloads every layer.
- Otherwise it offloads the share of layers that fits, and llama.cpp keeps the rest on the CPU.
- With no usable GPU it uses `0`.

```python
# Auto (recommended)
llm = LowRAMLLM.from_gguf("model.gguf", n_ram_gb=16)

# Force CPU
llm = LowRAMLLM.from_gguf("model.gguf", n_ram_gb=16, n_gpu_layers=0)

# Offload every layer
llm = LowRAMLLM.from_gguf("model.gguf", n_ram_gb=16, n_gpu_layers=-1)

# Manual split
llm = LowRAMLLM.from_gguf("model.gguf", n_ram_gb=16, n_gpu_layers=35)
```

`llm.config["warnings"]` holds anything detection noticed, and each warning is also logged.

### You need GPU builds of the backends

The default wheels from PyPI are CPU-only on Windows and Linux. If `nvidia-smi` finds a GPU but the installed backend can't use it, digvijay_llm warns and falls back to the CPU.

- **GGUF (llama-cpp-python):** build it with CUDA:
  ```bash
  CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
  ```
  (Use `-DGGML_METAL=on` on macOS, or `-DGGML_HIPBLAS=on` for ROCm.)
- **Safetensors (PyTorch):** install a CUDA wheel from the PyTorch index, for example:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  ```
  Check it with `python -c "import torch; print(torch.cuda.is_available())"`. AMD ROCm GPUs appear as `cuda` in PyTorch.

---

## Optional — Check your RAM budget

```python
from digvijay_llm import plan_for_budget

plan = plan_for_budget(
    model_path           = "models/llama-3-70b-hf/",
    total_params_billion = 70,
    n_layers             = 80,
    bytes_per_param      = 2.0,   # fp16; use 0.6 for Q4 quantized
    n_ram_gb             = 16,
)
print(plan)
for w in plan.warnings:
    print("WARNING:", w)
```

This warns you, before you try to load the model, if your disk doesn't have enough free space or your RAM budget is too tight.

> **This is an estimate, not a guarantee.** It assumes each layer is total weights ÷ `n_layers`, which ignores that embeddings and the output head are usually larger than one block. It reserves a fixed overhead margin instead of modeling the real KV cache (which grows with `n_ctx`), activations, or framework overhead. Measure actual memory use before you rely on a plan.

---

## Minimum requirements

| | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| Disk | 45 GB free (for 70B Q4) | NVMe SSD |
| Python | 3.9+ | 3.11+ |
| OS | Windows / Mac / Linux | any |
| GPU | not required | any CUDA GPU helps |

---

## Practical tips for 70B on 16GB

1. **Use Q4_K_M quantization** — shrinks the model from ~140 GB (fp16) to ~38 GB on disk. Biggest single lever.
2. **Use an NVMe SSD**, not a spinning HDD — random-access read speed directly sets your tokens/sec.
3. **Keep context length modest** — the KV cache also eats RAM. `digvijay_llm` auto-shrinks context on tight budgets.
4. **Close other RAM-heavy apps** while running — the OS needs free RAM as page cache for the mmap'd weights.
5. **Any GPU helps** — even offloading a few layers to a small GPU (`n_gpu_layers=20`) meaningfully improves speed.

---

## Testing

```bash
python -m unittest discover -s tests
```

The unit tests use fakes for torch, nvidia-smi and llama.cpp, so they run anywhere. The tests that need torch skip themselves when it isn't installed. The end-to-end tests load real models and are skipped unless you point them at small checkpoints:

```bash
DIGVIJAY_LLM_TEST_GGUF=models/stories15M-q8_0.gguf \
DIGVIJAY_LLM_TEST_HF=models/tiny-random-LlamaForCausalLM \
DIGVIJAY_LLM_TEST_RAM_GB=4 \
python -m unittest tests.test_integration -v
```

They exercise generate, stream, early stop, chat and benchmark on each backend. Set `DIGVIJAY_LLM_TEST_RAM_GB` to also check peak memory.

---

## Project layout

```
digvijay_llm/               ← repo root
├── digvijay_llm/           ← the package
│   ├── __init__.py
│   ├── api.py
│   ├── context.py
│   ├── engine_gguf.py
│   ├── engine_safetensors.py
│   ├── hardware.py
│   └── ram_planner.py
├── examples/
│   ├── run_gguf_example.py
│   └── run_safetensors_example.py
├── tests/
│   ├── test_api_and_engines.py
│   ├── test_gpu_detection.py
│   ├── test_hardware_detection.py
│   ├── test_integration.py ← opt-in, needs real models
│   └── test_ram_planner.py
├── .github/
│   └── workflows/
│       └── publish.yml
├── detect_params.py        ← run this first!
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── README.md
└── setup.py
```

---

## License

MIT — see [LICENSE](LICENSE)

---

## Author

Built by [Digvijay Phutane](https://github.com/DigvijayPhutane)
