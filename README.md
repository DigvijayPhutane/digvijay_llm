# digvijay_llm

Run big open-source LLMs — including 70B-class models — on machines with as little as **16 GB of RAM**, by streaming weights from disk instead of loading the full model into RAM.

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

## Step 1 — Detect your parameters (run this first!)

Not sure what values to use for your machine? Run this before anything else:

```bash
pip install psutil
python detect_params.py
```

It auto-detects your RAM, GPU, VRAM, disk space, and CPU cores, then prints the **exact parameters to copy-paste** into your code:

```
📦 YOUR SYSTEM
   RAM   : 16.0 GB total  |  12.3 GB free
   Disk  : 512.0 GB total |  180.0 GB free
   GPU   : NVIDIA RTX 3060  (CUDA)
   VRAM  : 12.0 GB total  |  11.2 GB free

[70B model]  ✅ CAN RUN  ⚡ Medium

   👉 Use these parameters:
      llm = LowRAMLLM.from_gguf(
          "path/to/model.Q4_K_M.gguf",
          n_ram_gb     = 16,
          n_gpu_layers = 45,
          n_threads    = 8,
      )
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
    n_ram_gb     = 16,
    n_gpu_layers = 0,    # set this from detect_params.py output
    n_threads    = 8,    # set this from detect_params.py output
)

# Generate text
print(llm.generate("Explain quantum entanglement simply."))

# Stream token by token
for token in llm.stream("Write a short poem about the sea."):
    print(token, end="", flush=True)

# Chat interface
reply = llm.chat([{"role": "user", "content": "Hello!"}])
print(reply)
```

### Option B — Raw HuggingFace safetensors checkpoint (no conversion needed)

```python
from digvijay_llm import LowRAMLLM

llm = LowRAMLLM.from_safetensors("models/llama-3-70b-hf/", n_ram_gb=16)
print(llm.generate("Write a haiku about the ocean."))
```

### Option C — Auto-detect format

```python
from digvijay_llm import LowRAMLLM

llm = LowRAMLLM.auto("models/llama-3-70b.Q4_K_M.gguf", n_ram_gb=16)
# or
llm = LowRAMLLM.auto("models/llama-3-70b-hf/", n_ram_gb=16)
```

---

## GPU support

digvijay_llm works on **every machine** — no GPU, small GPU, or big GPU. Just set `n_gpu_layers` accordingly (or let `detect_params.py` pick it for you):

| Your GPU VRAM | n_gpu_layers |
|---|---|
| No GPU | 0 |
| 4 GB | 10–15 |
| 6 GB | 20–25 |
| 8 GB | 30–35 |
| 12 GB | 45–50 |
| 16 GB | 60–65 |
| 24 GB+ | 80 (all layers, no disk streaming needed) |

Layers that fit in VRAM run at full GPU speed. Layers that don't fit are streamed from disk automatically. You always get the best of both.

```python
# No GPU
llm = LowRAMLLM.from_gguf("model.gguf", n_ram_gb=16, n_gpu_layers=0)

# Small GPU (8GB VRAM)
llm = LowRAMLLM.from_gguf("model.gguf", n_ram_gb=16, n_gpu_layers=35)

# Big GPU (24GB+ VRAM) — everything fits, no disk streaming needed
llm = LowRAMLLM.from_gguf("model.gguf", n_ram_gb=16, n_gpu_layers=80)
```

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

Warns you if your disk doesn't have enough free space or your RAM is too tight before you even try to load the model.

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

## Project layout

```
digvijay_llm/               ← repo root
├── digvijay_llm/           ← the package
│   ├── __init__.py
│   ├── api.py
│   ├── engine_gguf.py
│   ├── engine_safetensors.py
│   └── ram_planner.py
├── examples/
│   ├── run_gguf_example.py
│   └── run_safetensors_example.py
├── tests/
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
