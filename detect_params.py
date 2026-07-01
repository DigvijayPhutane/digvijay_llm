"""
detect_params.py
================
Run this script BEFORE using digvijay_llm.
It will auto-detect your system (RAM, GPU, disk) and tell you
EXACTLY what parameters to pass to LowRAMLLM.

Usage:
    pip install psutil
    python detect_params.py
"""

import os
import sys

# ── 1. RAM ────────────────────────────────────────────────────────────────────
try:
    import psutil
    total_ram_gb  = psutil.virtual_memory().total  / (1024**3)
    avail_ram_gb  = psutil.virtual_memory().available / (1024**3)
except ImportError:
    print("psutil not found. Install it: pip install psutil")
    sys.exit(1)

# ── 2. GPU / VRAM ─────────────────────────────────────────────────────────────
gpu_name      = None
total_vram_gb = 0.0
free_vram_gb  = 0.0
gpu_backend   = None

try:
    import torch
    if torch.cuda.is_available():
        gpu_name      = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free_vram_gb  = (torch.cuda.get_device_properties(0).total_memory
                         - torch.cuda.memory_reserved(0)) / (1024**3)
        gpu_backend   = "CUDA (NVIDIA)"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_name      = "Apple Silicon (MPS)"
        gpu_backend   = "MPS (Apple)"
        # Apple unified memory — use ~40% of total RAM as "VRAM"
        total_vram_gb = total_ram_gb * 0.4
        free_vram_gb  = avail_ram_gb * 0.4
except ImportError:
    pass  # torch not installed — that's fine

# ── 3. Disk ───────────────────────────────────────────────────────────────────
import shutil
disk          = shutil.disk_usage(".")
disk_free_gb  = disk.free  / (1024**3)
disk_total_gb = disk.total / (1024**3)

# ── 4. CPU ────────────────────────────────────────────────────────────────────
cpu_cores = os.cpu_count()

# ── 5. Compute recommendations ────────────────────────────────────────────────
def recommend(total_ram, avail_ram, vram, disk_free):
    """
    Returns recommended digvijay_llm parameters for common model sizes.
    """
    results = {}

    models = [
        # (label,  params_B, n_layers, q4_size_gb)
        ("7B  model", 7,  32, 4.1),
        ("13B model", 13, 40, 7.9),
        ("34B model", 34, 60, 20.0),
        ("70B model", 70, 80, 38.0),
    ]

    for label, params, layers, q4_gb in models:
        layer_size_gb = q4_gb / layers
        fits_on_disk  = disk_free >= q4_gb

        # How many layers fit in VRAM?
        gpu_layers = 0
        if vram > 0:
            gpu_layers = min(layers, int((vram * 0.85) // layer_size_gb))

        # Remaining layers stream from disk via RAM
        cpu_layers    = layers - gpu_layers
        ram_needed_gb = layer_size_gb * min(cpu_layers, 3)  # 3 layers max hot in RAM

        can_run  = fits_on_disk and (avail_ram >= 2.0)
        speed    = "🚀 Fast"   if gpu_layers == layers \
              else "⚡ Medium" if gpu_layers > layers // 2 \
              else "🐢 Slow"

        results[label] = {
            "q4_size_gb"  : q4_gb,
            "fits_on_disk": fits_on_disk,
            "gpu_layers"  : gpu_layers,
            "n_threads"   : cpu_cores,
            "n_ram_gb"    : int(total_ram),
            "can_run"     : can_run,
            "speed"       : speed,
        }

    return results


recs = recommend(total_ram_gb, avail_ram_gb, free_vram_gb, disk_free_gb)

# ── 6. Print report ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("   digvijay_llm — System Detector & Parameter Recommender")
print("="*60)

print(f"\n📦 YOUR SYSTEM")
print(f"   RAM        : {total_ram_gb:.1f} GB total  |  {avail_ram_gb:.1f} GB free")
print(f"   Disk       : {disk_total_gb:.1f} GB total  |  {disk_free_gb:.1f} GB free")
print(f"   CPU cores  : {cpu_cores}")

if gpu_name:
    print(f"   GPU        : {gpu_name}  ({gpu_backend})")
    print(f"   VRAM       : {total_vram_gb:.1f} GB total  |  {free_vram_gb:.1f} GB free")
else:
    print(f"   GPU        : None detected (CPU-only mode)")

print("\n" + "-"*60)
print("   RECOMMENDED PARAMETERS PER MODEL SIZE")
print("-"*60)

for label, r in recs.items():
    status = "✅ CAN RUN" if r["can_run"] else "❌ CANNOT RUN"
    disk_s = f"needs {r['q4_size_gb']}GB" if not r["fits_on_disk"] else "fits on disk ✅"

    print(f"\n  [{label}]  {status}  {r['speed']}")
    print(f"   Disk    : {disk_s}")

    if r["can_run"]:
        print(f"""
   👉 Use these parameters:

      from digvijay_llm import LowRAMLLM

      llm = LowRAMLLM.from_gguf(
          "path/to/model.Q4_K_M.gguf",
          n_ram_gb     = {r['n_ram_gb']},
          n_gpu_layers = {r['gpu_layers']},   # {'all layers on GPU 🚀' if r['gpu_layers'] == int(label.split('B')[0].strip().split()[-1])*10 else f'{r["gpu_layers"]} layers on GPU, rest streamed from disk'}
          n_threads    = {r['n_threads']},
      )""")
    else:
        reasons = []
        if not r["fits_on_disk"]:
            reasons.append(f"need {r['q4_size_gb']}GB free disk (you have {disk_free_gb:.1f}GB)")
        if avail_ram_gb < 2.0:
            reasons.append(f"need at least 2GB free RAM (you have {avail_ram_gb:.1f}GB)")
        print(f"   Reason  : {' | '.join(reasons)}")

print("\n" + "-"*60)
print("   QUICK COPY-PASTE CHEATSHEET")
print("-"*60)
best = next((r for r in recs.values() if r["can_run"]), None)
if best:
    print(f"""
   from digvijay_llm import LowRAMLLM, plan_for_budget

   # Optional: check budget first
   plan = plan_for_budget(
       model_path        = "path/to/model.gguf",
       total_params_billion = 70,
       n_layers          = 80,
       n_ram_gb          = {best['n_ram_gb']},
   )
   print(plan)

   # Load and run
   llm = LowRAMLLM.from_gguf(
       "path/to/model.gguf",
       n_ram_gb     = {best['n_ram_gb']},
       n_gpu_layers = {best['gpu_layers']},
       n_threads    = {best['n_threads']},
   )
   print(llm.generate("Hello!"))
""")
else:
    print("\n   ⚠️  No models can run on this machine right now.")
    print("   Free up disk space or RAM and try again.\n")

print("="*60 + "\n")
