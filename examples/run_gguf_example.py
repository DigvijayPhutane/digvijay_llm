"""
Example: run a quantized 70B GGUF model on a 16GB RAM machine.

Before running:
    pip install digvijay_llm[gguf]
    # download/produce a .gguf model, e.g. llama-3-70b.Q4_K_M.gguf
"""

from digvijay_llm import LowRAMLLM, plan_for_budget

MODEL_PATH = "models/llama-3-70b.Q4_K_M.gguf"

# Optional: sanity-check your RAM budget first
plan = plan_for_budget(
    model_path=MODEL_PATH,
    total_params_billion=70,
    n_layers=80,
    bytes_per_param=0.6,  # ~Q4 quantized
    n_ram_gb=16,
)
print(plan)
for w in plan.warnings:
    print("WARNING:", w)

llm = LowRAMLLM.from_gguf(MODEL_PATH, n_ram_gb=16, n_gpu_layers=0)
print(llm)

print("\n--- Non-streaming generation ---")
print(llm.generate("Explain quantum entanglement simply.", max_tokens=150))

print("\n--- Streaming generation ---")
for token in llm.stream("Write a 4 line poem about the ocean.", max_tokens=80):
    print(token, end="", flush=True)
print()

print("\n--- Chat ---")
reply = llm.chat([{"role": "user", "content": "Say hello in 3 languages."}])
print(reply)
