"""
Example: run a raw HuggingFace safetensors checkpoint (no GGUF conversion)
on a 16GB RAM machine, streaming one transformer layer at a time from disk.

Before running:
    pip install digvijay_llm[hf]
    huggingface-cli download <org>/<model> --local-dir models/my-model-hf
"""

from digvijay_llm import LowRAMLLM

MODEL_DIR = "models/my-model-hf"

llm = LowRAMLLM.from_safetensors(MODEL_DIR, n_ram_gb=16, dtype="float16")
print(llm)

print("\n--- Generation ---")
print(llm.generate("Write a haiku about the ocean.", max_tokens=60))

print("\n--- Streaming generation ---")
for token in llm.stream("List 3 facts about Mars.", max_tokens=80):
    print(token, end="", flush=True)
print()
