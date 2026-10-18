"""End-to-end tests against real models. Skipped unless pointed at one:

    DIGVIJAY_LLM_TEST_GGUF=path/to/small.gguf
    DIGVIJAY_LLM_TEST_HF=path/to/small-hf-checkpoint-dir
    DIGVIJAY_LLM_TEST_RAM_GB=4      # optional: assert peak RSS stays under this

Any small model works (e.g. a Q4 TinyLlama GGUF, or a tiny HF checkpoint
such as hf-internal-testing/tiny-random-LlamaForCausalLM).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from digvijay_llm import LowRAMLLM

GGUF_PATH = os.environ.get("DIGVIJAY_LLM_TEST_GGUF")
HF_DIR = os.environ.get("DIGVIJAY_LLM_TEST_HF")
RAM_LIMIT_GB = float(os.environ.get("DIGVIJAY_LLM_TEST_RAM_GB", "0") or 0)


def peak_rss_gb():
    import psutil

    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None) or info.rss  # peak_wset: Windows only
    return peak / (1024**3)


class BackendContract:
    """Shared checks every backend must pass on a real checkpoint."""

    llm = None

    def test_generate_returns_text(self):
        text = self.llm.generate("The capital of France is", max_tokens=8, temperature=0)
        self.assertIsInstance(text, str)
        self.assertTrue(text.strip())

    def test_stream_matches_callback(self):
        seen = []
        chunks = list(self.llm.stream("Count: one, two,", max_tokens=8, temperature=0, callback=seen.append))
        self.assertTrue(chunks)
        self.assertEqual(chunks, seen)

    def test_stream_can_stop_early(self):
        for _chunk in self.llm.stream("Tell me a long story.", max_tokens=64):
            break  # must not hang or raise

    def test_chat(self):
        reply = self.llm.chat([{"role": "user", "content": "Say hi."}], max_tokens=8, temperature=0)
        self.assertIsInstance(reply, str)

    def test_chat_history_longer_than_context(self):
        # 2x the context in history: the oldest turns must be dropped, not crash.
        turn = "Tell me more about the sea and the ships that sail on it. " * 3
        msgs = [{"role": "system", "content": "You are brief."}]
        for _ in range(40):
            msgs += [{"role": "user", "content": turn}, {"role": "assistant", "content": turn}]
        msgs.append({"role": "user", "content": "Say hi."})
        with self.assertLogs("digvijay_llm", level="WARNING"):
            reply = self.llm.chat(msgs, max_tokens=8, temperature=0)
        self.assertIsInstance(reply, str)

    def test_benchmark_counts_tokens(self):
        result = self.llm.benchmark("Hello", max_tokens=8)
        self.assertGreater(result["prompt_tokens"], 0)
        self.assertGreater(result["completion_tokens"], 0)
        self.assertLessEqual(result["completion_tokens"], 8)
        self.assertLessEqual(result["ttft_s"], result["total_s"])

    def test_ram_budget(self):
        if not RAM_LIMIT_GB:
            self.skipTest("DIGVIJAY_LLM_TEST_RAM_GB not set")
        self.assertLess(peak_rss_gb(), RAM_LIMIT_GB)


@unittest.skipUnless(GGUF_PATH, "DIGVIJAY_LLM_TEST_GGUF not set")
class TestGGUFEndToEnd(BackendContract, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm = LowRAMLLM.from_gguf(GGUF_PATH, n_ctx=512)
        print("\n[gguf] config:", {k: v for k, v in cls.llm.config.items() if k != "system_info"})


@unittest.skipUnless(HF_DIR, "DIGVIJAY_LLM_TEST_HF not set")
class TestSafetensorsEndToEnd(BackendContract, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm = LowRAMLLM.from_safetensors(HF_DIR, n_ctx=512, dtype="float32")
        print("\n[hf] config:", {k: v for k, v in cls.llm.config.items() if k != "system_info"})


if __name__ == "__main__":
    unittest.main()
