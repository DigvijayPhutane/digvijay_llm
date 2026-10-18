import os
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from digvijay_llm import api, hardware
from digvijay_llm.context import apply_keep, fit_messages, fit_to_context
from digvijay_llm.engine_gguf import GGUFStreamingEngine
from digvijay_llm.engine_safetensors import SafetensorsLayerStreamer
from digvijay_llm.hardware import SystemInfo

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401

    HAS_HF = True
except ImportError:
    HAS_HF = False


def gpu_box(llama_offload=True):
    return SystemInfo(
        os_family="windows", os_version="test", cpu_cores=8, cpu_model="x",
        ram_total_gb=16.0, ram_available_gb=10.0, storage_total_gb=100.0, storage_free_gb=50.0,
        gpu_name="NVIDIA GeForce RTX 4060", gpu_vendor="nvidia", vram_total_gb=8.0, vram_free_gb=6.0,
        llama_cpp_gpu_offload=llama_offload,
    )


class RecordingEngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TestFromGGUFGpuLayers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "tiny.gguf")
        with open(self.path, "wb") as f:
            f.write(b"\0" * 1024)
        for p in (
            mock.patch.object(hardware, "detect_system", return_value=gpu_box()),
            mock.patch("digvijay_llm.engine_gguf.GGUFStreamingEngine", RecordingEngine),
        ):
            p.start()
            self.addCleanup(p.stop)

    def test_default_auto_selects_gpu_layers(self):
        # Regression: the default was n_gpu_layers=0, which detect_params
        # treated as an explicit override, so GPU offload never happened.
        llm = api.LowRAMLLM.from_gguf(self.path)
        self.assertEqual(llm._engine.kwargs["n_gpu_layers"], -1)
        self.assertEqual(llm.config["device"], "cuda")

    def test_explicit_zero_forces_cpu(self):
        llm = api.LowRAMLLM.from_gguf(self.path, n_gpu_layers=0)
        self.assertEqual(llm._engine.kwargs["n_gpu_layers"], 0)

    def test_explicit_value_is_kept(self):
        llm = api.LowRAMLLM.from_gguf(self.path, n_gpu_layers=12, n_ram_gb=8)
        self.assertEqual(llm._engine.kwargs["n_gpu_layers"], 12)
        self.assertEqual(llm._engine.kwargs["n_ram_gb"], 8)

    def test_auto_passes_through(self):
        llm = api.LowRAMLLM.auto(self.path)
        self.assertEqual(llm._engine.kwargs["n_gpu_layers"], -1)

    def test_detect_false_uses_cpu(self):
        llm = api.LowRAMLLM.from_gguf(self.path, detect=False)
        self.assertEqual(llm._engine.kwargs["n_gpu_layers"], 0)
        self.assertGreaterEqual(llm._engine.kwargs["n_threads"], 1)


class FakeLlama:
    """Mimics the llama_cpp.Llama methods the benchmark uses."""

    BOS = 1
    EOS = 2

    def __init__(self, tokens):
        self._tokens = tokens
        self.calls = []

    def tokenize(self, data, add_bos=True, special=False):
        return ([self.BOS] if add_bos else []) + list(range(100, 100 + len(data.split())))

    def token_bos(self):
        return self.BOS

    def token_eos(self):
        return self.EOS

    def generate(self, tokens, **kwargs):
        self.calls.append(list(tokens))
        yield from self._tokens

    def __call__(self, prompt, max_tokens, stream=False, **kwargs):
        self.calls.append((prompt, max_tokens))
        if stream:
            return iter([{"choices": [{"text": "a"}]}, {"choices": [{"text": "b"}]}])
        return {"choices": [{"text": "ab"}]}

    @staticmethod
    def chat_handler(llama, messages, **kwargs):
        """Like llama.cpp's handlers: render, tokenize, then create_completion.
        The fake template costs 2 tokens per message plus BOS."""
        text = " ".join("<t> " + m["content"] for m in messages)
        prompt = llama.tokenize(text.encode("utf-8"))
        return llama.create_completion(prompt=prompt, max_tokens=kwargs.get("max_tokens"))

    def create_completion(self, prompt, max_tokens=None, **kwargs):
        return {"choices": [{"text": "unused"}]}

    def create_chat_completion(self, messages, max_tokens, **kwargs):
        self.calls.append(("chat", list(messages), max_tokens))
        return {"choices": [{"message": {"content": "hello"}}]}


def gguf_engine(tokens=(), n_ctx=2048):
    engine = object.__new__(GGUFStreamingEngine)
    engine._llm = FakeLlama(list(tokens))
    engine._ctx_size = n_ctx
    engine.device = "cpu"
    engine.n_gpu_layers = 0
    return engine


class TestGGUFBenchmark(unittest.TestCase):
    def test_counts_model_tokens_not_words(self):
        # One "word" made of several tokens: the old split() count said 1.
        engine = gguf_engine([10, 11, 12, 13, FakeLlama.EOS])
        result = engine.benchmark("hello world there", max_tokens=64)
        self.assertEqual(result["prompt_tokens"], 4)  # BOS + 3
        self.assertEqual(result["completion_tokens"], 4)
        self.assertLessEqual(result["ttft_s"], result["total_s"])

    def test_respects_max_tokens(self):
        engine = gguf_engine(list(range(10, 50)))
        self.assertEqual(engine.benchmark("hi", max_tokens=5)["completion_tokens"], 5)

    def test_max_tokens_capped_to_context(self):
        # Regression: max_tokens past the context made llama_decode fail.
        engine = gguf_engine(list(range(10, 500)), n_ctx=32)
        result = engine.benchmark("a b c", max_tokens=5000)
        self.assertEqual(result["completion_tokens"], 32 - 4)

    def test_immediate_eos(self):
        result = gguf_engine([FakeLlama.EOS]).benchmark("hi")
        self.assertEqual(result["completion_tokens"], 0)
        self.assertEqual(result["tokens_per_sec"], 0.0)


class TestGGUFGenerate(unittest.TestCase):
    def test_generate_calls_callback(self):
        # Regression: the non-streaming GGUF path ignored callback.
        seen = []
        engine = gguf_engine()
        self.assertEqual(engine.generate("hi", max_tokens=4, callback=seen.append), "ab")
        self.assertEqual(seen, ["ab"])

    def test_prompt_that_fits_is_passed_as_text(self):
        engine = gguf_engine()
        engine.generate("a b c", max_tokens=8)
        self.assertEqual(engine._llm.calls[-1], ("a b c", 8))

    def test_long_prompt_truncated_keeping_bos(self):
        # Regression: llama.cpp raised "Requested tokens exceed context window".
        engine = gguf_engine(n_ctx=16)
        list(engine.generate(" ".join(["w"] * 40), max_tokens=4, stream=True))
        prompt, max_tokens = engine._llm.calls[-1]
        self.assertEqual(len(prompt), 12)
        self.assertEqual(prompt[0], FakeLlama.BOS)
        self.assertEqual(prompt[-1], 100 + 39)  # most recent token kept
        self.assertEqual(max_tokens, 4)


class TestFitToContext(unittest.TestCase):
    def test_fits(self):
        self.assertEqual(fit_to_context(10, 20, 100), (None, 20))

    def test_max_tokens_capped(self):
        self.assertEqual(fit_to_context(10, 500, 100), (None, 90))

    def test_long_prompt_reserves_room(self):
        with self.assertLogs("digvijay_llm", level="WARNING"):
            keep, max_tokens = fit_to_context(500, 500, 100, first_is_bos=True)
        self.assertEqual(keep, (1, 49))
        self.assertEqual(max_tokens, 50)

    def test_apply_keep(self):
        self.assertEqual(apply_keep([1, 5, 6, 7, 8], (1, 2)), [1, 7, 8])
        self.assertEqual(apply_keep([5, 6], None), [5, 6])

    def test_tiny_context_rejected(self):
        with self.assertRaises(ValueError):
            fit_to_context(1, 1, 1)


def words(n, role="user"):
    return {"role": role, "content": " ".join(["w"] * n)}


def count_words(messages):
    return sum(len(m["content"].split()) for m in messages)


class TestFitMessages(unittest.TestCase):
    def test_fits_unchanged(self):
        msgs = [words(3, "system"), words(5), words(5, "assistant"), words(5)]
        self.assertEqual(fit_messages(msgs, count_words, 10, 100), (msgs, 10))

    def test_drops_oldest_keeps_system_and_starts_at_user(self):
        msgs = [words(3, "system"), words(40), words(40, "assistant"), words(10), words(10, "assistant"), words(5)]
        with self.assertLogs("digvijay_llm", level="WARNING"):
            kept, max_tokens = fit_messages(msgs, count_words, 20, 60)
        self.assertEqual(kept, [msgs[0], msgs[3], msgs[4], msgs[5]])  # 28 tokens fit in 40
        self.assertEqual(max_tokens, 20)

    def test_max_tokens_capped(self):
        kept, max_tokens = fit_messages([words(30)], count_words, 500, 100)
        self.assertEqual(max_tokens, 70)

    def test_falls_back_to_newest_message(self):
        msgs = [words(30), words(30, "assistant"), words(5, "assistant")]
        with self.assertLogs("digvijay_llm", level="WARNING"):
            kept, _ = fit_messages(msgs, count_words, 10, 40)
        self.assertEqual(kept, [msgs[2]])

    def test_newest_message_too_long_raises(self):
        with self.assertRaises(ValueError) as ctx:
            fit_messages([words(5, "system"), words(5), words(200)], count_words, 10, 100)
        self.assertIn("newest message", str(ctx.exception))

    def test_few_token_counts_for_long_history(self):
        calls = []

        def counting(messages):
            calls.append(1)
            return count_words(messages)

        msgs = [words(10, "user" if i % 2 == 0 else "assistant") for i in range(2000)]
        with self.assertLogs("digvijay_llm", level="WARNING"):
            kept, _ = fit_messages(msgs, counting, 100, 1000)
        self.assertLessEqual(count_words(kept), 900)
        self.assertEqual(kept[0]["role"], "user")
        self.assertLess(len(calls), 20)


class TestGGUFChat(unittest.TestCase):
    def test_prompt_length_uses_chat_handler(self):
        engine = gguf_engine()
        # BOS + ("<t>" + 3 words) + ("<t>" + 1 word)
        self.assertEqual(engine._chat_prompt_len([words(3), words(1, "assistant")]), 1 + 4 + 2)

    def test_prompt_length_estimate_without_handler(self):
        engine = gguf_engine()
        engine._llm.chat_handler = None  # no handler and no chat_format: estimate
        self.assertEqual(engine._chat_prompt_len([words(3)]), 3 + 8)

    def test_long_chat_trimmed_before_llama_cpp(self):
        engine = gguf_engine(n_ctx=64)
        msgs = [words(4, "system")] + [words(10, "user" if i % 2 == 0 else "assistant") for i in range(20)]
        with self.assertLogs("digvijay_llm", level="WARNING"):
            self.assertEqual(engine.chat(msgs, max_tokens=16), "hello")
        _, sent, max_tokens = engine._llm.calls[-1]
        self.assertEqual(sent[0], msgs[0])
        self.assertEqual(sent[-1], msgs[-1])
        self.assertLessEqual(engine._chat_prompt_len(sent), 64 - 16)
        self.assertEqual(max_tokens, 16)

    def test_short_chat_passed_through(self):
        engine = gguf_engine()
        msgs = [words(3)]
        engine.chat(msgs, max_tokens=8)
        self.assertEqual(engine._llm.calls[-1], ("chat", msgs, 8))


class TestSafetensorsMaxMemory(unittest.TestCase):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(mem_get_info=lambda i: (6 * 1024**3, 8 * 1024**3))
    )

    def test_cuda_uses_integer_key_and_free_vram(self):
        mm = SafetensorsLayerStreamer._max_memory(self.fake_torch, "cuda", 16.0)
        self.assertEqual(mm, {"cpu": "8.0GiB", 0: "5.00GiB"})

    def test_cuda_index(self):
        mm = SafetensorsLayerStreamer._max_memory(self.fake_torch, "cuda:1", 16.0)
        self.assertIn(1, mm)

    def test_cpu_and_mps(self):
        self.assertEqual(SafetensorsLayerStreamer._max_memory(self.fake_torch, "cpu", 16.0), {"cpu": "8.0GiB"})
        self.assertIn("mps", SafetensorsLayerStreamer._max_memory(self.fake_torch, "mps", 16.0))

    def test_unknown_device_rejected(self):
        with self.assertRaises(ValueError):
            SafetensorsLayerStreamer._max_memory(self.fake_torch, "rocm", 16.0)


class TestSafetensorsHelpers(unittest.TestCase):
    def test_version_tuple(self):
        vt = SafetensorsLayerStreamer._version_tuple
        self.assertEqual(vt("4.56.0.dev0"), (4, 56))
        self.assertEqual(vt("5.0.0rc1"), (5, 0))
        self.assertEqual(vt("4.38.2"), (4, 38))
        self.assertGreaterEqual(vt("4.100.0"), (4, 56))

    def fake_model(self):
        embedding = object()
        modules = [("", object()), ("model", object()), ("model.embed_tokens", embedding), ("lm_head", object())]
        return types.SimpleNamespace(get_input_embeddings=lambda: embedding, named_modules=lambda: iter(modules))

    def test_embedding_device_from_device_map(self):
        dev = SafetensorsLayerStreamer._embedding_device
        model = self.fake_model()
        self.assertEqual(dev(model, {"model.embed_tokens": 0, "model.layers": "disk"}), "cuda:0")
        self.assertEqual(dev(model, {"model.embed_tokens": 1}, "xpu"), "xpu:1")
        self.assertEqual(dev(model, {"model": "mps"}), "mps")
        self.assertEqual(dev(model, {"": "cpu"}), "cpu")
        self.assertEqual(dev(model, {"model.embed_tokens": "disk"}), "cpu")
        # "model.embed" must not match "model.embed_tokens" by plain prefix.
        self.assertEqual(dev(model, {"model.embed": 0, "": "cpu"}), "cpu")


@unittest.skipUnless(HAS_HF, "torch and transformers are not installed")
class TestSafetensorsGeneration(unittest.TestCase):
    def make_engine(self, model_generate, n_ctx=64):
        import torch

        class Tok:
            eos_token_id = 0
            chat_template = None

            def __call__(self, prompt, return_tensors=None, add_special_tokens=True, **kwargs):
                ids = torch.arange(1, len(prompt.split()) + 1).unsqueeze(0)
                if return_tensors is None:  # like real tokenizers: plain lists
                    return {"input_ids": ids[0].tolist()}
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

            def decode(self, ids, skip_special_tokens=True):
                return " ".join(f"t{int(i)}" for i in ids)

        engine = object.__new__(SafetensorsLayerStreamer)
        engine._torch = torch
        engine._tokenizer = Tok()
        engine._model = types.SimpleNamespace(generate=model_generate)
        engine.n_ctx = n_ctx
        engine.device = "cpu"
        return engine

    def test_generate_calls_callback(self):
        import torch

        def generate(input_ids, **kwargs):
            return torch.cat([input_ids, torch.tensor([[7, 8]])], dim=1)

        seen = []
        text = self.make_engine(generate).generate("a b c", max_tokens=2, callback=seen.append)
        self.assertEqual(text, "t7 t8")
        self.assertEqual(seen, ["t7 t8"])

    def test_greedy_does_not_pass_sampling_args(self):
        import torch

        captured = {}

        def generate(input_ids, **kwargs):
            captured.update(kwargs)
            return input_ids

        self.make_engine(generate).generate("a", max_tokens=1, temperature=0)
        self.assertFalse(captured["do_sample"])
        self.assertNotIn("temperature", captured)

    def test_prompt_truncated_to_context(self):
        import torch

        captured = {}

        def generate(input_ids, **kwargs):
            captured["len"] = input_ids.shape[1]
            return input_ids

        with self.assertLogs("digvijay_llm", level="WARNING"):
            self.make_engine(generate, n_ctx=10).generate(" ".join(["w"] * 20), max_tokens=4)
        self.assertEqual(captured["len"], 6)

    def test_max_tokens_capped_instead_of_raising(self):
        # Regression: max_tokens >= n_ctx raised ValueError.
        captured = {}

        def generate(input_ids, **kwargs):
            captured.update(kwargs, len=input_ids.shape[1])
            return input_ids

        self.make_engine(generate, n_ctx=10).generate("a b c", max_tokens=1000)
        self.assertEqual(captured["len"], 3)
        self.assertEqual(captured["max_new_tokens"], 7)

    def test_long_chat_trimmed(self):
        captured = {}

        def generate(input_ids, **kwargs):
            captured.update(kwargs, len=input_ids.shape[1])
            return input_ids

        msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": "w w w w w w w w"} for i in range(21)]
        with self.assertLogs("digvijay_llm", level="WARNING") as logs:
            self.make_engine(generate, n_ctx=64).chat(msgs, max_tokens=16)
        self.assertTrue(any("dropped" in line for line in logs.output))
        self.assertFalse(any("keeping the last" in line for line in logs.output))  # no token-level cut
        self.assertLessEqual(captured["len"], 64 - 16)
        self.assertEqual(captured["max_new_tokens"], 16)

    def test_stream_reraises_thread_error(self):
        # Regression: an exception in the generation thread left the
        # consumer blocked on the streamer forever.
        def generate(**kwargs):
            raise MemoryError("out of memory")

        with self.assertRaises(RuntimeError) as ctx:
            list(self.make_engine(generate).stream_generate("a b", max_tokens=4))
        self.assertIsInstance(ctx.exception.__cause__, MemoryError)

    def test_stream_early_break_cancels(self):
        import threading
        import torch

        stopped = threading.Event()

        def generate(input_ids, streamer, stopping_criteria, **kwargs):
            streamer.put(input_ids)  # prompt, skipped
            for i in range(1000):
                streamer.put(torch.tensor([10 + i]))
                if bool(stopping_criteria[0](input_ids, None)[0]):
                    stopped.set()
                    break
            streamer.end()

        gen = self.make_engine(generate, n_ctx=2048).stream_generate("a b", max_tokens=1000)
        next(gen)
        gen.close()
        self.assertTrue(stopped.is_set())


if __name__ == "__main__":
    unittest.main()
