import os
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from digvijay_llm import hardware
from digvijay_llm.hardware import detect_params, detect_system, read_gguf_layer_count, recommend_config


def fake_torch(cuda=False, mps=False, xpu=False, hip=None, fail_on_name=False, total=8 * 1024**3, free=6 * 1024**3):
    """Build a stand-in `torch` module exposing only what hardware.py touches."""

    def get_device_name(_index):
        if fail_on_name:
            raise RuntimeError("driver error")
        return "Fake GPU"

    torch = types.ModuleType("torch")
    torch.__version__ = "2.4.0+fake"
    torch.version = types.SimpleNamespace(hip=hip)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        get_device_name=get_device_name,
        get_device_properties=lambda _i: types.SimpleNamespace(total_memory=total),
        mem_get_info=lambda _i: (free, total),
        memory_reserved=lambda _i: 0,
    )
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    torch.xpu = types.SimpleNamespace(
        is_available=lambda: xpu,
        get_device_properties=lambda _i: types.SimpleNamespace(total_memory=total),
    )
    return torch


def write_gguf(path, block_count=40, padding_bytes=0, context_length=None):
    """Write a minimal GGUF v3 header with a string array before block_count."""

    def gguf_str(s):
        b = s.encode()
        return struct.pack("<Q", len(b)) + b

    kvs = [
        gguf_str("general.architecture") + struct.pack("<I", 8) + gguf_str("llama"),
        gguf_str("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2) + gguf_str("a") + gguf_str("bc"),
        gguf_str("tokenizer.ggml.scores") + struct.pack("<IIQ", 9, 6, 2) + struct.pack("<ff", 0.0, 1.0),
        gguf_str("llama.block_count") + struct.pack("<I", 4) + struct.pack("<I", block_count),
        # A different arch's key must not be mistaken for this model's.
        gguf_str("clip.context_length") + struct.pack("<I", 4) + struct.pack("<I", 7),
    ]
    if context_length is not None:
        kvs.append(gguf_str("llama.context_length") + struct.pack("<I", 4) + struct.pack("<I", context_length))
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + b"".join(kvs))
        f.write(b"\0" * padding_bytes)


class DetectionPatches(unittest.TestCase):
    """Run each test with a controlled torch / nvidia-smi / llama.cpp view."""

    torch_module = None  # None -> "torch is not installed"
    smi = None
    llama_offload = None

    def setUp(self):
        patches = [
            mock.patch.dict(sys.modules, {"torch": self.torch_module}),
            mock.patch.object(hardware, "_probe_nvidia_smi", return_value=self.smi),
            mock.patch.object(hardware, "_llama_cpp_gpu_offload", return_value=self.llama_offload),
            mock.patch.object(hardware, "_get_ram_info", return_value=(32.0, 20.0)),
            mock.patch.object(hardware.platform, "machine", return_value="AMD64"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)


class TestAppleSilicon(DetectionPatches):
    torch_module = fake_torch(mps=True)

    def test_mps_branch_uses_detected_ram(self):
        # Regression: this branch used an undefined `vm` and raised NameError.
        info = detect_system()
        self.assertEqual(info.gpu_backend, "metal")
        self.assertAlmostEqual(info.vram_total_gb, 32.0 * 0.4)
        self.assertAlmostEqual(info.vram_free_gb, 20.0 * 0.4)
        self.assertTrue(info.torch_available)
        cfg = recommend_config(info, backend="safetensors")
        self.assertEqual(cfg["device"], "mps")


class TestIntelXPU(DetectionPatches):
    torch_module = fake_torch(xpu=True, total=16 * 1024**3)

    def test_xpu_branch_reads_device_memory(self):
        info = detect_system()
        self.assertEqual(info.gpu_backend, "xpu")
        self.assertAlmostEqual(info.vram_total_gb, 16.0)


class TestROCm(DetectionPatches):
    torch_module = fake_torch(cuda=True, hip="6.1")

    def test_rocm_is_detected_through_cuda_api(self):
        # Regression: ROCm torch reports cuda.is_available(), so the old
        # dedicated ROCm branch was unreachable and AMD was labeled NVIDIA.
        info = detect_system()
        self.assertEqual(info.gpu_vendor, "amd")
        self.assertEqual(info.gpu_backend, "rocm")
        cfg = recommend_config(info, backend="safetensors")
        self.assertEqual(cfg["device"], "cuda")  # torch addresses ROCm GPUs as "cuda"


class TestPartialFailure(DetectionPatches):
    torch_module = fake_torch(cuda=True, fail_on_name=True)

    def test_failure_mid_detection_clears_all_gpu_fields(self):
        info = detect_system()
        self.assertIsNone(info.gpu_name)
        self.assertIsNone(info.gpu_vendor)
        self.assertIsNone(info.gpu_backend)
        self.assertEqual(info.vram_total_gb, 0.0)
        self.assertEqual(info.vram_free_gb, 0.0)
        self.assertTrue(info.torch_available)


class TestNvidiaWithoutTorch(DetectionPatches):
    torch_module = None
    smi = ("NVIDIA GeForce RTX 4060", 8.0, 6.0)
    llama_offload = True

    def test_safetensors_warns_and_uses_cpu(self):
        info = detect_system()
        self.assertEqual(info.gpu_name, "NVIDIA GeForce RTX 4060")
        self.assertIsNone(info.gpu_backend)  # torch cannot use it
        cfg = recommend_config(info, backend="safetensors")
        self.assertEqual(cfg["device"], "cpu")
        self.assertEqual(len(cfg["warnings"]), 1)
        self.assertIn("PyTorch is not installed", cfg["warnings"][0])
        self.assertIn("download.pytorch.org", cfg["warnings"][0])

    def test_gguf_offloads_without_torch(self):
        info = detect_system()
        cfg = recommend_config(info, backend="gguf", model_size_gb=4.0, n_layers=32)
        self.assertEqual(cfg["device"], "cuda")
        self.assertEqual(cfg["n_gpu_layers"], -1)  # 4 GB fits in 6 GB free - 1 GB headroom

    def test_gguf_partial_offload_is_proportional(self):
        info = detect_system()
        cfg = recommend_config(info, backend="gguf", model_size_gb=20.0, n_layers=80)
        self.assertEqual(cfg["n_gpu_layers"], int(80 * (6.0 - 1.0) / 20.0))

    def test_detect_params_logs_warning(self):
        with self.assertLogs("digvijay_llm", level="WARNING") as logs:
            cfg = detect_params(model_path="some/dir", backend="safetensors")
        self.assertIn("PyTorch is not installed", logs.output[0])
        self.assertEqual(len(cfg["warnings"]), 1)

    def test_explicit_device_suppresses_warning(self):
        cfg = detect_params(model_path="some/dir", backend="safetensors", device="cpu")
        self.assertEqual(cfg["warnings"], [])


class TestNvidiaWithCpuOnlyTorch(DetectionPatches):
    torch_module = fake_torch(cuda=False)
    smi = ("NVIDIA GeForce RTX 4060", 8.0, 6.0)

    def test_warning_names_cpu_only_build(self):
        cfg = recommend_config(detect_system(), backend="safetensors")
        self.assertIn("CPU-only build", cfg["warnings"][0])
        self.assertIn("2.4.0+fake", cfg["warnings"][0])


class TestLlamaCppWithoutGpu(DetectionPatches):
    smi = ("NVIDIA GeForce RTX 4060", 8.0, 6.0)
    llama_offload = False

    def test_gguf_stays_on_cpu_and_warns(self):
        cfg = recommend_config(detect_system(), backend="gguf", model_size_gb=4.0)
        self.assertEqual(cfg["n_gpu_layers"], 0)
        self.assertEqual(cfg["device"], "cpu")
        self.assertIn("GGML_CUDA", cfg["warnings"][0])


class TestNoGpu(DetectionPatches):
    def test_cpu_only_machine(self):
        cfg = recommend_config(detect_system(), backend="gguf", model_size_gb=4.0)
        self.assertEqual(cfg["n_gpu_layers"], 0)
        self.assertEqual(cfg["device"], "cpu")
        self.assertEqual(cfg["warnings"], [])


class TestModelInspection(DetectionPatches):
    smi = ("NVIDIA GeForce RTX 4060", 8.0, 6.0)
    llama_offload = True

    def test_reads_block_count_after_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.gguf")
            write_gguf(path, block_count=40)
            self.assertEqual(read_gguf_layer_count(path), 40)

    def test_non_gguf_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.gguf")
            with open(path, "wb") as f:
                f.write(b"nope")
            self.assertIsNone(read_gguf_layer_count(path))

    def test_detect_params_uses_real_file_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.gguf")
            write_gguf(path, block_count=40, padding_bytes=1024)
            cfg = detect_params(model_path=path)
        # A tiny file fits entirely, so every layer is offloaded.
        self.assertEqual(cfg["n_gpu_layers"], -1)

    def test_safetensors_dir_layer_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                f.write('{"num_hidden_layers": 28}')
            self.assertEqual(hardware._model_layer_count(tmp), 28)


class TestTorchImportFailure(DetectionPatches):
    def test_broken_torch_install_does_not_crash(self):
        # A CUDA DLL failing to load raises OSError, not ImportError.
        with mock.patch("builtins.__import__", side_effect=self._import_raising_oserror):
            with self.assertLogs("digvijay_llm", level="WARNING"):
                info = detect_system()
        self.assertFalse(info.torch_available)

    @staticmethod
    def _import_raising_oserror(name, *args, _real=__import__, **kwargs):
        if name == "torch":
            raise OSError("[WinError 126] fbgemm.dll could not be loaded")
        return _real(name, *args, **kwargs)


class TestModelMetadata(unittest.TestCase):
    def test_gguf_context_length_uses_model_arch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.gguf")
            write_gguf(path, context_length=4096)
            self.assertEqual(hardware.read_gguf_metadata(path), {"block_count": 40, "context_length": 4096})

    def test_null_text_config_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                f.write('{"text_config": null, "num_hidden_layers": 12, "max_position_embeddings": 1024}')
            self.assertEqual(hardware._model_layer_count(tmp), 12)
            self.assertEqual(hardware._model_context_length(tmp), 1024)

    def test_nested_text_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                f.write('{"text_config": {"num_hidden_layers": 30}, "num_hidden_layers": 2}')
            self.assertEqual(hardware._model_layer_count(tmp), 30)

    def test_size_does_not_double_count_bin_and_safetensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("model.safetensors", "pytorch_model.bin"):
                with open(os.path.join(tmp, name), "wb") as f:
                    f.write(b"\0" * 1024)
            self.assertAlmostEqual(hardware._model_size_gb(tmp), 1024 / 1024**3)

    def test_uppercase_extension_is_gguf(self):
        self.assertTrue(hardware.is_gguf_path("MODEL.Q4_K_M.GGUF"))
        self.assertFalse(hardware.is_gguf_path("model-dir"))


class TestContextAndBatch(DetectionPatches):
    def test_batch_is_not_tiny(self):
        # Regression: n_batch was 4, so llama.cpp evaluated prompts 4 tokens at a time.
        cfg = recommend_config(detect_system(), backend="gguf", model_size_gb=4.0)
        self.assertGreaterEqual(cfg["n_batch"], 128)

    def test_context_follows_requested_ram_budget(self):
        # Machine has 32 GB, but the caller asked to stay within 8 GB.
        cfg = detect_params(model_path="x.gguf", n_ram_gb=8)
        self.assertEqual(cfg["n_ctx"], 1024)
        self.assertEqual(cfg["n_ram_gb"], 8)

    def test_context_clamped_to_model_training_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.gguf")
            write_gguf(path, context_length=256)
            cfg = detect_params(model_path=path)
        self.assertEqual(cfg["n_ctx"], 256)
        self.assertLessEqual(cfg["n_batch"], 256)

    def test_explicit_context_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.gguf")
            write_gguf(path, context_length=256)
            self.assertEqual(detect_params(model_path=path, n_ctx=512)["n_ctx"], 512)

    def test_ram_recommendation_never_exceeds_physical_ram(self):
        with mock.patch.object(hardware, "_get_ram_info", return_value=(4.0, 2.0)):
            cfg = recommend_config(detect_system(), backend="gguf")
        self.assertLessEqual(cfg["n_ram_gb"], 4.0)


if __name__ == "__main__":
    unittest.main()
