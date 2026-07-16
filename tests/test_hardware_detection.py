import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from digvijay_llm.hardware import detect_system, recommend_config, detect_params


class TestHardwareDetection(unittest.TestCase):
    def test_detect_system_reports_core_hardware(self):
        info = detect_system()
        self.assertGreater(info.ram_total_gb, 0)
        self.assertGreater(info.cpu_cores, 0)
        self.assertTrue(info.os_family in {"windows", "linux", "macos", "unknown"})
        self.assertTrue(info.storage_free_gb >= 0)

    def test_recommend_config_returns_sane_defaults(self):
        info = detect_system()
        cfg = recommend_config(info, model_size_gb=20.0)
        self.assertIn("backend", cfg)
        self.assertIn("device", cfg)
        self.assertGreaterEqual(cfg["n_threads"], 1)
        self.assertGreaterEqual(cfg["n_ctx"], 512)
        self.assertGreaterEqual(cfg["n_batch"], 1)

    def test_detect_params_uses_manual_overrides(self):
        cfg = detect_params(model_path="model.gguf", n_gpu_layers=8, n_threads=4, n_ctx=1024)
        self.assertEqual(cfg["n_gpu_layers"], 8)
        self.assertEqual(cfg["n_threads"], 4)
        self.assertEqual(cfg["n_ctx"], 1024)


if __name__ == "__main__":
    unittest.main()
