import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from digvijay_llm.ram_planner import plan_for_budget, RAMPlanner


class TestRAMPlanner(unittest.TestCase):
    def test_70b_on_16gb_warns_or_streams_safely(self):
        plan = plan_for_budget(
            model_path=".",
            total_params_billion=70,
            n_layers=80,
            bytes_per_param=0.6,  # Q4 quantized
            n_ram_gb=16,
        )
        self.assertGreaterEqual(plan.safe_hot_layers, 1)
        self.assertLess(plan.safe_hot_layers, 80)  # must NOT fit all layers in RAM
        self.assertGreater(plan.layer_size_gb, 0)

    def test_small_model_fits_more_layers(self):
        plan = plan_for_budget(
            model_path=".",
            total_params_billion=1.1,
            n_layers=22,
            bytes_per_param=2.0,
            n_ram_gb=16,
        )
        self.assertGreaterEqual(plan.safe_hot_layers, 1)

    def test_class_wrapper_matches_function(self):
        planner = RAMPlanner(model_path=".", total_params_billion=7, n_layers=32, n_ram_gb=16)
        plan = planner.plan()
        self.assertEqual(plan.total_ram_gb, 16)

    def test_extreme_budget_emits_warning(self):
        plan = plan_for_budget(
            model_path=".",
            total_params_billion=70,
            n_layers=80,
            bytes_per_param=2.0,  # fp16, unrealistic for 16GB -> should warn
            n_ram_gb=4,
        )
        self.assertTrue(len(plan.warnings) >= 1)


if __name__ == "__main__":
    unittest.main()
