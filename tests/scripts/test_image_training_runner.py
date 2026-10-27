"""Reject image benchmark reports that misidentify precision or training work."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "image_training_runner", ROOT / "scripts/benchmarks/run_image_training.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def report(precision="fp32", device="cpu"):
    cuda = device == "cuda:0"
    return {
        "kind": "liuding-image-training-benchmark-sample",
        "workload": {"descriptor": {
            "model": "vit_giant_patch14_dinov2-random-init", "device": device,
            "source_mode": "image-folder", "precision": precision,
            "parameter_dtype": "float16" if precision == "cuda-fp16-storage" else "float32",
            "model_input_dtype": "float16" if precision == "cuda-fp16-storage" else "float32",
            "grad_scaler_init_scale": 1.0 if precision == "fp32" else 128.0,
            "samples_per_epoch": 3, "batch_size": 2, "epochs": 2,
        }},
        "execution": {"device": device},
        "validation": {
            "observed_samples": 6, "parameter_updated": True, "completed_epochs": 2,
            "global_step": 4, "optimizer_completed_steps": 4,
            "final_loss_scale": 1.0 if precision == "fp32" else 128.0,
            "final_parameter_sha256": "a" * 64,
        },
        "metrics": {
            "peak_rss_bytes": {"status": "available"},
            "workload_seconds": 2.0, "fit_seconds": 1.0,
            "source_preparation_seconds": 0.1, "workload_images_per_second": 3.0,
            "cached_source_bytes": 0,
            "peak_cuda_allocated_bytes": {"status": "available" if cuda else "unavailable", "value": 128},
            "peak_cuda_reserved_bytes": {"status": "available" if cuda else "unavailable", "value": 256},
        },
    }


class ImageTrainingReportTest(unittest.TestCase):
    def run_report(self, payload, precision="fp32", device="cpu"):
        completed = SimpleNamespace(stdout=json.dumps(payload))
        with mock.patch.object(runner.subprocess, "run", return_value=completed):
            return runner.run_sample(["benchmark"], "vit_giant_patch14_dinov2-random-init",
                                     device, "image-folder", precision)

    def test_accepts_each_declared_parameter_storage_mode(self):
        for precision, device in (("fp32", "cpu"), ("cuda-fp16", "cuda:0"),
                                  ("cuda-fp16-storage", "cuda:0")):
            with self.subTest(precision=precision):
                payload = report(precision, device)
                self.assertEqual(self.run_report(payload, precision, device), payload)

    def test_rejects_skipped_or_miscounted_updates(self):
        for key in ("global_step", "optimizer_completed_steps"):
            with self.subTest(key=key):
                payload = report()
                payload["validation"][key] = 3
                with self.assertRaisesRegex(ValueError, "optimizer updates"):
                    self.run_report(payload)

    def test_rejects_false_precision_and_dtype_claims(self):
        changes = (("precision", "cuda-fp16"), ("parameter_dtype", "float16"),
                   ("model_input_dtype", "float16"),
                   ("grad_scaler_init_scale", 128.0))
        for key, value in changes:
            with self.subTest(key=key):
                payload = report()
                payload["workload"]["descriptor"][key] = value
                with self.assertRaises(ValueError):
                    self.run_report(payload)

    def test_rejects_invalid_dynamic_scales(self):
        for scale in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(scale=scale):
                payload = report("cuda-fp16-storage", "cuda:0")
                payload["validation"]["final_loss_scale"] = scale
                with self.assertRaisesRegex(ValueError, "loss scale"):
                    self.run_report(payload, "cuda-fp16-storage", "cuda:0")

    def test_rejects_mismatched_model_identity(self):
        payload = report()
        payload["workload"]["descriptor"]["model"] = "linear64"
        with self.assertRaisesRegex(ValueError, "different model"):
            self.run_report(payload)

    def test_rejects_cuda_precision_on_cpu_before_running_a_binary(self):
        for precision in ("cuda-fp16", "cuda-fp16-storage"):
            arguments = ["runner", "--binary", "unused", "--image-root", "unused",
                         "--output", "unused", "--precision", precision]
            with self.subTest(precision=precision), mock.patch.object(runner.sys, "argv", arguments), \
                    mock.patch.object(runner.subprocess, "run") as run, \
                    mock.patch.object(runner.sys, "stderr"):
                with self.assertRaises(SystemExit) as raised:
                    runner.main()
                self.assertEqual(raised.exception.code, 2)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
