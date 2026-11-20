"""Reject image benchmark reports that misidentify precision or training work."""

import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/benchmarks"))
SPEC = importlib.util.spec_from_file_location(
    "image_training_runner", ROOT / "scripts/benchmarks/run_image_training.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def report(precision="fp32", device="cpu", model="vit_giant_patch14_dinov2-random-init"):
    cuda = device == "cuda:0"
    return {
        "kind": "liuding-image-training-benchmark-sample",
        "build_fingerprint": {"accelerator_runtime": "cuda:available:1" if cuda else "cpu"},
        "workload": {"descriptor": {
            "model": model, "device": device,
            "source_mode": "image-folder", "precision": precision,
            "diagnostics": "off",
            "parameter_dtype": "float16" if precision == "cuda-fp16-storage" else "float32",
            "model_input_dtype": "float16" if precision == "cuda-fp16-storage" else "float32",
            "grad_scaler_init_scale": 1.0 if precision == "fp32" else 128.0,
            "samples_per_epoch": 3, "batch_size": 2, "epochs": 2,
            "threads": 1,
        }},
        "execution": {"device": device},
        "diagnostics": {"norm_samples": 0},
        "validation": {
            "observed_samples": 6, "parameter_updated": True, "completed_epochs": 2,
            "parameters_finite": True,
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
    def run_report(self, payload, precision="fp32", device="cpu",
                   model="vit_giant_patch14_dinov2-random-init", diagnostics="off"):
        with mock.patch.object(runner.resource_protocol, "run_process", return_value=payload):
            return runner.run_sample(["benchmark"], model,
                                     device, "image-folder", precision, diagnostics, Path("unused"))

    def rocm_report(self):
        payload = report("cuda-fp16", "cuda:0")
        payload["build_fingerprint"] = {
            "jyppx_version": "fixture", "git_commit": "fixture", "compiler": "fixture",
            "cxx_standard": "202002", "os_arch": "Windows-x86_64", "build_type": "Release",
            "libtorch_version": "fixture", "libtorch_build": "rocm-fixture",
            "accelerator_runtime": "rocm:available:1", "dependency_lock_digest": "fixture"}
        payload["execution"].update({
            "intraop_threads": 1, "interop_threads": 1, "timing": "HIP synchronize",
            "rocm": {"device_index": 0, "device_name": "Fixture GPU",
                "hip_uuid": "00112233445566778899aabbccddeeff", "pci_bus_id": "0000:03:00.0",
                "architecture": "gfx1100", "hip_driver_api_version": 1, "hip_runtime_version": 2,
                "compile_hip_version": 3, "total_memory_bytes": 4096}})
        for suffix in ("allocated_bytes", "reserved_bytes"):
            payload["metrics"]["peak_rocm_" + suffix] = payload["metrics"].pop("peak_cuda_" + suffix)
        return payload

    def test_accepts_native_hip_identity_and_independent_metrics(self):
        payload = self.rocm_report()
        self.assertEqual(self.run_report(payload, "cuda-fp16", "cuda:0"), payload)

    def test_rejects_false_hip_identity_timing_and_memory(self):
        for section, field, value in (("rocm", "hip_uuid", "0" * 32),
                                      ("rocm", "pci_bus_id", "invalid"),
                                      ("rocm", "hip_runtime_version", 0),
                                      ("execution", "timing", "CUDA sync"),
                                      ("execution", "compute_capability", "11.0"),
                                      ("build", "libtorch_build", "cuda-fixture")):
            with self.subTest(field=field):
                payload = self.rocm_report()
                target = (payload["execution"]["rocm"] if section == "rocm" else
                          payload["execution"] if section == "execution" else payload["build_fingerprint"])
                target[field] = value
                with self.assertRaises(ValueError):
                    self.run_report(payload, "cuda-fp16", "cuda:0")
        for value in (8192, float("nan"), True):
            payload = self.rocm_report()
            payload["metrics"]["peak_rocm_reserved_bytes"]["value"] = value
            with self.assertRaises(ValueError):
                self.run_report(payload, "cuda-fp16", "cuda:0")

    def test_rejects_mixed_or_relabelled_allocator_backend(self):
        for modification in ("mixed", "cuda", "cpu", "missing"):
            payload = self.rocm_report()
            if modification == "mixed":
                payload["metrics"]["peak_cuda_allocated_bytes"] = {"status": "available", "value": 128}
            elif modification == "missing":
                del payload["execution"]["rocm"]
            else:
                payload["build_fingerprint"]["accelerator_runtime"] = modification
            with self.subTest(modification=modification), self.assertRaises(ValueError):
                self.run_report(payload, "cuda-fp16", "cuda:0")

    def test_preflight_rejects_changed_hip_device_and_build(self):
        payload = self.rocm_report()
        identity = {"schema_version": 1, "build_fingerprint": copy.deepcopy(payload["build_fingerprint"]),
                    "rocm": copy.deepcopy(payload["execution"]["rocm"]),
                    "runtime": {"intraop_threads": 1, "interop_threads": 1}}
        runner.validate_preflight(payload, identity)
        changed_sample = copy.deepcopy(payload)
        changed_sample["execution"]["intraop_threads"] = 2
        with self.assertRaisesRegex(ValueError, "thread configuration"):
            runner.validate_preflight(changed_sample, identity)
        for section, key, value in (("rocm", "hip_uuid", "112233445566778899aabbccddeeff00"),
                                     ("rocm", "pci_bus_id", "0000:04:00.0"),
                                     ("rocm", "hip_runtime_version", 3),
                                     ("build_fingerprint", "git_commit", "changed")):
            changed = copy.deepcopy(identity)
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_preflight(payload, changed)

    def test_stability_keeps_ten_percent_gate(self):
        payload = self.rocm_report()
        names = ("images_per_second", "workload_images_per_second", "fit_seconds", "workload_seconds",
                 "epoch_seconds", "peak_rss_bytes", "peak_rocm_allocated_bytes", "peak_rocm_reserved_bytes")
        document = {"warmup": payload, "summary": {key: {"coefficient_of_variation": 0.10} for key in names}}
        self.assertTrue(runner.stability(document)["passed"])
        document["summary"]["fit_seconds"]["coefficient_of_variation"] = 0.10001
        self.assertEqual(runner.stability(document)["unstable_metrics"], ["fit_seconds"])

    def test_validates_all_diagnostic_policies_and_sampling_boundaries(self):
        for policy, samples in (("off", 0), ("counters", 0),
                                ("sampled", 2), ("each-step", 34)):
            with self.subTest(policy=policy):
                payload = report()
                payload["workload"]["descriptor"].update(
                    diagnostics=policy, samples_per_epoch=34)
                payload["validation"].update(
                    observed_samples=68, global_step=34, optimizer_completed_steps=34)
                payload["diagnostics"]["norm_samples"] = samples
                self.assertEqual(self.run_report(payload, diagnostics=policy), payload)
                payload["diagnostics"]["norm_samples"] = samples + 1
                with self.assertRaisesRegex(ValueError, "diagnostic samples"):
                    self.run_report(payload, diagnostics=policy)

    def test_rejects_a_different_diagnostic_policy(self):
        payload = report()
        payload["workload"]["descriptor"]["diagnostics"] = "each-step"
        with self.assertRaisesRegex(ValueError, "different diagnostic policy"):
            self.run_report(payload)

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
        for expected in ("vit_giant_patch14_dinov2", "dinov3_vitl16", "dinov3_vitl16plus", "dinov3_vith16plus",
                         "dinov3_convnext_base", "dinov3_convnext_large"):
            for actual in ("flatten-linear64-relu-linear", "dinov3_vits16-random-init"):
                with self.subTest(expected=expected, actual=actual):
                    with self.assertRaisesRegex(ValueError, "different model"):
                        self.run_report(report(model=actual), model=expected + "-random-init")

    def test_rejects_cuda_precision_on_cpu_before_running_a_binary(self):
        for precision in ("cuda-fp16", "cuda-fp16-storage"):
            arguments = ["runner", "--binary", "unused", "--image-root", "unused",
                         "--output", "unused", "--precision", precision]
            with self.subTest(precision=precision), mock.patch.object(runner.sys, "argv", arguments), \
                    mock.patch.object(runner.resource_protocol, "run_process") as run, \
                    mock.patch.object(runner.sys, "stderr"):
                with self.assertRaises(SystemExit) as raised:
                    runner.main()
                self.assertEqual(raised.exception.code, 2)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
