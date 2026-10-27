#!/usr/bin/env python3
"""Run one warm-cache image-training workload in independent processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MODEL_SPECS = {
    "linear64": ("flatten-linear64-relu-linear", 1, 1),
    "resnet18": ("resnet18-random-init", 64, 1),
    "vit_small_patch14_dinov2": ("vit_small_patch14_dinov2-random-init", 14, 14),
    "vit_base_patch14_dinov2": ("vit_base_patch14_dinov2-random-init", 14, 14),
    "vit_large_patch14_dinov2": ("vit_large_patch14_dinov2-random-init", 14, 14),
    "vit_giant_patch14_dinov2": ("vit_giant_patch14_dinov2-random-init", 14, 14),
    "dinov3_vits16": ("dinov3_vits16-random-init", 16, 16),
    "dinov3_vits16plus": ("dinov3_vits16plus-random-init", 16, 16),
    "dinov3_vitb16": ("dinov3_vitb16-random-init", 16, 16),
    "dinov3_vitl16": ("dinov3_vitl16-random-init", 16, 16),
    "dinov3_vitl16plus": ("dinov3_vitl16plus-random-init", 16, 16),
    "dinov3_convnext_tiny": ("dinov3_convnext_tiny-random-init", 32, 1),
    "dinov3_convnext_small": ("dinov3_convnext_small-random-init", 32, 1),
    "dinov3_convnext_base": ("dinov3_convnext_base-random-init", 32, 1),
    "dinov3_convnext_large": ("dinov3_convnext_large-random-init", 32, 1),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed in the dataset: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
        size += path.stat().st_size
        count += 1
    if count < 2:
        raise ValueError("at least two JPEG/PNG images are required")
    return digest.hexdigest(), count, size


def git_output(source_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=source_root, text=True, encoding="utf-8"
    ).strip()


def run_sample(command: list[str], expected_model: str, expected_device: str,
               expected_source: str, expected_precision: str) -> dict:
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout.strip())
    if result.get("kind") != "liuding-image-training-benchmark-sample":
        raise ValueError("benchmark returned an unexpected result kind")
    validation = result["validation"]
    workload = result["workload"]["descriptor"]
    if workload["model"] != expected_model:
        raise ValueError("benchmark reported a different model than requested")
    if workload["device"] != expected_device or result["execution"]["device"] != expected_device:
        raise ValueError("benchmark reported a different device than requested")
    if workload["source_mode"] != expected_source:
        raise ValueError("benchmark reported a different source mode than requested")
    if workload["precision"] != expected_precision:
        raise ValueError("benchmark reported a different precision than requested")
    expected_dtype = "float16" if expected_precision == "cuda-fp16-storage" else "float32"
    if workload["parameter_dtype"] != expected_dtype:
        raise ValueError("benchmark reported a different parameter dtype than requested")
    if workload["model_input_dtype"] != expected_dtype:
        raise ValueError("benchmark reported a different model input dtype than requested")
    expected_scale = 1.0 if expected_precision == "fp32" else 128.0
    if workload["grad_scaler_init_scale"] != expected_scale:
        raise ValueError("benchmark reported a different initial loss scale")
    expected = workload["samples_per_epoch"] * workload["epochs"]
    if validation["observed_samples"] != expected or not validation["parameter_updated"]:
        raise ValueError("benchmark did not validate its training work")
    if validation["completed_epochs"] != workload["epochs"]:
        raise ValueError("benchmark did not complete all requested epochs")
    expected_steps = ((workload["samples_per_epoch"] + workload["batch_size"] - 1) //
                      workload["batch_size"]) * workload["epochs"]
    if (validation["global_step"] != expected_steps or
            validation["optimizer_completed_steps"] != expected_steps):
        raise ValueError("benchmark skipped or miscounted optimizer updates")
    if (not math.isfinite(validation["final_loss_scale"]) or
            validation["final_loss_scale"] <= 0):
        raise ValueError("invalid final loss scale")
    if len(validation["final_parameter_sha256"]) != 64:
        raise ValueError("missing final parameter digest")
    if result["metrics"]["peak_rss_bytes"]["status"] != "available":
        raise ValueError("peak process RSS is unavailable")
    metrics = result["metrics"]
    if not (metrics["workload_seconds"] >= metrics["fit_seconds"] > 0 and
            metrics["source_preparation_seconds"] > 0 and
            metrics["workload_images_per_second"] > 0):
        raise ValueError("invalid workload/source timing")
    if ((metrics["cached_source_bytes"] > 0) !=
            (expected_source != "image-folder")):
        raise ValueError("invalid Tensor cache accounting")
    for name in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
        status = result["metrics"][name]["status"]
        if status != ("available" if expected_device == "cuda:0" else "unavailable"):
            raise ValueError(f"unexpected {name} status for {expected_device}")
    if expected_device == "cuda:0":
        metrics = result["metrics"]
        if (metrics["peak_cuda_allocated_bytes"]["value"] <= 0 or
                metrics["peak_cuda_reserved_bytes"]["value"] <
                metrics["peak_cuda_allocated_bytes"]["value"]):
            raise ValueError("invalid CUDA allocator peaks")
    return result


def summary(samples: list[dict], name: str) -> dict:
    values = [float(sample["metrics"][name]) for sample in samples]
    return summarize_values(values)


def summarize_values(values: list[float]) -> dict:
    mean = statistics.mean(values)
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "coefficient_of_variation": statistics.stdev(values) / mean if mean else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), default="linear64")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "cuda-fp16", "cuda-fp16-storage"),
                        default="fp32")
    parser.add_argument("--source-mode", choices=("image-folder", "tensor-cache",
                                                  "tensor-cache-resized"),
                        default="image-folder")
    parser.add_argument("--max-cached-bytes", type=int, default=512 << 20)
    args = parser.parse_args()
    if args.precision != "fp32" and args.device != "cuda:0":
        parser.error("CUDA precision requires --device cuda:0")
    source_root = Path(__file__).resolve().parents[2]
    binary = args.binary.resolve(strict=True)
    root = args.image_root.resolve(strict=True)
    output = args.output.resolve()
    if output.is_relative_to(source_root):
        parser.error("--output must be outside the code repository")
    if not root.is_dir():
        parser.error("--image-root must be a directory")
    if output.exists():
        parser.error("--output already exists; use a fresh evidence path")
    if any(value <= 0 for value in (args.batch_size, args.image_size, args.epochs, args.threads)):
        parser.error("batch size, image size, epochs and threads must be positive")
    if not 0 <= args.workers <= 16 or not 1 <= args.prefetch_batches <= 4:
        parser.error("workers must be 0..16 and prefetch batches must be 1..4")
    if not 0 < args.max_cached_bytes <= 64 << 30:
        parser.error("max cached bytes must be in 1..64 GiB")
    expected_model, minimum_image_size, image_size_multiple = MODEL_SPECS[args.model]
    if args.image_size < minimum_image_size:
        parser.error(f"{args.model} requires image size >= {minimum_image_size}")
    if args.image_size % image_size_multiple:
        parser.error(f"{args.model} requires image size divisible by {image_size_multiple}")

    content_sha256, image_count, encoded_bytes = dataset_digest(root)
    command = [
        str(binary), "--image-root", str(root), "--batch-size", str(args.batch_size),
        "--image-size", str(args.image_size), "--epochs", str(args.epochs),
        "--threads", str(args.threads), "--workers", str(args.workers),
        "--prefetch-batches", str(args.prefetch_batches), "--model", args.model,
        "--device", args.device, "--precision", args.precision,
        "--source-mode", args.source_mode,
        "--max-cached-bytes", str(args.max_cached_bytes),
    ]
    warmup = run_sample(command, expected_model, args.device, args.source_mode, args.precision)
    samples = [run_sample(command, expected_model, args.device, args.source_mode, args.precision)
               for _ in range(5)]
    final_sha256, final_count, final_bytes = dataset_digest(root)
    if (final_sha256, final_count, final_bytes) != (content_sha256, image_count, encoded_bytes):
        raise ValueError("dataset files changed during the benchmark run")
    identity = (warmup["workload"]["digest"], warmup["build_fingerprint"],
                warmup["execution"])
    if any((sample["workload"]["digest"], sample["build_fingerprint"],
            sample["execution"]) != identity
           for sample in samples):
        raise ValueError("independent processes reported different workload/build identities")
    if args.device == "cpu" and any(
            sample["validation"]["final_parameter_sha256"] !=
            warmup["validation"]["final_parameter_sha256"] for sample in samples):
        raise ValueError("independent CPU processes produced different parameters")
    commit = git_output(source_root, "rev-parse", "HEAD")
    if not warmup["build_fingerprint"]["git_commit"].startswith(commit):
        raise ValueError("binary build fingerprint does not match current commit")
    report = {
        "schema_version": 1,
        "kind": "liuding-image-training-benchmark-run",
        "cache_state": "warm: all selected image files read for SHA-256 before warmup",
        "source": {
            "commit": commit,
            "dirty": bool(git_output(source_root, "status", "--porcelain")),
            "binary_sha256": sha256_file(binary),
        },
        "dataset": {
            "root": str(root), "content_sha256": content_sha256,
            "content_digest_scope": "all recursive .jpg/.jpeg/.png files, including hidden paths",
            "image_files": image_count, "encoded_bytes": encoded_bytes,
        },
        "environment": {"platform": platform.platform(), "python": sys.version},
        "command": command,
        "warmup": warmup,
        "samples": samples,
        "summary": {
            "images_per_second": summary(samples, "images_per_second"),
            "workload_images_per_second": summary(samples, "workload_images_per_second"),
            "fit_seconds": summary(samples, "fit_seconds"),
            "workload_seconds": summary(samples, "workload_seconds"),
            "source_preparation_seconds": summary(samples, "source_preparation_seconds"),
            "cached_source_bytes": summary(samples, "cached_source_bytes"),
            "epoch_seconds": summary(samples, "epoch_seconds"),
            "data_wait_seconds": summary(samples, "data_wait_seconds"),
            "peak_rss_bytes": summarize_values([
                float(sample["metrics"]["peak_rss_bytes"]["value"]) for sample in samples
            ]),
            "peak_cuda_allocated_bytes": (
                summarize_values([float(sample["metrics"]["peak_cuda_allocated_bytes"]["value"])
                                  for sample in samples]) if args.device == "cuda:0"
                else {"status": "not_applicable"}),
            "peak_cuda_reserved_bytes": (
                summarize_values([float(sample["metrics"]["peak_cuda_reserved_bytes"]["value"])
                                  for sample in samples]) if args.device == "cuda:0"
                else {"status": "not_applicable"}),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(json.dumps({"output": str(output), "summary": report["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
