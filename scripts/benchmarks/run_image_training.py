#!/usr/bin/env python3
"""Run one warm-cache image-training workload in independent processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


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


def run_sample(command: list[str]) -> dict:
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout.strip())
    if result.get("kind") != "liuding-image-training-benchmark-sample":
        raise ValueError("benchmark returned an unexpected result kind")
    validation = result["validation"]
    workload = result["workload"]["descriptor"]
    expected = workload["samples_per_epoch"] * workload["epochs"]
    if validation["observed_samples"] != expected or not validation["parameter_updated"]:
        raise ValueError("benchmark did not validate its training work")
    if validation["completed_epochs"] != workload["epochs"]:
        raise ValueError("benchmark did not complete all requested epochs")
    if result["metrics"]["peak_rss_bytes"]["status"] != "available":
        raise ValueError("peak process RSS is unavailable")
    return result


def summary(samples: list[dict], name: str) -> dict:
    values = [float(sample["metrics"][name]) for sample in samples]
    return summarize_values(values)


def summarize_values(values: list[float]) -> dict:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "coefficient_of_variation": statistics.stdev(values) / statistics.mean(values),
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
    args = parser.parse_args()
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

    content_sha256, image_count, encoded_bytes = dataset_digest(root)
    command = [
        str(binary), "--image-root", str(root), "--batch-size", str(args.batch_size),
        "--image-size", str(args.image_size), "--epochs", str(args.epochs),
        "--threads", str(args.threads),
    ]
    warmup = run_sample(command)
    samples = [run_sample(command) for _ in range(5)]
    final_sha256, final_count, final_bytes = dataset_digest(root)
    if (final_sha256, final_count, final_bytes) != (content_sha256, image_count, encoded_bytes):
        raise ValueError("dataset files changed during the benchmark run")
    identity = (warmup["workload"]["digest"], warmup["build_fingerprint"])
    if any((sample["workload"]["digest"], sample["build_fingerprint"]) != identity
           for sample in samples):
        raise ValueError("independent processes reported different workload/build identities")
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
            "fit_seconds": summary(samples, "fit_seconds"),
            "epoch_seconds": summary(samples, "epoch_seconds"),
            "data_wait_seconds": summary(samples, "data_wait_seconds"),
            "peak_rss_bytes": summarize_values([
                float(sample["metrics"]["peak_rss_bytes"]["value"]) for sample in samples
            ]),
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
