"""Test-only process checker. All product conversion executes the native tool."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile

SOURCE_SHA = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
OUTPUT_SHA = "f3ef1ae01fd4af3cd15926a0d5e631fa0caea50ca68b29db8a68c8c84c9810e5"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def invoke(binary, source=None, output=None, report=None, extra=None):
    args = [str(binary)]
    if source is not None:
        args += ["--source", str(source), "--destination", str(output), "--report", str(report)]
    return subprocess.run(args + (extra or []), capture_output=True, text=True,
                          encoding="utf-8", timeout=120)


def negatives(binary, root):
    source, output, report = root / "bad.safetensors", root / "out", root / "report.json"
    source.write_bytes(b"invalid")
    require(invoke(binary, extra=["--help"]).returncode == 0, "help")
    require(invoke(binary).returncode == 2, "required options")
    require(invoke(binary, source, output, report, ["--source", str(source)]).returncode == 2,
            "duplicate/extra options")
    failed = invoke(binary, source, output, report)
    require(failed.returncode == 1 and not output.exists() and not report.exists(),
            "reject invalid source without publications")
    source.write_bytes(b"retained source")
    output.write_bytes(b"retained output")
    require(invoke(binary, source, output, report).returncode == 1, "overwrite")
    require(output.read_bytes() == b"retained output" and source.read_bytes() == b"retained source",
            "keep existing files")
    output.unlink()
    report.write_bytes(b"retained report")
    require(invoke(binary, source, output, report).returncode == 1 and not output.exists(),
            "report collision")
    require(report.read_bytes() == b"retained report", "keep report")
    require(invoke(binary, source, output, output).returncode == 1, "same outputs")
    # Exact source length still must fail the hard complete-digest gate.
    with source.open("wb") as stream:
        stream.truncate(88249960)
    require(invoke(binary, source, root / "sized-output", root / "sized-report").returncode == 1,
            "correct size is not identity")
    require(not (root / "sized-output").exists(), "SHA failure must not publish")


def positive(binary, source, reference, root):
    require(source.is_file() and source.stat().st_size == 88249960 and sha(source) == SOURCE_SHA,
            "opt-in test requires exact pinned official source")
    output, report = root / "原生 weights.safetensors", root / "原生 report.json"
    completed = invoke(binary, source, output, report)
    require(completed.returncode == 0, completed.stderr)
    require(output.stat().st_size == 88240512 and sha(output) == OUTPUT_SHA, "canonical bytes")
    result = json.loads(report.read_text(encoding="utf-8"))
    require(json.loads(completed.stdout)["output"] == result["output"], "stdout contract")
    require(result["source"]["sha256"] == SOURCE_SHA and len(result["mappings"]) == 174,
            "complete provenance")
    spec = importlib.util.spec_from_file_location("independent_conversion_reference", reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected_file, expected_report = root / "reference.safetensors", root / "reference.json"
    expected = module.convert(source, expected_file, expected_report)
    require(result == expected and sha(expected_file) == sha(output), "independent full report/bytes")
    require(report.read_bytes() == expected_report.read_bytes(), "canonical report serialization")
    require(invoke(binary, source, output, report).returncode == 1 and sha(output) == OUTPUT_SHA,
            "repeat refuses overwrite")
    # A native run must work with an empty PATH: no Python/external executable.
    env = os.environ.copy()
    env["PATH"] = ""
    standalone = subprocess.run([str(binary), "--source", str(source), "--destination",
        str(root / "standalone.safetensors"), "--report", str(root / "standalone.json")],
        env=env, capture_output=True, text=True, timeout=120)
    require(standalone.returncode == 0, standalone.stderr)
    # Race two valid native publishers: exactly one complete pair wins. Each
    # failed publisher must leave the winner's model/report unchanged.
    race_output, race_report = root / "race.safetensors", root / "race.json"
    command = [str(binary), "--source", str(source), "--destination", str(race_output),
               "--report", str(race_report)]
    runs = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
    for process in runs:
        process.communicate(timeout=120)
    require(sorted(process.returncode for process in runs) == [0, 1], "no-clobber concurrent publish")
    require(sha(race_output) == OUTPUT_SHA and json.loads(race_report.read_text()) == expected,
            "winner pair intact")
    # Nonempty directory races with report publication after staging begins.
    # Poll only for a real model stage: source identity checks cannot be bypassed.
    rollback_output, rollback_report = root / "rollback.safetensors", root / "rollback.json"
    existing_stages = set(root.glob(".liuding-dinov2-*"))
    run = subprocess.Popen([str(binary), "--source", str(source), "--destination", str(rollback_output),
                            "--report", str(rollback_report)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    import time
    deadline = time.monotonic() + 30
    staged = False
    while time.monotonic() < deadline and run.poll() is None:
        if set(root.glob(".liuding-dinov2-*")) - existing_stages:
            rollback_report.mkdir(exist_ok=True)
            (rollback_report / "keep").write_text("collision")
            staged = True
            break
        time.sleep(0.001)
    stdout, stderr = run.communicate(timeout=120)
    require(staged and run.returncode == 1, "publication collision injection")
    require(not rollback_output.exists() and (rollback_report / "keep").read_text() == "collision",
            "rollback own model, preserve collision")
    new_stages = set(root.glob(".liuding-dinov2-*")) - existing_stages
    require(b"retained model stage=" in stderr and any((stage / "FAILURE.json").is_file() for stage in new_stages),
            "retain explicit failure evidence")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="native-dinov2-conversion-") as folder:
        root = Path(folder)
        negatives(args.binary.resolve(), root)
        if args.source:
            require(args.reference is not None, "independent reference required")
            positive(args.binary.resolve(), args.source.resolve(), args.reference.resolve(), root)


if __name__ == "__main__":
    main()
