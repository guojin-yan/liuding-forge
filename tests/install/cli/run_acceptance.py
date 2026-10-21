#!/usr/bin/env python3
"""Install and execute the product tool outside its source/build tree."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--torch-runtime", required=True, type=Path)
    args = parser.parse_args()
    subprocess.run([args.cmake, "--install", str(args.build_dir), "--config", args.config,
                    "--prefix", str(args.prefix)], check=True, timeout=120)
    cli = args.prefix / "bin" / ("jyppx.exe" if os.name == "nt" else "jyppx")
    assert cli.is_file(), "Installed product CLI is missing"
    for forbidden in ("scripts", "tests", "tools", "generated-private"):
        assert not (args.prefix / forbidden).exists(), f"Developer-only {forbidden} was installed"
    assert not list((args.prefix / "include").rglob("build_metadata.hpp")), "Private build metadata header leaked"
    assert not list((args.prefix / "include").rglob("metadata_validation.hpp")), "Private interop header leaked"
    environment = os.environ.copy()
    variable = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    environment[variable] = str(args.torch_runtime) + os.pathsep + environment.get(variable, "")
    with tempfile.TemporaryDirectory(prefix="jyppx-installed-cli-") as cwd:
        def invoke(*arguments: str) -> dict:
            process = subprocess.run([str(cli.resolve()), *arguments, "--format", "json"],
                cwd=cwd, env=environment, text=True, encoding="utf-8", capture_output=True, timeout=30)
            assert process.returncode == 0, process.stdout + process.stderr
            result = json.loads(process.stdout)
            assert result["schema_version"] == 1 and result["status"] == "ok", result
            return result["data"]
        version = invoke("--version")
        assert version["version"], version
        doctor = invoke("doctor", "--device", "cpu")
        assert doctor["checks"] and all(check["status"] == "pass" for check in doctor["checks"]), doctor
        print(json.dumps({"status": "passed", "installed_cli": str(cli), "version": version["version"]}))


if __name__ == "__main__":
    main()
