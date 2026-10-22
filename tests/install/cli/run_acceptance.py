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
    parser.add_argument("--runtime-stager", required=True, type=Path)
    args = parser.parse_args()
    subprocess.run([args.cmake, "--install", str(args.build_dir), "--config", args.config,
                    "--prefix", str(args.prefix)], check=True, timeout=120)
    suffix = ".exe" if os.name == "nt" else ""
    cli = args.prefix / "bin" / ("liuding-forge" + suffix)
    legacy_cli = args.prefix / "bin" / ("jyppx" + suffix)
    assert cli.is_file(), "Installed Liuding Forge CLI is missing"
    assert legacy_cli.is_file(), "Installed jyppx compatibility CLI is missing"
    for forbidden in ("scripts", "tests", "tools", "generated-private"):
        assert not (args.prefix / forbidden).exists(), f"Developer-only {forbidden} was installed"
    assert not list((args.prefix / "include").rglob("build_metadata.hpp")), "Private build metadata header leaked"
    assert not list((args.prefix / "include").rglob("metadata_validation.hpp")), "Private interop header leaked"
    if os.name == "nt":
        # The SDK install does not redistribute LibTorch. Like the installed
        # C++ consumers, this fixture deploys its explicitly supplied external
        # runtime beside the executable, including dynamically loaded CUDA DLLs.
        subprocess.run([args.cmake,
                        f"-DJYPPX_RUNTIME_SOURCE_DIR={args.torch_runtime.resolve(strict=True)}",
                        f"-DJYPPX_RUNTIME_OUTPUT_DIR={(args.prefix / 'bin').resolve(strict=True)}",
                        "-P", str(args.runtime_stager.resolve(strict=True))],
                       check=True, timeout=120)
    environment = os.environ.copy()
    variable = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    environment[variable] = str(args.torch_runtime) + os.pathsep + environment.get(variable, "")
    with tempfile.TemporaryDirectory(prefix="jyppx-installed-cli-") as cwd:
        def invoke(*arguments: str, executable: Path = cli) -> dict:
            process = subprocess.run([str(executable.resolve()), *arguments, "--format", "json"],
                cwd=cwd, env=environment, text=True, encoding="utf-8", capture_output=True, timeout=30)
            assert process.returncode == 0, process.stdout + process.stderr
            result = json.loads(process.stdout)
            assert result["schema_version"] == 1 and result["status"] == "ok", result
            return result["data"]
        version = invoke("--version")
        assert version["version"], version
        assert version["display_name"] == "六丁炼算", version
        assert version["display_name_en"] == "Liuding Forge", version
        assert version["artifact_prefix"] == "liuding-forge", version
        assert version["legacy_name"] == "JYPPX Training Framework", version
        assert version["build"]["jyppx_version"] == version["version"], version
        assert invoke("--version", executable=legacy_cli) == version
        help_data = invoke("--help")
        assert help_data["usage"].startswith("liuding-forge "), help_data
        assert invoke("--help", executable=legacy_cli) == help_data
        doctor = invoke("doctor", "--device", "cpu")
        assert doctor["checks"] and all(check["status"] == "pass" for check in doctor["checks"]), doctor
        legacy_doctor = invoke("doctor", "--device", "cpu", executable=legacy_cli)
        assert legacy_doctor["checks"] and all(check["status"] == "pass" for check in legacy_doctor["checks"]), legacy_doctor
        print(json.dumps({"status": "passed", "installed_cli": str(cli),
                          "compatibility_cli": str(legacy_cli), "version": version["version"]}))


if __name__ == "__main__":
    main()
