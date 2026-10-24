# Repository development scripts

These commands maintain and validate the JYPPX repository. They are developer infrastructure, separate from the user-facing library tools in [tools](../tools/README.md), and are not installed as product commands.

Run the commands below from the repository root. Python script defaults also locate their repository inputs relative to the script rather than the caller's working directory.

| Directory | Commands and purpose |
| --- | --- |
| `validation/` | Compatibility and dependency manifests, component/include/link graphs, header guards, checkpoint fixtures, coverage gates, and the Linux coverage/TSan/fuzz lanes. |
| `benchmarks/` | Windows baseline orchestration, CUDA benchmark orchestration, and statistical comparison of measured baseline samples. |
| `interop/` | Official safetensors interoperability qualification and ONNX environment probes, with the qualification dependency lock and provenance. |
| `release/` | Offline candidate preparation, installed-file inventory, evidence and dependency bindings, and independent release verification. |
| `datasets/` | 显式下载 MNIST、核对 SHA-256 并有界解压到仓库外的数据根；见 [MNIST 示例](../examples/mnist/README.md)。 |

[Resource baseline sampling](benchmarks/RESOURCE_BASELINE.md) covers stable storage/checkpoint/telemetry and the three optional Previews in independent processes, using existing Release binaries after chapter-level validation.

Examples:

```sh
python scripts/validation/compatibility.py --file compatibility.json --validate
python scripts/validation/dependency_manifest.py --file dependency-manifest.json
python scripts/validation/header_guards.py
python tests/scripts/test_component_graph.py
bash scripts/validation/run-coverage.sh
```

```powershell
pwsh -NoProfile -File scripts/benchmarks/Run-Baseline.ps1
```

Script unit tests live in `tests/scripts/`. Telemetry backend acceptance lives in `tests/integration/telemetry_sidecar/`; the runtime adapter remains in
`tools/telemetry_sidecar/`. Use `out/` for all generated reports, logs, downloaded
dependencies and qualification environments. Existing persistent build
directories are reused unless a particular clean-install gate requires its own
isolated consumer directory.
