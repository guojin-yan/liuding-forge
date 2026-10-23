# Telemetry backend acceptance

This directory qualifies the optional runtime adapter in [`tools/telemetry_sidecar`](../../../tools/telemetry_sidecar). Tests and the official Collector fixture are development infrastructure and are not installed with that adapter.

Use a dedicated CPython 3.11 environment, install the runtime's exact
`requirements.lock` with `pip --require-hashes --only-binary=:all:`, and supply
the pinned official Collector Contrib 0.135.0 executable through
`JYPPX_LIUDING_TEST_OTELCOL`. Verify its download against `collector-artifacts.json`.
Supply the built product executable through `JYPPX_LIUDING_TEST_CLI`. The configured CMake gate supplies both paths and fails when either prerequisite is absent.

The 2026-10-21 dependency preparation installed the locked environment on both platforms, passed `pip check` and official module imports, and verified Collector archive hashes and `--version` output. Exact interpreter distributions, package wheels and report digests are recorded in [`optional-runtime-provenance.json`](../../../scripts/interop/optional-runtime-provenance.json). This preparation did not execute the acceptance entry point or the real Collector protocol fixture.

Reuse `out/envs/telemetry-py311/Scripts/python.exe` on Windows or
`out/envs/telemetry-py311/bin/python` on Ubuntu. The prepared Collector is at
`out/deps/otelcol-contrib-0.135.0/windows/otelcol-contrib.exe` or
`out/deps/otelcol-contrib-0.135.0/linux/otelcol-contrib`, respectively. Set
`JYPPX_LIUDING_TEST_OTELCOL` to its absolute path.

From the repository root:

```sh
<venv-python> tests/integration/telemetry_sidecar/run_acceptance.py
```

The acceptance entry point rejects missing prerequisites, version mismatches and skipped tests. It runs protocol tests, official TensorBoard readback, generated OTLP schema consumers, and an actual Collector file-export roundtrip. It also runs actual C++ training/recovery, reads its canonical SQLite events, exports through the product CLI and OS process transport into the official sidecar, and compares official TensorBoard/Collector readback with that evidence. The tests have moved from the runtime directory as part of chapter integration; their execution results are recorded by the chapter gate.
