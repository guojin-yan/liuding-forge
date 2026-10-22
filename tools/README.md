# 六丁炼算工具（Liuding Forge tools）

This directory contains tools delivered with Liuding Forge for library users. It is
not the location for repository build, CI, test, coverage, benchmark-baseline,
or release-acceptance scripts.

| Directory | Purpose |
| --- | --- |
| `cli/` | C++ source for the `liuding-forge` command-line tool (with the installed `jyppx` compatibility copy): configuration, local runs, evaluation, data, model and checkpoint operations. |
| `telemetry_sidecar/` | Optional runtime adapter using the official TensorBoard and OpenTelemetry libraries, including its runtime dependency lock. |

These tools are under active integration. Their presence in the source tree
does not establish installation or qualification of an optional backend.

Repository-maintenance commands live in [scripts](../scripts/README.md).
Tests and acceptance fixtures live in [tests](../tests).
Generated output, virtual environments and dependency downloads belong in
`out/`, outside the tracked tool sources.
