# Linux coverage lane

Run `scripts/validation/run-coverage.sh` from a clean committed checkout, with
`CMAKE_PREFIX_PATH` pointing to one LibTorch installation. Use the established
CPU LibTorch prefix for the CPU coverage lane. The script sets both
`CMAKE_PREFIX_PATH` and `Torch_DIR` to that installation, enables all three
Previews, and disables incompatible sanitizer instrumentation. It reuses
`out/coverage/linux/build`; the default build and gcovr parallelism is two.

The existing cache retains other configured options, including
`JYPPX_LIUDING_MODELS_REFERENCE_DIR`. Provision and configure the official model
reference fixtures before collecting that part of the evidence. CPU-only CUDA
skips remain visible in `coverage-ctest.xml` and the CTest log; coverage is not
evidence of actual CUDA execution. The source coverage filter covers library
`src/jyppx/liuding` and `include/jyppx/liuding`, excluding CLI and the Python sidecar.

The script clears only this cache's `.gcda` counters after a successful build.
Both gcovr commands search only this cache, so other Release, sanitizer, fuzz or
consumer builds cannot contribute counters. All enabled library paths remain
in the report. The checked-in owner policy rejects unowned files, reduced
denominators, covered-count regressions and percentage regressions.

The fixed artifact/reference workloads hash more than a GiB. Their 22,615,392
SHA-256 blocks invoke 576 rotations each, producing exactly 13,026,465,792
positive hits. The gcovr suspicious-hit ceiling is a finite 20 billion to retain
these verified hot-loop counts; errors above it still fail. No source path or
counter is discarded for exceeding gcovr's default heuristic.

CTest failure does not prevent collection of the coverage report and policy
result. The overall exit remains nonzero if either CTest or the policy failed.
Configure/build/gcovr failures stop collection. `coverage-metadata.txt` is
written on failure too, with the source identity, dependency prefix, last stage,
individual test/policy exits and overall exit. `not_run` never means pass.

When a separate chapter gate has already passed the complete packaging
matrices, set `JYPPX_COVERAGE_EXCLUDE_PACKAGING=1` to omit their nested builds
from the coverage run. This exact label exclusion is recorded in metadata and
must be paired with the separate packaging evidence. The default excludes no
label. Do not use coverage collection to relabel failed chapter tests as passed.

Reports and logs are under `out/coverage/linux/report`. Archive a completed
observation with its source identity before reusing those report names. After
a reviewed policy-only correction, rerun `coverage_gate.py` against the saved
summary; no recompilation or CTest repeat is needed.
