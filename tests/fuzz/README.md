# Bounded parser fuzzing

The original `device_parse`, `data_pipeline_state` and
`iterable_pipeline_state` targets remain available. Six additional targets
exercise production parsers without filesystem/network activity:

| Target suffix | Production boundary | Successful-input invariants |
| --- | --- | --- |
| `telemetry_codec` | Binary envelope and metric-definition decoders | Exact canonical bytes, SHA-256, encode/decode stability and exact byte budgets |
| `checkpoint_manifest` | Strict checkpoint JSON parser | Canonical parse/encode/reparse and exact byte budgets |
| `model_bundle_manifest` | Strict model-bundle JSON parser | Canonical JSON, stable signing payload and exact byte budgets |
| `safetensors` | Complete header/shape/extent parser | Metadata, tensor association and opaque payload preservation; canonical stability and exact file budgets |
| `evaluation_json` | Suite, measured-input and quality-report parsers | Canonical JSON, exact byte budgets, report self-comparison and escaped HTML |
| `backend_protocol` | `OfficialTelemetrySink` reply parser | Actual emit/flush/close receipt validation, immutable replay, exact reply budgets, counter monotonicity and rejection atomicity |

Every new harness accepts raw libFuzzer bytes up to 64 KiB with explicit small
parser collection/string/shape limits. Each also creates and validates real
production encodings once, then applies up to eight edits to a selected valid
seed on every input. Binary edits can preserve the domain/framing prefix to
reach nested fields while the raw-input path mutates all bytes. Corpus files
include valid payloads of each supported kind and short malformed boundary
inputs. `generate_corpus.py` regenerates those data-only fixtures; actual codec
generated seeds are validated by the C++ executables themselves.

Only documented input-rejection `jyppx::Error` codes are caught. Internal
errors, unexpected framework categories, standard exceptions, invariant
failures and sanitizer errors remain visible. Backend-reported errors are
accepted only by the protocol harness. Once an input parses, later
serialization/reparse failures cannot be reclassified as malformed input.

Canonical output has a separate bounded budget because compact source JSON,
number spelling, escaping and safetensors layout can change its byte length.
The exact-size success and one-byte-too-small rejection checks use that output
size. HTML has its own escaped output budget. Tensor values remain opaque bits,
including signed zero and nonfinite floating-point encodings.

`JYPPX_BUILD_FUZZERS=ON` requires Clang and the complete SDK. Only that opt-in
build instruments framework implementation archives with `fuzzer-no-link`,
in addition to the harness coverage and libFuzzer entry point. Third-party
libraries are not rebuilt with libFuzzer coverage.

```sh
CMAKE_PREFIX_PATH=/path/to/libtorch JYPPX_FUZZ_RUNS=10000 \
  bash scripts/validation/run-fuzz.sh
```

The runner always reuses `out/build/fuzz-linux` and preserves discovered corpora
under `out/fuzz/linux`. It runs all nine targets with ASan/UBSan, a fixed seed,
per-input timeout and process RSS bound. `JYPPX_BUILD_PARALLEL_LEVEL`,
`JYPPX_FUZZ_RUNS`, `JYPPX_FUZZ_SEED` and `JYPPX_FUZZ_RSS_LIMIT_MB` customize the
bounded run. Configure/build logs, per-target dictionaries/corpus counts/fuzzer
logs/artifacts and overall success/failure metadata are retained. A failed
target stops the gate and retains its reproducer; no failure is skipped.

These tests cover the C++ official-backend reply boundary. They do not invoke
official Python libraries or replace the separate sidecar request/protocol
integration gate.
