# Library command-line tools

The product CLI is `liuding-forge` (`liuding-forge.exe` on Windows), branded
六丁炼算（Liuding Forge）. It composes installed library components to inspect
the environment, resolve configuration, manage local runs, evaluate measured
evidence and inspect models/checkpoints. The old `jyppx` executable name is not
installed.

Configure with `JYPPX_LIUDING_BUILD_TOOLS=ON` to build and install it. The default is ON
when all SDK components are enabled, and OFF for partial component profiles.
Explicit ON requires the complete SDK closure; it never silently enables a
disabled library component. `JYPPX_LIUDING_BUILD_TOOLS=OFF` leaves the library usable.
The executable is installed in `bin/`, without development scripts or tests.

Use `liuding-forge --help --format text` for commands and `liuding-forge --version --format json`
for the configured build identity. Commands accept `--format text|json`; text
uses readable field/value lines, JSON uses a single schema-1 envelope:

```json
{"schema_version":1,"status":"ok","data":{}}
```

An error uses `status: "error"` and an `error` object with stable numeric code,
name, redacted message, component, operation and hint. Input content and generic
library exception dumps are not echoed. Explicit producer annotations still
must be redacted before ingestion. Exit codes are 0 for success, 1 for a
completed failed/incomplete verdict, 2 for invalid/unsupported input, 3 for
operational failure, 124 for deadline and 130 for cancellation.

```sh
liuding-forge doctor --device cpu --format text
liuding-forge doctor --device cuda --expected-libtorch 2.5.1 --format json
liuding-forge config lint --schema config-schema.json --file training.json --format text
liuding-forge config render --schema config-schema.json --set /training/rate=0.01 --format json
liuding-forge config explain --schema config-schema.json --path /training/rate --format text
liuding-forge config migrate --input old.json --plan migration-plan.json --to-version 3 --format json
liuding-forge api scan --root application-src --format json
liuding-forge run list --root local-runs --format text
liuding-forge model verify --path weights.safetensors --format json
liuding-forge model bundle verify --path immutable-model-bundle --format json
liuding-forge checkpoint verify --path checkpoint.jyck --format json
```

`doctor` exercises a real tensor/autograd/serialization operation on the
selected device. An unavailable CUDA device or expected LibTorch mismatch is
an explicit failed check. The CLI does not install a driver or substitute CPU
for requested CUDA.

Config schemas use schema 1 with `schema_id`, positive `version` and a `root`
value schema. `defaults`, `environment_bindings` and `secret_paths` are optional.
Value schema kinds are dictionary/list/tuple/integer/floating/boolean/string.
Dictionary fields, list element and tuple elements are declared explicitly.
The resolver applies defaults, ordered files, declared environment bindings and
ordered CLI overrides. Rendering/explanation use the library's redacted values;
the resulting digest is a redacted configuration identity.

Run metadata inputs require a caller-supplied `created_at_unix_ms`; retries do
not fabricate a new start time. See [local run storage](../api/local_run_storage.md)
for query/export/backup and [evaluation](../api/evaluation.md) for measured-input
schemas, immutable reports and data quality. The [offline workflow](../examples/offline_workflow.md)
exercises training and recovery together.

The optional telemetry runtime may be installed separately with
`JYPPX_LIUDING_INSTALL_TELEMETRY_SIDECAR=ON`. Its source and locked requirements land in
`share/jyppx/telemetry_sidecar/`; no Python packages are installed by CMake.
The caller explicitly supplies a locked interpreter and backend configuration.
Neither Python nor its optional SDK packages are a core library runtime
dependency. See [process transport](../api/telemetry_sidecar_process.md).

`model bundle inspect|verify` performs the same complete bounded directory,
hash, size and safetensors mapping verification before returning metadata.
The result separates integrity from signature authentication and runtime
compatibility. The CLI has no configured cryptographic trust provider, so an
unsigned bundle reports `unsigned`, a signature reports `present_unverified`,
and `--require-trusted true` fails. Applications can supply the library's
explicit trusted signature-verifier callback. Declarative schema payloads and
ONNX execution remain the responsibility of their named owners.

Repository maintenance commands belong in `scripts/` and acceptance fixtures
in `tests/`. They are not subcommands of the installed product tool.
