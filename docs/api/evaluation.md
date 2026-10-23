# Local evaluation evidence

Target: `jyppx::liuding::experiment`; header: `jyppx/experiment/evaluation.hpp`.

An `EvaluationSuite` freezes its task, dataset digest, environment, seed and ordered rules. A rule selects an exact neutral metric definition and context, including unit, reduction, direction, parameters and required sample count. Changing these fields changes the canonical suite digest. The producer must assign a new suite version when changing its contract.

`evaluate_observations` validates caller measurements. `run_evaluation` first
validates the suite and identity, invokes one trusted C++ adapter, and checks
cancellation/deadline again before evaluating the returned measurements. It
does not preempt arbitrary caller code. The CLI accepts a measured-input file;
it does not load executable adapters from JSON.

`EvaluationInput` schema 1 contains `schema_version`, `identity`, and
`observations`. Use `evaluation_input_json` and `parse_evaluation_input_json`
for this wire format. Identity contains evaluation/run IDs, exact model and
config SHA-256 digests, actual producer build fingerprint, and adapter ID/version.
An observation carries its complete metric definition, typed step, value status,
context and optional sample count. NaN, infinity and missing values retain their
status with a null numerical value; they never become a passing zero.

All JSON codecs reject duplicate, missing and unknown keys, malformed UTF-8,
unsupported schemas, negative unsigned values and configured resource limits.
There is at most one final observation per metric-definition/context key.
Unknown observations cannot be silently dropped from an evaluation.

Encoders and decoders apply the same byte, node, container-depth and UTF-8
string-byte budgets. A node is a container, an object key or a scalar value;
the root container has depth one. String budgets include JSON field names and
telemetry parameter/context keys and values. Successfully encoded evidence can
be decoded with the same limits. A report counts its complete embedded suite
and all retained observations together, including repeated metric definitions;
`evaluate_observations` rejects a report that exceeds this aggregate budget.

## Reports and comparisons

`QualityReport` retains the full suite, producer provenance and one result per
rule. Its JSON decoder and encoder recompute rule outcomes and the overall
verdict; edited verdicts and reasons are rejected. `quality_report_html` embeds
the escaped canonical report in standalone HTML with no scripts or external
resources. A report describes supplied evidence; it does not authenticate its
producer or imply model inference took place inside the report codec.

| Verdict | Meaning |
| --- | --- |
| `pass` | All measured applicable rules met their inclusive bounds. |
| `fail` | A finite measurement or directional regression violated a bound. |
| `incomplete` | A required observation, finite value or known sample count is missing. |
| `warn` | Reserved usable evidence with an explicit advisory limitation. |
| `invalid` | Reserved incompatible evidence; malformed contracts throw typed errors. |

`compare_quality_reports` requires equal suite digests and adapter contracts,
and equal step meanings for paired observations. Different model/config/build identities remain visible. It checks both candidate thresholds and declared absolute/relative directional regression tolerances. Relative loss from a zero baseline is incomplete when it cannot be defined. These comparisons make no statistical significance or confidence interval claim.

## CLI and persistence

```sh
liuding-forge evaluation run --suite suite.json --input measured.json \
  --output quality.json --html quality.html --format json
liuding-forge evaluation show --report quality.json --format text
liuding-forge evaluation compare --baseline baseline.json --candidate quality.json --format json
```

To retain a report in an existing local run, add `--root RUNS --reference-id ID --attached-ms UNIX_MS`. The input identity supplies the run ID. The command publishes canonical report bytes to CAS and an `evaluation.report` relationship. Reuse the same reference ID, timestamp and content for retries; different content under that relationship is rejected. The returned digest identifies the exact canonical report. Output files require an existing parent directory; identical output permits replay and a different existing output is rejected. Each invocation exclusively owns its output paths. Local presentation files are separate effects from the durable CAS relationship.

Successful commands use exit 0; completed failing/incomplete evidence uses 1; invalid input uses 2; operational failure uses 3. Cancellation/deadline use 130/124. Both text and schema-1 JSON output expose the same verdict.

## Basic projected data quality

`inspect_data_quality` takes a closed numeric field schema and bounded projected
rows. Each row has a sample ID, split, immutable content digest, scalar fields
and optional label. It counts missing/nonfinite/out-of-range/unknown fields,
duplicate IDs/content, exact cross-split content leakage and conflicting labels.
An empty dataset is incomplete. Reports contain aggregate counts; no sample
IDs, labels, scalar values or raw payloads are copied into them.

`liuding-forge data inspect|verify --input projection.json --format text|json` uses
schema 1: `schema_version`, `schema`, `rows`. A schema field contains exactly
`required`, `minimum`, `maximum` (nullable bounds). Each row contains exactly
`sample_id`, `split`, `content_digest` (`algorithm`, `hex`), `fields`, and
nullable `label`. Scalar values are finite JSON numbers or null. The C++ API also accepts explicit IEEE nonfinite scalar inputs. Tensor dtype/rank checks belong to data/model schemas; near duplicates, fairness, robustness and statistical inference are outside this basic data projection checker.

This chapter is awaiting consolidated Windows and Ubuntu/CUDA qualification.
