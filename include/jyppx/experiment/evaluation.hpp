// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_EXPERIMENT_EVALUATION_HPP
#define JYPPX_EXPERIMENT_EVALUATION_HPP

/** @file evaluation.hpp
 * @brief Bounded local evaluation evidence, data checks, and baseline
 * comparison.
 */

#include <jyppx/foundation/artifact.hpp>
#include <jyppx/foundation/telemetry.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::experiment {

/** @brief Machine verdict; unavailable evidence never becomes a pass. */
enum class QualityVerdict : std::uint8_t {
  pass = 1,        ///< All applicable requirements were met.
  fail = 2,        ///< A measured requirement was violated.
  warn = 3,        ///< Usable evidence with a declared advisory limitation.
  incomplete = 4,  ///< Required observations or samples are unavailable.
  invalid = 5,     ///< Evidence definitions or provenance are incompatible.
};

/** @brief Renders a stable lowercase verdict.
 * @param verdict Verdict to render.
 * @return Stable name, or unknown for an invalid value.
 */
std::string_view quality_verdict_name(QualityVerdict verdict) noexcept;

/** @brief One immutable rule selecting a complete metric definition/context. */
struct EvaluationRule {
  std::string rule_id;  ///< Unique identity within the suite version.
  TelemetryMetricDefinition
      definition;  ///< Exact unit/reduction/parameter schema.
  TelemetryAttributes
      context;  ///< Exact slice/split context, without wildcards.
  std::uint64_t minimum_samples = 1U;  ///< Required known sample count.
  std::optional<double> minimum;  ///< Inclusive lower threshold, if present.
  std::optional<double> maximum;  ///< Inclusive upper threshold, if present.
  std::optional<double> max_absolute_regression;  ///< Allowed directional loss.
  std::optional<double>
      max_relative_regression;  ///< Allowed loss / abs(baseline).
};

/** @brief Immutable versioned local suite; changing a rule changes its digest.
 */
struct EvaluationSuite {
  std::uint32_t schema_version = 1U;  ///< Local suite wire schema.
  std::string suite_id;               ///< Stable suite identity.
  std::uint32_t version = 1U;         ///< Positive immutable suite version.
  std::string task;               ///< Trusted caller adapter's declared task.
  ArtifactDigest dataset_digest;  ///< Immutable evaluated dataset manifest.
  std::string environment_id;     ///< Comparable runtime/workload environment.
  std::uint64_t seed =
      0U;  ///< Declared evaluation seed, not an implicit RNG reset.
  std::vector<EvaluationRule> rules;  ///< Ordered, nonempty metric rules.
};

/** @brief Exact producer identities for one independent evaluation execution.
 */
struct EvaluationIdentity {
  std::string
      evaluation_id;   ///< Stable idempotency identity for this execution.
  std::string run_id;  ///< Source training/local run identity.
  ArtifactDigest model_digest;    ///< Exact model/weights/bundle evaluated.
  std::string build_fingerprint;  ///< Actual evaluator build identity.
  std::string adapter_id;         ///< Trusted task adapter identity.
  std::uint32_t adapter_version = 1U;  ///< Positive adapter contract version.
  ArtifactDigest config_digest;  ///< Frozen redacted evaluator configuration.
};

/** @brief Portable measured input; parsing it never runs a model or plugin. */
struct EvaluationInput {
  std::uint32_t schema_version = 1U;  ///< Measured-input wire schema.
  EvaluationIdentity
      identity;  ///< Actual producer/run/model/config provenance.
  std::vector<TelemetryMetricPoint>
      observations;  ///< Final caller measurements.
};

/** @brief One report rule, retaining unavailable/nonfinite observations. */
struct QualityRuleResult {
  std::string rule_id;  ///< Matching immutable suite rule.
  QualityVerdict verdict = QualityVerdict::incomplete;  ///< Machine outcome.
  std::string reason;  ///< Redacted explanatory reason, empty for pass.
  std::optional<TelemetryMetricPoint>
      observation;  ///< Original measured result.
};

/** @brief Immutable, fully self-describing local evidence report. */
struct QualityReport {
  std::uint32_t schema_version = 1U;  ///< Local report wire schema.
  EvaluationSuite suite;  ///< Full immutable suite, including all thresholds.
  EvaluationIdentity
      identity;  ///< Actual producer/model/run/config provenance.
  QualityVerdict verdict = QualityVerdict::incomplete;  ///< Aggregate outcome.
  std::vector<QualityRuleResult>
      results;  ///< Exactly one ordered result per rule.
};

/** @brief Per-rule directional differences between compatible reports. */
struct QualityComparisonResult {
  std::string rule_id;  ///< Exact suite rule identity.
  QualityVerdict verdict = QualityVerdict::incomplete;  ///< Threshold outcome.
  std::optional<double>
      candidate_minus_baseline;      ///< Finite signed difference.
  std::optional<double> regression;  ///< Positive directional degradation.
  std::string reason;  ///< Unavailable or failed comparison explanation.
};

/** @brief Detached comparison retaining both immutable report identities. */
struct QualityComparison {
  ArtifactDigest baseline_report_digest;   ///< Exact baseline canonical JSON.
  ArtifactDigest candidate_report_digest;  ///< Exact candidate canonical JSON.
  QualityVerdict verdict = QualityVerdict::incomplete;  ///< Aggregate outcome.
  std::vector<QualityComparisonResult> results;  ///< Suite-ordered comparisons.
};

/** @brief Explicit resource limits for objects, parsers, and reports. */
struct EvaluationLimits {
  std::size_t max_rules = 1024U;       ///< Maximum suite rules/observations.
  std::size_t max_text_bytes = 4096U;  ///< Maximum one identity/reason/string.
  std::size_t max_json_bytes =
      8U * 1024U * 1024U;  ///< Maximum one wire artifact.
  std::size_t max_json_nodes =
      262144U;                       ///< Maximum parser nodes, keys included.
  std::size_t max_json_depth = 24U;  ///< Maximum container nesting depth.
  std::size_t max_data_rows = 1000000U;  ///< Maximum projected data rows.
  std::size_t max_data_fields = 256U;    ///< Maximum projected scalar columns.
  std::size_t max_data_text_bytes =
      64U * 1024U * 1024U;      ///< Aggregate row text.
  TelemetryLimits telemetry{};  ///< Reused metric semantic/UTF-8/byte bounds.
};

/** @brief Serializes measured inputs with the shared neutral metric semantics.
 * @param input Actual provenance and bounded measured observations.
 * @param limits Explicit resource/semantic limits.
 * @return Canonical schema-v1 JSON; unavailable values retain status fields.
 * @throws Error On invalid identity, observations, schema, or resource limits.
 */
std::string evaluation_input_json(const EvaluationInput &input,
                                  const EvaluationLimits &limits = {});

/** @brief Parses strict measured-input JSON without executing caller code.
 * @param text Strict JSON with no duplicate, missing, or unknown keys.
 * @param limits Explicit parser, identity and observation limits.
 * @return Detached validated input for evaluate_observations.
 * @throws Error On malformed input, unsupported schema, or resource limits.
 */
EvaluationInput parse_evaluation_input_json(
    std::string_view text, const EvaluationLimits &limits = {});

/** @brief Validates suite identities, rules, finite bounds and metric contexts.
 * @param suite Complete immutable suite.
 * @param limits Explicit metadata/resource bounds.
 * @throws Error For unsupported schemas, malformed contracts, or limits.
 */
void validate_evaluation_suite(const EvaluationSuite &suite,
                               const EvaluationLimits &limits = {});

/** @brief Encodes an immutable suite as canonical bounded strict JSON.
 * @param suite Valid suite with complete semantic definitions.
 * @param limits Explicit resource bounds.
 * @return Deterministic UTF-8 JSON; negative and positive zero normalize alike.
 * @throws Error For invalid objects, resource limits, or encoding failures.
 */
std::string evaluation_suite_json(const EvaluationSuite &suite,
                                  const EvaluationLimits &limits = {});

/** @brief Parses exact schema-v1 suite fields with bounded allocation work.
 * @param text Strict UTF-8 JSON; duplicate and unknown keys are rejected.
 * @param limits Explicit parser and semantic bounds.
 * @return Detached validated suite.
 * @throws Error For malformed JSON, schema mismatch, or resource limits.
 */
EvaluationSuite parse_evaluation_suite_json(
    std::string_view text, const EvaluationLimits &limits = {});

/** @brief Hashes the complete canonical immutable suite including thresholds.
 * @param suite Valid immutable suite.
 * @param limits Explicit resource bounds.
 * @return SHA-256 of its canonical JSON.
 */
ArtifactDigest evaluation_suite_digest(const EvaluationSuite &suite,
                                       const EvaluationLimits &limits = {});

/** @brief Evaluates caller-measured task observations against the frozen suite.
 * @param suite Frozen rules/dataset/environment/seed.
 * @param identity Actual producer/run/model/config provenance.
 * @param observations Complete final observations, at most one per rule key.
 * @param stop Cooperative cancellation.
 * @param deadline Optional absolute operation deadline.
 * @param limits Explicit semantic and resource bounds.
 * @return Self-describing report; missing/nonfinite results are incomplete.
 * @note This function does no model inference and does not execute input code.
 * @throws Error On malformed/duplicate/unknown observations or cancellation.
 */
QualityReport evaluate_observations(
    const EvaluationSuite &suite, const EvaluationIdentity &identity,
    std::span<const TelemetryMetricPoint> observations,
    std::stop_token stop = {}, TelemetryDeadline deadline = std::nullopt,
    const EvaluationLimits &limits = {});

/** @brief Runs one trusted in-process task adapter and creates its report.
 * @param suite Frozen suite passed to the adapter.
 * @param identity Actual producer provenance, independent of adapter outputs.
 * @param evaluator Trusted caller code; must cooperate with stop/deadline.
 * @param stop Cooperative cancellation.
 * @param deadline Optional absolute deadline, also checked after adapter
 * return.
 * @param limits Resource bounds for returned observations and report.
 * @return Immutable evidence report after complete validation.
 * @note No process isolation or preemption is claimed. Untrusted code belongs
 * in a separately managed sidecar and supplies measured observations instead.
 * @throws Error On failure, malformed adapter output, or cancellation/deadline.
 */
QualityReport run_evaluation(
    const EvaluationSuite &suite, const EvaluationIdentity &identity,
    const std::function<std::vector<TelemetryMetricPoint>(
        const EvaluationSuite &, std::stop_token, TelemetryDeadline)>
        &evaluator,
    std::stop_token stop = {}, TelemetryDeadline deadline = std::nullopt,
    const EvaluationLimits &limits = {});

/** @brief Validates and serializes full report; results are recomputed to
 * audit.
 * @param report Complete immutable report.
 * @param limits Explicit resource bounds.
 * @return Canonical UTF-8 schema-v1 JSON.
 * @throws Error For invalid provenance, tampered verdicts, or resource limits.
 */
std::string quality_report_json(const QualityReport &report,
                                const EvaluationLimits &limits = {});

/** @brief Parses and verifies report definitions, rule results and verdicts.
 * @param text Strict schema-v1 UTF-8 JSON without duplicate/unknown fields.
 * @param limits Explicit parser/resource bounds.
 * @return Detached verified report.
 * @throws Error For malformed or internally inconsistent evidence.
 */
QualityReport parse_quality_report_json(std::string_view text,
                                        const EvaluationLimits &limits = {});

/** @brief Produces escaped standalone HTML of the verified machine report.
 * @param report Immutable report whose machine verdict is not modified.
 * @param limits Output and semantic bounds.
 * @return Self-contained HTML without scripts, remote resources or raw samples.
 * @throws Error For invalid report evidence or resource limits.
 */
std::string quality_report_html(const QualityReport &report,
                                const EvaluationLimits &limits = {});

/** @brief Compares equal suite/dataset/environment/adapter definitions.
 * @param baseline Verified baseline report.
 * @param candidate Verified candidate report with a different model if desired.
 * @param limits Explicit resource bounds.
 * @return Directional differences and rule threshold verdicts.
 * @note Different builds/configs are recorded, not silently equated. Equal
 * environment IDs assert comparable workloads; no statistical claim is made.
 * @throws CompatibilityError For suite or adapter contract mismatches.
 * @throws Error For invalid reports or resource limits.
 */
QualityComparison compare_quality_reports(const QualityReport &baseline,
                                          const QualityReport &candidate,
                                          const EvaluationLimits &limits = {});

/** @brief A required/optional scalar projection used for basic data checks. */
struct DataQualityField {
  bool required = true;  ///< Absence or null counts as missing when required.
  std::optional<double> minimum;  ///< Inclusive allowed finite minimum.
  std::optional<double> maximum;  ///< Inclusive allowed finite maximum.
};

/** @brief One caller-projected sample; payload bytes are never copied to
 * report. */
struct DataQualityRow {
  std::string sample_id;          ///< Stable dataset sample identity.
  std::string split;              ///< Exact logical split name.
  ArtifactDigest content_digest;  ///< Exact canonical content identity.
  std::map<std::string, std::optional<double>> fields;  ///< Scalar projection.
  std::optional<std::string> label;  ///< Optional categorical label projection.
};

/** @brief Basic schema/missing/range/duplicate/leakage counts only. */
struct DataQualityReport {
  std::uint64_t checked_rows = 0U;      ///< Rows inspected.
  std::uint64_t missing_values = 0U;    ///< Required absent/null scalar values.
  std::uint64_t nonfinite_values = 0U;  ///< Present NaN/Inf scalar values.
  std::uint64_t out_of_range_values = 0U;  ///< Finite values outside bounds.
  std::uint64_t unknown_fields = 0U;       ///< Undeclared scalar fields.
  std::uint64_t duplicate_ids =
      0U;  ///< Repeated (split, sample ID) occurrences.
  std::uint64_t duplicate_contents =
      0U;  ///< Repeated (split, digest) occurrences.
  std::uint64_t cross_split_leaks =
      0U;  ///< Content encountered in another split.
  std::uint64_t conflicting_labels = 0U;  ///< Same digest with unequal labels.
  QualityVerdict verdict =
      QualityVerdict::incomplete;  ///< Empty means incomplete.
};

/** @brief Checks bounded scalar projections and exact content-based split
 * leaks.
 * @param schema Closed numeric scalar field schema and inclusive ranges.
 * @param rows Caller-projected sample IDs/digests/scalars; no raw samples.
 * @param stop Cooperative cancellation checked while iterating.
 * @param deadline Optional absolute operation deadline.
 * @param limits Explicit row/field/string resource bounds.
 * @return Counts with no sample IDs, labels, values or payloads in the report.
 * @note Does not infer near duplicates or assert tensor dtype/rank validation.
 * @throws Error For malformed schema, identifiers, digests, or resource limits.
 */
DataQualityReport inspect_data_quality(
    const std::map<std::string, DataQualityField> &schema,
    std::span<const DataQualityRow> rows, std::stop_token stop = {},
    TelemetryDeadline deadline = std::nullopt,
    const EvaluationLimits &limits = {});

}  // namespace jyppx::experiment

#endif  // JYPPX_EXPERIMENT_EVALUATION_HPP
