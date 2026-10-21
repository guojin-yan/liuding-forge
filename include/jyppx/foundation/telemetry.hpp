// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_FOUNDATION_TELEMETRY_HPP
#define JYPPX_FOUNDATION_TELEMETRY_HPP

/**
 * @file telemetry.hpp
 * @brief Draft bounded, storage-neutral observation contracts.
 *
 * This file is a next-chapter draft outside the production repository. No
 * implementation, compatibility, or release gate is claimed by its presence.
 */

#include <jyppx/foundation/artifact.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace jyppx {

/** @brief Bounded UTF-8 scalar annotations, ordered by exact key bytes.
 * @note Values are already redacted by the producer; this map is not a secret
 * detector. Correlation identifiers are not automatically metric labels.
 */
using TelemetryAttributes = std::map<std::string, std::string>;

/** @brief Metric units without implicit scaling or conversion. */
enum class TelemetryMetricUnit : std::uint8_t {
  dimensionless = 1,     ///< Unitless scalar without an asserted ratio range.
  count = 2,             ///< Number of items or events.
  seconds = 3,           ///< Duration in SI seconds.
  bytes = 4,             ///< Number of bytes.
  ratio = 5,             ///< Ratio convention, distinct from percent.
  percent = 6,           ///< Percentage convention, distinct from ratio.
  count_per_second = 7,  ///< Item or event rate per SI second.
  bytes_per_second = 8,  ///< Byte rate per SI second.
  unspecified = 9,  ///< Historical producer did not declare units; never infer.
};

/** @brief Declares how the producer calculated a scalar observation.
 * @note A sink must not sum means or average ratios without sufficient
 * statistics. These values do not authorize reaggregation across contexts.
 */
enum class TelemetryAggregation : std::uint8_t {
  last = 1,         ///< Latest independently measured value.
  sum = 2,          ///< Sum over the producer's declared population or window.
  mean = 3,         ///< Arithmetic mean over the declared population or window.
  minimum = 4,      ///< Minimum over the declared population or window.
  maximum = 5,      ///< Maximum over the declared population or window.
  unspecified = 6,  ///< Historical reduction semantics were not declared.
};

/** @brief Explicit comparison direction, without an inferred objective. */
enum class TelemetryMetricDirection : std::uint8_t {
  none = 1,      ///< No improvement ordering is declared.
  minimize = 2,  ///< Smaller compatible finite observations are preferable.
  maximize = 3,  ///< Larger compatible finite observations are preferable.
};

/** @brief Preserves nonfinite and unavailable observations symbolically. */
enum class TelemetryValueStatus : std::uint8_t {
  finite = 1,        ///< A present, finite IEEE-754 binary64 value.
  not_a_number = 2,  ///< The producer observed NaN, without its payload bits.
  positive_infinity = 3,  ///< The producer observed positive infinity.
  negative_infinity = 4,  ///< The producer observed negative infinity.
  missing = 5,            ///< No value was produced; a reason is mandatory.
};

/** @brief Defines the counter measured by a metric's step value. */
enum class TelemetryStepKind : std::uint8_t {
  micro_step = 1,           ///< Microbatch progress before optimizer commit.
  global_step = 2,          ///< Number of committed optimizer updates.
  epoch = 3,                ///< Producer-defined epoch index.
  sample = 4,               ///< Number of consumed samples.
  elapsed_nanoseconds = 5,  ///< Duration since the producer's declared origin.
};

/** @brief Log severity without backend-specific integer conventions. */
enum class TelemetryLogSeverity : std::uint8_t {
  debug = 1,    ///< Diagnostic detail, disabled by many display policies.
  info = 2,     ///< Ordinary progress or informative observation.
  warning = 3,  ///< A recoverable or evidence-limiting condition.
  error = 4,    ///< An operation failed; this does not change run state itself.
};

/** @brief Outcome of one completed trace span. */
enum class TelemetrySpanStatus : std::uint8_t {
  unset = 1,      ///< The producer made no success/failure assertion.
  ok = 2,         ///< The measured operation completed successfully.
  error = 3,      ///< The measured operation failed.
  cancelled = 4,  ///< The measured operation honored cancellation.
};

/** @brief Observation reliability for an evidence sink.
 * @note Required lifecycle/checkpoint/error evidence cannot be sampled out.
 * Optional visualization sinks can report an unsupported payload explicitly.
 */
enum class TelemetryReliability : std::uint8_t {
  required = 1,  ///< Evidence sinks must retain or reject this observation.
  best_effort =
      2,  ///< A declared sampling/drop policy may omit the observation.
};

/** @brief Stable tags for the supported schema-v1 payload alternatives. */
enum class TelemetryPayloadKind : std::uint8_t {
  lifecycle = 1,           ///< Lifecycle snapshot and audited decisions.
  metric_batch = 2,        ///< One bounded vector of scalar metric points.
  log = 3,                 ///< One structured, already redacted log body.
  span = 4,                ///< One completed operation span.
  resource_sample = 5,     ///< Resource measurements for one logical resource.
  artifact_reference = 6,  ///< Reference to committed immutable content.
};

/** @brief Self-describing identity of one immutable metric definition.
 *
 * Changing units, aggregation, parameters, direction, or context keys requires
 * a new definition version. A store compares the definition digest as well as
 * its human-readable ID/version and rejects conflicting redefinitions.
 */
struct TelemetryMetricDefinition {
  std::uint32_t schema_version = 1U;  ///< Definition schema, independent of ID.
  std::string metric_id;              ///< Stable semantic metric identity.
  std::uint32_t version = 1U;  ///< Positive immutable definition version.
  TelemetryMetricUnit unit = TelemetryMetricUnit::dimensionless;  ///< Unit.
  TelemetryAggregation aggregation =
      TelemetryAggregation::last;  ///< Reduction.
  TelemetryMetricDirection direction =
      TelemetryMetricDirection::none;  ///< Goal.
  TelemetryAttributes parameters;  ///< Semantic parameters, such as averaging.
  std::set<std::string>
      context_keys;  ///< Exact required per-point context keys.
};

/** @brief A scalar observation with a portable nonfinite/missing encoding.
 * @note `value` is present only for finite status. No raw NaN/Inf enters the
 * canonical representation, and a missing value always has a nonempty reason.
 */
struct TelemetryMetricValue {
  TelemetryValueStatus status = TelemetryValueStatus::missing;  ///< Value kind.
  std::optional<double>
      value;           ///< Present finite value; signed zeros hash alike.
  std::string reason;  ///< Redacted explanation, mandatory for missing.
};

/** @brief One typed nonnegative progress coordinate. */
struct TelemetryStep {
  TelemetryStepKind kind =
      TelemetryStepKind::global_step;  ///< Counter meaning.
  std::uint64_t value = 0U;  ///< Count or index, never inferred from wall time.
};

/** @brief A scalar observation carrying its complete definition and context. */
struct TelemetryMetricPoint {
  TelemetryMetricDefinition definition;  ///< Immutable semantics to persist.
  TelemetryStep step;                    ///< Coordinate and its counter kind.
  TelemetryMetricValue observation;      ///< Finite/nonfinite/missing result.
  TelemetryAttributes context;  ///< Exact key set declared by the definition.
  std::optional<std::uint64_t> sample_count;  ///< Known contributing count.
};

/** @brief Stable producer identity and retry coordinates for one observation.
 *
 * A stream belongs to one producer attempt; sequence starts at one and never
 * resets within it. `event_id`, `(stream_id, sequence)`, and the idempotency
 * tuple must identify the same immutable envelope. Delivery retries reuse the
 * entire envelope, including timestamps. A new run attempt uses a new stream.
 */
struct TelemetryIdentity {
  std::string stream_id;  ///< Stable producer stream/attempt identity.
  std::string event_id;   ///< Stable producer event identity in the stream.
  std::string idempotency_key;  ///< Caller operation key, scoped to the stream.
  std::uint64_t sequence = 0U;  ///< Positive gap-detectable producer sequence.
};

/** @brief Observation timestamps without inventing missing historical clocks.
 *
 * Wall time is signed Unix nanoseconds. Monotonic time is comparable only
 * within the same nonempty clock ID. A restarted process must use a new clock
 * ID. Old persisted observations may have neither timestamp; their adapter
 * records an explicit provenance annotation instead of sampling the load time.
 */
struct TelemetryTime {
  std::optional<std::int64_t>
      wall_time_unix_ns;  ///< Producer wall-clock sample.
  std::optional<std::uint64_t>
      monotonic_time_ns;  ///< Producer monotonic sample.
  std::string clock_id;   ///< Required exactly when monotonic time is present.
};

/** @brief Optional opaque correlation IDs; none are inferred metric labels.
 *
 * Trace/span IDs are native IDs, not promises of an OTel wire representation.
 * Exporters preserve them in attributes when mapping to backend trace IDs.
 */
struct TelemetryCorrelation {
  std::string project_id;  ///< Stable local project, if applicable.
  std::string run_id;      ///< Stable local run, if applicable.
  std::string attempt_id;  ///< Execution attempt; requires a run ID.
  std::string trace_id;    ///< Logical distributed or local trace context.
  std::string span_id;     ///< Current span; requires a trace ID.
  std::string
      parent_span_id;  ///< Parent span, if any; must differ from span ID.
};

/** @brief One neutral audited producer decision, without training enums.
 * @note The vector order is meaningful. At most one decision is effective in
 * a lifecycle payload; a training adapter preserves dispatcher priority/order.
 */
struct TelemetryDecision {
  std::string actor_id;  ///< Callback, evaluator, or other deciding actor.
  std::string action;  ///< Stable domain action name, without executable code.
  std::string reason;  ///< Nonempty redacted explanation.
  bool effective =
      false;  ///< Whether this decision controls the domain outcome.
};

/** @brief One committed lifecycle observation, optionally including metrics. */
struct TelemetryLifecycle {
  std::string name;  ///< Domain event name, such as `training.epoch_finished`.
  bool terminal = false;  ///< Producer declares its stream terminal here.
  std::vector<TelemetryMetricPoint>
      metrics;  ///< Metrics on this same snapshot.
  std::vector<TelemetryDecision>
      decisions;  ///< Ordered, immutable decision audit.
};

/** @brief A nonempty bounded scalar metric batch. */
struct TelemetryMetricBatch {
  std::vector<TelemetryMetricPoint> points;  ///< Producer-ordered observations.
};

/** @brief A redacted structured log with envelope-level correlation/attributes.
 */
struct TelemetryLog {
  TelemetryLogSeverity severity =
      TelemetryLogSeverity::info;  ///< Log severity.
  std::string
      body;  ///< Nonempty bounded UTF-8 text; no raw tensor/sample dump.
};

/** @brief A completed span measured in the envelope's monotonic clock domain.
 */
struct TelemetrySpan {
  std::string name;  ///< Stable operation name, not an unbounded request URL.
  std::uint64_t start_monotonic_ns = 0U;  ///< Inclusive operation start sample.
  std::uint64_t end_monotonic_ns = 0U;  ///< End sample, at or after the start.
  TelemetrySpanStatus status = TelemetrySpanStatus::unset;  ///< Domain outcome.
  std::string reason;  ///< Optional redacted outcome explanation.
};

/** @brief A nonempty metric vector measured for one logical resource. */
struct TelemetryResourceSample {
  std::string resource_id;  ///< Stable CPU/device/worker identity within a run.
  std::vector<TelemetryMetricPoint> measurements;  ///< Resource scalar points.
};

/** @brief A relationship to bytes already committed in an artifact store.
 * @note Validation checks metadata only; the receiving durable store verifies
 * the digest against committed bytes before acknowledging a relationship.
 */
struct TelemetryArtifactReference {
  std::string role;  ///< Semantic role, such as `checkpoint.last` or `report`.
  ArtifactRef artifact;  ///< Foundation identity; no path, credential, or blob.
};

/** @brief Schema-v1 observation payload, without an unbounded JSON escape
 * hatch. */
using TelemetryPayload =
    std::variant<TelemetryLifecycle, TelemetryMetricBatch, TelemetryLog,
                 TelemetrySpan, TelemetryResourceSample,
                 TelemetryArtifactReference>;

/** @brief One immutable-in-use, versioned observation accepted by all sinks.
 * @note Fields own their memory. Producers may build/move values freely before
 * enqueue; after acceptance, retained copies and retries must remain identical.
 */
struct TelemetryEnvelope {
  std::uint32_t schema_version = 1U;  ///< Neutral envelope schema version.
  TelemetryIdentity identity;  ///< Stable stream, event, retry, sequence.
  TelemetryTime time;          ///< Captured once before durable acceptance.
  TelemetryCorrelation correlation;  ///< Opaque lineage and trace coordinates.
  TelemetryReliability reliability =
      TelemetryReliability::required;  ///< Policy.
  TelemetryAttributes attributes;      ///< Redacted bounded scalar annotations.
  TelemetryPayload payload;  ///< Exactly one supported typed alternative.
};

/** @brief Limits validated before any durable acknowledgement or export.
 * @note Limits apply to already-owned objects; parsers must enforce the same
 * bounds while decoding, before allocating attacker-declared collections.
 */
struct TelemetryLimits {
  std::size_t max_identifier_bytes =
      256U;                              ///< One nonempty identity/key bytes.
  std::size_t max_text_bytes = 4096U;    ///< One redacted UTF-8 text value.
  std::size_t max_attributes = 64U;      ///< Envelope attributes or parameters.
  std::size_t max_context_fields = 32U;  ///< Metric context schema/value count.
  std::size_t max_metrics_per_envelope = 4096U;  ///< Total metric observations.
  std::size_t max_decisions_per_envelope =
      1024U;  ///< Ordered lifecycle decisions.
  std::size_t max_batch_events =
      1024U;  ///< Maximum envelopes per sink operation.
  std::uint64_t max_envelope_bytes =
      1024U * 1024U;  ///< Canonical envelope bytes.
  std::uint64_t max_batch_bytes =
      16U * 1024U * 1024U;  ///< Sum of envelope bytes.
  std::uint64_t max_artifact_reference_bytes = 1ULL
                                               << 40U;  ///< Declared blob size.
};

/** @brief Returns the stable tag of the active payload alternative.
 * @param envelope Observation whose typed payload is inspected.
 * @return Persisted payload tag, or no value for a valueless variant.
 */
std::optional<TelemetryPayloadKind> telemetry_payload_kind(
    const TelemetryEnvelope& envelope) noexcept;

/** @brief Normalizes a raw scalar without preserving NaN implementation bits.
 * @param value Finite/nonfinite raw scalar, or no value for missing.
 * @param reason Redacted explanation; mandatory when value is missing.
 * @return Symbolic observation suitable for validation/canonical hashing.
 * @throws InvalidArgument If a missing observation has no explanation.
 */
TelemetryMetricValue make_telemetry_metric_value(std::optional<double> value,
                                                 std::string reason = {});

/** @brief Validates positive bounds and their batch/envelope relationship.
 * @param limits Bounds to inspect without modification.
 * @throws InvalidArgument If a bound is zero or batch bytes are too small.
 */
void validate_telemetry_limits(const TelemetryLimits& limits);

/** @brief Validates a complete immutable metric definition.
 * @param definition Definition to inspect without modification.
 * @param limits String, collection, and canonical byte bounds.
 * @throws Error For an invalid field, unknown schema, or resource violation.
 */
void validate_telemetry_metric_definition(
    const TelemetryMetricDefinition& definition,
    const TelemetryLimits& limits = {});

/** @brief Validates semantics, UTF-8, identity, payload, and encoded byte
 * limits.
 * @param envelope Observation to inspect without mutation or external I/O.
 * @param limits Explicit resource bounds.
 * @throws InvalidArgument On invalid enum/text/identity or inconsistent values.
 * @throws CompatibilityError On an unsupported schema version.
 * @throws ResourceExhausted On a configured resource bound violation.
 */
void validate_telemetry_envelope(const TelemetryEnvelope& envelope,
                                 const TelemetryLimits& limits = {});

/** @brief Preflights an entire bounded batch before a sink makes side effects.
 *
 * Identical retries within a batch are valid. Reusing a stream/key, stream/
 * sequence, or stream/event ID for different content is a compatibility error.
 * Durable cross-batch sequence, terminal, and conflict checks belong to stores.
 * @param envelopes Input batch; an empty batch is a valid no-op.
 * @param limits Per-envelope, aggregate byte, and count limits.
 * @throws Error If any input is invalid, oversized, or conflicts in the batch.
 */
void validate_telemetry_batch(std::span<const TelemetryEnvelope> envelopes,
                              const TelemetryLimits& limits = {});

/** @brief Encodes one validated observation as bounded canonical schema-v1
 * bytes.
 * @param envelope Complete immutable observation to encode.
 * @param limits Field, collection, and aggregate byte bounds.
 * @return Canonical bytes whose SHA-256 equals `telemetry_content_digest`.
 * @note This internal persistence codec does not replace JSONL/TensorBoard/OTel
 * exporters. No producer-specific callback or database fields are serialized.
 * @throws Error On invalid input, unsupported schema, or a resource violation.
 */
std::vector<std::byte> encode_telemetry_envelope(
    const TelemetryEnvelope& envelope, const TelemetryLimits& limits = {});

/** @brief Decodes one complete bounded canonical observation before side
 * effects.
 * @param bytes Exactly one canonical envelope; trailing bytes are rejected.
 * @param limits Bounds checked before variable-length allocation.
 * @return Fully validated detached observation with its original identity/time.
 * @throws SerializationError On truncation, noncanonical encoding, or malformed
 * UTF-8/scalar/collection data.
 * @throws CompatibilityError On an unsupported schema or critical payload tag.
 * @throws ResourceExhausted On any declared or actual resource violation.
 */
TelemetryEnvelope decode_telemetry_envelope(std::span<const std::byte> bytes,
                                            const TelemetryLimits& limits = {});

/** @brief Encodes one validated metric definition as bounded canonical bytes.
 * @param definition Immutable self-describing definition to persist.
 * @param limits String, collection, and aggregate byte bounds.
 * @return Canonical bytes whose SHA-256 equals its definition digest.
 * @throws Error On invalid fields, unsupported schema, or a resource violation.
 */
std::vector<std::byte> encode_telemetry_metric_definition(
    const TelemetryMetricDefinition& definition,
    const TelemetryLimits& limits = {});

/** @brief Decodes one canonical metric definition with strict allocation
 * bounds.
 * @param bytes Exactly one canonical definition, without trailing bytes.
 * @param limits Bounds checked before variable-length allocation.
 * @return Fully validated immutable definition.
 * @throws Error On malformed/noncanonical data, version mismatch, or limits.
 */
TelemetryMetricDefinition decode_telemetry_metric_definition(
    std::span<const std::byte> bytes, const TelemetryLimits& limits = {});

/** @brief Hashes the canonical complete metric definition with SHA-256.
 * @param definition Immutable metric definition to validate and identify.
 * @param limits Definition resource bounds.
 * @return Foundation content digest with a domain-separated schema-v1 encoding.
 */
ArtifactDigest telemetry_metric_definition_digest(
    const TelemetryMetricDefinition& definition,
    const TelemetryLimits& limits = {});

/** @brief Hashes the full envelope, including immutable IDs/time/provenance.
 * @param envelope Validated immutable observation to identify.
 * @param limits Envelope resource bounds.
 * @return SHA-256 content identity, independent of platform JSON formatting.
 * @note Hash identity detects corruption/conflicts; it does not authenticate
 * a producer. Changing timestamps on retry changes the digest and is rejected.
 */
ArtifactDigest telemetry_content_digest(const TelemetryEnvelope& envelope,
                                        const TelemetryLimits& limits = {});

/** @brief Hashes the length-delimited stream/key tuple for retry deduplication.
 * @param identity Producer retry identity, with all identity fields validated.
 * @param limits Identity resource bounds.
 * @return SHA-256 operation identity, deliberately independent of event
 * content.
 * @note A store must compare the content digest on a matching operation key;
 * key equality alone must never acknowledge different data as a duplicate.
 */
ArtifactDigest telemetry_idempotency_digest(const TelemetryIdentity& identity,
                                            const TelemetryLimits& limits = {});

/** @brief An absolute steady-clock deadline; absence means no time bound.
 * @note Deadline values are operation-local and are never persisted as event
 * identity or compared with observation monotonic timestamps.
 */
using TelemetryDeadline = std::optional<std::chrono::steady_clock::time_point>;

/** @brief Checks cancellation, then an absolute operation deadline.
 * @param stop Cooperative cancellation token.
 * @param deadline Optional absolute steady-clock deadline.
 * @throws Cancelled When cancellation was already requested.
 * @throws DeadlineExceeded When the deadline has been reached.
 */
void check_telemetry_operation(std::stop_token stop = {},
                               TelemetryDeadline deadline = std::nullopt);

/** @brief Role controlling whether a sink may omit required evidence. */
enum class TelemetrySinkRole : std::uint8_t {
  evidence =
      1,  ///< Required observations must be retained durably or rejected.
  observer = 2,  ///< Optional display/export; unsupported input is counted.
};

/** @brief Stable destination descriptor, without secrets or filesystem paths.
 */
struct TelemetrySinkDescriptor {
  std::uint32_t schema_version = 1U;  ///< Sink protocol schema.
  std::string type_id;         ///< Stable implementation/output-schema ID.
  std::string sink_id;         ///< Unique logical identity in one fanout.
  std::string destination_id;  ///< Stable redacted destination identity.
  TelemetrySinkRole role = TelemetrySinkRole::observer;  ///< Evidence policy.
};

/** @brief Receipt for one envelope, at the same index as the input batch. */
enum class TelemetryDisposition : std::uint8_t {
  accepted = 1,   ///< Sink owns the whole immutable envelope and its retries.
  duplicate = 2,  ///< Same operation and same content were already accepted.
  sampled_out =
      3,  ///< Declared observer/best-effort sampling policy omitted it.
  unsupported = 4,  ///< Optional observer cannot represent this payload.
};

/** @brief Complete positional receipts; there are no partial-success returns.
 */
struct TelemetryEmitResult {
  std::vector<TelemetryDisposition> dispositions;  ///< One per input envelope.
};

/** @brief Cumulative counters after a successful flush with no pending work. */
struct TelemetryFlushResult {
  std::uint64_t exported_events =
      0U;  ///< Accepted envelopes at final boundary.
  std::uint64_t sampled_out_events = 0U;  ///< Explicit policy omissions so far.
  std::uint64_t unsupported_events =
      0U;  ///< Unsupported observer inputs so far.
};

/** @brief Validates destination identity and protocol/role values.
 * @param descriptor Immutable logical destination description.
 * @param limits Identity field bounds.
 * @throws Error For an unsupported schema, malformed identity, or invalid role.
 */
void validate_telemetry_sink_descriptor(
    const TelemetrySinkDescriptor& descriptor,
    const TelemetryLimits& limits = {});

/** @brief Validates complete receipts and required-evidence preservation.
 * @param descriptor Sink whose acceptance policy was applied.
 * @param envelopes Original immutable batch, validated in full.
 * @param result Successful positional receipt vector.
 * @param limits Input and descriptor resource bounds.
 * @throws CompatibilityError For incomplete/unknown receipts or omitted
 * required observations in an evidence sink.
 */
void validate_telemetry_emit_result(
    const TelemetrySinkDescriptor& descriptor,
    std::span<const TelemetryEnvelope> envelopes,
    const TelemetryEmitResult& result, const TelemetryLimits& limits = {});

/**
 * @brief Narrow neutral sink, shared by training, evaluation, and local tools.
 *
 * Calls are serialized by the owner; implementations need not be thread-safe.
 * Batch validation precedes side effects. An evidence `accepted` receipt means
 * the envelope is durably recoverable now, not merely queued in memory. An
 * observer receipt transfers retry ownership; observer failure never changes
 * the immutable training outcome. Exporters expose omissions in receipts and
 * counters instead of silently treating unsupported data as delivered.
 *
 * Exceptions may follow a durable prefix/commit whose acknowledgement was
 * interrupted. The caller retains and retries the same entire batch. Evidence
 * sinks deduplicate stable keys and compare content hashes. Exactly-once
 * external effects are not claimed. Once emit returns, inputs are not retained
 * by borrowed reference. Destruction performs no fallible flush or I/O.
 */
class TelemetrySink {
 public:
  /** @brief Enables destruction through the narrow sink interface. */
  virtual ~TelemetrySink() = default;

  /** @brief Returns stable logical destination and evidence policy.
   * @return Borrowed descriptor valid for this sink's lifetime.
   */
  virtual const TelemetrySinkDescriptor& descriptor() const noexcept = 0;

  /** @brief Accepts a whole bounded batch or throws with retry ambiguity.
   * @param envelopes Immutable producer observations; empty is a no-op.
   * @param stop Cooperative cancellation, checked before blocking operations.
   * @param deadline Absolute deadline checked before each bounded blocking
   * step.
   * @return Complete positional receipts conforming to the descriptor policy.
   * @throws Error On invalid input, conflict, cancellation/deadline, or I/O.
   */
  virtual TelemetryEmitResult emit(
      std::span<const TelemetryEnvelope> envelopes, std::stop_token stop = {},
      TelemetryDeadline deadline = std::nullopt) = 0;

  /** @brief Reaches the declared durable/export boundary for accepted work.
   * @param stop Cooperative cancellation; an abort preserves pending state.
   * @param deadline Absolute bound; timeout does not claim a complete flush.
   * @return Cumulative counters after all accepted pending work has drained.
   * @throws Error On cancellation/deadline or incomplete flush.
   */
  virtual TelemetryFlushResult flush(
      std::stop_token stop = {}, TelemetryDeadline deadline = std::nullopt) = 0;

  /** @brief Idempotently flushes, releases resources, and rejects future emit.
   * @param stop Cooperative cancellation; failed close can be retried.
   * @param deadline Absolute bound for draining and finalization.
   * @return Same final cumulative counters on repeated successful calls.
   * @throws Error On cancellation/deadline or incomplete finalization.
   */
  virtual TelemetryFlushResult close(
      std::stop_token stop = {}, TelemetryDeadline deadline = std::nullopt) = 0;
};

}  // namespace jyppx

#endif  // JYPPX_FOUNDATION_TELEMETRY_HPP
