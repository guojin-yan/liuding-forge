// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_INTEROP_EXPORT_BACKEND_HPP
#define JYPPX_INTEROP_EXPORT_BACKEND_HPP

/**
 * @file export_backend.hpp
 * @brief Defines explicit export discovery and versioned sidecar preflight.
 *
 * This contract owns artifact metadata only. Invocation/value schemas and
 * examples are separately versioned ArtifactRefs produced by their owners;
 * interop does not depend on core, nn, LibTorch, Python, protobuf, or ONNX.
 */

#include <jyppx/foundation/artifact.hpp>

#include <chrono>
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

namespace jyppx::interop {

/** @brief Stable artifact format identities; never inferred from extensions. */
enum class ArtifactFormat : std::uint8_t {
  jyppx_checkpoint = 1,    ///< Versioned participant checkpoint.
  jyppx_model_bundle = 2,  ///< Pure-data model bundle and referenced artifacts.
  safetensors = 3,         ///< Official safetensors tensor-only bytes.
  libtorch_archive = 4,    ///< Supported-matrix LibTorch compatibility archive.
  torchscript = 5,         ///< Existing TorchScript compatibility artifact.
  onnx = 6,                ///< Official ONNX schema, never handwritten wire.
};

/** @brief Explicit source identity, independent of runtime module C++ types. */
enum class ExportSourceKind : std::uint8_t {
  native_state = 1,          ///< Tensor/state data, not an executable graph.
  python_factory = 2,        ///< Trusted importable Python factory plus state.
  exported_program = 3,      ///< Existing trusted torch.export ExportedProgram.
  torchscript = 4,           ///< Existing trusted compatibility graph.
  native_graph_adapter = 5,  ///< Explicit model-specific graph adapter/plugin.
};

/** @brief Distinguishes export discovery from artifact runtime discovery. */
enum class BackendRole : std::uint8_t {
  exporter = 1,  ///< Produces new artifacts from explicitly supported sources.
  runtime = 2,   ///< Loads/runs/inspects existing artifacts only.
};

/** @brief Operations are independent capabilities, including trace opt-in. */
enum class BackendOperation : std::uint8_t {
  export_model = 1,  ///< Export an explicit source to a new target artifact.
  load = 2,          ///< Load an existing supported artifact.
  run = 3,           ///< Execute an existing supported artifact.
  inspect = 4,       ///< Inspect existing format/schema information.
  trace = 5,         ///< Deprecated, explicit model-specific tracing only.
};

/** @brief Declares process placement without claiming a security sandbox. */
enum class BackendExecution : std::uint8_t {
  in_process = 1,      ///< Built-in or separately approved trusted ABI plugin.
  out_of_process = 2,  ///< Isolated runner with enforced resource limits.
};

/** @brief One exact package/runtime requirement or real observed version. */
struct BackendDependency {
  std::string name;     ///< Lowercase machine identity, e.g. onnxruntime.
  std::string version;  ///< Exact version, never wildcard/latest/range.
};

/**
 * @brief Bounded, machine-readable declaration, distinct from runtime evidence.
 *
 * A descriptor does not prove the environment or executable exists. Only
 * preflight against a real current probe can return supported. Advertised
 * operations remain constrained to these explicit source/target matrices.
 */
struct BackendDescriptor {
  static constexpr std::uint32_t format_version = 1U;  ///< Descriptor schema.
  std::uint32_t schema_version = format_version;       ///< Serialized schema.
  std::string backend_id;       ///< Stable explicit registry identity.
  std::string backend_version;  ///< Exact implementation version.
  BackendRole role = BackendRole::exporter;  ///< Export or runtime registry.
  BackendExecution execution =
      BackendExecution::out_of_process;   ///< Placement.
  std::uint32_t protocol_version = 1U;    ///< Sidecar v1; in-process uses zero.
  ArtifactDigest environment_lock;        ///< Exact lock artifact identity.
  std::vector<ArtifactFormat> formats;    ///< Supported target/runtime formats.
  std::vector<ExportSourceKind> sources;  ///< Explicit supported source kinds.
  std::vector<BackendOperation>
      operations;                    ///< Never implies other operations.
  std::vector<std::string> devices;  ///< Device families such as cpu/cuda.
  std::vector<std::string> dtypes;   ///< Scalar tokens such as float32.
  bool dynamic_shapes = false;       ///< Declared dynamic-shape implementation.
  std::vector<std::string> quantization_modes;  ///< Explicit modes incl. none.
  std::vector<std::string> prepost_modes;  ///< Explicit modes incl. external.
  std::map<std::string, std::vector<std::string>>
      allowed_options;                          ///< Exact values.
  std::vector<BackendDependency> dependencies;  ///< Pinned environment inputs.
  bool compatibility_only = false;  ///< Existing-artifact compatibility scope.
  bool deprecated = false;          ///< Backend is on a deprecation path.
  std::uint64_t max_artifact_bytes = 8ULL << 30U;  ///< Each input/output bound.
  std::size_t max_verification_cases =
      64U;  ///< Finite case-artifact inventory.
};

/** @brief Resource requirements which the external runner must actually
 * enforce. */
struct SidecarResources {
  std::uint32_t cpu_threads = 1U;  ///< Positive CPU concurrency bound.
  std::uint64_t memory_bytes = 4ULL << 30U;  ///< Positive process memory bound.
  std::uint64_t wall_time_ms = 60000U;  ///< Positive wall-clock deadline delta.
  std::uint64_t max_output_bytes = 8ULL
                                   << 30U;  ///< Positive total output bound.
  std::size_t max_control_bytes = 1024U * 1024U;  ///< UTF-8 stdin/stdout bound.
};

/** @brief Explicit source and verification inputs for one backend operation. */
struct ExportRequest {
  static constexpr std::uint32_t format_version = 1U;  ///< Request schema.
  std::uint32_t schema_version = format_version;       ///< Serialized schema.
  std::string request_id;  ///< Stable request correlation token.
  std::string backend_id;  ///< Exact explicitly selected backend.
  BackendOperation operation = BackendOperation::export_model;  ///< Operation.
  ExportSourceKind source_kind = ExportSourceKind::native_state;  ///< Required.
  ArtifactRef source;  ///< Code/state/program reference, never a source path.
  std::optional<ArtifactRef> state;  ///< Factory tensor state, when applicable.
  std::string factory_entry_point;  ///< Python module:factory, otherwise empty.
  ArtifactFormat target = ArtifactFormat::onnx;  ///< Explicit requested format.
  std::string device = "cpu";                    ///< Requested device family.
  std::string dtype = "float32";  ///< Requested scalar matrix entry.
  bool dynamic_shapes = false;    ///< Requires fixed and dynamic cases.
  std::string quantization_mode = "none";  ///< Explicit quantization policy.
  std::string prepost_mode = "external";   ///< Explicit pre/post policy.
  std::optional<ArtifactRef>
      invocation_schema;  ///< Owner's versioned contract.
  std::vector<ArtifactRef>
      verification_cases;  ///< Bounded referenced examples.
  std::map<std::string, std::string>
      options;                ///< Allow-listed exact parameters.
  bool trust_source = false;  ///< Explicit authorization for executable inputs.
  bool trace_opt_in = false;  ///< Explicit tracing consent, never inferred.
  ArtifactDigest environment_lock;  ///< Requested exact sidecar/runtime lock.
  SidecarResources
      resources;  ///< Enforced by the process owner, not this codec.
  double absolute_tolerance = 1e-5;  ///< Finite nonnegative element tolerance.
  double relative_tolerance = 1e-4;  ///< Finite nonnegative relative tolerance.
};

/**
 * @brief Real environment observations supplied by a trusted runner/prober.
 *
 * The caller must run the actual backend probe, verify the referenced evidence
 * bytes/digest, and bind it to the current process/environment. Deserializing
 * this object is syntax validation only and cannot establish that trust.
 */
struct BackendProbe {
  static constexpr std::uint32_t format_version = 1U;  ///< Probe schema.
  std::uint32_t schema_version = format_version;       ///< Serialized schema.
  std::string backend_id;               ///< Observed executable identity.
  std::string backend_version;          ///< Observed executable version.
  std::uint32_t protocol_version = 1U;  ///< Observed protocol; in-process zero.
  ArtifactDigest environment_lock;      ///< Lock verified by the real probe.
  std::vector<BackendDependency> dependencies;  ///< Real imported versions.
  std::vector<BackendOperation> operations;     ///< Actually probed operations.
  std::vector<std::string> devices;  ///< Actually available device families.
  std::vector<std::string> dtypes;   ///< Actually verified scalar encodings.
  bool dynamic_shapes_verified = false;  ///< Real dynamic validation available.
  bool official_onnx_exporter = false;   ///< Official torch.export ONNX path.
  bool onnx_checker = false;  ///< Official ONNX checker import/probe succeeded.
  bool onnx_runtime = false;  ///< Actual ORT import/session probe succeeded.
  std::optional<ArtifactDigest>
      request_digest;  ///< Exact model preflight scope.
  bool model_preflight_passed =
      false;  ///< Real model/operator/schema preflight.
  std::vector<std::string>
      unsupported_operators;  ///< Explicit real graph findings.
  ArtifactRef evidence;  ///< Nonempty independently verified probe evidence.
};

/** @brief Tri-state support applies to preflight, never to export success. */
enum class SupportDecision : std::uint8_t {
  supported = 1,    ///< Declared request matches the verified real environment.
  unsupported = 2,  ///< A concrete requirement is violated.
  conditional = 3,  ///< Real environment evidence is still missing.
};

/** @brief Classifies preflight issues for CLI/doctor machine consumers. */
enum class BackendIssueKind : std::uint8_t {
  dependency = 1,  ///< Missing or incompatible required package.
  source = 2,      ///< Source kind/format is unsupported.
  operation = 3,   ///< Operation is absent, including unknown model operators.
  device = 4,      ///< Requested device family is unavailable.
  dtype = 5,       ///< Requested scalar encoding is unsupported.
  dynamic_shape = 6,  ///< Dynamic-shape requirement is unavailable.
  option = 7,       ///< Quantization, pre/post, or named option is unsupported.
  environment = 8,  ///< Environment lock/version/evidence is incompatible.
  protocol = 9,     ///< Sidecar version cannot be negotiated.
  trust = 10,       ///< Executable source trust or trace opt-in is missing.
  verification = 11,  ///< Required independent official validation is absent.
};

/** @brief Bounded structured requirement without secrets or stack traces. */
struct BackendIssue {
  BackendIssueKind kind = BackendIssueKind::environment;  ///< Machine category.
  std::string code;      ///< Stable machine code, e.g. dependency.missing.
  std::string subject;   ///< Dependency, dtype, source, or option identity.
  std::string expected;  ///< Required bounded value.
  std::string actual;    ///< Observed bounded value or explicit missing marker.
};

/** @brief Complete preflight result for one explicit backend/request pair. */
struct SupportReport {
  SupportDecision decision = SupportDecision::conditional;  ///< Tri-state.
  std::string backend_id;       ///< Selected explicit registry identity.
  std::string backend_version;  ///< Selected implementation version.
  std::vector<BackendIssue>
      issues;  ///< Complete discovered unmet requirements.
};

/** @brief Official shape-inference stage may be explicitly inapplicable. */
enum class ShapeInferenceStatus : std::uint8_t {
  not_checked = 1,     ///< No validation evidence; cannot publish.
  passed = 2,          ///< Official shape inference completed successfully.
  not_applicable = 3,  ///< Recorded reason explains an inapplicable stage.
  failed = 4,          ///< Official inference failed; cannot publish.
};

/** @brief Evidence records actual validation and binds it to exact bytes. */
struct ExportVerification {
  ArtifactDigest
      request_digest;  ///< SHA-256 of exact canonical request envelope.
  ArtifactDigest artifact_digest;   ///< Must match the returned ONNX artifact.
  ArtifactDigest environment_lock;  ///< Exact environment used for validation.
  bool schema_passed = false;   ///< Official protobuf/schema parsing succeeded.
  bool checker_passed = false;  ///< Official onnx.checker accepted the graph.
  ShapeInferenceStatus shape_inference = ShapeInferenceStatus::not_checked;
  ///< Explicit official inference result.
  std::string shape_inference_reason;  ///< Required for not_applicable.
  bool runtime_passed = false;  ///< Real ORT executed every verification case.
  bool numerical_passed =
      false;  ///< Every output satisfied requested tolerance.
  std::uint64_t cases_run = 0U;  ///< Exact number of referenced cases executed.
  std::uint64_t cases_passed = 0U;  ///< Exact number of cases that passed.
  std::uint64_t dynamic_cases_passed =
      0U;  ///< Dynamic subset; static also required.
  double absolute_tolerance =
      0.0;  ///< Actual tolerance used, must equal request.
  double relative_tolerance =
      0.0;               ///< Actual tolerance used, must equal request.
  ArtifactRef evidence;  ///< Independently verified official validation record.
};

/**
 * @brief Staged export candidate, never proof that execution or validation ran.
 *
 * The runner must resolve and independently verify the artifact/evidence bytes,
 * check metadata against the request/probe, and only then atomically publish.
 * No implementation in this contract fabricates a graph or successful export.
 */
struct ExportCandidate {
  static constexpr std::uint32_t format_version = 1U;  ///< Candidate schema.
  std::uint32_t schema_version = format_version;       ///< Serialized schema.
  std::string request_id;       ///< Exact request correlation identity.
  std::string backend_id;       ///< Actual backend used.
  std::string backend_version;  ///< Actual implementation version used.
  ArtifactRef
      artifact;  ///< Staged output; not yet a published final reference.
  ExportVerification
      verification;  ///< Validation records bound to exact bytes.
};

/** @brief Explicit descriptor catalog; export and runtime catalogs are
 * separate. */
class BackendRegistry final {
 public:
  /** @brief Creates an empty explicit catalog with one role.
   * @param role Exporter or runtime identity accepted by this registry.
   * @throws InvalidArgument If role is invalid.
   */
  explicit BackendRegistry(BackendRole role);

  /** @brief Registers one validated descriptor without discovering code.
   * @param descriptor Bounded declaration with a unique explicit ID.
   * @throws InvalidArgument For duplicates, role mismatch, or invalid metadata.
   * @throws ResourceExhausted If the 64-backend bound is exceeded.
   */
  void add(BackendDescriptor descriptor);

  /** @brief Finds one explicit registered declaration.
   * @param backend_id Exact stable identity; no suffix/type inference occurs.
   * @return Pointer stable until registry destruction, or null if absent.
   */
  const BackendDescriptor* find(std::string_view backend_id) const;

 private:
  BackendRole role_;  ///< Immutable role enforced for every registration.
  std::map<std::string, BackendDescriptor, std::less<>> descriptors_;
  ///< Explicit declarations only; no dynamic-library handles.
};

/** @brief Abstract exporter; concrete process owners implement real execution.
 */
class ExportBackend {
 public:
  /** @brief Releases the concrete backend and its owned resources. */
  virtual ~ExportBackend() = default;
  /** @brief Describes this backend without claiming runtime availability.
   * @return Bounded declaration for explicit discovery.
   */
  virtual BackendDescriptor descriptor() const = 0;
  /** @brief Performs actual request/environment preflight.
   * @param request Explicit source, target, schemas, options, and limits.
   * @return Tri-state report with structured diagnostics.
   */
  virtual SupportReport supports(const ExportRequest& request) const = 0;
  /** @brief Runs the real official exporter and verification in its owner.
   * @param request Complete preflighted request.
   * @param cancellation Cooperative cancellation observed by the process owner.
   * @param deadline Absolute monotonic deadline enforced by the process owner.
   * @return Staged candidate with real evidence; publication belongs to runner.
   * @throws Error On unsupported input, cancellation, deadline, or backend
   * error.
   */
  virtual ExportCandidate export_model(
      const ExportRequest& request, std::stop_token cancellation,
      std::chrono::steady_clock::time_point deadline) = 0;
};

/** @brief Validates descriptor enums, ownership, resource bounds, and exact
 * pins.
 * @param descriptor Declaration to validate without modification.
 * @throws Error If metadata is malformed, unsupported, or exceeds bounds.
 */
void validate_backend_descriptor(const BackendDescriptor& descriptor);

/** @brief Declares the official ONNX sidecar path, without probing/installing
 * it.
 * @param backend_version Exact implementation version.
 * @param environment_lock Digest of the independently approved environment
 * lock.
 * @param dependencies Exact
 * python/torch/onnx/onnxscript/onnxruntime/safetensors pins.
 * @return Out-of-process PythonFactory/ExportedProgram exporter declaration.
 * @throws Error If a required dependency/pin or lock is invalid/missing.
 */
BackendDescriptor make_onnx_sidecar_descriptor(
    std::string backend_version, ArtifactDigest environment_lock,
    std::span<const BackendDependency> dependencies);

/** @brief Declares only deprecated TorchScript load/run/inspect compatibility.
 * @param backend_version Exact adapter implementation version.
 * @param environment_lock Exact approved runtime lock digest.
 * @param libtorch_version Exact runtime requirement.
 * @return Runtime-role descriptor; no general export or trace is advertised.
 */
BackendDescriptor make_torchscript_compatibility_descriptor(
    std::string backend_version, ArtifactDigest environment_lock,
    std::string libtorch_version);

/** @brief Compares the complete request with declaration and real probe
 * metadata.
 * @param descriptor Explicit selected backend declaration.
 * @param request Explicit source/target/options/verification request.
 * @param probe Current real verified probe, or null to report conditional.
 * @return Complete preflight diagnostics; supported is not export success.
 * @throws Error If caller metadata/schema/limits are invalid.
 */
SupportReport preflight_backend(const BackendDescriptor& descriptor,
                                const ExportRequest& request,
                                const BackendProbe* probe = nullptr);

/** @brief Binds real model preflight and validation to one complete request.
 * @param request Complete validated request, including cases and tolerances.
 * @return Foundation SHA-256 of `serialize_sidecar_request()` bytes exactly,
 * including the envelope and excluding any transport-added whitespace. Other
 * languages hash those original bytes rather than reserializing JSON floats.
 */
ArtifactDigest backend_request_digest(const ExportRequest& request);

/** @brief Encodes the full declaration for CLI/doctor machine consumers.
 * @param descriptor Validated explicit backend declaration, not a real probe.
 * @param maximum_bytes Host metadata byte bound.
 * @return Canonical schema-v1 JSON with status declared-unprobed.
 */
std::string serialize_backend_descriptor(const BackendDescriptor& descriptor,
                                         std::size_t maximum_bytes = 1024U *
                                                                     1024U);

/** @brief Encodes a bounded v1 control request; tensor/sample bytes stay
 * external.
 * @param request Explicit out-of-process export request.
 * @return Canonical UTF-8 control envelope, protocol jyppx.export.sidecar v1.
 * @throws Error For invalid/unsupported metadata or resource exhaustion.
 */
std::string serialize_sidecar_request(const ExportRequest& request);

/** @brief Reads an exact v1 control request, rejecting unknown/duplicate
 * fields.
 * @param json Bounded UTF-8 control envelope, never a shell command.
 * @param maximum_bytes Independent host control-message byte limit.
 * @return Validated request metadata, with no artifact or code loading.
 * @throws Error For malformed versions, fields, UTF-8, or resource bounds.
 */
ExportRequest deserialize_sidecar_request(std::string_view json,
                                          std::size_t maximum_bytes = 1024U *
                                                                      1024U);

/** @brief Encodes actual probe observations; this never runs a probe itself.
 * @param probe Real runner observations and verified evidence reference.
 * @param maximum_bytes Host control-message byte bound.
 * @return Canonical v1 probe envelope without paths, samples, or stack traces.
 */
std::string serialize_sidecar_probe(const BackendProbe& probe,
                                    std::size_t maximum_bytes = 1024U * 1024U);

/** @brief Reads bounded probe metadata without establishing its authenticity.
 * @param json Exact versioned probe envelope.
 * @param maximum_bytes Host control-message byte bound.
 * @return Validated observation metadata; caller must verify real evidence.
 */
BackendProbe deserialize_sidecar_probe(std::string_view json,
                                       std::size_t maximum_bytes = 1024U *
                                                                   1024U);

/** @brief Encodes a staged result as candidate, never an unverified success.
 * @param candidate Actual staged artifact and validation observation metadata.
 * @param maximum_bytes Host control-message byte bound.
 * @return Canonical v1 export.candidate envelope.
 */
std::string serialize_sidecar_candidate(const ExportCandidate& candidate,
                                        std::size_t maximum_bytes = 1024U *
                                                                    1024U);

/** @brief Reads a staged candidate without publishing or trusting output bytes.
 * @param json Exact v1 candidate envelope with complete validation metadata.
 * @param maximum_bytes Host control-message byte bound.
 * @return Candidate syntax only; subsequent request/evidence checks are
 * required.
 */
ExportCandidate deserialize_sidecar_candidate(
    std::string_view json, std::size_t maximum_bytes = 1024U * 1024U);

/**
 * @brief Rejects incomplete, stale, or incorrectly bound ONNX candidate
 * metadata.
 * @param descriptor Explicit selected ONNX exporter declaration.
 * @param request Original canonical request and exact tolerances/cases.
 * @param probe Real verified current runtime observations.
 * @param candidate Staged output and actual validation evidence metadata.
 * @throws CompatibilityError If preflight or any required verification fails.
 * @throws Error For malformed metadata or resource limits.
 *
 * This validates records only. The runner must independently resolve/hash
 * referenced bytes and ensure official parsing/checker/ORT/numeric validation
 * actually ran before accepting or atomically publishing an artifact.
 */
void validate_export_candidate_metadata(const BackendDescriptor& descriptor,
                                        const ExportRequest& request,
                                        const BackendProbe& probe,
                                        const ExportCandidate& candidate);

}  // namespace jyppx::interop

#endif  // JYPPX_INTEROP_EXPORT_BACKEND_HPP
