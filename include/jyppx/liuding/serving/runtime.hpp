// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#ifndef JYPPX_LIUDING_SERVING_RUNTIME_HPP
#define JYPPX_LIUDING_SERVING_RUNTIME_HPP
/** @file runtime.hpp @brief Optional in-process inference, ownership and
 * limits. */
#include <jyppx/liuding/foundation/telemetry.hpp>
#include <jyppx/liuding/interop/model_bundle.hpp>
#include <jyppx/liuding/interop/torchscript_compatibility.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>
/** @namespace jyppx::liuding::serving
 * @brief Opt-in bounded inference runtimes, batching and host protocol
 * adapters.
 */
namespace jyppx::liuding::serving {
/** @brief Optional monotonic deadline; absence means no caller time bound. */
using Deadline = std::optional<std::chrono::steady_clock::time_point>;
/** @brief Correlation and caller security scope, never raw bearer tokens. */
struct RequestContext {
  std::string request_id;  ///< Caller request identity propagated to the reply.
  std::string trace_id;    ///< Optional redacted tracing correlation.
  std::string project_id;  ///< Host-authenticated project security scope.
  std::string principal_id;   ///< Host-authenticated caller security scope.
  std::string deployment_id;  ///< Optional deployment correlation.
  std::string model_version;  ///< Optional exact expected model version.
  std::string priority_class = "normal";  ///< Batch compatibility priority key.
  std::string deadline_class =
      "default";  ///< Batch compatibility deadline key.
};
/** @brief Logical bounds checked before tensor cloning, not backend RSS limits.
 */
struct RequestLimits {
  std::uint64_t max_tensor_bytes = 64ULL
                                   << 20U;  ///< Aggregate logical tensor bytes.
  std::size_t max_tensors = 64U;            ///< Tensor leaf count ceiling.
  std::size_t max_rank = 16U;     ///< Per-tensor dimension count ceiling.
  std::size_t max_nodes = 4096U;  ///< Recursively visited value-node ceiling.
  std::uint64_t max_dimension = 1ULL
                                << 20U;  ///< Extent ceiling for each dimension.
  std::uint64_t max_elements = 1ULL
                               << 26U;    ///< Element ceiling for each tensor.
  std::size_t max_string_bytes = 65536U;  ///< Bytes in a string or keyword.
  bool require_finite = true;     ///< Reject nonfinite floating/complex values.
  std::optional<double> minimum;  ///< Optional inclusive numeric lower bound.
  std::optional<double> maximum;  ///< Optional inclusive numeric upper bound.
};
/** @brief Exact boundary and backend declarations, without fallback inference.
 */
struct RuntimeDescriptor {
  std::string
      backend_id;  ///< Explicit native or compatibility backend identity.
  std::string backend_version;  ///< Exact backend implementation version.
  std::string format;           ///< Explicit state/archive format identity.
  ArtifactDigest
      environment_lock;  ///< Owner-verified dependency environment identity.
  std::vector<interop::BundleDependency>
      dependencies;          ///< Exact declared dependency versions.
  Device device{c10::kCPU};  ///< Selected device; no automatic fallback.
  std::vector<Dtype> dtypes{
      torch::kFloat32};        ///< Accepted tensor element types.
  bool dynamic_shapes = true;  ///< Declared schema supports dynamic extents.
  bool tensor_batching =
      false;  ///< Owner asserts independent rows and batch-preserving stages.
  std::size_t max_batch_size = 1U;  ///< Owner-approved inference batch ceiling.
  bool oom_retry_safe =
      false;  ///< Only a declared pure native model may opt in.
};
/** @brief One call uses the existing nn Invocation, not another tensor map. */
struct PredictRequest {
  nn::Invocation
      input;  ///< Invocation values validated at the request boundary.
  RequestContext context;  ///< Redacted caller correlation and security scope.
  Deadline
      deadline;  ///< Cooperative request deadline, never kernel preemption.
};
/** @brief Stage measurements in seconds, without fabricated transport timing.
 */
struct TimingBreakdown {
  double queue = 0;        ///< Admission/scheduler wait seconds.
  double decode = 0;       ///< Host protocol decode seconds.
  double preprocess = 0;   ///< Trusted input-stage seconds.
  double compute = 0;      ///< Model invocation seconds.
  double postprocess = 0;  ///< Trusted output-stage seconds.
  double encode = 0;       ///< Host protocol encode seconds.
  double total = 0;        ///< Locally measured complete call seconds.
};
/** @brief Lifecycle state; draining refuses new work while retaining leases. */
enum class HealthStatus : std::uint8_t {
  ready = 1,     ///< Validated and accepting inference.
  draining = 2,  ///< Admission stopped while existing leases remain.
  closed = 3,    ///< No session work can be admitted.
  degraded = 4   ///< Runtime is live but no longer advertises readiness.
};
/** @brief Owning diagnostics that survive a session close. */
struct ModelRuntimeInfo {
  std::string model_version;     ///< Loaded model content/version identity.
  std::string pipeline_version;  ///< Exact trusted pipeline identity.
  RuntimeDescriptor
      runtime;  ///< Selected backend/device/capability declaration.
  HealthStatus health = HealthStatus::closed;  ///< Current lifecycle snapshot.
  std::uint64_t accepted = 0;   ///< Requests admitted since publication.
  std::uint64_t completed = 0;  ///< Successful independent responses.
  std::uint64_t failed = 0;     ///< Admitted requests ending in an error.
  std::uint64_t active = 0;     ///< Outstanding admission leases.
  std::uint64_t observation_failures =
      0;  ///< Suppressed observation callback errors.
};
/** @brief Independent response values and original request correlation. */
struct PredictResponse {
  nn::InvocationResult output;  ///< Owned independent result snapshot.
  RequestContext context;       ///< Original caller request correlation.
  ModelRuntimeInfo runtime;     ///< Runtime identity/counters at completion.
  TimingBreakdown timing;       ///< Local stage measurements in seconds.
};
/** @brief Caller bridges these redacted records into foundation envelopes.
 * @note Callback must be bounded/nonblocking and must not reenter this
 * instance. Exceptions are counted, never allowed to replace an inference
 * result.
 */
using ObservationHook = std::function<void(TelemetryLog, TelemetryCorrelation,
                                           TelemetryAttributes)>;
/** @brief An exact versioned owner descriptor and its verified bytes identity.
 */
struct DescriptorBinding {
  std::string schema_id;  ///< Exact owner descriptor schema identity.
  std::uint32_t schema_version = 1U;  ///< Supported descriptor schema version.
  ArtifactDigest digest;  ///< Hash of the verified descriptor bytes.
};
/** @brief Owner-supplied stages; empty functions mean explicitly bound
 * identity.
 * @note Stages run serially and must not retain mutable per-request references.
 * They are trusted application code, not code read from a bundle.
 */
struct Pipeline {
  std::string version;       ///< Immutable trusted pipeline version.
  DescriptorBinding input;   ///< Binding for the external request schema.
  DescriptorBinding output;  ///< Binding for the external response schema.
  DescriptorBinding preprocessing;   ///< Binding for the trusted input stage.
  DescriptorBinding postprocessing;  ///< Binding for the trusted output stage.
  nn::InvocationSchema
      request_schema;           ///< Boundary checked before preprocessing.
  ValueSchema response_schema;  ///< Boundary checked after postprocessing.
  std::function<nn::Invocation(const nn::Invocation&)>
      preprocess;  ///< Optional trusted input transform.
  std::function<nn::InvocationResult(const nn::InvocationResult&)>
      postprocess;  ///< Optional trusted result transform.
};
/** @brief Owns exact hash-verified file snapshots passed to a native factory.
 */
struct VerifiedBundle {
  interop::ModelBundleVerification
      verification;  ///< Verified manifest and selection report.
  std::map<std::string, std::vector<std::byte>>
      files;  ///< Path-keyed owned file snapshots.
};
/** @brief Explicitly created module and owner-decoded pipeline schema. */
struct NativeModel {
  std::shared_ptr<nn::ModelAdapter>
      adapter;  ///< Explicit owner-created callable module.
  Pipeline
      pipeline;  ///< Owner-decoded verified descriptor bindings and stages.
};
/** @brief Fixed native model creation, never discovered from artifact code.
 * The factory owns/copies any data retained past this call; it must not retain
 * references or spans into the borrowed VerifiedBundle argument.
 */
using NativeModelFactory =
    std::function<NativeModel(const VerifiedBundle&, const RuntimeDescriptor&)>;
/** @brief One supplied known vector, checked before publication as ready. */
struct WarmupCase {
  nn::Invocation input;  ///< Known request exercised before readiness.
  c10::IValue expected;  ///< Independently supplied expected result.
  double relative_tolerance =
      1e-5;  ///< Floating comparison relative tolerance.
  double absolute_tolerance =
      1e-6;  ///< Floating comparison absolute tolerance.
};
/** @brief Bounds for serving state/input/output and owner observation bridge.
 */
struct LoadOptions {
  RequestLimits request_limits;  ///< External request resource limits.
  RequestLimits output_limits;   ///< Intermediate/final result resource limits.
  std::uint64_t max_state_bytes = 1ULL
                                  << 30U;  ///< Native model state byte ceiling.
  std::uint64_t max_snapshot_bytes =
      512ULL << 20U;                   ///< Owned verified file byte ceiling.
  std::size_t max_warmup_cases = 16U;  ///< Known-vector warmup count ceiling.
  std::size_t max_active_requests = 32U;  ///< Includes serialization waiters.
  std::stop_token stop;     ///< Load/warmup cooperative cancellation only.
  Deadline deadline;        ///< Load/warmup cooperative deadline only.
  ObservationHook observe;  ///< Optional bounded redacted observation bridge.
};
/** @brief Confirmed allocation failure from an explicitly retry-safe forward.
 * Budget violations and output/stage errors never produce this retry marker.
 */
class RetryableOutOfMemory : public ResourceExhausted {
 public:
  // Retain both base constructors and their respective parameter contracts.
  using ResourceExhausted::ResourceExhausted;
};

/** @brief Owns one serialized, validated inference session.
 * @note No move/copy. A supplied deadline cannot preempt trusted code/kernels;
 * a timed-out close leaves the instance draining. Destruction joins safely.
 * Exclude concurrent destruction. Input storage must remain quiescent during
 * snapshotting. Value snapshots deliberately do not preserve storage aliases.
 */
class Predictor final {
 public:
  /** @brief Stops admission and joins all existing leases before destruction.
   */
  ~Predictor();
  /** @brief Copying a serialized inference owner is forbidden. */
  Predictor(const Predictor&) = delete;
  /** @brief Replacing a serialized inference owner by copy is forbidden. */
  Predictor& operator=(const Predictor&) = delete;
  /** @brief Validates, snapshots, executes and snapshots one independent
   * result.
   * @param request Caller-owned invocation and request context.
   * @param stop Cooperative cancellation request.
   * @return Independent validated prediction and timing metadata.
   */
  PredictResponse predict(const PredictRequest& request,
                          std::stop_token stop = {});
  /** @brief Owning lifecycle/capability/counter snapshot.
   * @return Current runtime identity, state and counters.
   */
  ModelRuntimeInfo info() const;
  /** @brief Input boundary before preprocessing.
   * @return Invocation schema accepted by this predictor.
   */
  nn::InvocationSchema input_schema() const;
  /** @brief Validated response boundary after postprocessing.
   * @return Schema of the published prediction values.
   */
  ValueSchema output_schema() const;
  /** @brief Immutable logical input budget used by schedulers.
   * @return Frozen logical tensor and byte limits.
   */
  RequestLimits request_limits() const;
  /** @brief Stops admission and waits for existing leases; false on
   * timeout/cancel.
   * @param deadline Optional cooperative absolute close deadline.
   * @param stop Cooperative cancellation while waiting.
   * @return True after all leases drain; false while work remains.
   */
  bool close(Deadline deadline = {}, std::stop_token stop = {});
  /** @brief Validates/clones a queued request before the caller can mutate it.
   * @param request Invocation and context to validate and copy.
   * @return Independent request value without caller-owned tensor aliases.
   */
  PredictRequest snapshot_request(const PredictRequest& request) const;

 private:
  /** @brief Private serialized session and lease state. */
  class Impl;
  std::shared_ptr<Impl>
      impl_;  ///< Shared state retained by outstanding leases.
  /** @brief Adopts a fully initialized private session.
   * @param impl Shared validated lifecycle and execution owner.
   */
  explicit Predictor(std::shared_ptr<Impl> impl);
  /** @brief Native backend may publish a verified session. */
  friend class NativeStateBackend;
  /** @brief Trusted JIT adapter may publish a validated session. */
  friend std::shared_ptr<Predictor> load_torchscript_predictor(
      std::span<const std::byte> archive, const ArtifactDigest& expected,
      const nn::InvocationSchema& input_schema,
      const ValueSchema& output_schema,
      const interop::TorchScriptLoadOptions& runtime_options,
      RuntimeDescriptor descriptor, std::string model_version,
      std::span<const WarmupCase> warmup, const LoadOptions& options);
};
/** @brief Explicit backend selection; no dtype/device/provider fallback. */
class RuntimeBackend {
 public:
  /** @brief Destroys the backend descriptor/factory owner. */
  virtual ~RuntimeBackend() = default;
  /** @brief Returns the exact backend, device and capability declaration.
   * @return Frozen backend execution contract.
   */
  virtual RuntimeDescriptor descriptor() const = 0;
  /** @brief Verifies, loads and warms a session before returning it as ready.
   * @param directory Data-only bundle directory.
   * @param verify Integrity and resource verification policy.
   * @param warmup Known input/output cases to execute before publication.
   * @param options Session resource and lifetime policy.
   * @return Ready serialized predictor after every check succeeds.
   */
  virtual std::shared_ptr<Predictor> load(
      const std::filesystem::path& directory,
      const interop::ModelBundleVerifyOptions& verify,
      std::span<const WarmupCase> warmup,
      const LoadOptions& options = {}) const = 0;
};
/** @brief Verifies data-only bundle snapshots and strictly imports selected
 * state. */
class NativeStateBackend final : public RuntimeBackend {
 public:
  /** @brief Binds an explicit native factory and selected bundle model
   * identity.
   * @param descriptor Exact native execution declaration.
   * @param factory Trusted module and pipeline factory.
   * @param model_id Exact bundle model entry to import.
   */
  NativeStateBackend(RuntimeDescriptor descriptor, NativeModelFactory factory,
                     std::string model_id = "primary");
  /** @brief Returns the immutable native execution declaration.
   * @return Frozen native backend execution contract.
   */
  RuntimeDescriptor descriptor() const override;
  /** @brief Verifies snapshots, imports state and checks every known vector.
   * @param directory Data-only bundle directory.
   * @param verify Bundle integrity and resource verification policy.
   * @param warmup Known input/output validation cases.
   * @param options Session resource and lifetime policy.
   * @return Ready predictor with verified imported model state.
   */
  std::shared_ptr<Predictor> load(
      const std::filesystem::path& directory,
      const interop::ModelBundleVerifyOptions& verify,
      std::span<const WarmupCase> warmup,
      const LoadOptions& options = {}) const override;

 private:
  RuntimeDescriptor descriptor_;  ///< Immutable exact execution declaration.
  NativeModelFactory
      factory_;  ///< Trusted application module/pipeline factory.
  std::string
      model_id_;  ///< Exact model entry selected from a verified bundle.
};
/** @brief Really loads an explicitly trusted existing JIT archive outside
 * bundles.
 * @param archive Explicitly trusted executable JIT archive bytes.
 * @param expected Expected digest of the supplied archive bytes.
 * @param input_schema Invocation contract before preprocessing.
 * @param output_schema Published prediction contract.
 * @param runtime_options Explicit trust, device and JIT load policy.
 * @param descriptor Locked runtime and environment declaration.
 * @param model_version Stable model version reported by the predictor.
 * @param warmup Known input/output cases checked before publication.
 * @param options Session resource and lifetime policy.
 * @return Ready compatibility predictor after load and warmup.
 * @note Data-only bundles never carry executable TorchScript. This
 * compatibility adapter inherits exact JIT trust/schema/device/failure
 * semantics and no OOM retry. The descriptor pins a separately supplied
 * environment lock and LibTorch version; an archive digest never substitutes
 * for an environment identity.
 */
std::shared_ptr<Predictor> load_torchscript_predictor(
    std::span<const std::byte> archive, const ArtifactDigest& expected,
    const nn::InvocationSchema& input_schema, const ValueSchema& output_schema,
    const interop::TorchScriptLoadOptions& runtime_options,
    RuntimeDescriptor descriptor, std::string model_version,
    std::span<const WarmupCase> warmup, const LoadOptions& options = {});
/** @brief Validates resource/shape/numeric bounds without copying tensors.
 * @param input Invocation to inspect without mutation.
 * @param limits Logical tensor, element and byte limits.
 * @param runtime Required device and numeric execution contract.
 * @return Total logical tensor bytes in the validated invocation.
 */
std::uint64_t validate_request_resources(const nn::Invocation& input,
                                         const RequestLimits& limits,
                                         const RuntimeDescriptor& runtime);
/** @brief Lowercase configure-time OS/architecture token used in bundle matrix.
 * @return Stable platform token from the build configuration.
 */
std::string serving_platform_id();
/** @brief Truthful versioned preview capability manifest.
 * @return UTF-8 JSON with explicit implemented and unavailable capabilities.
 */
std::string serving_preview_manifest_json();
}  // namespace jyppx::liuding::serving
#endif
