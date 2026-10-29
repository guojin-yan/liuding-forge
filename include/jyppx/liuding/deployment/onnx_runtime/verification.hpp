// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DEPLOYMENT_ONNX_RUNTIME_VERIFICATION_HPP
#define JYPPX_LIUDING_DEPLOYMENT_ONNX_RUNTIME_VERIFICATION_HPP
/** @file verification.hpp @brief Bounded numerical and actual provider
 * evidence. */

#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>
#include <jyppx/liuding/foundation/version.hpp>

namespace jyppx::liuding::onnx_runtime {

/** @brief One caller-supplied named case; all payloads are borrowed and
 * immutable. */
struct VerificationCase {
  std::string_view id;  ///< Unique nonempty case identity, at most 128 bytes.
  std::span<const TensorInput> inputs;  ///< Complete graph inputs, any order.
  std::span<const TensorInput>
      expected;  ///< Complete expected outputs, any order.
};

/** @brief Per-element |actual-expected| <= atol + rtol*|expected| policy. */
struct VerificationOptions {
  double absolute_tolerance = 1e-6;  ///< Finite value in [0, 1].
  double relative_tolerance = 1e-5;  ///< Finite value in [0, 1].
  std::size_t max_cases = 64U;       ///< Positive bound, at most 1024 cases.
  std::uint64_t max_case_bytes = 128U
                                 << 20U;  ///< Inputs plus expected payloads.
  std::string
      reference;  ///< Caller-declared oracle identity, not authentication.
};

/** @brief Named FP32 payload identity; no sample values are retained. */
struct VerificationTensor {
  std::string name;                 ///< Graph name.
  std::vector<std::int64_t> shape;  ///< Concrete positive dimensions.
  ArtifactDigest digest;  ///< SHA-256 of contiguous little-endian FP32 bytes.
  std::uint64_t size_bytes = 0U;  ///< Logical payload bytes.
};

/** @brief One compared graph output; failures remain reportable observations.
 */
struct OutputComparison {
  VerificationTensor actual;    ///< Observed runtime output identity.
  VerificationTensor expected;  ///< Caller-supplied reference identity.
  std::uint64_t mismatched_elements =
      0U;                               ///< Values outside declared tolerance.
  double maximum_absolute_error = 0.0;  ///< Largest finite absolute difference.
};

/** @brief One complete execution; timing is not a benchmark qualification. */
struct CaseVerification {
  std::string id;                          ///< Stable supplied case identity.
  std::vector<VerificationTensor> inputs;  ///< Input identities, graph order.
  std::vector<OutputComparison> outputs;   ///< All comparisons, graph order.
  std::uint64_t elapsed_nanoseconds =
      0U;  ///< Session::run wall-clock duration.
};

/** @brief Actual Node/_kernel_time events aggregated by provider and operation.
 */
struct ProfileObservation {
  ArtifactDigest digest;  ///< Identity of the complete original ORT trace.
  std::uint64_t size_bytes = 0U;  ///< Exact encoded trace bytes.
  std::map<std::string, std::map<std::string, std::uint64_t>> kernels;
  ///< Provider -> operation -> observed event count, not unique graph nodes.
};

/** @brief Execution observations, without claims of task quality or producer
 * trust. */
struct VerificationReport {
  ArtifactDigest model_digest;  ///< Complete checked ONNX identity.
  BuildFingerprint build;       ///< Actual verification binary identity.
  std::string runtime_version;  ///< Actual loaded ORT version.
  ExecutionProvider requested_provider;  ///< Explicit provider and options.
  bool allow_cpu_fallback = false;  ///< Requested partition fallback policy.
  VerificationOptions policy;       ///< Exact comparison and input limits.
  std::vector<CaseVerification> cases;  ///< Executed cases; no Tensor storage.
  std::optional<ProfileObservation> profile;  ///< Actual trace, if supplied.
};

/** @brief Executes all cases and compares every named output without retaining
 * payloads.
 * @param session Existing model owner; graph signature and limits stay
 * immutable.
 * @param cases Complete borrowed cases, unchanged until return; preflight
 * covers all cases.
 * @param options Explicit tolerances, bounds and caller-declared reference
 * identity.
 * @param stop Cooperative cancellation, including preflight and comparisons.
 * @param deadline Absolute steady-clock deadline covering the whole operation.
 * @return Measurements for all cases, including numerical mismatches.
 * @throws Error Invalid cases, schema/count/shape/value errors, bounds or
 * runtime failure.
 * @note Reference values are assertions. Matching them does not qualify
 * arbitrary inputs, task accuracy, performance, memory ceilings or the
 * authenticity of the model/oracle.
 */
VerificationReport verify_session(
    const Session& session, std::span<const VerificationCase> cases,
    const VerificationOptions& options, std::stop_token stop = {},
    std::optional<std::chrono::steady_clock::time_point> deadline =
        std::nullopt);

/** @brief Parses bounded native ORT profiling data and requires actual kernel
 * events.
 * @param json Complete profiler output returned after Session::end_profiling().
 * @param maximum_bytes Positive encoded limit, at most 64 MiB; default 16 MiB.
 * @return Complete trace identity and observed provider/operation event counts.
 * @throws Error Invalid JSON, duplicate fields, missing kernel metadata or
 * excess resources.
 * @note A supplied profile is not authenticated or automatically bound to a
 * session. The producer must collect it from the same session after the
 * recorded executions.
 */
ProfileObservation inspect_profile(std::string_view json,
                                   std::size_t maximum_bytes = 16U << 20U);

/** @brief Reports whether every recorded output met the declared numerical
 * tolerance.
 * @param report Nonempty execution report.
 * @return False for no cases/outputs or any mismatched element.
 */
bool verification_passed(const VerificationReport& report) noexcept;

/** @brief Serializes execution observations with explicit unchecked/trust
 * boundaries.
 * @param report Native numerical observations and optional actual profile
 * summary.
 * @param maximum_bytes Positive encoded bound, at most 64 MiB; default 4 MiB.
 * @return Deterministic schema-v1 JSON; numerical status is derived from
 * comparisons.
 * @throws Error Invalid fields or exhausted limits.
 */
std::string serialize_verification_report(const VerificationReport& report,
                                          std::size_t maximum_bytes = 4U
                                                                      << 20U);

}  // namespace jyppx::liuding::onnx_runtime

#endif  // JYPPX_LIUDING_DEPLOYMENT_ONNX_RUNTIME_VERIFICATION_HPP
