// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#ifndef JYPPX_LIUDING_DEPLOYMENT_MIGRAPHX_SERVING_BACKEND_HPP
#define JYPPX_LIUDING_DEPLOYMENT_MIGRAPHX_SERVING_BACKEND_HPP
/** @file backend.hpp @brief Optional fixed-shape native MIGraphX bundle bridge.
 */
#include <jyppx/liuding/deployment/migraphx_runtime/environment.hpp>
#include <jyppx/liuding/serving/runtime.hpp>

namespace jyppx::liuding::serving {
/** @brief Native adapter identity, independently of any ORT EP. */
inline constexpr std::string_view kMIGraphXBackendId =
    "jyppx.liuding.migraphx_runtime";
/** @brief Trusted owner decodes the four verified pipeline descriptors.
 * Both arguments are borrowed immutable snapshots. The owner must verify actual
 * descriptor contents, not only copy hashes, and must not retain borrowed
 * references. Only same-schema trusted preprocessing/postprocessing is allowed.
 * Concurrent backend loads may invoke the factory concurrently; its owner must
 * synchronize any shared mutable state.
 */
using MIGraphXPipelineFactory = std::function<Pipeline(
    const VerifiedBundle&, const migraphx_runtime::EnvironmentLock&)>;
/** @brief Exact keyword-only single CPU FP32 fixed-shape input schema.
 * @param name Nonempty printable ASCII input name, at most 255 bytes.
 * @param shape Complete positive dense rank 1..8 shape.
 * @return Closed invocation schema with one required named tensor.
 * @throws Error For invalid text, rank, dimensions or element overflow.
 */
nn::InvocationSchema migraphx_invocation_schema(
    std::string_view name, std::span<const std::size_t> shape);
/** @brief Exact single CPU FP32 tensor response schema, without an invented
 * name.
 * @param shape Complete positive dense rank 1..8 output shape.
 * @return Required tensor schema shared by the owner and actual graph.
 * @throws Error For unsupported dimensions, rank or element overflow.
 */
ValueSchema migraphx_result_schema(std::span<const std::size_t> shape);
/** @brief Projects a canonical observed lock into an explicit Serving
 * descriptor.
 * @param environment Independently observed and trusted session environment.
 * @return Fixed CPU host boundary, native GPU execution identity, exact header/
 * runtime versions and SHA of serialize_environment_lock(environment).
 * @note Store those canonical bytes in the bundle. This helper validates a
 * declaration; it does not certify that its argument was observed.
 */
RuntimeDescriptor migraphx_runtime_descriptor(
    const migraphx_runtime::EnvironmentLock& environment);
/** @brief Loads verified self-contained ONNX bundles using native HIP/MIGraphX.
 * Independent default-off component requiring serving and migraphx_runtime.
 * Public tensors remain CPU FP32; the explicit Session device and checked lock
 * pin actual GPU execution. No ORT EP, implicit fallback, automatic tensor
 * batching, OOM retry or serialized-program loading is provided.
 *
 * Snapshot integrity, runtime declarations, the actual environment lock, exact
 * owner schemas/bindings and nonempty known vectors must all pass before
 * publication. Requests and responses use the common owned snapshot/lease/
 * close lifecycle. Cancellation/deadlines are cooperative boundaries around
 * blocking compile/run; they cannot preempt work. A completed synchronized run
 * interrupted at publication leaves the predictor usable; backend errors
 * degrade it. Logical budgets exclude backend RSS/VRAM/workspace and compile
 * time; use independently bounded processes for untrusted graphs.
 */
class MIGraphXRuntimeBackend final : public RuntimeBackend {
 public:
  /** @brief Freezes the explicit runtime, fixed specialization and trusted
   * owner.
   * @param descriptor Exact canonical-lock declaration and CPU host boundary.
   * @param session Explicit HIP ordinal, input name/shape and payload limits.
   * @param factory Trusted verified-descriptor decoder and same-schema stages.
   * @param model_id Single self-contained ONNX model variant in the bundle.
   * @throws Error For inconsistent or unsupported declarations.
   */
  MIGraphXRuntimeBackend(RuntimeDescriptor descriptor,
                         migraphx_runtime::SessionOptions session,
                         MIGraphXPipelineFactory factory,
                         std::string model_id = "primary");
  /** @brief Returns immutable host/GPU execution metadata. */
  RuntimeDescriptor descriptor() const override;
  /** @brief Reserves the one verified snapshot, without copying model bytes.
   * @param options Positive snapshot byte ceiling used by load.
   * @return max_snapshot_bytes; excludes compiler/device/workspace allocations.
   */
  std::uint64_t load_reservation_bytes(
      const LoadOptions& options) const override;
  /** @brief Verifies, compiles, warms and publishes one fixed GPU predictor.
   * @param directory Immutable data-only bundle directory.
   * @param verify Explicit integrity/signature and encoded-resource policy.
   * @param warmup Nonempty independently supplied known input/output vectors.
   * @param options Snapshot, host I/O and cooperative lifecycle bounds.
   * @return Ready predictor only after complete actual contract checks.
   * @throws Error For integrity, runtime/owner/schema/warmup/resource failure.
   */
  std::shared_ptr<Predictor> load(
      const std::filesystem::path& directory,
      const interop::ModelBundleVerifyOptions& verify,
      std::span<const WarmupCase> warmup,
      const LoadOptions& options = {}) const override;

 private:
  RuntimeDescriptor descriptor_;  ///< Frozen host and actual GPU declaration.
  migraphx_runtime::SessionOptions session_;  ///< Fixed device/shape/limits.
  MIGraphXPipelineFactory factory_;  ///< Explicit trusted pipeline owner.
  std::string model_id_;  ///< Exact selected single ONNX model variant.
};
}  // namespace jyppx::liuding::serving
#endif  // JYPPX_LIUDING_DEPLOYMENT_MIGRAPHX_SERVING_BACKEND_HPP
