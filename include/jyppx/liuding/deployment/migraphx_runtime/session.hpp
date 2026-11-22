// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DEPLOYMENT_MIGRAPHX_RUNTIME_SESSION_HPP
#define JYPPX_LIUDING_DEPLOYMENT_MIGRAPHX_RUNTIME_SESSION_HPP
/** @file session.hpp
 * @brief Optional native MIGraphX GPU adapter with fixed FP32 host I/O.
 */

#include <jyppx/liuding/foundation/artifact.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace jyppx::liuding::migraphx_runtime {

/** @brief Explicit device, specialization and host payload limits. */
struct SessionOptions {
  std::int32_t device_id = 0;  ///< Visible HIP ordinal, independently checked.
  std::string input_name = "input";      ///< Exact single graph input name.
  std::vector<std::size_t> input_shape;  ///< Complete positive rank 1..8 shape.
  std::uint64_t max_model_bytes = 256U << 20U;  ///< Bound before parsing.
  std::uint64_t max_input_bytes = 64U << 20U;   ///< Bound before GPU work.
  std::uint64_t max_output_bytes = 64U << 20U;  ///< Bound before execution.
  std::uint64_t max_batch_size = 1024U;  ///< Positive leading dimension bound.
};

/** @brief Owned, contiguous FP32 output and actual executed operation names. */
struct RunResult {
  std::vector<std::size_t> shape;  ///< Complete positive row-major shape.
  std::vector<float> values;  ///< Finite values, independent of session memory.
  std::vector<std::string> instructions;  ///< At most 10000 executed names.
};

/** @brief Owns one GPU program specialized to a single fixed input shape.
 * Requires the optional `jyppx::liuding::migraphx_runtime` target, external
 * MIGraphX 2.18.0 and a compatible HIP SDK/device. Pass independently checked
 * self-contained ONNX bytes; the digest verifies identity, not producer trust.
 * Exactly one dense FP32 input and output of rank 1..8 are supported. Each
 * different batch/shape needs a separate session. Compilation explicitly uses
 * gpu, offload_copy=true, fast_math=false and exhaustive_tune=false.
 *
 * Calls on one session serialize; different sessions own independent programs.
 * Calls bind the selected HIP device and restore the calling thread's prior
 * device. Inputs remain borrowed throughout admission and synchronous work.
 * Results own their storage. Destruction must not race a call. Graph parsing,
 * compilation and running block the calling thread without cancellation or a
 * hard execution deadline. Payload limits exclude backend/intermediate/device
 * allocations; separately bound processes are needed for untrusted graphs.
 * Model/operator/numerical support requires separate qualification. Bitwise
 * determinism across executions is not guaranteed by this adapter.
 * An incomplete/error run performs a device-wide synchronization before
 * releasing borrowed input and backend arguments; other work on that device
 * can therefore delay error cleanup.
 */
class Session final {
 public:
  /** @brief Validates identity/options and compiles one fixed GPU program.
   * @param model Stable self-contained model bytes, borrowed for this call.
   * @param digest Independently trusted SHA-256 identity of the complete bytes.
   * @param options Explicit visible device, shape and positive payload bounds.
   * @throws InvalidArgument For invalid options, shape, digest or empty bytes.
   * @throws CompatibilityError For a digest, model, type or signature mismatch.
   * @throws ResourceExhausted For payload bounds or host allocation failure.
   * @throws DeviceError For device discovery, selection or initialization.
   * @throws BackendError If the backend cannot compile the requested program.
   */
  Session(std::span<const std::byte> model, const ArtifactDigest& digest,
          const SessionOptions& options);
  /** @brief Releases the owned GPU program after all calls have ended. */
  ~Session();
  Session(const Session&) = delete;  ///< Exclusive program ownership.
  Session& operator=(const Session&) = delete;  ///< Copying is unsupported.
  Session(Session&&) = delete;  ///< Stable identity for serialized callers.
  Session& operator=(Session&&) = delete;  ///< Moving is unsupported.

  /** @brief Returns immutable specialization metadata for this lifetime. */
  const SessionOptions& options() const noexcept;
  /** @brief Returns the verified fixed output shape for this lifetime. */
  std::span<const std::size_t> output_shape() const noexcept;
  /** @brief Runs finite host FP32 values using the fixed specialization.
   * @param values Contiguous values, stable and unchanged until return.
   * @return Owned output and actual GPU dispatch; no backend buffer escapes.
   * @throws InvalidArgument For an incorrect element count/nonfinite input.
   * @throws NumericalError For a nonfinite output.
   * @throws CompatibilityError For a changed signature, CPU/reference route or
   * missing GPU arithmetic and synchronized host output in the actual trace.
   * @throws ResourceExhausted For host allocation/trace storage bounds.
   * @throws DeviceError For a device selection failure.
   * @throws BackendError For backend execution failure.
   */
  RunResult run(std::span<const float> values) const;

 private:
  class Impl;  ///< Private C API resources; SDK types never enter this ABI.
  std::unique_ptr<Impl> impl_;  ///< Owns specialization and run admission.
};
}  // namespace jyppx::liuding::migraphx_runtime
#endif  // JYPPX_LIUDING_DEPLOYMENT_MIGRAPHX_RUNTIME_SESSION_HPP
