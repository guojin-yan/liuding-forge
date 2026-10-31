// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DEPLOYMENT_ONNX_RUNTIME_SESSION_HPP
#define JYPPX_LIUDING_DEPLOYMENT_ONNX_RUNTIME_SESSION_HPP
/** @file session.hpp
 * @brief Optional independent ONNX Runtime adapter with explicit providers.
 */

#include <jyppx/liuding/foundation/artifact.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::onnx_runtime {

/** @brief One explicitly requested ORT provider; availability is not execution.
 */
struct ExecutionProvider {
  std::string name = "CPUExecutionProvider";  ///< Exact official provider name.
  std::map<std::string, std::string>
      options;  ///< Exact official key/value pairs.
};

/** @brief Native CUDA EP settings for host I/O and concurrent session runs. */
struct CudaProviderOptions {
  std::int32_t device_id = 0;  ///< Explicit nonnegative CUDA device ordinal.
  std::optional<std::uint64_t> arena_memory_limit_bytes;
  ///< Optional positive CUDA arena limit; allocations outside that arena and
  ///< total provider/driver memory are not bounded by this setting.
  bool allow_tf32 = false;  ///< Explicit precision opt-in on supported devices.
  bool allow_max_conv_workspace =
      false;  ///< Opt-in to cuDNN's larger workspace.
};

/** @brief Creates explicit CUDA options without loading or probing a device.
 * @param options Device, optional arena budget and precision/workspace policy.
 * @return Official CUDA V2 provider keys, heuristic convolution search and
 * synchronized default-stream copies. CPU fallback remains a session policy.
 * @throws InvalidArgument For a negative device or unrepresentable/zero budget.
 * @throws ResourceExhausted If provider option storage cannot be allocated.
 * @note Session creation still requires a complete ORT GPU SDK and compatible
 * CUDA/cuDNN/driver/device. The arena budget does not bound total device
 * memory. Raw provider maps remain available for other documented CUDA options,
 * except CUDA Graph, external streams/allocators and asynchronous copies, whose
 * lifetime/concurrency contracts are not supported by this host-I/O adapter.
 */
ExecutionProvider cuda_execution_provider(const CudaProviderOptions& options);

/** @brief Queries the providers compiled into the actually loaded ORT library.
 * @return Exact official names reported by OrtApi::GetAvailableProviders.
 * @throws Error If the runtime cannot report its available providers.
 * @note A listed GPU provider may still lack a device or dependent driver/SDK.
 * This query never establishes model, numerical or kernel qualification.
 */
std::vector<std::string> available_providers();

/** @brief Reports whether this adapter has an official append path for a name.
 * @param name Exact official provider name; no case/suffix inference occurs.
 * @return True for a wired official append API, independently of availability.
 * @note False means explicitly unsupported by this adapter version. True does
 * not prove an installed provider, compatible driver/device, or model support.
 */
bool provider_append_supported(std::string_view name) noexcept;

/** @brief Immutable session configuration; no provider is inferred or fetched.
 */
struct SessionOptions {
  ExecutionProvider
      provider;  ///< CPU by default; no implicit device selection.
  bool allow_cpu_fallback = false;  ///< Explicit non-CPU partition fallback.
  std::uint32_t cpu_threads =
      1U;  ///< Intra-op threads; inter-op is always one.
  std::uint64_t max_model_bytes = 256U << 20U;  ///< Bound before parsing bytes.
  std::uint64_t max_input_bytes = 64U << 20U;  ///< All inputs' FP32 byte bound.
  std::uint64_t max_output_bytes = 64U
                                   << 20U;  ///< All outputs' FP32 byte bound.
  std::uint64_t max_batch_size =
      1024U;  ///< Each input's leading dimension bound.
  std::uint32_t max_io_tensors =
      64U;  ///< Combined input/output count, 2..4096.
  bool disable_graph_optimizations =
      false;                             ///< Diagnostic option, not fallback.
  std::filesystem::path profile_prefix;  ///< Empty disables ORT profiling.
};

/** @brief Declared dense FP32 tensor signature, not a task contract. */
struct TensorSignature {
  std::string name;  ///< Exact nonempty graph input or output name.
  std::vector<std::int64_t> shape;  ///< Positive dimensions; -1 only for batch.
  std::string batch_symbol;  ///< Equal nonempty symbols bind a dynamic batch.
};

/** @brief Borrowed named contiguous FP32 input; storage must outlive the call.
 * @note Input order is arbitrary. Names bind tensors to the graph signature.
 * Values, dimensions and name must remain unchanged throughout execution.
 */
struct TensorInput {
  std::string_view name;  ///< Exact graph input name; no positional fallback.
  std::span<const float> values;  ///< Read-only contiguous row-major values.
  std::span<const std::int64_t> shape;  ///< Complete positive runtime shape.
};

/** @brief Owned contiguous row-major FP32 result independent of runtime memory.
 */
struct TensorOutput {
  std::vector<std::int64_t> shape;  ///< Complete positive runtime shape.
  std::vector<float> values;        ///< Exactly the shape's element count.
  std::string name;  ///< Exact graph output name, including single-output runs.
};

/**
 * @brief Owns one independently loaded ONNX Runtime session.
 *
 * Requires the separately enabled `jyppx::liuding::onnx_runtime` target and
 * exact supported ORT library. No Python, Trainer, Torch module, network
 * discovery or custom-op library is loaded by this adapter. The requested
 * provider must be compiled into that library and have a wired official append
 * API. Non-CPU requests prohibit CPU partition fallback unless explicitly
 * permitted. The graph has one or more FP32 inputs and outputs of rank 1..8,
 * bounded by max_io_tensors. Only named leading batch dimensions may be
 * dynamic; other dimensions are positive constants. Equal symbols bind equal
 * sizes across inputs; every dynamic output symbol must occur in an input.
 * Independent input batch symbols may have different sizes. Quantized internal
 * nodes are allowed without
 * asserting INT8 execution: that requires separate real provider/kernel
 * evidence.
 *
 * Inputs/outputs reside in host memory; ORT owns provider-specific transfers.
 * I/O limits do not bound graph/intermediate memory or device allocations.
 * Untrusted graphs require a separately bounded process and provenance policy.
 * SHA-256 verifies bytes, not producer trust. External-data models are not
 * supported: pass an independently checked self-contained ONNX artifact.
 */
class Session final {
 public:
  /** @brief Checks identity, parses bytes and verifies the restricted
   * signature.
   * @param model Complete stable self-contained ONNX bytes for this call.
   * @param expected_digest Independently trusted identity of exactly these
   * bytes.
   * @param options Explicit provider, allocation and optional profile settings.
   * @throws InvalidArgument For malformed identity, options or empty input.
   * @throws CompatibilityError For a hash/runtime/model/operator/schema
   * mismatch.
   * @throws ResourceExhausted For a model bound or allocation failure.
   * @throws DeviceError If an available non-CPU provider fails to initialize.
   * @throws Error For backend initialization or filesystem failures.
   * @note The constructor does not retain the caller's byte span. The caller
   * must not concurrently modify it while construction is in progress.
   */
  Session(std::span<const std::byte> model,
          const ArtifactDigest& expected_digest,
          const SessionOptions& options = {});

  /** @brief Releases session resources; all concurrent calls must have ended.
   */
  ~Session();
  Session(const Session&) =
      delete;  ///< One session owns its runtime resources.
  Session& operator=(const Session&) = delete;  ///< Copying is not supported.
  Session(Session&&) = delete;  ///< Stable identity for concurrent callers.
  Session& operator=(Session&&) = delete;  ///< Moving is not supported.

  /** @brief Returns all verified input signatures in graph declaration order.
   * @return Immutable borrowed metadata valid for this session's lifetime.
   */
  std::span<const TensorSignature> inputs() const noexcept;
  /** @brief Returns all verified output signatures in graph declaration order.
   * @return Immutable borrowed metadata valid for this session's lifetime.
   */
  std::span<const TensorSignature> outputs() const noexcept;

  /** @brief Returns the signature of a graph with exactly one input.
   * @return Reference valid for this session's lifetime; concurrently readable.
   * @throws InvalidArgument If the graph has multiple inputs; use inputs().
   */
  const TensorSignature& input() const;
  /** @brief Returns the signature of a graph with exactly one output.
   * @return Reference valid for this session's lifetime; concurrently readable.
   * @throws InvalidArgument If the graph has multiple outputs; use outputs().
   */
  const TensorSignature& output() const;
  /** @brief Returns the exact loaded ORT version, not the exporter version.
   * @return Reference valid for this session's lifetime; concurrently readable.
   */
  const std::string& runtime_version() const noexcept;
  /** @brief Returns the verified complete source-byte identity.
   * @return Reference valid for this session's lifetime; concurrently readable.
   */
  const ArtifactDigest& model_digest() const noexcept;

  /** @brief Returns the explicit primary provider and options used at creation.
   * @return Immutable request metadata; it does not prove actual kernel choice.
   */
  const ExecutionProvider& execution_provider() const noexcept;

  /** @brief Returns whether non-CPU graphs may partition onto the CPU provider.
   * @return The explicit fallback policy retained at construction.
   */
  bool cpu_fallback_allowed() const noexcept;

  /** @brief Executes all graph inputs and returns all outputs in graph order.
   * @param inputs Every named input exactly once, in any order; spans are
   * borrowed and no FP32 input payload is copied by the adapter.
   * @param stop Cancellation forwarded to this call's ORT run options only.
   * @param deadline Optional absolute steady-clock execution deadline.
   * @return Independently owned named finite results in output signature order.
   * @throws InvalidArgument For missing, duplicate or unknown names, invalid
   * shape/count/value, or inconsistent sizes for one dynamic batch symbol.
   * @throws ResourceExhausted If the combined input or output payload exceeds
   * its call-level bound; output bounds are checked before payload allocation.
   * @throws Cancelled If stop is observed before result publication.
   * @throws DeadlineExceeded If the deadline is reached before publication.
   * @throws NumericalError For any nonfinite graph result.
   * @throws Error For runtime/operator execution failures.
   * @note Concurrent calls have independent run options, symbol bindings and
   * output storage. Cancellation is cooperative. Destruction must not race a
   * call. I/O limits exclude metadata and graph/intermediate/provider memory.
   */
  std::vector<TensorOutput> run(
      std::span<const TensorInput> inputs, std::stop_token stop = {},
      std::optional<std::chrono::steady_clock::time_point> deadline =
          std::nullopt) const;

  /** @brief Executes the graph using caller-owned host input and owned output.
   * @param values Stable contiguous FP32 values retained unchanged for the
   * call.
   * @param shape Positive input dimensions matching the declared signature.
   * @param stop Cancellation forwarded to this call's ORT run options only.
   * @param deadline Optional absolute steady-clock execution deadline.
   * @return Owned finite values and shape; no ORT-owned tensor escapes.
   * @throws InvalidArgument If the graph has multiple inputs or outputs, or
   * for shape, element-count or nonfinite input errors.
   * @throws ResourceExhausted If I/O limits or allocation bounds are exceeded.
   * @throws Cancelled If stop is observed before result publication.
   * @throws DeadlineExceeded If the deadline is reached before publication.
   * @throws NumericalError For nonfinite graph results.
   * @throws Error For runtime/operator execution failures.
   * @note Concurrent calls use independent run options and I/O storage. ORT
   * cancellation is cooperative, not a hard wall-clock or process-memory quota.
   * Destruction must not race any call. No output is returned after an observed
   * cancellation/deadline, even if the backend finished concurrently.
   */
  TensorOutput run(std::span<const float> values,
                   std::span<const std::int64_t> shape,
                   std::stop_token stop = {},
                   std::optional<std::chrono::steady_clock::time_point>
                       deadline = std::nullopt) const;

  /** @brief Flushes and closes this session's optional native ORT profiler.
   * @return Actual profile filename. Repeated calls return the same filename.
   * @throws InvalidArgument If profiling was not explicitly enabled.
   * @throws Error If the backend cannot finish its profile file.
   * @note Waits for active run calls and serializes repeated closes. Inference
   * remains usable afterward. A profile is observation data, not proof of a
   * particular kernel unless its actual node/type/provider events are checked.
   */
  std::filesystem::path end_profiling();

 private:
  class Impl;  ///< Private ORT resources and synchronization; no SDK in this
               ///< ABI.
  std::unique_ptr<Impl> impl_;  ///< Exclusive lifetime owner.
};

}  // namespace jyppx::liuding::onnx_runtime

#endif  // JYPPX_LIUDING_DEPLOYMENT_ONNX_RUNTIME_SESSION_HPP
