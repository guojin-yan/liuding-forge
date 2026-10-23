// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_INTEROP_TORCHSCRIPT_COMPATIBILITY_HPP
#define JYPPX_LIUDING_INTEROP_TORCHSCRIPT_COMPATIBILITY_HPP
/**
 * @file torchscript_compatibility.hpp
 * @brief Loads and executes explicitly trusted existing TorchScript artifacts.
 * @note This optional interop->nn adapter is deprecated compatibility support.
 * It neither exports/traces models nor changes the data-only bundle format.
 */

#include <jyppx/liuding/foundation/artifact.hpp>
#include <jyppx/liuding/nn/invocation.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

namespace jyppx::liuding::interop {

/**
 * @brief Limits checked at the trusted in-process runtime boundary.
 * @note Archive size is a compressed-input limit. Parameter/buffer accounting
 * happens after LibTorch load; output accounting happens after execution.
 * These are not limits on backend allocations, decompression, graph execution,
 * process memory, or kernel duration. Use an isolated process for those limits.
 */
struct TorchScriptLimits {
  std::uint64_t max_artifact_bytes =
      512ULL << 20U;  ///< Complete serialized artifact snapshot.
  std::uint64_t max_state_tensor_bytes =
      1ULL << 30U;  ///< Post-load sum of named parameter/buffer logical bytes.
  std::uint64_t max_input_tensor_bytes =
      256ULL << 20U;  ///< Per-call input snapshot bytes, aliases counted again.
  std::uint64_t max_output_tensor_bytes =
      256ULL << 20U;  ///< Per-call independent output snapshot bytes.
  std::size_t max_arguments = 64U;  ///< Forward arguments excluding self.
  std::size_t max_state_tensors = 65536U;  ///< Named parameters plus buffers.
  std::size_t max_tensor_rank = 32U;  ///< Every boundary/state tensor rank.
  std::size_t max_string_bytes =
      64U << 10U;  ///< Total value strings and dictionary keys per boundary.
  ValueSchemaLimits value_limits;  ///< Total traversal/type/metadata budgets.
};

/** @brief Explicit trust and placement for one existing compatibility model. */
struct TorchScriptLoadOptions {
  bool trust_source =
      false;                 ///< Required opt-in; a hash is not authentication.
  Device device{c10::kCPU};  ///< Exactly cpu or an explicit cuda:N device.
  TorchScriptLimits limits;  ///< Boundary limits, not an execution sandbox.
};

/** @brief One actual JIT argument without exposing its default value. */
struct TorchScriptArgumentInfo {
  std::string name;                 ///< Actual forward argument name.
  std::string type;                 ///< Actual supported JIT type description.
  bool has_default = false;         ///< Artifact contains a real default value.
  bool bound_positionally = false;  ///< Member of the fixed positional prefix.
  bool required =
      true;  ///< Adapter requires an explicit argument on each call.
};

/** @brief Owning diagnostic snapshot; never proof about unexecuted cases. */
struct TorchScriptRuntimeInfo {
  ArtifactDigest artifact_digest;  ///< Hash of the exact bytes actually loaded.
  std::uint64_t artifact_bytes = 0U;  ///< Verified serialized snapshot length.
  std::string libtorch_version;       ///< Actual linked LibTorch build version.
  std::string device;  ///< Explicit placement, without fallback or dtype cast.
  std::vector<TorchScriptArgumentInfo> arguments;  ///< Actual forward boundary.
  std::string return_type;  ///< Actual single result type, possibly a tuple.
  std::uint64_t state_tensor_bytes =
      0U;  ///< Parameters/buffers, not all memory.
  std::size_t state_tensors =
      0U;             ///< Named entries; shared storage counted again.
  bool open = false;  ///< False after close or a failure during execution.
  bool compatibility_only = true;  ///< Existing artifacts only.
  bool deprecated = true;          ///< Not a recommended new export path.
};

/**
 * @brief Serialized inference runtime for an explicitly trusted JIT artifact.
 *
 * The supported boundary is Tensor, int, float, bool, string, None, optional
 * values, fixed tuples, homogeneous lists and fixed string-keyed dictionaries
 * of those types. Any, custom classes, device values, named tuples, variadic
 * methods and keyword-only JIT arguments are rejected. Tensors are dense,
 * contiguous CPU/CUDA inputs; outputs are copied to contiguous independent
 * storage. See the API documentation for the exact dtype and schema matrix.
 *
 * Inputs/defaults and outputs are value snapshots: tensor/storage aliasing
 * between fields is deliberately not preserved. Inputs must remain quiescent
 * during snapshotting. Returns are inference tensors, independent of module
 * state and input storage, and can outlive close. No autograd/training contract
 * is provided. Trusted scripted code may mutate its private model state.
 *
 * Operations on one object serialize. close waits for the current operation
 * and is idempotent. Move/destruction require exclusive access to the object.
 * Stop/deadline are checked while waiting and around backend calls; they cannot
 * preempt load, forward, CUDA synchronization or a running kernel. CUDA success
 * includes device synchronization, which also waits for unrelated device work.
 * Ambient autocast on the runtime device is rejected without changing it.
 * Any failure after entering forward closes the runtime: partial private state
 * changes are not rolled back. Backend error details remain a nested cause.
 */
class TorchScriptCompatibilityRuntime final {
 public:
  /** @brief Releases the retained compatibility module. */
  ~TorchScriptCompatibilityRuntime();
  /** @brief Transfers ownership; exclude concurrent access during the move. */
  TorchScriptCompatibilityRuntime(TorchScriptCompatibilityRuntime&&) noexcept;
  /** @brief Transfers ownership; exclude concurrent access during the move.
   * @return This runtime after replacement.
   */
  TorchScriptCompatibilityRuntime& operator=(
      TorchScriptCompatibilityRuntime&&) noexcept;
  /** @brief Runtime state cannot be copied. */
  TorchScriptCompatibilityRuntime(const TorchScriptCompatibilityRuntime&) =
      delete;
  /** @brief Runtime state cannot be copy assigned. */
  TorchScriptCompatibilityRuntime& operator=(
      const TorchScriptCompatibilityRuntime&) = delete;

  /**
   * @brief Verifies a private byte snapshot, then really loads a JIT module.
   * @param artifact Complete stable artifact bytes; retained only for this
   * call.
   * @param expected_digest Independently trusted SHA-256 of those exact bytes.
   * @param input_schema Fixed positional prefix and all remaining named args.
   * Extra keywords are forbidden. A keyword may be optional only if the actual
   * JIT argument has a default, which is validated and snapshotted at load.
   * @param output_schema Supported exact result category and constraints.
   * @param options Explicit executable-source trust, device and limits.
   * @param stop Cooperative boundary cancellation.
   * @param deadline Absolute cooperative boundary deadline.
   * @return A ready runtime with its real method/type boundary validated.
   * @throws Error For untrusted/invalid/oversize/corrupt/incompatible
   * artifacts, schema/device/limit failures, cancellation, deadline or backend
   * failure.
   * @note No path is reopened after verification; no code/plugin is taken from
   * a model bundle. LibTorch operator resolution uses the host process
   * registry.
   */
  static TorchScriptCompatibilityRuntime load(
      std::span<const std::byte> artifact,
      const ArtifactDigest& expected_digest,
      const nn::InvocationSchema& input_schema,
      const ValueSchema& output_schema,
      const TorchScriptLoadOptions& options = {}, std::stop_token stop = {},
      std::optional<std::chrono::steady_clock::time_point> deadline = {});

  /**
   * @brief Snapshots, validates and executes a real forward call.
   * @param input Exactly the declared positional prefix and permitted kwargs.
   * @param stop Cooperative boundary cancellation, including lock acquisition.
   * @param deadline Absolute cooperative boundary deadline.
   * @return Independent validated inference values and the declared schema.
   * @throws Error For invalid inputs, resource/schema/backend failures, a
   * closed runtime, cancellation or deadline. Pre-forward rejection keeps the
   * runtime open; post-forward failure closes it without state rollback.
   */
  nn::InvocationResult invoke(
      const nn::Invocation& input, std::stop_token stop = {},
      std::optional<std::chrono::steady_clock::time_point> deadline = {});

  /**
   * @brief Returns actual loaded metadata, also available after close.
   * @return An owning metadata snapshot without state/default values.
   * @throws NotReady For an object whose ownership has been moved away.
   */
  TorchScriptRuntimeInfo inspect() const;

  /**
   * @brief Waits for the active operation and releases module/default state.
   * @note Idempotent, also for a moved-from object; does not invalidate
   * returns.
   */
  void close();

 private:
  /** @brief Owns the official JIT module without exposing its public headers.
   */
  class Impl;
  /**
   * @brief Takes exclusive ownership of an already validated runtime.
   * @param impl Non-null ready implementation built by load.
   */
  explicit TorchScriptCompatibilityRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;  ///< Hidden official JIT module and lock.
};

}  // namespace jyppx::liuding::interop
#endif  // JYPPX_LIUDING_INTEROP_TORCHSCRIPT_COMPATIBILITY_HPP