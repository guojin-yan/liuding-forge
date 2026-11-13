// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_DEPLOYMENT_ONNX_RUNTIME_TYPED_SESSION_HPP
#define JYPPX_LIUDING_DEPLOYMENT_ONNX_RUNTIME_TYPED_SESSION_HPP
/** @file typed_session.hpp @brief Versioned native CPU dense FP32/int64/int32
 * I/O. */
#include <jyppx/liuding/foundation/artifact.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
namespace jyppx::liuding::onnx_runtime {
enum class TensorElementType { float32 = 1, int32 = 6, int64 = 7 };
struct TypedTensorSignature {
  std::string name;
  TensorElementType type = TensorElementType::float32;
  std::vector<std::int64_t> shape;
  std::string leading_symbol;
};
using TypedInputValues =
    std::variant<std::span<const float>, std::span<const std::int32_t>,
                 std::span<const std::int64_t>>;
using TypedOutputValues =
    std::variant<std::vector<float>, std::vector<std::int32_t>,
                 std::vector<std::int64_t>>;
struct TypedTensorInput {
  std::string_view name;
  TypedInputValues values;
  std::span<const std::int64_t> shape;
};
struct TypedTensorOutput {
  std::string name;
  std::vector<std::int64_t> shape;
  TypedOutputValues values;
};
struct TypedSessionOptions {
  std::uint32_t cpu_threads = 1, max_io_tensors = 64;
  std::uint64_t max_model_bytes = 256ULL << 20U, max_input_bytes = 64ULL << 20U,
                max_output_bytes = 64ULL << 20U, max_batch_size = 1024;
  bool disable_graph_optimizations = false;
  std::map<std::string, std::uint64_t>
      maximum_output_rows;  ///< Required for unbound dynamic leading output
                            ///< dimensions.
};
/** @brief v1 typed profile using the pinned official ORT CPU provider.
 * Rank 1..8, positive fixed axes, optional named leading dimension. Dynamic
 * output sizes may be input-bound or have explicit per-output row limits; zero
 * rows are valid only on dynamic outputs. Limits check signatures before load
 * consumption and actual output bytes after execution. ORT allocates dynamic
 * outputs/intermediates internally; these are logical I/O limits, not an arena
 * or process memory quota. Inputs are borrowed, outputs own aligned typed
 * vectors. Float inputs/outputs must be finite. Old FP32 Session is unchanged.
 * This profile does not append GPU providers or claim their qualification.
 */
class TypedSession final {
 public:
  TypedSession(std::span<const std::byte> model, const ArtifactDigest& expected,
               const TypedSessionOptions& options = {});
  ~TypedSession();
  TypedSession(const TypedSession&) = delete;
  TypedSession& operator=(const TypedSession&) = delete;
  const std::vector<TypedTensorSignature>& inputs() const noexcept;
  const std::vector<TypedTensorSignature>& outputs() const noexcept;
  std::vector<TypedTensorOutput> run(std::span<const TypedTensorInput> inputs,
                                     std::stop_token stop = {}) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace jyppx::liuding::onnx_runtime
#endif
