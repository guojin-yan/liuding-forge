// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_MODELS_TRANSFORMER_HPP
#define JYPPX_MODELS_TRANSFORMER_HPP
/** @file transformer.hpp
 * @brief A real batch-first post-norm Transformer encoder of embedded features.
 */
#include <jyppx/nn/model_adapter.hpp>

#include <torch/nn/modules/transformercoder.h>
#include <torch/nn/pimpl.h>

#include <cstdint>
namespace jyppx::models {
/** @brief Explicit floating feature encoder, without tokenization/embeddings.
 */
struct TransformerEncoderOptions {
  std::int64_t model_dimension = 512;         ///< Positive feature width.
  std::int64_t attention_heads = 8;           ///< Must divide model_dimension.
  std::int64_t feedforward_dimension = 2048;  ///< Positive hidden FFN width.
  std::int64_t layers = 6;  ///< Positive number of post-norm encoder blocks.
  double dropout = 0.1;     ///< Training-only attention/FFN/residual dropout.
  bool gelu = false;        ///< False selects ReLU; true selects exact GELU.
  bool final_norm = true;   ///< Add final LayerNorm, epsilon 1e-5.
};
/** @brief Registered LibTorch attention/FFN/norm stack with batch-first IO. */
class TransformerEncoderImpl final : public torch::nn::Module {
 public:
  /** @brief Constructs randomly initialized post-norm encoder blocks.
   * @param options Explicit bounded widths, heads, layers and normalization.
   */
  explicit TransformerEncoderImpl(TransformerEncoderOptions options = {});
  /** @brief Encodes embedded [batch,sequence,model_dimension] features.
   * @param input Positive dense floating features on module device/dtype.
   * @param attention_mask Optional bool [sequence,sequence], true means
   * blocked.
   * @param padding_mask Optional bool [batch,sequence], true means ignored key.
   * @return Features with the same shape/device/dtype; gradients preserved.
   * @note No token embedding, positional encoding, causal inference or pooling
   * is added. Callers own sequence meaning and mask correctness.
   */
  Tensor forward(const Tensor& input, const Tensor& attention_mask = {},
                 const Tensor& padding_mask = {});
  /** @brief Returns the immutable feature encoder configuration.
   * @return Constructor options retained for the module lifetime.
   */
  const TransformerEncoderOptions& options() const noexcept;

 private:
  TransformerEncoderOptions options_;  ///< Validated immutable configuration.
  torch::nn::TransformerEncoder encoder_{
      nullptr};  ///< Registered sequence-first stack.
};
/** @brief Shared owning LibTorch holder for the feature encoder. */
TORCH_MODULE(TransformerEncoder);
/** @brief Constructs the floating feature encoder.
 * @param options Explicit bounded widths, heads, layers and normalization.
 * @return Owning randomly initialized encoder without embeddings.
 */
TransformerEncoder transformer_encoder(TransformerEncoderOptions options = {});
/** @brief Adapts one positional feature tensor and optional named bool masks.
 * @param model Nonempty shared holder whose mode and identity are preserved.
 * @return Adapter retaining the same registered module and state.
 */
nn::ModelAdapter adapt(TransformerEncoder model);
}  // namespace jyppx::models
#endif
