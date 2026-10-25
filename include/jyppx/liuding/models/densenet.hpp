// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_MODELS_DENSENET_HPP
#define JYPPX_LIUDING_MODELS_DENSENET_HPP
/** @file densenet.hpp
 * @brief Trainable DenseNet121 image classifier and four dense-stage features.
 */
#include <jyppx/liuding/nn/model_adapter.hpp>

#include <torch/nn/modules/linear.h>
#include <torch/nn/pimpl.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace jyppx::liuding::models {
class DenseNetFeatures;
/** @brief Fixed DenseNet121 architecture with configurable input and head. */
struct DenseNetOptions {
  std::int64_t num_classes = 1000;  ///< Positive classifier width.
  std::int64_t input_channels = 3;  ///< Positive image channel count.
};

/** @brief DenseNet121 with 6/12/24/16 dense layers and growth rate 32. */
class DenseNetImpl final : public torch::nn::Module {
 public:
  /** @brief Constructs a randomly initialized classifier without downloading.
   * @param options Input channels and classifier width.
   */
  explicit DenseNetImpl(DenseNetOptions options = {});
  /** @brief Maps normalized NCHW images to unnormalized class logits.
   * @param input Floating image on the model's exact device and dtype.
   * @return Logits with shape [N,num_classes].
   */
  Tensor forward(const Tensor& input);
  /** @brief Returns outputs of four dense blocks before their transitions.
   * @param input Same image contract as forward.
   * @return NCHW maps at nominal strides 4,8,16,32.
   */
  std::vector<Tensor> forward_features(const Tensor& input);
  /** @brief Returns immutable constructor options.
   * @return Options retained for the module lifetime.
   */
  const DenseNetOptions& options() const noexcept;

 private:
  DenseNetOptions options_;
  std::shared_ptr<DenseNetFeatures> features_;
  torch::nn::Linear classifier_{nullptr};
};
TORCH_MODULE(DenseNet);

/** @brief Constructs the fixed DenseNet121 architecture.
 * @param options Input channels and classifier width.
 * @return Owning randomly initialized holder.
 */
DenseNet densenet121(DenseNetOptions options = {});

/** @brief Creates an owning NCHW/logit schema-checked adapter.
 * @param model Nonempty holder retained by the adapter.
 * @return Adapter over the same registered module and state.
 */
nn::ModelAdapter adapt(DenseNet model);
}  // namespace jyppx::liuding::models
#endif  // JYPPX_LIUDING_MODELS_DENSENET_HPP
