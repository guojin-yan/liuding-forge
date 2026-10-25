// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_MODELS_EFFICIENTNETV2_HPP
#define JYPPX_LIUDING_MODELS_EFFICIENTNETV2_HPP

/** @file efficientnetv2.hpp
 * @brief Fixed timm TensorFlow-compatible EfficientNetV2-S classifier.
 */
#include <jyppx/liuding/nn/model_adapter.hpp>

#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/pimpl.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace jyppx::liuding::models {
class EfficientNetV2Blocks;

struct EfficientNetV2Options {
  std::int64_t num_classes = 1000;  ///< Positive classifier width.
  std::int64_t input_channels = 3;  ///< Positive image channel count.
};

class EfficientNetV2Impl final : public torch::nn::Module {
 public:
  /** @brief Constructs the random tf_efficientnetv2_s architecture. */
  explicit EfficientNetV2Impl(EfficientNetV2Options options = {});
  /** @brief Maps floating NCHW images to unnormalized class logits. */
  Tensor forward(const Tensor& input);
  /** @brief Returns stage maps at strides 2,4,8,16,32. */
  std::vector<Tensor> forward_features(const Tensor& input);
  const EfficientNetV2Options& options() const noexcept;

 private:
  EfficientNetV2Options options_;
  torch::nn::Conv2d conv_stem_{nullptr};
  torch::nn::BatchNorm2d bn1_{nullptr};
  std::shared_ptr<EfficientNetV2Blocks> blocks_;
  torch::nn::Conv2d conv_head_{nullptr};
  torch::nn::BatchNorm2d bn2_{nullptr};
  torch::nn::Linear classifier_{nullptr};
};
TORCH_MODULE(EfficientNetV2);

EfficientNetV2 tf_efficientnetv2_s(EfficientNetV2Options options = {});
nn::ModelAdapter adapt(EfficientNetV2 model);
}  // namespace jyppx::liuding::models
#endif  // JYPPX_LIUDING_MODELS_EFFICIENTNETV2_HPP
