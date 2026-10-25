// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_MODELS_EFFICIENTNET_HPP
#define JYPPX_LIUDING_MODELS_EFFICIENTNET_HPP
/** @file efficientnet.hpp
 * @brief Fixed timm EfficientNet B0/B2 classifiers and multiscale features.
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
class EfficientNetBlocks;

struct EfficientNetOptions {
  std::int64_t num_classes = 1000;  ///< Positive classifier width.
  std::int64_t input_channels = 3;  ///< Positive image channel count.
};

enum class EfficientNetVariant { b0, b2 };

class EfficientNetImpl final : public torch::nn::Module {
 public:
  /** @brief Constructs the fixed random architecture without downloading. */
  EfficientNetImpl(EfficientNetVariant variant,
                   EfficientNetOptions options = {});
  /** @brief Maps floating NCHW images to unnormalized class logits. */
  Tensor forward(const Tensor& input);
  /** @brief Returns timm stage checkpoints at strides 2,4,8,16,32. */
  std::vector<Tensor> forward_features(const Tensor& input);
  const EfficientNetOptions& options() const noexcept;
  EfficientNetVariant variant() const noexcept;

 private:
  EfficientNetVariant variant_;
  EfficientNetOptions options_;
  torch::nn::Conv2d conv_stem_{nullptr};
  torch::nn::BatchNorm2d bn1_{nullptr};
  std::shared_ptr<EfficientNetBlocks> blocks_;
  torch::nn::Conv2d conv_head_{nullptr};
  torch::nn::BatchNorm2d bn2_{nullptr};
  torch::nn::Linear classifier_{nullptr};
};
TORCH_MODULE(EfficientNet);

EfficientNet efficientnet_b0(EfficientNetOptions options = {});
EfficientNet efficientnet_b2(EfficientNetOptions options = {});
nn::ModelAdapter adapt(EfficientNet model);
}  // namespace jyppx::liuding::models
#endif  // JYPPX_LIUDING_MODELS_EFFICIENTNET_HPP
