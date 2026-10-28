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
/** @brief Opaque fused-MBConv/MBConv stage implementation. */
class EfficientNetV2Blocks;

/** @brief TF EfficientNetV2-S input and optional classifier settings. */
struct EfficientNetV2Options {
  std::int64_t num_classes = 1000;  ///< Zero returns the pooled embedding.
  std::int64_t input_channels = 3;  ///< Positive image channel count.
};

/** @brief Trainable TF-S backbone with dynamic SAME padding. */
class EfficientNetV2Impl final : public torch::nn::Module {
 public:
  /** @brief Constructs the random tf_efficientnetv2_s architecture.
   * @param options Input channels and head width; zero selects an embedding.
   */
  explicit EfficientNetV2Impl(EfficientNetV2Options options = {});
  /** @brief Maps floating NCHW images to logits or a pooled embedding.
   * @param input Positive dense images matching module device/dtype/channels.
   * @return NC logits, or the 1280-wide pre-classifier pooled embedding.
   */
  Tensor forward(const Tensor& input);
  /** @brief Returns stage maps at strides 2,4,8,16,32.
   * @param input Same image contract as forward.
   * @return Five NCHW maps with normal autograd and TF SAME padding.
   */
  std::vector<Tensor> forward_features(const Tensor& input);
  /** @brief Returns current input/head settings.
   * @return Settings including the current classifier width.
   */
  const EfficientNetV2Options& options() const noexcept;
  /** @brief Replaces or removes the classifier on the current device/dtype.
   * @param num_classes Classes in [0,1000000]; zero selects the pooled
   * embedding.
   * @note Rebuild adapters, optimizers and checkpoint topology after mutation.
   */
  void reset_classifier(std::int64_t num_classes);

 private:
  EfficientNetV2Options options_;         ///< Current input/head settings.
  torch::nn::Conv2d conv_stem_{nullptr};  ///< Registered input projection.
  torch::nn::BatchNorm2d bn1_{nullptr};   ///< Registered stem normalization.
  std::shared_ptr<EfficientNetV2Blocks> blocks_;  ///< Fused/MBConv stages.
  torch::nn::Conv2d conv_head_{nullptr};   ///< Pre-pooling channel projection.
  torch::nn::BatchNorm2d bn2_{nullptr};    ///< Pre-pooling normalization.
  torch::nn::Linear classifier_{nullptr};  ///< Optional logits projection.
};
/** @brief Shared owning LibTorch holder for TF EfficientNetV2-S. */
TORCH_MODULE(EfficientNetV2);

/** @brief Constructs the fixed TF-S architecture without downloading.
 * @param options Input/head settings.
 * @return Owning randomly initialized holder.
 */
EfficientNetV2 tf_efficientnetv2_s(EfficientNetV2Options options = {});
/** @brief Adapts the current output width without copying model state.
 * @param model Nonempty concrete holder retained by the adapter.
 * @return NCHW-to-NC adapter with the current logits/embedding schema.
 */
nn::ModelAdapter adapt(EfficientNetV2 model);
}  // namespace jyppx::liuding::models
#endif  // JYPPX_LIUDING_MODELS_EFFICIENTNETV2_HPP
