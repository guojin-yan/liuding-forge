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
/** @brief Opaque registered MBConv stage implementation. */
class EfficientNetBlocks;

/** @brief Fixed backbone with configurable input and optional classifier. */
struct EfficientNetOptions {
  std::int64_t num_classes = 1000;  ///< Zero returns the pooled embedding.
  std::int64_t input_channels = 3;  ///< Positive image channel count.
};

/** @brief Supported width/depth scaling schedules. */
enum class EfficientNetVariant {
  b0,  ///< Base width/depth multipliers of one.
  b2,  ///< Width 1.1 and depth 1.2.
};

/** @brief Trainable EfficientNet B0/B2 with optional affine classification. */
class EfficientNetImpl final : public torch::nn::Module {
 public:
  /** @brief Constructs the fixed random architecture without downloading.
   * @param variant Supported width/depth schedule.
   * @param options Input channels and head width; zero selects an embedding.
   */
  EfficientNetImpl(EfficientNetVariant variant,
                   EfficientNetOptions options = {});
  /** @brief Maps floating NCHW images to logits or a pooled embedding.
   * @param input Positive dense images matching module device/dtype/channels.
   * @return NC logits, or the 1280/1408-wide pooled B0/B2 embedding.
   */
  Tensor forward(const Tensor& input);
  /** @brief Returns timm stage checkpoints at strides 2,4,8,16,32.
   * @param input Same image contract as forward.
   * @return Five NCHW maps retaining autograd and device/dtype.
   */
  std::vector<Tensor> forward_features(const Tensor& input);
  /** @brief Returns current input/head settings.
   * @return Settings including the current classifier width.
   */
  const EfficientNetOptions& options() const noexcept;
  /** @brief Replaces or removes the classifier on the current device/dtype.
   * @param num_classes Classes in [0,1000000]; zero selects the pooled
   * embedding.
   * @note Rebuild adapters, optimizers and checkpoint topology after mutation.
   */
  void reset_classifier(std::int64_t num_classes);

  /** @brief Returns the fixed width/depth schedule.
   * @return Variant chosen at construction.
   */
  EfficientNetVariant variant() const noexcept;

 private:
  EfficientNetVariant variant_;           ///< Fixed scaling schedule.
  EfficientNetOptions options_;           ///< Current input/head settings.
  torch::nn::Conv2d conv_stem_{nullptr};  ///< Registered input projection.
  torch::nn::BatchNorm2d bn1_{nullptr};   ///< Registered stem normalization.
  std::shared_ptr<EfficientNetBlocks> blocks_;  ///< MBConv stages.
  torch::nn::Conv2d conv_head_{nullptr};   ///< Pre-pooling channel projection.
  torch::nn::BatchNorm2d bn2_{nullptr};    ///< Pre-pooling normalization.
  torch::nn::Linear classifier_{nullptr};  ///< Optional logits projection.
};
/** @brief Shared owning LibTorch holder for EfficientNet. */
TORCH_MODULE(EfficientNet);

/** @brief Constructs the fixed B0 architecture.
 * @param options Input/head settings.
 * @return Owning randomly initialized holder.
 */
EfficientNet efficientnet_b0(EfficientNetOptions options = {});
/** @brief Constructs the fixed B2 architecture.
 * @param options Input/head settings.
 * @return Owning randomly initialized holder.
 */
EfficientNet efficientnet_b2(EfficientNetOptions options = {});
/** @brief Adapts the current output width without copying model state.
 * @param model Nonempty concrete holder retained by the adapter.
 * @return NCHW-to-NC adapter with the current logits/embedding schema.
 */
nn::ModelAdapter adapt(EfficientNet model);
}  // namespace jyppx::liuding::models
#endif  // JYPPX_LIUDING_MODELS_EFFICIENTNET_HPP
