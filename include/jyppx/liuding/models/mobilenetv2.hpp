// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_MODELS_MOBILENETV2_HPP
#define JYPPX_LIUDING_MODELS_MOBILENETV2_HPP
/** @file mobilenetv2.hpp
 * @brief Fixed timm MobileNetV2 1.0 classifier and multiscale features.
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
class MobileNetV2Blocks;
/** @brief Fixed 1.0-width backbone with configurable input and classifier. */
struct MobileNetV2Options {
  std::int64_t num_classes = 1000;  ///< Positive classifier width.
  std::int64_t input_channels = 3;  ///< Positive image channel count.
};

/** @brief MobileNetV2 with 1/2/3/4/3/3/1 depthwise and inverted blocks. */
class MobileNetV2Impl final : public torch::nn::Module {
 public:
  /** @brief Constructs a randomly initialized classifier without downloading.
   * @param options Input channels and classifier width.
   */
  explicit MobileNetV2Impl(MobileNetV2Options options = {});
  /** @brief Maps normalized NCHW images to unnormalized class logits.
   * @param input Floating image on the model's exact device and dtype.
   * @return Logits with shape [N,num_classes].
   */
  Tensor forward(const Tensor& input);
  /** @brief Returns timm feature checkpoints at strides 2,4,8,16,32.
   * @param input Same image contract as forward.
   * @return NCHW maps with channels 16,24,32,96,320.
   */
  std::vector<Tensor> forward_features(const Tensor& input);
  /** @brief Returns immutable constructor options. */
  const MobileNetV2Options& options() const noexcept;

 private:
  MobileNetV2Options options_;
  torch::nn::Conv2d conv_stem_{nullptr};
  torch::nn::BatchNorm2d bn1_{nullptr};
  std::shared_ptr<MobileNetV2Blocks> blocks_;
  torch::nn::Conv2d conv_head_{nullptr};
  torch::nn::BatchNorm2d bn2_{nullptr};
  torch::nn::Linear classifier_{nullptr};
};
TORCH_MODULE(MobileNetV2);

/** @brief Constructs the fixed timm mobilenetv2_100 architecture.
 * @param options Input channels and classifier width.
 * @return Owning randomly initialized holder.
 */
MobileNetV2 mobilenetv2_100(MobileNetV2Options options = {});

/** @brief Creates an owning NCHW/logit schema-checked adapter.
 * @param model Nonempty holder retained by the adapter.
 * @return Adapter over the same registered module and state.
 */
nn::ModelAdapter adapt(MobileNetV2 model);
}  // namespace jyppx::liuding::models
#endif  // JYPPX_LIUDING_MODELS_MOBILENETV2_HPP
