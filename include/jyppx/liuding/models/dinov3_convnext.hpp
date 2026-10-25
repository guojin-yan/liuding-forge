// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#ifndef JYPPX_LIUDING_MODELS_DINOV3_CONVNEXT_HPP
#define JYPPX_LIUDING_MODELS_DINOV3_CONVNEXT_HPP
/** @file dinov3_convnext.hpp
 * @brief Independent DINOv3 ConvNeXt-T/S backbone contracts.
 */

#include <jyppx/liuding/nn/model_adapter.hpp>

#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/pimpl.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace jyppx::liuding::models {

/** @brief Low-resource DINOv3 ConvNeXt capacities selected for v1.0. */
enum class DinoV3ConvNeXtVariant : std::uint8_t {
  tiny = 1,   ///< depths [3,3,9,3], channels [96,192,384,768].
  small = 2,  ///< depths [3,3,27,3], channels [96,192,384,768].
  base = 3,   ///< depths [3,3,27,3], channels [128,256,512,1024].
  large = 4,  ///< depths [3,3,27,3], channels [192,384,768,1536].
};

/** @brief Input, head and initialization settings for ConvNeXt backbones. */
struct DinoV3ConvNeXtOptions {
  std::int64_t num_classes = 0;
  ///< Zero returns the global pooled feature; positive values add a head.
  std::int64_t input_channels = 3;  ///< Positive input channel count.
  double layer_scale_init = 1e-6;   ///< Positive LayerScale initialization.
};

/** @brief DINOv3 ConvNeXt encoder exposing four spatial stages. */
class DinoV3ConvNeXtImpl final : public torch::nn::Module {
 public:
  explicit DinoV3ConvNeXtImpl(DinoV3ConvNeXtVariant variant,
                              DinoV3ConvNeXtOptions options = {});

  /**
   * @brief Returns four NCHW stage tensors after explicit floor cropping.
   * @return Stage channels follow the selected variant and total stride is
   * 4, 8, 16 and 32 respectively.
   * @note H and W are cropped to their largest positive multiples of 32.
   */
  std::vector<Tensor> forward_features(const Tensor& input);

  /** @brief Returns global average pooled stage-four features or logits. */
  Tensor forward(const Tensor& input);

  const DinoV3ConvNeXtOptions& options() const noexcept;
  DinoV3ConvNeXtVariant variant() const noexcept;
  void reset_classifier(std::int64_t num_classes);

 private:
  DinoV3ConvNeXtOptions options_;
  DinoV3ConvNeXtVariant variant_;
  std::vector<std::int64_t> channels_;
  torch::nn::Conv2d stem_{nullptr};
  torch::nn::LayerNorm stem_norm_{nullptr};
  torch::nn::ModuleList downsample_;
  torch::nn::ModuleList stages_;
  torch::nn::LayerNorm final_norm_{nullptr};
  torch::nn::Linear head_{nullptr};
};

/** @brief Shared owning holder for a DINOv3 ConvNeXt encoder. */
TORCH_MODULE(DinoV3ConvNeXt);

DinoV3ConvNeXt dinov3_convnext_tiny(DinoV3ConvNeXtOptions options = {});
DinoV3ConvNeXt dinov3_convnext_small(DinoV3ConvNeXtOptions options = {});
DinoV3ConvNeXt dinov3_convnext_base(DinoV3ConvNeXtOptions options = {});
DinoV3ConvNeXt dinov3_convnext_large(DinoV3ConvNeXtOptions options = {});

/** @brief Creates a schema-checked adapter for a ConvNeXt holder. */
nn::ModelAdapter adapt(DinoV3ConvNeXt model);

}  // namespace jyppx::liuding::models

#endif  // JYPPX_LIUDING_MODELS_DINOV3_CONVNEXT_HPP
