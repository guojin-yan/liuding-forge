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

/** @brief Fixed DINOv3 ConvNeXt backbone capacities. */
enum class DinoV3ConvNeXtVariant : std::uint8_t {
  tiny = 1,            ///< depths [3,3,9,3], channels [96,192,384,768].
  small_capacity = 2,  ///< depths [3,3,27,3], channels [96,192,384,768].
  base = 3,            ///< depths [3,3,27,3], channels [128,256,512,1024].
  large = 4,           ///< depths [3,3,27,3], channels [192,384,768,1536].
};

/** @brief Construction settings; pretrained weights are loaded separately. */
struct DinoV3ConvNeXtOptions {
  std::int64_t num_classes = 0;
  ///< Classes in [0, 1000000]; zero returns the normalized pooled feature.
  std::int64_t input_channels = 3;  ///< Input channels in [1, 1024].
  double layer_scale_init = 1e-6;   ///< Finite LayerScale value in (0, 1].
};

/**
 * @brief DINOv3 ConvNeXt encoder exposing four spatial stages.
 * @note Inputs are NCHW floating tensors on the parameter device and dtype,
 * with H/W at least 32. Callers synchronize concurrent state access, training,
 * mode changes and head replacement. No preprocessing or weight download is
 * performed by this module.
 */
class DinoV3ConvNeXtImpl final : public torch::nn::Module {
 public:
  /**
   * @brief Constructs and randomly initializes the selected capacity.
   * @param variant Tiny, Small, Base or Large architecture.
   * @param options Input channels, classifier width and LayerScale settings.
   * @throws InvalidArgument Unsupported capacity or settings.
   * @throws ResourceExhausted Insufficient model allocation memory.
   * @throws BackendError Underlying LibTorch construction failure.
   */
  explicit DinoV3ConvNeXtImpl(DinoV3ConvNeXtVariant variant,
                              DinoV3ConvNeXtOptions options = {});

  /**
   * @brief Returns four NCHW stage tensors after explicit floor cropping.
   * @param input Image batch satisfying this class's input contract.
   * @return Stage channels follow the selected variant and total stride is
   * 4, 8, 16 and 32 respectively.
   * @note H and W are cropped to their largest positive multiples of 32.
   * Outputs retain normal autograd on the input device.
   * @throws InvalidArgument Input shape, device or dtype mismatch.
   * @throws ResourceExhausted Insufficient forward computation memory.
   * @throws BackendError Underlying LibTorch computation failure.
   */
  std::vector<Tensor> forward_features(const Tensor& input);

  /**
   * @brief Returns normalized stage-four pooled features or classifier logits.
   * @param input Image batch satisfying the forward_features() contract.
   * @return [N, num_classes], or [N, final-stage channels] for a zero head.
   * @throws InvalidArgument Input shape, device or dtype mismatch.
   * @throws ResourceExhausted Insufficient forward computation memory.
   * @throws BackendError Underlying LibTorch computation failure.
   */
  Tensor forward(const Tensor& input);

  /** @brief Queries current construction and head settings.
   * @return Borrowed reference valid no longer than this model.
   */
  const DinoV3ConvNeXtOptions& options() const noexcept;
  /** @brief Queries the fixed architecture.
   * @return Capacity selected during construction.
   */
  DinoV3ConvNeXtVariant variant() const noexcept;
  /**
   * @brief Replaces the Linear head; zero selects normalized pooled features.
   * @param num_classes Classes in [0, 1000000].
   * @note New weights are random and retain device, dtype and mode. Rebuild
   * optimizers for the new parameters; use the unified reset_classifier API
   * to refresh an existing adapter's output schema.
   * @throws InvalidArgument Class count outside the supported range.
   * @throws ResourceExhausted Insufficient new-head allocation memory.
   * @throws BackendError Underlying LibTorch replacement failure.
   */
  void reset_classifier(std::int64_t num_classes);

 private:
  DinoV3ConvNeXtOptions
      options_;  ///< Current channel, head and scale settings.
  DinoV3ConvNeXtVariant variant_;       ///< Fixed backbone capacity.
  std::vector<std::int64_t> channels_;  ///< Four selected stage widths.
  torch::nn::Conv2d stem_{nullptr};     ///< Shared registered stem convolution.
  torch::nn::ModuleList downsample_layers_;  ///< Stem and three reductions.
  torch::nn::ModuleList stages_;  ///< Four registered residual stages.
  torch::nn::LayerNorm final_norm_{nullptr};  ///< Pooled-feature normalization.
  torch::nn::Linear head_{nullptr};  ///< Optional registered classifier.
};

/** @brief Shared owning holder for a DINOv3 ConvNeXt encoder. */
TORCH_MODULE(DinoV3ConvNeXt);

/** @brief Creates a randomly initialized DINOv3 ConvNeXt Tiny.
 * @param options Construction settings with DinoV3ConvNeXtImpl constraints.
 * @return Shared owning LibTorch holder; no pretrained weights are loaded.
 */
DinoV3ConvNeXt dinov3_convnext_tiny(DinoV3ConvNeXtOptions options = {});
/** @brief Creates a randomly initialized DINOv3 ConvNeXt Small.
 * @param options Construction settings with DinoV3ConvNeXtImpl constraints.
 * @return Shared owning LibTorch holder; no pretrained weights are loaded.
 */
DinoV3ConvNeXt dinov3_convnext_small(DinoV3ConvNeXtOptions options = {});
/** @brief Creates a randomly initialized DINOv3 ConvNeXt Base.
 * @param options Construction settings with DinoV3ConvNeXtImpl constraints.
 * @return Shared owning LibTorch holder; resource qualification is separate.
 */
DinoV3ConvNeXt dinov3_convnext_base(DinoV3ConvNeXtOptions options = {});
/** @brief Creates a randomly initialized DINOv3 ConvNeXt Large.
 * @param options Construction settings with DinoV3ConvNeXtImpl constraints.
 * @return Shared owning LibTorch holder; resource qualification is separate.
 */
DinoV3ConvNeXt dinov3_convnext_large(DinoV3ConvNeXtOptions options = {});

/** @brief Creates a schema-checked adapter sharing the supplied module.
 * @param model Nonempty holder whose parameters remain shared with callers.
 * @return Single-Tensor model entry with channel and output-width constraints.
 * @throws InvalidArgument The supplied holder is empty.
 */
nn::ModelAdapter adapt(DinoV3ConvNeXt model);

}  // namespace jyppx::liuding::models

#endif  // JYPPX_LIUDING_MODELS_DINOV3_CONVNEXT_HPP
