// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#ifndef JYPPX_LIUDING_MODELS_SWIN_HPP
#define JYPPX_LIUDING_MODELS_SWIN_HPP

/** @file swin.hpp
 * @brief Trainable timm-aligned Swin-T patch4/window7 image classifier.
 */

#include <jyppx/liuding/nn/model_adapter.hpp>

#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/pimpl.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace jyppx::liuding::models {

/** @brief Construction settings; pretrained weights are loaded separately. */
struct SwinOptions {
  std::int64_t num_classes = 1000;  ///< Classes in [0, 1000000]; zero embeds.
  std::int64_t input_channels = 3;  ///< Input channels in [1, 1024].
  bool dynamic_image_size = false;  ///< Accept patch-aligned images.
};

class SwinPatchEmbed;
class SwinHead;

/**
 * @brief Swin Tiny with four shifted-window stages and an optional classifier.
 * @note Inputs are NCHW floating tensors on the parameter device and dtype.
 * Default H/W is exactly 224; dynamic mode accepts positive multiples of 4.
 * Callers synchronize training, mode/head changes and concurrent state access.
 */
class SwinTinyImpl final : public torch::nn::Module {
 public:
  /** @brief Constructs randomly initialized modules with depths 2, 2, 6, 2.
   * @param options Input channels, classifier width and spatial-size policy.
   * @throws InvalidArgument Unsupported settings.
   * @throws ResourceExhausted Insufficient model allocation memory.
   * @throws BackendError Underlying LibTorch construction failure.
   */
  explicit SwinTinyImpl(SwinOptions options = {});

  /** @brief Computes four NCHW stage features with normal autograd.
   * @param input Image batch satisfying this class's input contract.
   * @return Channels 96, 192, 384, 768 at nominal reductions 4, 8, 16, 32;
   * odd patch grids are padded before each merging reduction.
   * @throws InvalidArgument Input shape, device or dtype mismatch.
   * @throws ResourceExhausted Insufficient forward computation memory.
   * @throws BackendError Underlying LibTorch computation failure.
   */
  std::vector<Tensor> forward_features(const Tensor& input);
  /** @brief Computes logits without softmax or the normalized pooled feature.
   * @param input Image batch satisfying the forward_features() contract.
   * @return [N, num_classes], or [N, 768] when num_classes is zero.
   * @throws InvalidArgument Input shape, device or dtype mismatch.
   * @throws ResourceExhausted Insufficient forward computation memory.
   * @throws BackendError Underlying LibTorch computation failure.
   */
  Tensor forward(const Tensor& input);
  /** @brief Replaces the Linear classifier; zero keeps only feature pooling.
   * @param num_classes Classes in [0, 1000000].
   * @note New weights are random and retain device, dtype and mode. Rebuild
   * optimizers for the new parameters; use the unified reset_classifier API
   * to refresh an existing adapter's output schema.
   * @throws InvalidArgument Class count outside the supported range.
   * @throws ResourceExhausted Insufficient new-head allocation memory.
   * @throws BackendError Underlying LibTorch replacement failure.
   */
  void reset_classifier(std::int64_t num_classes);
  /** @brief Queries settings, including the current classifier width.
   * @return Borrowed reference valid no longer than this model.
   */
  const SwinOptions& options() const noexcept;

 private:
  SwinOptions options_;  ///< Current input, head and spatial-size settings.
  std::shared_ptr<SwinPatchEmbed> patch_embed_;  ///< Registered patch stem.
  torch::nn::ModuleList layers_{nullptr};        ///< Four hierarchical stages.
  torch::nn::LayerNorm norm_{nullptr};  ///< Final channel normalization.
  std::shared_ptr<SwinHead> head_;      ///< Registered pooling and classifier.
};

/** @brief Shared owning LibTorch holder for a Swin Tiny module. */
TORCH_MODULE(SwinTiny);

/** @brief Creates a randomly initialized patch4/window7 Swin Tiny.
 * @param options Construction settings with the SwinTinyImpl constraints.
 * @return Shared owning holder; no pretrained weights or preprocessing.
 */
SwinTiny swin_tiny_patch4_window7_224(SwinOptions options = {});
/** @brief Creates a single-Tensor adapter sharing the supplied module.
 * @param model Nonempty holder; parameters remain shared with callers.
 * @return Model entry with input-channel and output-width constraints.
 * @throws InvalidArgument The supplied holder is empty.
 */
nn::ModelAdapter adapt(SwinTiny model);

}  // namespace jyppx::liuding::models

#endif  // JYPPX_LIUDING_MODELS_SWIN_HPP
