// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#ifndef JYPPX_LIUDING_MODELS_CONVNEXT_HPP
#define JYPPX_LIUDING_MODELS_CONVNEXT_HPP

/** @file convnext.hpp
 * @brief Fixed timm ConvNeXt Tiny and Small classifier architectures.
 */

#include <jyppx/liuding/nn/model_adapter.hpp>

#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/pimpl.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace jyppx::liuding::models {

enum class ConvNeXtVariant : std::uint8_t { tiny = 1, small = 2 };

struct ConvNeXtOptions {
  std::int64_t num_classes = 1000;  ///< Zero returns the pooled embedding.
  std::int64_t input_channels = 3;
  double layer_scale_init = 1e-6;
};

class ConvNeXtHead;

class ConvNeXtImpl final : public torch::nn::Module {
 public:
  explicit ConvNeXtImpl(ConvNeXtVariant variant, ConvNeXtOptions options = {});

  /** @brief Four NCHW stages with reductions 4, 8, 16 and 32. */
  std::vector<Tensor> forward_features(const Tensor& input);
  /** @brief Unnormalized logits or the 768-wide pooled embedding. */
  Tensor forward(const Tensor& input);
  void reset_classifier(std::int64_t num_classes);
  const ConvNeXtOptions& options() const noexcept;
  ConvNeXtVariant variant() const noexcept;

 private:
  ConvNeXtOptions options_;
  ConvNeXtVariant variant_;
  torch::nn::Sequential stem_{nullptr};
  torch::nn::ModuleList stages_{nullptr};
  std::shared_ptr<ConvNeXtHead> head_;
};

TORCH_MODULE(ConvNeXt);

ConvNeXt convnext_tiny(ConvNeXtOptions options = {});
ConvNeXt convnext_small(ConvNeXtOptions options = {});
nn::ModelAdapter adapt(ConvNeXt model);

}  // namespace jyppx::liuding::models

#endif  // JYPPX_LIUDING_MODELS_CONVNEXT_HPP
