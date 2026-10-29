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

/** @brief Fixed Tiny and Small backbone architectures. */
enum class ConvNeXtVariant : std::uint8_t {
  tiny = 1,           ///< Stage depths 3, 3, 9, 3.
  small_capacity = 2  ///< Stage depths 3, 3, 27, 3.
};

/** @brief Construction settings; weights are randomly initialized. */
struct ConvNeXtOptions {
  std::int64_t num_classes = 1000;  ///< Classes in [0, 1000000]; zero embeds.
  std::int64_t input_channels = 3;  ///< Input channels in [1, 1024].
  double layer_scale_init =
      1e-6;  ///< Finite LayerScale initial value in (0, 1].
};

class ConvNeXtHead;

/**
 * @brief ConvNeXt backbone with classification or pooled-feature output.
 * @note Inputs are NCHW floating tensors on the parameter device and dtype,
 * with spatial dimensions at least 32. Callers synchronize concurrent state
 * access, mode changes, head replacement and training.
 */
class ConvNeXtImpl final : public torch::nn::Module {
 public:
  /**
   * @brief Constructs and randomly initializes the selected capacity.
   * @param variant Tiny or Small architecture.
   * @param options Channels, classes and LayerScale configuration.
   * @throws InvalidArgument Unsupported capacity or settings.
   * @throws ResourceExhausted Insufficient model allocation memory.
   * @throws BackendError Underlying LibTorch construction failure.
   */
  explicit ConvNeXtImpl(ConvNeXtVariant variant, ConvNeXtOptions options = {});

  /**
   * @brief Returns four NCHW stage features with normal autograd.
   * @param input Image batch satisfying this class's input contract.
   * @return Channels 96, 192, 384, 768 at reductions 4, 8, 16, 32.
   * @throws InvalidArgument Input shape, device or dtype mismatch.
   * @throws ResourceExhausted Insufficient forward computation memory.
   * @throws BackendError Underlying LibTorch computation failure.
   */
  std::vector<Tensor> forward_features(const Tensor &input);
  /**
   * @brief Returns classification logits without softmax or pooled features.
   * @param input Image batch satisfying this class's input contract.
   * @return [N, num_classes], or [N, 768] when num_classes is zero.
   * @throws InvalidArgument Input shape, device or dtype mismatch.
   * @throws ResourceExhausted Insufficient forward computation memory.
   * @throws BackendError Underlying LibTorch computation failure.
   */
  Tensor forward(const Tensor &input);
  /**
   * @brief Replaces the Linear head; zero keeps only pooling and normalization.
   * @param num_classes Classes in [0, 1000000].
   * @note New weights are random and retain device, dtype and mode. Existing
   * optimizers do not acquire new parameters automatically. Use the unified
   * reset_classifier API to refresh an existing adapter's schema.
   * @throws InvalidArgument Class count outside the supported range.
   * @throws ResourceExhausted Insufficient new-head allocation memory.
   * @throws BackendError Underlying LibTorch replacement failure.
   */
  void reset_classifier(std::int64_t num_classes);
  /**
   * @brief Queries settings, including the current classifier width.
   * @return Borrowed reference valid no longer than this model.
   */
  const ConvNeXtOptions &options() const noexcept;
  /**
   * @brief Queries the fixed architecture.
   * @return Capacity selected during construction.
   */
  ConvNeXtVariant variant() const noexcept;

 private:
  ConvNeXtOptions options_;                ///< Current model settings.
  ConvNeXtVariant variant_;                ///< Fixed capacity.
  torch::nn::Sequential stem_{nullptr};    ///< Initial patch stem.
  torch::nn::ModuleList stages_{nullptr};  ///< Four backbone stages.
  std::shared_ptr<ConvNeXtHead> head_;     ///< Pooling, normalization and head.
};

/** @brief Shared owning LibTorch holder for a ConvNeXt module. */
TORCH_MODULE(ConvNeXt);

/**
 * @brief Creates a randomly initialized ConvNeXt Tiny.
 * @param options Construction settings with the ConvNeXtImpl constraints.
 * @return LibTorch holder sharing ownership of the model.
 */
ConvNeXt convnext_tiny(ConvNeXtOptions options = {});
/**
 * @brief Creates a randomly initialized ConvNeXt Small.
 * @param options Construction settings with the ConvNeXtImpl constraints.
 * @return LibTorch holder sharing ownership of the model.
 */
ConvNeXt convnext_small(ConvNeXtOptions options = {});
/**
 * @brief Adapts a nonempty model to one Tensor input and output.
 * @param model Nonempty holder; the adapter shares its module and parameters.
 * @return Model entry with input-channel and output-width constraints.
 * @throws InvalidArgument The supplied model is empty.
 */
nn::ModelAdapter adapt(ConvNeXt model);

}  // namespace jyppx::liuding::models

#endif  // JYPPX_LIUDING_MODELS_CONVNEXT_HPP
