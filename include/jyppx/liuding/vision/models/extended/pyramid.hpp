// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_MODELS_EXTENDED_PYRAMID_HPP
#define JYPPX_LIUDING_VISION_MODELS_EXTENDED_PYRAMID_HPP
/** @file pyramid.hpp @brief Owning feature bridges and five-level necks. */
#include <jyppx/liuding/models/registry.hpp>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/conv.h>
namespace jyppx::liuding::models::extended {
/** @brief A light token pyramid has its own identity, distinct from ViT-Adapter. */
enum class PyramidKind { fpn, bifpn, token_pyramid };
struct PyramidOptions {
  std::string backbone="resnet50";
  PyramidKind kind=PyramidKind::fpn;
  std::int64_t channels=256;
  std::int64_t repeats=3;
  bool softmax_fusion=false;  ///< Independent normalized softmax variant.
  bool extra_from_p5=true;  ///< FCOS uses P5, RetinaNet v1 uses C5.
};
/** @brief Owns one registry backbone; returns P3..P7 at strides 8..128.
 * NTC conversion uses explicit input H/W, declared stride and prefix count.
 * BiFPN uses nonnegative normalized two/three-input fusion, epsilon 1e-4,
 * separable convolution, BN, SiLU and repeated top-down/bottom-up topology.
 */
class FeaturePyramidImpl final:public torch::nn::Module {
 public:
  explicit FeaturePyramidImpl(PyramidOptions options={});
  ~FeaturePyramidImpl() override;
  std::vector<Tensor> forward(const Tensor& images);
  const PyramidOptions& options()const noexcept;
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
TORCH_MODULE(FeaturePyramid);
/** @brief Converts declared row-major patch tokens; no square-grid inference. */
Tensor token_grid(const Tensor& tokens,std::array<std::int64_t,2> grid,
                   std::int64_t prefix_tokens=0);
}
#endif
