// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_EXAMPLE_DETECTION_MODEL_DETECTOR_HPP
#define JYPPX_LIUDING_EXAMPLE_DETECTION_MODEL_DETECTOR_HPP

#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/conv.h>

namespace jyppx::liuding::examples::detection {

// Direct RGB-to-box example model. Targets and instance masks never enter
// forward(); they belong to the training/evaluation driver.
class Detector final : public torch::nn::Module {
 public:
  Detector();
  torch::Tensor forward(const torch::Tensor& images);

 private:
  torch::nn::Sequential body{nullptr};
  torch::nn::Conv2d head{nullptr};
};

}  // namespace jyppx::liuding::examples::detection

#endif  // JYPPX_LIUDING_EXAMPLE_DETECTION_MODEL_DETECTOR_HPP
