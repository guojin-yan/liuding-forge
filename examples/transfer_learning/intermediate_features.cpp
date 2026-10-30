// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/models/models.hpp>

#include <torch/torch.h>

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  try {
    namespace models = jyppx::liuding::models;
    const std::string name = argc == 1 ? "vit_small_patch14_dinov2" : argv[1];
    if (argc > 2 ||
        (name != "vit_small_patch14_dinov2" && name != "dinov3_vits16"))
      throw std::invalid_argument(
          "Use vit_small_patch14_dinov2 or dinov3_vits16");
    torch::set_num_threads(2);
    torch::manual_seed(267);
    auto model = models::create_model(name);
    model.state().eval();
    // Both adapters own the same complete model. Selection does not freeze
    // parameters or change the module's training/evaluation mode.
    const models::DinoIntermediateOptions selection{
        .block_indices = {8, 2}, .out_indices = {0, 2}, .normalize = true};
    const auto spec = models::model_intermediate_feature_spec(model, selection);
    auto extractor =
        models::create_intermediate_feature_extractor(model, selection);
    const auto v2 = name == "vit_small_patch14_dinov2";
    // Deterministic synthetic pixels and random weights teach the API only.
    const auto images =
        torch::linspace(-1.0, 1.0, 3 * (v2 ? 28 : 35) * (v2 ? 42 : 50))
            .view({1, 3, v2 ? 28 : 35, v2 ? 42 : 50});
    torch::NoGradGuard no_grad;
    const auto values =
        extractor.invoke({.positional = {images}}).value.toTuple();
    if (values->elements().size() != spec.outputs.size() ||
        extractor.state().shared_module() != model.state().shared_module())
      throw std::runtime_error("Intermediate output/state contract mismatch");
    std::cout
        << "{\"scope\":\"synthetic_api_demonstration\",\"weights\":\"random\","
           "\"device\":\"cpu\",\"model\":\""
        << name << "\",\"shared_state\":true,\"normalized\":true,\"outputs\":[";
    for (std::size_t index = 0; index < spec.outputs.size(); ++index) {
      const auto tensor = values->elements()[index].toTensor();
      if (!torch::isfinite(tensor).all().item<bool>())
        throw std::runtime_error("Nonfinite feature");
      if (index) std::cout << ',';
      std::cout << "{\"name\":\"" << spec.outputs[index].name
                << "\",\"shape\":[";
      for (std::int64_t axis = 0; axis < tensor.dim(); ++axis) {
        if (axis) std::cout << ',';
        std::cout << tensor.size(axis);
      }
      std::cout << "]}";
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
