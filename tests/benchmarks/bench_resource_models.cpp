// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0.
#include "resource_torch_support.hpp"

#include <jyppx/liuding/models/registry.hpp>

#include <algorithm>
#include <array>
#include <optional>
#include <vector>

int main(int argc, char** argv) {
  try {
    using namespace jyppx::liuding::benchmarks;
    const auto args = arguments(argc, argv);
    constexpr std::array<std::string_view, 5> names{
        "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
        "transformer_encoder"};
    require(std::find(names.begin(), names.end(), args.profile) != names.end(),
            "unknown model profile");
    const bool transformer = args.profile == "transformer_encoder";
    const std::size_t steps = transformer ? 256 : 32;
    const std::vector<std::int64_t> shape =
        transformer ? std::vector<std::int64_t>{2, 16, 32}
                    : std::vector<std::int64_t>{2, 3, 32, 32};
    auto sample = report(args,
                         {{"model", args.profile},
                          {"input_shape", shape},
                          {"seed", 20261021},
                          {"input", "arange%31 /31 -0.5 FP32"},
                          {"classes", 7},
                          {"warmup_forwards", 1},
                          {"forwards", steps},
                          {"transformer",
                           {{"dimension", 32},
                            {"heads", 4},
                            {"feedforward", 64},
                            {"layers", 2},
                            {"dropout", 0.0},
                            {"gelu", true},
                            {"final_norm", true}}}},
                         steps);
    const auto device = configure_device(args, sample);
    torch::NoGradGuard guard;
    jyppx::liuding::models::ModelOptions options;
    options.resnet.num_classes = 7;
    options.mobile.num_classes = 7;
    options.mobile.dropout = 0;
    options.transformer = {.model_dimension = 32,
                           .attention_heads = 4,
                           .feedforward_dimension = 64,
                           .layers = 2,
                           .dropout = 0,
                           .gelu = true,
                           .final_norm = true};
    std::optional<jyppx::liuding::nn::ModelAdapter> model;
    auto& metrics = sample["metrics"];
    metrics["load_seconds"] =
        metric(device_seconds(device,
                              [&] {
                                model.emplace(jyppx::liuding::models::create_model(
                                    args.profile, options));
                                model->state().eval();
                                model->state().to(device);
                              }),
               "seconds", "lower", false,
               "architecture construction, random initialization and device "
               "transfer; no pretrained load");
    std::int64_t input_elements = 1;
    for (const auto dimension : shape) input_elements *= dimension;
    const auto input =
        (torch::arange(input_elements, torch::kFloat32).remainder(31) / 31.0 -
         0.5)
            .reshape(shape)
            .to(device);
    const auto forward = [&] {
      return model->invoke({.positional = {input}, .keyword = {}})
          .value.toTensor();
    };
    torch::Tensor warmed;
    metrics["warmup_seconds"] =
        metric(device_seconds(device, [&] { warmed = forward(); }), "seconds",
               "lower", false);
    const auto output_elements = transformer ? 2 * 16 * 32 : 2 * 7;
    verify_tensor(warmed, device, output_elements);
    const std::vector<std::int64_t> expected_shape =
        transformer ? shape : std::vector<std::int64_t>{2, 7};
    require(warmed.sizes().vec() == expected_shape,
            "model output shape mismatch");
    std::vector<torch::Tensor> outputs;
    outputs.reserve(steps);
    const auto forward_time = device_seconds(device, [&] {
      for (std::size_t step = 0; step < steps; ++step)
        outputs.push_back(forward());
    });
    for (const auto& output : outputs) {
      verify_tensor(output, device, output_elements);
      require(torch::allclose(output, warmed, 1e-5, 1e-6),
              "eval output changed for identical inputs");
    }
    metrics["forward_seconds"] = metric(forward_time / steps, "seconds");
    metrics["throughput_items_per_second"] =
        metric(2 * steps / forward_time, "items/second", "higher");
    metrics["parameter_elements"] = metric(
        jyppx::liuding::models::parameter_count(model->state().module()), "elements");
    metrics["output_bytes"] = metric(output_elements * sizeof(float), "bytes");
    sample["execution"]["path"] =
        "ModelAdapter eval FP32; randomly initialized, no pretrained quality "
        "claim";
    sample["validation"]["output_sum"] = warmed.sum().item<double>();
    sample["validation"]["output_kind"] =
        "finite FP32 tensor with fixed shape and repeatability";
    cuda_metrics(device, sample);
    finish(sample, outputs.size());
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "models resource benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
