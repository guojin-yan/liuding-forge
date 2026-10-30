// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");

#include <jyppx/liuding/quantization/quantization.hpp>

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void Require(bool condition, const char* operation) {
  if (!condition) throw std::runtime_error(operation);
}
}  // namespace

int main() {
  try {
    namespace q = jyppx::liuding::quantization;
    q::CalibrationSession session(
        {.dataset_version = "installed-consumer/v1",
         .dataset_digest = {.hex = std::string(64U, 'a')},
         .preprocessing_fingerprint = "identity/v1",
         .seed = 1U});
    const std::array<float, 2U> activation{1.0F, -1.0F};
    session.observe("sample-0", activation);
    const std::array<float, 2U> weights{1.0F, -1.0F};
    const std::array<float, 1U> bias{0.25F};
    const auto artifact = q::quantize_linear({.in_features = 2U,
                                              .out_features = 1U,
                                              .weights = weights,
                                              .bias = bias},
                                             session.finalize());
    const auto bytes = q::serialize_linear_artifact(artifact);
    q::StorageQuantizedLinearRuntime runtime(
        q::deserialize_linear_artifact(bytes));
    const auto output = runtime.forward(activation);
    Require(output.size() == 1U && std::abs(output.front() - 2.25F) < 1.0e-6F,
            "installed-quantization-fp32-linear");
    Require(!runtime.compute_quantized() && !runtime.qat_supported(),
            "installed-quantization-storage-only");
    const auto report = q::quantization_report_json(runtime.report(weights));
    Require(report.find("\"compute_quantized\":false") != std::string::npos,
            "installed-quantization-report-claims");
    const std::array<float, 4U> channel_weights{127.0F, -127.0F, 0.125F,
                                                -0.25F};
    const std::array<float, 2U> channel_bias{0.25F, -0.5F};
    const auto channels =
        q::quantize_linear_per_output_channel({.in_features = 2U,
                                               .out_features = 2U,
                                               .weights = channel_weights,
                                               .bias = channel_bias},
                                              session.finalize());
    const auto channel_bytes =
        q::serialize_per_output_channel_linear_artifact(channels);
    const auto restored_channels =
        q::deserialize_per_output_channel_linear_artifact(channel_bytes);
    Require(
        q::serialize_per_output_channel_linear_artifact(restored_channels) ==
                channel_bytes &&
            restored_channels.channels.size() == 2U &&
            restored_channels.channels[0].weight_params.scale == 1.0 &&
            restored_channels.channels[1].weight_params.scale == 0.25 / 127.0,
        "installed-per-output-channel-canonical-scales");
    std::cout << report << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
