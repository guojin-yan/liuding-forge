// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/quantization_torch/static_linear.hpp>

#include <array>
#include <exception>
#include <iostream>
#include <string>

int main() {
  try {
    namespace j = jyppx::liuding;
    namespace q = j::quantization;
    q::CalibrationLineage lineage{
        .dataset_version = "installed-fixture/v1",
        .dataset_digest = {.hex = std::string(64U, 'a')},
        .preprocessing_fingerprint = "identity-fp32/v1",
        .seed = 1U};
    q::CalibrationSession inputs(lineage), outputs(lineage);
    const std::array<float, 2U> training{127.F, -127.F};
    const std::array<float, 1U> fp32_output{254.F};
    inputs.observe("training-0", training);
    outputs.observe("training-0", fp32_output);
    const std::array<float, 2U> weights{1.F, -1.F};
    const auto source = q::serialize_linear_artifact(q::quantize_linear(
        {.in_features = 2U, .out_features = 1U, .weights = weights, .bias = {}},
        inputs.finalize()));
    const auto calibration =
        q::serialize_calibration_result(outputs.finalize());
    auto runtime = j::quantization_torch::StaticLinear::load_per_tensor(
        source, {.hex = j::sha256_hex(source)}, calibration,
        {.hex = j::sha256_hex(calibration)});
    const auto result = runtime.forward_quantized(
        runtime.quantize_input(at::tensor({1.F, -1.F}).reshape({1, 2})));
    return result.is_quantized() && result.q_zero_point() == 128 &&
                   result.q_scale() == 2.0 &&
                   result.dequantize().item<float>() == 2.F
               ? 0
               : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
