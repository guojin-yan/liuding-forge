// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include <jyppx/liuding/deployment/onnx_export/static_int8_linear.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/quantization/quantization.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>

int main() {
  try {
    namespace q = jyppx::liuding::quantization;
    namespace ex = jyppx::liuding::onnx_export;
    namespace rt = jyppx::liuding::onnx_runtime;

    // 1. Freeze activation calibration and its exact data/preprocessing
    // identity.
    const std::array<float, 3U> calibration_values{-127.0F, 0.0F, 127.0F};
    q::CalibrationSession calibration(
        {.dataset_version = "teaching-linear-three-values/v1",
         .dataset_digest = {.hex = jyppx::liuding::sha256_hex(
                                std::as_bytes(std::span(calibration_values)))},
         .preprocessing_fingerprint = "identity-fp32-three-features/v1",
         .seed = 20261030U});
    calibration.observe("fixture/0", calibration_values);

    // 2. Each output row has a different magnitude. A shared weight scale
    // would round the entire second row to zero. Channel scales retain it.
    const std::array<float, 6U> weights{127, -127, 1, 0.125F, -0.25F, 0.375F};
    const std::array<float, 2U> bias{0.25F, -0.5F};
    const auto model =
        q::quantize_linear_per_output_channel({.in_features = 3U,
                                               .out_features = 2U,
                                               .weights = weights,
                                               .bias = bias},
                                              calibration.finalize());
    const auto bytes = q::serialize_per_output_channel_linear_artifact(model);
    const jyppx::liuding::ArtifactDigest source_digest{
        .hex = jyppx::liuding::sha256_hex(bytes)};
    const auto restored =
        q::deserialize_per_output_channel_linear_artifact(bytes);
    if (q::serialize_per_output_channel_linear_artifact(restored) != bytes)
      throw std::runtime_error("Canonical aggregate roundtrip changed bytes");

    // 3. C++ export creates one U8U8 MatMulInteger plus per-channel scaling.
    // No Python exporter or FP32 dot-product fallback participates.
    const std::array<std::int64_t, 2U> shape{2, 3};
    const auto exported = ex::export_static_int8_linear_per_output_channel(
        bytes, source_digest, shape);
    rt::SessionOptions options;
    options.provider.name = "CPUExecutionProvider";
    options.allow_cpu_fallback = false;
    options.cpu_threads = 1U;
    options.disable_graph_optimizations = true;
    rt::Session runtime(exported.artifact.onnx, exported.artifact.digest,
                        options);
    const std::array<float, 6U> input{1, 2, 3, -1, 0, 1};
    const auto result = runtime.run(input, shape);

    // 4. Independent integer golden values use qweights [127,-127,1] and
    // [42,-85,127], activation scale 1 and per-row reconstruction scales.
    const std::array<float, 4U> expected{
        -124.0F + 0.25F, 253.0F * static_cast<float>(0.375 / 127.0) - 0.5F,
        -126.0F + 0.25F, 85.0F * static_cast<float>(0.375 / 127.0) - 0.5F};
    if (result.values.size() != expected.size())
      throw std::runtime_error("Incorrect INT8 result shape");
    for (std::size_t i = 0U; i < expected.size(); ++i)
      if (std::abs(result.values[i] - expected[i]) > 1.0e-6F)
        throw std::runtime_error(
            "CPU output disagrees with integer golden values");
    std::cout << "profile: " << exported.profile_id
              << "\nsource_sha256: " << source_digest.hex
              << "\nonnx_sha256: " << exported.artifact.digest.hex
              << "\nruntime: " << runtime.runtime_version()
              << "\nprovider: " << runtime.execution_provider().name
              << "\nweight_scales: " << restored.channels[0].weight_params.scale
              << ", " << restored.channels[1].weight_params.scale
              << "\ninteger oracle: passed\n"
                 "Teaching fixture only; qualify kernel trace, task accuracy, "
                 "latency and memory separately.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Per-output-channel INT8 example failed: " << error.what()
              << '\n';
    return 1;
  }
}
