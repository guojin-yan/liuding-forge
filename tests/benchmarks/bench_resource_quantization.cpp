// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0.
#include "resource_support.hpp"

#include <jyppx/liuding/quantization/quantization.hpp>

#include <memory>
#include <numeric>
#include <vector>

int main(int argc, char** argv) {
  try {
    using namespace jyppx::liuding::benchmarks;
    namespace q = jyppx::liuding::quantization;
    const auto args = arguments(argc, argv);
    require(args.profile == "quantization" && args.device == "cpu",
            "quantization profile supports CPU FP32 fallback only");
    constexpr std::size_t inputs = 64, outputs = 32, batch = 16, steps = 512,
                          codec_rounds = 128;
    auto sample = report(args,
                         {{"in_features", inputs},
                          {"out_features", outputs},
                          {"batch", batch},
                          {"forwards", steps},
                          {"codec_rounds", codec_rounds},
                          {"weights", "((i*13)%3)-1"},
                          {"input", "(((i*7)%17)-8)/8"},
                          {"bias", "(o%7)/8"},
                          {"warmup_forwards", 1},
                          {"calibration_samples", 2}},
                         steps);
    std::vector<float> weights(inputs * outputs), bias(outputs),
        data(batch * inputs);
    for (std::size_t i = 0; i < weights.size(); ++i)
      weights[i] = static_cast<int>((i * 13) % 3) - 1;
    for (std::size_t i = 0; i < bias.size(); ++i)
      bias[i] = static_cast<float>(i % 7) / 8;
    for (std::size_t i = 0; i < data.size(); ++i)
      data[i] = static_cast<float>(static_cast<int>((i * 7) % 17) - 8) / 8;
    std::vector<float> expected(batch * outputs);
    for (std::size_t row = 0; row < batch; ++row)
      for (std::size_t out = 0; out < outputs; ++out) {
        double value = bias[out];
        for (std::size_t in = 0; in < inputs; ++in)
          value += static_cast<double>(data[row * inputs + in]) *
                   weights[out * inputs + in];
        expected[row * outputs + out] = static_cast<float>(value);
      }
    q::QuantizedLinearArtifact artifact;
    auto& metrics = sample["metrics"];
    metrics["calibration_conversion_seconds"] =
        metric(seconds([&] {
                 const auto data_bytes = std::as_bytes(std::span(data));
                 q::CalibrationSession calibration(
                     {.dataset_version = "resource-linear/v1",
                      .dataset_digest = {.hex = jyppx::liuding::sha256_hex(data_bytes)},
                      .preprocessing_fingerprint = "fixed-fp32-divide8/v1",
                      .seed = 20261021});
                 calibration.observe("sample-1",
                                     std::span(data).first(data.size() / 2));
                 calibration.observe("sample-2",
                                     std::span(data).subspan(data.size() / 2));
                 artifact = q::quantize_linear({.in_features = inputs,
                                                .out_features = outputs,
                                                .weights = weights,
                                                .bias = bias},
                                               calibration.finalize());
               }),
               "seconds", "lower", false);
    std::vector<std::byte> encoded;
    metrics["serialization_seconds"] =
        metric(seconds([&] {
                 for (std::size_t i = 0; i < codec_rounds; ++i)
                   encoded = q::serialize_linear_artifact(artifact);
               }) / codec_rounds,
               "seconds");
    std::ofstream file(args.directory / "linear.qint8", std::ios::binary);
    file.write(reinterpret_cast<const char*>(encoded.data()),
               static_cast<std::streamsize>(encoded.size()));
    file.close();
    require(file.good(), "quantization artifact write failed");
    std::unique_ptr<q::StorageQuantizedLinearRuntime> runtime;
    metrics["load_seconds"] =
        metric(seconds([&] {
                 auto decoded = q::deserialize_linear_artifact(encoded);
                 require(q::serialize_linear_artifact(decoded) == encoded,
                         "artifact canonical roundtrip mismatch");
                 runtime = std::make_unique<q::StorageQuantizedLinearRuntime>(
                     std::move(decoded));
               }),
               "seconds", "lower", false,
               "canonical decode, verification and FP32 weight reconstruction");
    std::vector<float> result;
    metrics["warmup_seconds"] =
        metric(seconds([&] { result = runtime->forward(data, batch); }),
               "seconds", "lower", false);
    require(result == expected, "quantization known-vector output mismatch");
    std::size_t completed = 0;
    const auto elapsed = seconds([&] {
      for (std::size_t step = 0; step < steps; ++step) {
        result = runtime->forward(data, batch);
        require(result == expected, "FP32 fallback changed exact known vector");
        ++completed;
      }
    });
    for (const auto value : result)
      require(std::isfinite(value), "non-finite FP32 fallback output");
    const auto claims = runtime->report(weights);
    require(!runtime->compute_quantized() && !runtime->qat_supported() &&
                claims.execution_claim == "storage-quantized" &&
                !claims.compute_quantized && !claims.qat &&
                claims.fallback_modules == 1,
            "unsupported quantized compute claim");
    metrics["forward_seconds"] =
        metric(elapsed / steps, "seconds", "lower", true,
               "FP32 fallback batch plus exact known-vector comparison; no "
               "speedup comparison");
    metrics["throughput_items_per_second"] =
        metric(batch * steps / elapsed, "items/second", "higher");
    metrics["output_bytes"] = metric(encoded.size(), "bytes");
    sample["execution"]["path"] = std::string(runtime->execution_path());
    sample["execution"]["compute_quantized"] = false;
    sample["execution"]["qat"] = false;
    sample["execution"]["fallback_modules"] = claims.fallback_modules;
    sample["validation"]["artifact_digest"] =
        "sha256:" + jyppx::liuding::sha256_hex(encoded);
    sample["validation"]["output_sum"] =
        std::accumulate(result.begin(), result.end(), 0.0);
    sample["validation"]["output_kind"] = "finite exact FP32 known vector";
    finish(sample, completed);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "quantization resource benchmark failed: " << error.what()
              << '\n';
    return 1;
  }
}
