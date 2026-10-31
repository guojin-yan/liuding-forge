// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/quantization_torch/static_linear.hpp>

#include <array>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace j = jyppx::liuding;
namespace q = j::quantization;
namespace qt = j::quantization_torch;

j::ArtifactDigest digest(std::span<const std::byte> bytes) {
  return {.hex = j::sha256_hex(bytes)};
}

// The caller has just created this directory and owns its exclusive use.
void save(const std::filesystem::path& path, std::span<const std::byte> bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  file.close();
  if (!file) throw std::runtime_error("Cannot completely save demo artifact");
}

std::vector<std::byte> load(const std::filesystem::path& path) {
  const auto size = std::filesystem::file_size(path);
  if (size == 0U || size > q::QuantizationLimits{}.max_artifact_bytes)
    throw std::runtime_error("Demo artifact exceeds bounded read size");
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::ifstream file(path, std::ios::binary);
  file.read(reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  if (!file || file.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("Cannot read exact complete demo artifact");
  return bytes;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr
        << "Usage: liuding-forge-example-libtorch-static-int8 NEW_DIRECTORY\n";
    return 1;
  }
  try {
    const auto capabilities = qt::quantized_cpu_capabilities();
    std::cout << "LibTorch: " << capabilities.libtorch_version
              << ", current engine: " << capabilities.current_engine << '\n';
    // The application fixes global engine configuration before concurrent work.
    if (capabilities.current_engine != "ONEDNN" &&
        capabilities.current_engine != "X86")
      throw std::runtime_error(
          "This demo requires an initialized ONEDNN or X86 engine");
    const std::filesystem::path directory(argv[1]);
    if (!std::filesystem::create_directory(directory))
      throw std::runtime_error("Use a new directory; demo refuses overwrite");
    j::ArtifactDigest tensor_digest, channel_digest, output_digest;
    {
      constexpr std::string_view manifest =
          "training-0:-4,3,2,-1\ntraining-1:1,-2,3,4\ntraining-2:4,2,-4,-2\n";
      const q::CalibrationLineage lineage{
          .dataset_version = "training-demo/v1",
          .dataset_digest = digest(std::as_bytes(std::span(manifest))),
          .preprocessing_fingerprint = "demo-fp32/v1",
          .seed = 7U};
      const std::array<float, 8U> weights{1.F, 0.F,   -1.F, 0.F,
                                          0.F, 0.25F, 0.F,  -0.25F};
      const std::array<float, 2U> bias{0.5F, -0.5F};
      const std::array<std::array<float, 4U>, 3U> training_samples{
          {{-4.F, 3.F, 2.F, -1.F},
           {1.F, -2.F, 3.F, 4.F},
           {4.F, 2.F, -4.F, -2.F}}};
      q::CalibrationSession inputs(lineage), outputs(lineage);
      for (std::size_t i = 0U; i < training_samples.size(); ++i) {
        const auto& input = training_samples[i];
        // Original FP32 model outputs, with no evaluation labels or tuning.
        std::array<float, 2U> output{};
        for (std::size_t row = 0U; row < output.size(); ++row) {
          output[row] = bias[row];
          for (std::size_t column = 0U; column < input.size(); ++column)
            output[row] += input[column] * weights[row * input.size() + column];
        }
        const auto sample_id = "training-" + std::to_string(i);
        inputs.observe(sample_id, input);
        outputs.observe(sample_id, output);
      }
      const q::FloatLinearModel fp32{.in_features = 4U,
                                     .out_features = 2U,
                                     .weights = weights,
                                     .bias = bias};
      const auto input_calibration = inputs.finalize();
      const auto tensor_bytes = q::serialize_linear_artifact(
          q::quantize_linear(fp32, input_calibration));
      const auto channel_bytes =
          q::serialize_per_output_channel_linear_artifact(
              q::quantize_linear_per_output_channel(fp32, input_calibration));
      const auto output_bytes =
          q::serialize_calibration_result(outputs.finalize());
      // For these self-generated demo files, preserve identity before writing.
      // External files need independently trusted, preapproved digests.
      tensor_digest = digest(tensor_bytes);
      channel_digest = digest(channel_bytes);
      output_digest = digest(output_bytes);
      save(directory / "linear.qint8", tensor_bytes);
      save(directory / "linear.per-output-channel.qint8", channel_bytes);
      save(directory / "output.calibration", output_bytes);
    }  // All source/calibration objects and original serialized bytes are gone.

    const auto tensor_bytes = load(directory / "linear.qint8");
    const auto channel_bytes =
        load(directory / "linear.per-output-channel.qint8");
    const auto output_bytes = load(directory / "output.calibration");
    for (const bool per_channel : {false, true}) {
      auto runtime =
          per_channel
              ? qt::StaticLinear::load_per_output_channel(
                    channel_bytes, channel_digest, output_bytes, output_digest)
              : qt::StaticLinear::load_per_tensor(tensor_bytes, tensor_digest,
                                                  output_bytes, output_digest);
      const auto input = at::tensor({1.F, 2.F, 3.F, 4.F}).reshape({1, 4});
      const auto output =
          runtime.forward_quantized(runtime.quantize_input(input));
      const auto& info = runtime.info();
      std::cout << "Profile: " << info.profile_id
                << ", packed engine: " << info.engine
                << "\nSource SHA-256: " << info.source_artifact_digest.hex
                << "\nOutput calibration SHA-256: "
                << info.output_calibration_digest.hex
                << "\nOfficial quantized output: " << output.is_quantized()
                << ", scale: " << output.q_scale()
                << ", zero point: " << output.q_zero_point()
                << "\nFinal FP32 output:\n"
                << output.dequantize() << '\n';
      if (!output.is_quantized()) return 2;
    }
    // Kernel execution, task error, speed and RSS remain separate acceptances.
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
