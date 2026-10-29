// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

/** @file checkpoint_onnx.cpp
 * @brief Restores native model state, exports a multi-input graph and runs ORT.
 */

#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/verification.hpp>
#include <jyppx/liuding/interop/file_state_archive.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>

#include <torch/torch.h>

#include <ATen/Parallel.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
namespace forge = jyppx::liuding;
namespace exporter = forge::onnx_export;
namespace runtime = forge::onnx_runtime;
namespace fs = std::filesystem;
using Tensors = std::vector<at::Tensor>;

constexpr std::array<float, 3> kGain{0.5F, -1.0F, 2.0F};
constexpr std::array<float, 3> kOffset{-0.25F, 0.5F, 1.0F};

// Application-owned forward code: no model registry or dedicated exporter.
class PairCalibration final : public torch::nn::Module {
 public:
  PairCalibration()
      : gain_(register_parameter("gain", at::zeros({3}, at::kFloat))),
        offset_(register_buffer("offset", at::zeros({3}, at::kFloat))) {}

  void initialize_demo_state() {
    const torch::NoGradGuard guard;
    gain_.copy_(at::tensor({kGain[0], kGain[1], kGain[2]}));
    offset_.copy_(at::tensor({kOffset[0], kOffset[1], kOffset[2]}));
  }

  Tensors forward(const at::Tensor& left, const at::Tensor& right) const {
    return {at::relu(left * gain_ + right + offset_), left - right};
  }

 private:
  at::Tensor gain_;
  at::Tensor offset_;
};

struct Sample {
  std::string id;
  std::array<std::int64_t, 2> shape;
  std::vector<float> left;
  std::vector<float> right;
  std::vector<float> calibrated;
  std::vector<float> difference;
};

Sample MakeSample(std::int64_t batch) {
  Sample sample;
  sample.id = "batch-" + std::to_string(batch);
  sample.shape = {batch, 3};
  for (std::int64_t index = 0; index < batch * 3; ++index) {
    const auto feature = static_cast<std::size_t>(index % 3);
    const auto left = static_cast<float>(index) / 4.0F - 1.0F;
    const auto right = static_cast<float>(index) / 8.0F + 0.125F;
    sample.left.push_back(left);
    sample.right.push_back(right);
    // The oracle uses scalar arithmetic, independently of LibTorch and ORT.
    sample.calibrated.push_back(
        std::max(0.0F, left * kGain[feature] + right + kOffset[feature]));
    sample.difference.push_back(left - right);
  }
  return sample;
}

Tensors BorrowTensors(Sample& sample) {
  const auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  // from_blob borrows the stable sample buffers; forward must not mutate them.
  return {at::from_blob(sample.left.data(), sample.shape, options),
          at::from_blob(sample.right.data(), sample.shape, options)};
}

void CheckNativeOutputs(PairCalibration& model, Sample& sample) {
  const auto inputs = BorrowTensors(sample);
  const auto outputs = model.forward(inputs[0], inputs[1]);
  const std::array<std::span<const float>, 2> expected{sample.calibrated,
                                                       sample.difference};
  if (outputs.size() != expected.size())
    throw std::runtime_error("Restored model output count differs");
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const auto output = outputs[index].contiguous();
    if (output.sizes().vec() != std::vector<std::int64_t>(sample.shape.begin(),
                                                          sample.shape.end()) ||
        output.numel() != static_cast<std::int64_t>(expected[index].size()))
      throw std::runtime_error("Restored model output shape differs");
    const auto values =
        std::span(output.const_data_ptr<float>(), expected[index].size());
    for (std::size_t element = 0; element < values.size(); ++element)
      if (!std::isfinite(values[element]) ||
          std::abs(values[element] - expected[index][element]) > 1e-6F)
        throw std::runtime_error("Restored model differs from scalar oracle");
  }
}

exporter::ExportResult RestoreAndExport(const fs::path& checkpoint,
                                        std::vector<Sample>& samples) {
  const torch::NoGradGuard guard;
  forge::nn::ModelCheckpointParticipantOptions state_options;
  state_options.max_tensors = 16;
  state_options.max_topology_bytes = 64U << 10U;
  state_options.max_tensor_bytes = 1U << 20U;
  state_options.max_total_bytes = 1U << 20U;
  forge::interop::FileStateArchiveOptions archive_options;
  archive_options.max_artifact_bytes = 4U << 20U;
  archive_options.max_field_bytes = 1U << 20U;

  {
    auto original = std::make_shared<PairCalibration>();
    original->initialize_demo_state();
    auto state = std::make_shared<forge::nn::ModelStateView>(original);
    forge::nn::ModelCheckpointParticipant participant(state, state_options);
    forge::interop::FileStateArchive archive(checkpoint, archive_options);
    archive.begin({participant.describe_state()});
    participant.save_state(archive.writer_for(participant.participant_id()));
    archive.commit();
  }  // Release the original model and archive before reconstruction.

  auto restored = std::make_shared<PairCalibration>();
  auto state = std::make_shared<forge::nn::ModelStateView>(restored);
  forge::nn::ModelCheckpointParticipant participant(state, state_options);
  {
    archive_options.read_only = true;
    forge::interop::FileStateArchive archive(checkpoint, archive_options);
    participant.load_state(archive.reader_for(participant.participant_id()),
                           forge::RestorePolicy::exact_only);
  }  // Release the checkpoint snapshot before tracing.
  restored->eval();
  for (auto& sample : samples) CheckNativeOutputs(*restored, sample);

  exporter::ExportOptions options;
  options.input_names = {"left", "right"};
  options.output_names = {"calibrated", "difference"};
  options.minimum_batch = 1;
  options.maximum_batch = 8;
  options.max_output_bytes = 4U << 20U;
  options.dynamic_axes = {{"left", {{0, "batch"}}},
                          {"right", {{0, "batch"}}},
                          {"calibrated", {{0, "batch"}}},
                          {"difference", {{0, "batch"}}}};
  const auto examples = BorrowTensors(samples[1]);  // Trace batch 2.
  return exporter::export_module(
      *restored,
      [restored](std::span<const at::Tensor> inputs) {
        return restored->forward(inputs[0], inputs[1]);
      },
      examples, options);
}  // Release the restored model before creating the ORT session.

void WriteBytes(const fs::path& path, std::span<const std::byte> bytes) {
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.close();
  if (!output) throw std::runtime_error("Cannot save example artifact");
}

void WriteText(const fs::path& path, std::string_view text) {
  WriteBytes(path, std::as_bytes(std::span(text.data(), text.size())));
}

std::string ReadProfile(const fs::path& path) {
  const auto length = fs::file_size(path);
  if (length == 0 || length > (16U << 20U))
    throw std::runtime_error("ORT profile exceeds the example byte bound");
  std::string text(static_cast<std::size_t>(length), '\0');
  std::ifstream input(path, std::ios::binary);
  input.read(text.data(), static_cast<std::streamsize>(text.size()));
  if (!input || input.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("Cannot read complete ORT profile");
  return text;
}

void RunExample(const fs::path& output) {
  if (!output.is_absolute())
    throw std::invalid_argument("--output requires an absolute directory path");
  if (!fs::create_directory(output))
    throw std::invalid_argument("--output must be a new directory");
  at::set_num_threads(1);
  std::vector<Sample> samples;
  for (const auto batch : {1, 2, 5}) samples.push_back(MakeSample(batch));
  const auto artifact = RestoreAndExport(output / "checkpoint.jyck", samples);
  if (!artifact.report)
    throw std::runtime_error("Missing native export report");
  exporter::verify_export_report_model(*artifact.report, artifact.onnx);
  WriteBytes(output / "model.onnx", artifact.onnx);
  WriteText(output / "export-report.json",
            exporter::serialize_export_report(*artifact.report));

  runtime::SessionOptions options;
  options.provider.name = "CPUExecutionProvider";
  options.max_batch_size = 8;
  options.max_model_bytes = 4U << 20U;
  options.max_input_bytes = 1U << 20U;
  options.max_output_bytes = 1U << 20U;
  options.profile_prefix = output / "ort-profile";
  runtime::Session session(artifact.onnx, artifact.digest, options);
  std::vector<std::array<runtime::TensorInput, 2>> inputs;
  std::vector<std::array<runtime::TensorInput, 2>> expected;
  for (const auto& sample : samples) {
    // Names bind inputs: their order deliberately differs from graph order.
    inputs.push_back({{{"right", sample.right, sample.shape},
                       {"left", sample.left, sample.shape}}});
    expected.push_back({{{"calibrated", sample.calibrated, sample.shape},
                         {"difference", sample.difference, sample.shape}}});
  }
  std::vector<runtime::VerificationCase> cases;
  for (std::size_t index = 0; index < samples.size(); ++index)
    cases.push_back({samples[index].id, inputs[index], expected[index]});
  runtime::VerificationOptions policy;
  policy.absolute_tolerance = 1e-6;
  policy.relative_tolerance = 1e-5;
  policy.max_cases = 3;
  policy.max_case_bytes = 1U << 20U;
  policy.reference = "Independent scalar formulas in checkpoint_onnx.cpp";
  auto report = runtime::verify_session(
      session, cases, policy, {},
      std::chrono::steady_clock::now() + std::chrono::seconds(30));
  const auto generated_profile = session.end_profiling();
  const auto profile_text = ReadProfile(generated_profile);
  report.profile = runtime::inspect_profile(profile_text);
  WriteText(output / "runtime-profile.json", profile_text);
  fs::remove(generated_profile);
  WriteText(output / "verification-report.json",
            runtime::serialize_verification_report(report));
  if (!runtime::verification_passed(report))
    throw std::runtime_error("ORT results differ from scalar oracle");
  if (report.profile->kernels.size() != 1 ||
      !report.profile->kernels.contains(options.provider.name))
    throw std::runtime_error("Expected CPU provider kernels were not observed");
  std::cout << "Checkpoint restored; native ONNX export completed.\n"
            << "Validated batches: 1, 2, 5; inputs: 2; outputs: 2.\n"
            << "Provider: " << options.provider.name
            << "; ORT: " << session.runtime_version() << "\n"
            << "All outputs match the independent scalar oracle.\n";
}

int Main(std::span<const fs::path> arguments) {
  constexpr auto usage =
      "Usage: liuding-forge-example-checkpoint-onnx --output "
      "ABSOLUTE_DIRECTORY\n";
  if (arguments.size() == 1 && arguments[0] == "--help") {
    std::cout << usage;
    return 0;
  }
  if (arguments.size() != 2 || arguments[0] != "--output") {
    std::cerr << usage;
    return 2;
  }
  try {
    RunExample(arguments[1]);
    return 0;
  } catch (const std::invalid_argument& error) {
    std::cerr << error.what() << '\n';
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "Deployment example failed: " << error.what()
              << "\nA newly created output directory may contain partial "
                 "artifacts.\n";
    return 1;
  }
}
}  // namespace

#if defined(_WIN32)
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  const std::vector<fs::path> arguments(argv + 1, argv + argc);
  return Main(arguments);
}
