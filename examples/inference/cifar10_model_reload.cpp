// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
/** @file cifar10_model_reload.cpp
 * @brief Reloads a recipe's complete ResNet18 state and evaluates CIFAR-10. */

#include "cifar10_model_support.hpp"

namespace {
using namespace cifar10_example;

int Run(const fs::path& model_path, std::string_view expected_sha,
        const fs::path& validation_root) {
  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  at::globalContext().setDeterministicAlgorithms(true, false);
  at::globalContext().setDeterministicCuDNN(true);
  at::globalContext().setBenchmarkCuDNN(false);
  at::globalContext().setAllowTF32CuDNN(false);
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setFloat32MatmulPrecision("highest");
  forge::core::RngGuard seeded(kSeed, true);
  forge::models::ResNetOptions architecture;
  architecture.num_classes = 10;
  architecture.input_channels = 3;
  auto model = forge::models::adapt(forge::models::resnet18(architecture));
  const auto imported = LoadModel(model, model_path, expected_sha);
  model.state().eval();
  const auto before = StateDigest(model);
  Images images(validation_root);
  forge::data::samplers::RandomSampler sampler(kSamples, kSeed);
  sampler.set_epoch(0);
  const auto order = sampler.indices();
  std::size_t offset = 0;
  std::size_t last_batch_samples = 0;
  std::optional<image::ImageBatch> current;
  const auto scores = forge::training::evaluate_classification(
      model,
      [&](std::stop_token stop)
          -> std::optional<forge::training::ClassificationEvaluationBatch> {
        current.reset();
        if (offset == kSamples) return std::nullopt;
        const auto count = std::min(kBatchSize, kSamples - offset);
        current = images.batch(std::span(order).subspan(offset, count), stop);
        offset += count;
        last_batch_samples = count;
        return forge::training::ClassificationEvaluationBatch{
            .invocation = {.positional = {current->images}, .keyword = {}},
            .targets = current->labels,
            .source_indices = {}};
      },
      {.num_classes = 10,
       .max_batches = kBatches,
       .max_samples = kSamples,
       .max_batch_samples = kBatchSize});
  current.reset();
  Require(offset == kSamples && scores.batches == kBatches &&
              scores.rows_seen == kSamples &&
              scores.overall.samples == kSamples &&
              scores.overall.accuracy.has_value() && last_batch_samples == 16,
          "Evaluation did not consume exactly 10000 images");
  const auto after = StateDigest(model);
  Require(after == before, "Evaluation changed model state");
  const auto accuracy = *scores.overall.accuracy;
  Require(std::isfinite(accuracy), "Nonfinite accuracy");
  std::cout << std::setprecision(17)
            << "{\"schema_id\":\"jyppx.examples.cifar10-model-reload\","
               "\"schema_version\":1,\"device\":\"cpu\",\"model_sha256\":\""
            << expected_sha << "\",\"build\":"
            << forge::build_fingerprint_json(forge::current_build_fingerprint())
            << ",\"applied_tensors\":" << imported.applied_tensors
            << ",\"import_staging_bytes\":" << imported.charged_staging_bytes
            << ",\"parameter_tensors\":"
            << model.state().named_parameters().size()
            << ",\"buffer_tensors\":" << model.state().named_buffers().size()
            << ",\"state_sha256_before\":\"" << before
            << "\",\"state_sha256_after\":\"" << after
            << "\",\"model_state_unchanged\":true,\"training_updates\":0,"
               "\"forward_calls\":"
            << scores.batches
            << ",\"validation_samples\":" << scores.overall.samples
            << ",\"tail_batch_samples\":" << last_batch_samples
            << ",\"accuracy\":" << accuracy
            << ",\"minimum_accuracy\":0.65,\"quality_passed\":"
            << (accuracy >= .65 ? "true" : "false") << ",\"confusion\":[";
  for (std::size_t row = 0; row < scores.overall.confusion.size(); ++row) {
    if (row != 0) std::cout << ',';
    std::cout << '[';
    for (std::size_t column = 0; column < scores.overall.confusion[row].size();
         ++column) {
      if (column != 0) std::cout << ',';
      std::cout << scores.overall.confusion[row][column];
    }
    std::cout << ']';
  }
  std::cout << "]}\n";
  return accuracy >= .65 ? 0 : 4;
}

template <class Character>
int Main(int argc, Character* argv[]) {
  constexpr auto usage =
      "usage: liuding-forge-example-cifar10-model-reload "
      "MODEL_SAFETENSORS MODEL_SHA256 VALIDATION_ROOT\n";
  try {
    if (argc == 2 && fs::path(argv[1]) == fs::path("--help")) {
      std::cout << usage;
      return 0;
    }
    if (argc != 4) {
      std::cerr << usage;
      return 2;
    }
    std::string expected_sha;
    for (const auto* value = argv[2]; *value != 0; ++value) {
      const auto code = static_cast<unsigned int>(*value);
      Require(expected_sha.size() < 64 && ((code >= '0' && code <= '9') ||
                                           (code >= 'a' && code <= 'f')),
              "Expected a lowercase SHA-256 from the recipe result");
      expected_sha.push_back(static_cast<char>(code));
    }
    Require(expected_sha.size() == 64, "Expected a complete model SHA-256");
    return Run(fs::path(argv[1]), expected_sha, fs::path(argv[3]));
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) { return Main(argc, argv); }
#else
int main(int argc, char* argv[]) { return Main(argc, argv); }
#endif
