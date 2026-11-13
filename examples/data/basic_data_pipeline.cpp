// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

/** @file basic_data_pipeline.cpp
 * @brief Demonstrates worker collation and persistent committed-cursor replay.
 */

#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/random_sampler.hpp>
#include <jyppx/liuding/data/tensor_dataset.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/interop/file_state_archive.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace liuding = jyppx::liuding;
namespace data = liuding::data;
namespace interop = liuding::interop;

constexpr std::size_t kSamples = 10;
constexpr std::uint64_t kEpoch = 3;

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

data::BatchSchema batch_schema() {
  return {.id = "example.tensor-pair", .version = 1,
          .value = liuding::ValueSchema::tuple(
              {liuding::ValueSchema::tensor(
                   {.dtype = torch::kFloat32, .device = torch::Device(torch::kCPU),
                    .rank = 2, .shape = {{std::nullopt, 2}}},
                   "features", "features"),
               liuding::ValueSchema::tensor(
                   {.dtype = torch::kInt64, .device = torch::Device(torch::kCPU),
                    .rank = 1, .shape = {{std::nullopt}}},
                   "targets", "target")}, "pair")};
}

std::unique_ptr<data::DataPipeline> make_pipeline(
    bool workers, std::shared_ptr<std::atomic_bool> fail = {}) {
  const auto ids = torch::arange(static_cast<std::int64_t>(kSamples),
                                 torch::kFloat32);
  auto dataset = std::make_shared<data::TensorDataset>(
      torch::stack({ids, ids + 10}, 1), ids.to(torch::kInt64).remainder(3));
  auto sampler = std::make_shared<data::samplers::RandomSampler>(kSamples, 7);
  data::DataPipelineOptions options{
      .pipeline_id = "example.basic_data_pipeline",
      .dataset_version = "example.tensor-pair/v1",
      .dataset_size = dataset->size(),
      .schema = batch_schema(),
      .batch_factory_id = "example.stack-pair/v1",
      .batch_size = 4,
      .num_workers = workers ? 2U : 0U,
      .prefetch_batches = 3,
      .persistent_workers = workers,
      .worker_seed = 7};
  return std::make_unique<data::DataPipeline>(
      std::move(sampler), std::move(options),
      [dataset, fail](std::span<const std::size_t> indices,
                      const data::DataWorkerContext& context,
                      std::stop_token stop) -> c10::IValue {
        if (fail && fail->load() && context.sequence == 1) {
          throw std::runtime_error("example batch factory failure");
        }
        std::vector<torch::Tensor> features;
        std::vector<torch::Tensor> targets;
        features.reserve(indices.size());
        targets.reserve(indices.size());
        for (const auto index : indices) {
          if (stop.stop_requested()) {
            throw liuding::Cancelled("Example collation cancelled");
          }
          const auto sample = dataset->get(index);
          features.push_back(sample.data);
          targets.push_back(sample.target);
        }
        return c10::ivalue::Tuple::create(
            {torch::stack(features), torch::stack(targets)});
      });
}

void check_batch(const data::BatchEnvelope& batch, std::uint64_t sequence) {
  const auto count = sequence == 2 ? 2U : 4U;
  require(batch.epoch == kEpoch && batch.sequence == sequence &&
              batch.sample_ids.size() == count,
          "Unexpected batch boundary");
  const auto& values = batch.value.toTuple()->elements();
  const auto features = values.at(0).toTensor();
  const auto targets = values.at(1).toTensor();
  require(features.sizes() == torch::IntArrayRef({count, 2}) &&
              targets.sizes() == torch::IntArrayRef({count}),
          "Unexpected tensor shape");
  for (std::size_t row = 0; row < count; ++row) {
    const auto id = batch.sample_ids.at(row);
    require(id < kSamples &&
                features[row][0].item<float>() == static_cast<float>(id) &&
                features[row][1].item<float>() == static_cast<float>(id + 10) &&
                targets[row].item<std::int64_t>() ==
                    static_cast<std::int64_t>(id % 3),
            "Sample ID and tensor values disagree");
  }
}

std::vector<data::BatchEnvelope> reference_batches(bool workers) {
  auto pipeline = make_pipeline(workers);
  require(pipeline->capability().restore_level == liuding::RestoreLevel::exact,
          "This example requires exact replay");
  pipeline->set_epoch(kEpoch);
  std::vector<data::BatchEnvelope> batches;
  std::vector<std::uint64_t> ids;
  while (auto batch = pipeline->next()) {
    check_batch(*batch, batches.size());
    ids.insert(ids.end(), batch->sample_ids.begin(), batch->sample_ids.end());
    batches.push_back(std::move(*batch));
  }
  require(batches.size() == 3 && ids.size() == kSamples,
          "Unexpected epoch length");
  std::sort(ids.begin(), ids.end());
  for (std::size_t i = 0; i < ids.size(); ++i) {
    require(ids[i] == i, "Epoch omitted or repeated a sample");
  }
  pipeline->close();
  return batches;
}

interop::FileStateArchiveOptions archive_options(bool read_only) {
  // The limits must also accept the participant's declared field maxima.
  return {.max_artifact_bytes = 64ULL << 20U,
          .max_field_bytes = 32ULL << 20U,
          .read_only = read_only};
}

void save(const std::filesystem::path& path, bool workers) {
  auto pipeline = make_pipeline(workers);
  pipeline->set_epoch(kEpoch);
  const auto batch = pipeline->next();
  require(batch.has_value(), "Missing first batch");
  check_batch(*batch, 0);
  require(pipeline->state().next_cursor == 4 &&
              pipeline->state().next_sequence == 1,
          "Unexpected committed cursor");
  interop::FileStateArchive archive(path, archive_options(false));
  archive.begin({pipeline->describe_state()});
  pipeline->save_state(archive.writer_for(pipeline->participant_id()));
  archive.commit();
  pipeline->close();
  std::cout << "saved cursor=4 sequence=1\n";
}

void restore(const std::filesystem::path& path, bool workers, bool inject_failure) {
  const auto reference = reference_batches(workers);
  auto fail = std::make_shared<std::atomic_bool>(inject_failure);
  auto pipeline = make_pipeline(workers, fail);
  interop::FileStateArchive archive(path, archive_options(true));
  const auto& reader = archive.reader_for(pipeline->participant_id());
  pipeline->load_state(reader, liuding::RestorePolicy::exact_only);
  require(pipeline->state().next_cursor == 4 &&
              pipeline->state().next_sequence == 1,
          "Saved committed boundary was not restored");
  if (inject_failure) {
    bool rejected = false;
    try {
      (void)pipeline->next();
    } catch (const liuding::BackendError&) {
      rejected = true;
    }
    require(rejected && pipeline->state().next_cursor == 4 &&
                pipeline->state().next_sequence == 1,
            "Failed batch changed the committed boundary");
    fail->store(false);
    pipeline->load_state(reader, liuding::RestorePolicy::exact_only);
    std::cout << "failure preserved cursor=4 sequence=1\n";
  }
  for (std::size_t i = 1; i < reference.size(); ++i) {
    const auto batch = pipeline->next();
    require(batch.has_value(), "Restored pipeline ended too early");
    check_batch(*batch, i);
    const auto& actual = batch->value.toTuple()->elements();
    const auto& expected = reference[i].value.toTuple()->elements();
    require(batch->sample_ids == reference[i].sample_ids &&
                torch::equal(actual[0].toTensor(), expected[0].toTensor()) &&
                torch::equal(actual[1].toTensor(), expected[1].toTensor()),
            "Restored batch differs from continuous execution");
  }
  require(!pipeline->next() && pipeline->state().next_cursor == kSamples,
          "Restored epoch did not finish at the expected boundary");
  pipeline->close();
  std::cout << "restored batches=2 samples=6 tail=2 exact=true\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    torch::set_num_threads(1);
    const std::string_view mode = argc == 1 ? "inline" : argv[1];
    if (argc <= 2 && (mode == "inline" || mode == "workers")) {
      const auto batches = reference_batches(mode == "workers");
      for (const auto& batch : batches) {
        std::cout << "sequence=" << batch.sequence << " samples="
                  << batch.sample_ids.size() << " ids=";
        for (const auto id : batch.sample_ids) {
          std::cout << id << ',';
        }
        std::cout << '\n';
      }
      std::cout << "batches=3 samples=10 tail=2\n";
    } else {
      require(argc == 4, "Usage: program [inline|workers] OR "
                         "[save|restore|failure] STATE [inline|workers]");
      const std::string_view topology = argv[3];
      require(topology == "inline" || topology == "workers", "Invalid topology");
      if (mode == "save") {
        save(argv[2], topology == "workers");
      } else {
        require(mode == "restore" || mode == "failure", "Invalid mode");
        restore(argv[2], topology == "workers", mode == "failure");
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
