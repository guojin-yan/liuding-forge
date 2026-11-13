// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/batch_sampler.hpp>
#include <jyppx/liuding/data/samplers/random_sampler.hpp>
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>

#include <algorithm>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::data;

class MemoryState final : public StateWriter, public StateReader {
 public:
  explicit MemoryState(StateManifest manifest)
      : manifest_(std::move(manifest)) {}
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field =
        std::find_if(manifest_.fields.begin(), manifest_.fields.end(),
                     [name](const auto& item) { return item.name == name; });
    if (field == manifest_.fields.end() || data.size() > field->max_bytes ||
        !blobs_
             .emplace(std::string(name),
                      std::vector<std::byte>(data.begin(), data.end()))
             .second) {
      throw std::runtime_error("Invalid state write");
    }
  }
  const StateManifest& manifest() const noexcept override { return manifest_; }
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t limit) const override {
    const auto& data = blobs_.at(std::string(name));
    if (data.size() > limit) throw std::runtime_error("Oversized state read");
    return data;
  }

 private:
  StateManifest manifest_;
  std::map<std::string, std::vector<std::byte>> blobs_;
};

void require(bool condition) {
  if (!condition)
    throw std::runtime_error("Installed BatchSampler contract failed");
}

DataPipelineOptions options(std::size_t workers, bool drop_last) {
  return {.pipeline_id = "installed.batch-sampler",
          .dataset_version = "indices/v1",
          .dataset_size = 6,
          .schema = BatchSchema::any("sample-indices", 1),
          .batch_factory_id = "indices/v1",
          .batch_size = 4,
          .drop_last = drop_last,
          .num_workers = workers,
          .prefetch_batches = 2};
}

const DataBatchFactory factory = [](std::span<const std::size_t> indices,
                                    const DataWorkerContext&, std::stop_token) {
  c10::List<std::int64_t> values;
  for (const auto index : indices)
    values.push_back(static_cast<std::int64_t>(index));
  return c10::IValue(values);
};

void check_batch(const std::optional<BatchEnvelope>& batch,
                 const std::vector<std::uint64_t>& ids, std::uint64_t epoch,
                 std::uint64_t sequence) {
  require(batch.has_value());
  require(batch->sample_ids == ids && batch->epoch == epoch &&
          batch->sequence == sequence);
  const std::vector<std::int64_t> expected(ids.begin(), ids.end());
  require(batch->value.toIntVector() == expected);
}

void tail() {
  for (const auto workers : {0U, 2U}) {
    for (const auto drop_last : {false, true}) {
      auto sampler = std::make_shared<samplers::BatchSampler>(
          std::make_shared<samplers::SequentialSampler>(6), 4, drop_last);
      require(sampler->size() == (drop_last ? 1U : 2U));
      require(sampler->sample_count() == (drop_last ? 4U : 6U));
      DataPipeline pipeline(sampler, options(workers, drop_last), factory);
      require(pipeline.capability().restore_level == RestoreLevel::exact);
      pipeline.set_epoch(7);
      check_batch(pipeline.next(), {0, 1, 2, 3}, 7, 0);
      if (!drop_last) check_batch(pipeline.next(), {4, 5}, 7, 1);
      require(!pipeline.next());
      std::cout << "tail workers=" << workers << " drop_last=" << drop_last
                << '\n';
    }
  }
}

void resume() {
  for (const auto workers : {0U, 2U}) {
    for (const auto drop_last : {false, true}) {
      const auto create = [&](std::int64_t seed) {
        return std::make_unique<DataPipeline>(
            std::make_shared<samplers::BatchSampler>(
                std::make_shared<samplers::RandomSampler>(6, seed), 4,
                drop_last),
            options(workers, drop_last), factory);
      };
      auto source = create(17);
      source->set_epoch(4);
      // Independently frozen SplitMix64/Fisher-Yates vectors for seed 17.
      const std::vector<std::size_t> initial =
          drop_last ? std::vector<std::size_t>{0, 2, 5, 3}
                    : std::vector<std::size_t>{0, 2, 5, 3, 4, 1};
      require(source->state().epoch_indices == initial);
      check_batch(source->next(), {0, 2, 5, 3}, 4, 0);
      MemoryState state(source->describe_state());
      source->save_state(state);
      auto target = create(999);
      target->load_state(state, RestorePolicy::exact_only);
      require(target->state().next_cursor == 4 &&
              target->state().next_sequence == 1);
      if (!drop_last) {
        check_batch(source->next(), {4, 1}, 4, 1);
        check_batch(target->next(), {4, 1}, 4, 1);
      }
      require(!source->next() && !target->next());
      source->set_epoch(9);
      target->set_epoch(9);
      const std::vector<std::size_t> next =
          drop_last ? std::vector<std::size_t>{1, 5, 3, 0}
                    : std::vector<std::size_t>{1, 5, 3, 0, 2, 4};
      require(source->state().epoch_indices == next);
      require(target->state().epoch_indices == next);
      std::cout << "restore workers=" << workers << " drop_last=" << drop_last
                << '\n';
    }
  }
}

void rejected() {
  auto sampler = std::make_shared<samplers::BatchSampler>(
      std::make_shared<samplers::SequentialSampler>(6), 4, false);
  for (const auto mismatch_tail : {false, true}) {
    auto config = options(0, mismatch_tail);
    if (!mismatch_tail) config.batch_size = 2;
    bool rejected = false;
    try {
      DataPipeline pipeline(sampler, config, factory);
    } catch (const InvalidArgument&) {
      rejected = true;
    }
    require(rejected);
  }
  DataPipeline source(sampler, options(0, false), factory);
  source.set_epoch(7);
  require(source.next().has_value());
  MemoryState saved(source.describe_state());
  source.save_state(saved);
  DataPipeline target(
      std::make_shared<samplers::BatchSampler>(
          std::make_shared<samplers::RandomSampler>(6, 17), 4, false),
      options(0, false), factory);
  target.set_epoch(3);
  const auto before = target.state();
  bool incompatible = false;
  try {
    target.load_state(saved, RestorePolicy::exact_only);
  } catch (const CompatibilityError&) {
    incompatible = true;
  }
  require(incompatible);
  const auto after = target.state();
  require(after.epoch == before.epoch &&
          after.next_cursor == before.next_cursor &&
          after.epoch_indices == before.epoch_indices &&
          after.sampler_state == before.sampler_state);
  std::cout << "rejected batch-size/tail mismatch and incompatible base "
               "without mutation\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string_view mode = argc == 2 ? argv[1] : "";
    if (mode == "tail")
      tail();
    else if (mode == "resume")
      resume();
    else if (mode == "rejected")
      rejected();
    else
      throw std::invalid_argument("Expected tail|resume|rejected");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
