// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "fuzz_support.hpp"

#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kMaxFuzzInputLength = 4096;

/** @brief In-memory checkpoint store used by the fuzz harness. */
class FuzzStateStore final : public jyppx::liuding::StateWriter,
                             public jyppx::liuding::StateReader {
 public:
  /** @brief Creates an empty store governed by @p manifest. */
  explicit FuzzStateStore(jyppx::liuding::StateManifest manifest)
      : manifest_(std::move(manifest)) {}

  /** @brief Replaces one state field with bytes emitted by the participant. */
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    blobs_[std::string(name)] =
        std::vector<std::byte>(data.begin(), data.end());
  }

  /** @brief Returns the immutable source manifest. */
  const jyppx::liuding::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  /** @brief Reports whether a named field is present. */
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  /** @brief Returns a bounded field view or rejects the malformed store. */
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto found = blobs_.find(std::string(name));
    if (found == blobs_.end() || found->second.size() > max_bytes) {
      throw jyppx::liuding::SerializationError("Fuzz state field is unavailable");
    }
    return found->second;
  }

  /** @brief Returns mutable bytes for one existing field. */
  std::vector<std::byte>& mutable_blob(std::string_view name) {
    return blobs_.at(std::string(name));
  }

 private:
  jyppx::liuding::StateManifest manifest_;  ///< Manifest presented to the participant.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Mutable topology and cursor fields.
};

/** @brief Builds one valid mid-epoch state used as the mutation seed. */
FuzzStateStore make_baseline() {
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(4),
      {.pipeline_id = "data.pipeline.fuzz",
       .dataset_version = "dataset.fuzz/v1",
       .dataset_size = 4,
       .schema = jyppx::liuding::data::BatchSchema::any("batch.fuzz", 1),
       .batch_factory_id = "factory.fuzz/v1",
       .batch_size = 2},
      [](std::span<const std::size_t> indices,
         const jyppx::liuding::data::DataWorkerContext&, std::stop_token) {
        return c10::IValue(static_cast<std::int64_t>(indices.size()));
      });
  pipeline.set_epoch(2);
  static_cast<void>(pipeline.next());
  FuzzStateStore store(pipeline.describe_state());
  pipeline.save_state(store);
  return store;
}

/** @brief Returns the process-lifetime valid mutation seed. */
const FuzzStateStore& baseline() {
  static const FuzzStateStore value = make_baseline();
  return value;
}

/** @brief Copies fuzzer bytes into a state field from a wrapped offset. */
void overwrite(std::vector<std::byte>& field, const std::uint8_t* data,
               std::size_t size) {
  if (field.empty() || size == 0) {
    return;
  }
  const auto offset = static_cast<std::size_t>(data[0]) % field.size();
  for (std::size_t index = 1; index < size; ++index) {
    field[(offset + index - 1) % field.size()] =
        static_cast<std::byte>(data[index]);
  }
}

/** @brief Applies one bounded structural mutation to a valid state store. */
void mutate(FuzzStateStore& store, const std::uint8_t* data, std::size_t size) {
  if (size == 0) {
    return;
  }
  auto& topology = store.mutable_blob("topology");
  auto& cursor = store.mutable_blob("cursor");
  const auto* payload = data + 1;
  const auto payload_size = size - 1;
  switch (data[0] % 6U) {
    case 0:
      topology.assign(
          reinterpret_cast<const std::byte*>(payload),
          reinterpret_cast<const std::byte*>(payload + payload_size));
      break;
    case 1:
      cursor.assign(reinterpret_cast<const std::byte*>(payload),
                    reinterpret_cast<const std::byte*>(payload + payload_size));
      break;
    case 2:
      overwrite(topology, payload, payload_size);
      break;
    case 3:
      overwrite(cursor, payload, payload_size);
      break;
    case 4:
      if (payload_size > 0) {
        topology.resize(static_cast<std::size_t>(payload[0]) %
                        (topology.size() + 1U));
      }
      break;
    case 5:
      for (std::size_t index = 0; index < payload_size; ++index) {
        cursor.push_back(static_cast<std::byte>(payload[index]));
      }
      break;
  }
}

}  // namespace

/**
 * @brief Exercises DataPipeline state decoding with bounded mutated blobs.
 * @param data Byte sequence generated by libFuzzer.
 * @param size Number of bytes available at @p data.
 * @return Always zero after expected contract errors are handled.
 */
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data,
                                      std::size_t size) {
  if (size > kMaxFuzzInputLength) {
    return 0;
  }
  auto store = baseline();
  mutate(store, data, size);
  try {
    jyppx::liuding::data::DataPipeline pipeline(
        std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(4),
        {.pipeline_id = "data.pipeline.fuzz",
         .dataset_version = "dataset.fuzz/v1",
         .dataset_size = 4,
         .schema = jyppx::liuding::data::BatchSchema::any("batch.fuzz", 1),
         .batch_factory_id = "factory.fuzz/v1",
         .batch_size = 2},
        [](std::span<const std::size_t> indices,
           const jyppx::liuding::data::DataWorkerContext&, std::stop_token) {
          return c10::IValue(static_cast<std::int64_t>(indices.size()));
        });
    pipeline.load_state(store, jyppx::liuding::RestorePolicy::exact_only);
  } catch (const jyppx::liuding::Error& error) {
    switch (error.code()) {
      case jyppx::liuding::ErrorCode::invalid_argument:
      case jyppx::liuding::ErrorCode::serialization:
      case jyppx::liuding::ErrorCode::compatibility:
      case jyppx::liuding::ErrorCode::resource_exhausted:
        break;
      default:
        throw;
    }
  }
  return 0;
}
