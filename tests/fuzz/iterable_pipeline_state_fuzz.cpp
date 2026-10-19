// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/data/iterable_data_pipeline.hpp>
#include <jyppx/foundation/error.hpp>

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

/** @brief Small exact cursor used only to create a valid mutation seed. */
class FuzzCursor final : public jyppx::data::IterableSourceCursor {
 public:
  /** @brief Returns the next integer and its decimal position watermark. */
  std::optional<jyppx::data::IterableSample> next(
      std::stop_token stop) override {
    if (stop.stop_requested() || position_ == 8U) {
      return std::nullopt;
    }
    const auto position = position_++;
    return jyppx::data::IterableSample{
        .value = c10::IValue(static_cast<std::int64_t>(position)),
        .sample_id = position,
        .watermark = std::to_string(position + 1U)};
  }

  /** @brief Encodes the cursor position as decimal text. */
  std::string save_state() const override { return std::to_string(position_); }

  /** @brief Validates and applies a bounded decimal position. */
  void load_state(std::string_view state) override {
    const auto value = std::stoull(std::string(state));
    if (value > 8U) {
      throw jyppx::CompatibilityError("fuzz cursor position out of range");
    }
    position_ = static_cast<std::size_t>(value);
  }

  /** @brief Needs no wakeup because fuzz pulls never block. */
  void close() noexcept override {}

 private:
  std::size_t position_ = 0;  ///< Next source position.
};

/** @brief Immutable source factory for the fuzzing pipeline. */
class FuzzSource final : public jyppx::data::IterableSource {
 public:
  /** @brief Returns the exact source contract used by the seed state. */
  jyppx::data::IterableSourceDescriptor descriptor() const override {
    return {.source_id = "fuzz.iterable/v1",
            .finite_epoch = true,
            .deterministic_epoch = true,
            .checkpointable = true,
            .restore_level = jyppx::RestoreLevel::exact,
            .max_replayed_samples = 0,
            .max_omitted_samples = 0};
  }

  /** @brief Opens a fresh eight-item cursor for any logical epoch. */
  std::unique_ptr<jyppx::data::IterableSourceCursor> open_epoch(
      std::uint64_t) const override {
    return std::make_unique<FuzzCursor>();
  }
};

/** @brief In-memory reader/writer for fuzz-mutated participant fields. */
class FuzzStateStore final : public jyppx::StateWriter,
                             public jyppx::StateReader {
 public:
  /** @brief Creates a field store from a validated participant manifest. */
  explicit FuzzStateStore(jyppx::StateManifest manifest)
      : manifest_(std::move(manifest)) {}

  /** @brief Copies one participant field into mutable storage. */
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    blobs_[std::string(name)] =
        std::vector<std::byte>(data.begin(), data.end());
  }

  /** @brief Returns the participant manifest. */
  const jyppx::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  /** @brief Reports field presence. */
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  /** @brief Exposes one bounded field to the participant decoder. */
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto found = blobs_.find(std::string(name));
    if (found == blobs_.end() || found->second.size() > max_bytes) {
      throw jyppx::SerializationError("fuzz field is unavailable");
    }
    return found->second;
  }

  /** @brief Returns a mutable field for bounded mutation. */
  std::vector<std::byte>& mutable_blob(std::string_view name) {
    return blobs_.at(std::string(name));
  }

  /** @brief Returns the mutable manifest used to select the N-1 decoder. */
  jyppx::StateManifest& mutable_manifest() noexcept { return manifest_; }

 private:
  jyppx::StateManifest manifest_;  ///< Participant descriptor under test.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Mutable encoded fields.
};

FuzzStateStore make_current_baseline() {
  jyppx::data::IterableDataPipeline pipeline(
      std::make_shared<FuzzSource>(),
      {.pipeline_id = "data.iterable.fuzz",
       .dataset_version = "dataset.fuzz/v1",
       .schema = jyppx::data::BatchSchema::any("batch.fuzz", 1),
       .batch_factory_id = "factory.fuzz/v1",
       .batch_size = 2,
       .prefetch_batches = 2},
      [](std::span<const jyppx::data::IterableSample>, std::uint64_t,
         std::uint64_t,
         std::stop_token) { return c10::IValue(std::int64_t{2}); });
  pipeline.set_epoch(2);
  static_cast<void>(pipeline.next());
  FuzzStateStore store(pipeline.describe_state());
  pipeline.save_state(store);
  return store;
}

/** @brief Reads one little-endian string size from a valid seed topology. */
std::uint64_t read_seed_u64(const std::vector<std::byte>& value,
                            std::size_t offset) {
  std::uint64_t result = 0;
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    result |= static_cast<std::uint64_t>(
                  std::to_integer<std::uint8_t>(value.at(offset + shift / 8)))
              << shift;
  }
  return result;
}

/** @brief Writes one little-endian string size into a valid seed topology. */
void write_seed_u64(std::vector<std::byte>& value, std::size_t offset,
                    std::uint64_t replacement) {
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    value.at(offset + shift / 8) =
        static_cast<std::byte>(static_cast<std::uint8_t>(replacement >> shift));
  }
}

/** @brief Replaces one length-prefixed string in a valid seed topology. */
void replace_seed_string(std::vector<std::byte>& value, std::size_t index,
                         std::string_view replacement) {
  std::size_t offset = 0;
  for (std::size_t current = 0; current <= index; ++current) {
    const auto length_offset = offset;
    const auto length = read_seed_u64(value, length_offset);
    offset += sizeof(std::uint64_t);
    if (length > value.size() - offset) {
      throw jyppx::InternalError("valid iterable fuzz seed is malformed");
    }
    if (current == index) {
      const auto begin = value.begin() + static_cast<std::ptrdiff_t>(offset);
      const auto end = begin + static_cast<std::ptrdiff_t>(length);
      value.erase(begin, end);
      value.insert(value.begin() + static_cast<std::ptrdiff_t>(offset),
                   reinterpret_cast<const std::byte*>(replacement.data()),
                   reinterpret_cast<const std::byte*>(replacement.data() +
                                                      replacement.size()));
      write_seed_u64(value, length_offset, replacement.size());
      return;
    }
    offset += static_cast<std::size_t>(length);
  }
}

/** @brief Derives a valid N-1 seed matching the historical schema-v2 codec. */
FuzzStateStore make_previous_baseline() {
  auto store = make_current_baseline();
  auto& manifest = store.mutable_manifest();
  manifest.schema_version = 2U;
  manifest.fields.front().max_bytes = 64U * 1024U;
  auto& topology = store.mutable_blob("topology");
  auto& cursor = store.mutable_blob("cursor");
  replace_seed_string(topology, 0U, "JYPPX-ITERABLE-PIPELINE-TOPOLOGY/v2");
  replace_seed_string(topology, 3U, "batch.fuzz/v1");
  replace_seed_string(cursor, 0U, "JYPPX-ITERABLE-PIPELINE-CURSOR/v2");
  return store;
}

const FuzzStateStore& current_baseline() {
  static const FuzzStateStore value = make_current_baseline();
  return value;
}

const FuzzStateStore& previous_baseline() {
  static const FuzzStateStore value = make_previous_baseline();
  return value;
}

/** @brief Copies input bytes into one field without exceeding its size. */
void overwrite(std::vector<std::byte>& field, const std::uint8_t* data,
               std::size_t size) {
  if (field.empty() || size == 0) {
    return;
  }
  const auto offset = static_cast<std::size_t>(data[0]) % field.size();
  for (std::size_t index = 1; index < size; ++index) {
    field[(offset + index - 1U) % field.size()] =
        static_cast<std::byte>(data[index]);
  }
}

/** @brief Applies one bounded structural mutation to valid state blobs. */
void mutate(FuzzStateStore& store, const std::uint8_t* data, std::size_t size) {
  if (size == 0) {
    return;
  }
  auto& topology = store.mutable_blob("topology");
  auto& cursor = store.mutable_blob("cursor");
  const auto* payload = data + 1;
  const auto payload_size = size - 1U;
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
 * @brief Exercises iterable pipeline state decoding with bounded mutations.
 * @param data Byte sequence supplied by libFuzzer.
 * @param size Number of valid bytes at @p data.
 * @return Always zero after expected contract rejection.
 */
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data,
                                      std::size_t size) {
  if (size > kMaxFuzzInputLength) {
    return 0;
  }
  const auto use_previous = size > 0 && (data[0] & 0x80U) != 0U;
  auto store = use_previous ? previous_baseline() : current_baseline();
  if (size > 0) {
    mutate(store, data + 1U, size - 1U);
  }
  try {
    jyppx::data::IterableDataPipeline pipeline(
        std::make_shared<FuzzSource>(),
        {.pipeline_id = "data.iterable.fuzz",
         .dataset_version = "dataset.fuzz/v1",
         .schema = jyppx::data::BatchSchema::any("batch.fuzz", 1),
         .batch_factory_id = "factory.fuzz/v1",
         .batch_size = 2,
         .prefetch_batches = 2},
        [](std::span<const jyppx::data::IterableSample>, std::uint64_t,
           std::uint64_t,
           std::stop_token) { return c10::IValue(std::int64_t{2}); });
    pipeline.load_state(store, jyppx::RestorePolicy::exact_only);
  } catch (const std::exception&) {
    // Malformed state is expected; sanitizer failures must remain visible.
  }
  return 0;
}
