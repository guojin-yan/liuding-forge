// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/data/data_pipeline.hpp>
#include <jyppx/foundation/error.hpp>

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace jyppx::data {
namespace {

constexpr char kComponent[] = "data.pipeline";
constexpr char kTopologyMagic[] = "JYPPX-DATA-PIPELINE-TOPOLOGY/v1";
constexpr char kCursorMagic[] = "JYPPX-DATA-PIPELINE-CURSOR/v1";
constexpr std::uint64_t kMaxTopologyBytes = 32U * 1024U * 1024U;
constexpr std::uint64_t kMaxCursorBytes = 128U;
constexpr std::uint64_t kMaxIndexCount = 2U * 1024U * 1024U;
constexpr std::uint64_t kMaxMetadataBytes = 4096U;
constexpr std::uint64_t kMaxSamplerIdentityBytes = 4096U;
constexpr std::uint64_t kMaxSamplerStateBytes = 8U * 1024U * 1024U;

void append_byte(std::string& output, std::uint8_t value) {
  output.push_back(static_cast<char>(value));
}

void append_u64(std::string& output, std::uint64_t value) {
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    append_byte(output, static_cast<std::uint8_t>(value >> shift));
  }
}

void append_string(std::string& output, const std::string& value) {
  append_u64(output, static_cast<std::uint64_t>(value.size()));
  output.append(value);
}

std::uint8_t read_byte(std::span<const std::byte> input, std::size_t& offset) {
  if (offset >= input.size()) {
    throw SerializationError("DataPipeline state ended while reading a byte");
  }
  return std::to_integer<std::uint8_t>(input[offset++]);
}

std::uint64_t read_u64(std::span<const std::byte> input, std::size_t& offset) {
  if (input.size() - offset < sizeof(std::uint64_t)) {
    throw SerializationError(
        "DataPipeline state ended while reading an integer");
  }
  std::uint64_t value = 0;
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    value |= static_cast<std::uint64_t>(read_byte(input, offset)) << shift;
  }
  return value;
}

std::string read_string(std::span<const std::byte> input, std::size_t& offset,
                        std::uint64_t max_bytes) {
  const auto size = read_u64(input, offset);
  if (size > max_bytes) {
    throw ResourceExhausted("DataPipeline state string exceeds its bound",
                            ErrorDetails{.component = kComponent,
                                         .operation = "decode.string",
                                         .expected = "bounded string",
                                         .actual = std::to_string(size),
                                         .hint = "Use a valid current state."});
  }
  if (size > input.size() - offset) {
    throw SerializationError("DataPipeline state ended while reading a string");
  }
  const auto begin = input.begin() + static_cast<std::ptrdiff_t>(offset);
  const auto end = begin + static_cast<std::ptrdiff_t>(size);
  std::string value;
  value.reserve(static_cast<std::size_t>(size));
  for (auto cursor = begin; cursor != end; ++cursor) {
    value.push_back(static_cast<char>(std::to_integer<unsigned char>(*cursor)));
  }
  offset += static_cast<std::size_t>(size);
  return value;
}

void require_magic(const std::string& actual, const char* expected) {
  if (actual != expected) {
    throw SerializationError(
        "DataPipeline state has an unknown encoding",
        ErrorDetails{.component = kComponent,
                     .operation = "decode.magic",
                     .expected = expected,
                     .actual = actual,
                     .hint = "Use a state produced by the current schema."});
  }
}

void require_end(std::span<const std::byte> input, std::size_t offset) {
  if (offset != input.size()) {
    throw SerializationError("DataPipeline state contains trailing bytes");
  }
}

std::string encode_topology(const DataPipelineState& state) {
  std::string output;
  append_string(output, kTopologyMagic);
  append_string(output, state.pipeline_id);
  append_string(output, state.dataset_version);
  append_u64(output, state.dataset_size);
  append_string(output, state.schema_id);
  append_string(output, state.sampler_identity);
  append_u64(output, state.batch_size);
  append_byte(output, state.drop_last ? 1 : 0);
  append_byte(output, state.sampler_state_supported ? 1 : 0);
  append_string(output, state.sampler_state);
  append_u64(output, static_cast<std::uint64_t>(state.epoch_indices.size()));
  for (const auto index : state.epoch_indices) {
    append_u64(output, static_cast<std::uint64_t>(index));
  }
  return output;
}

std::string encode_cursor(const DataPipelineState& state) {
  std::string output;
  append_string(output, kCursorMagic);
  append_byte(output, state.prepared ? 1 : 0);
  append_u64(output, state.epoch);
  append_u64(output, state.next_cursor);
  append_u64(output, state.next_sequence);
  return output;
}

DataPipelineState decode_state(std::span<const std::byte> topology,
                               std::span<const std::byte> cursor) {
  DataPipelineState state;
  std::size_t topology_offset = 0;
  require_magic(read_string(topology, topology_offset, 128), kTopologyMagic);
  state.pipeline_id = read_string(topology, topology_offset, kMaxMetadataBytes);
  state.dataset_version =
      read_string(topology, topology_offset, kMaxMetadataBytes);
  const auto dataset_size = read_u64(topology, topology_offset);
  if (dataset_size > std::numeric_limits<std::size_t>::max()) {
    throw SerializationError("DataPipeline dataset size is out of range");
  }
  state.dataset_size = static_cast<std::size_t>(dataset_size);
  state.schema_id = read_string(topology, topology_offset, kMaxMetadataBytes);
  state.sampler_identity =
      read_string(topology, topology_offset, kMaxSamplerIdentityBytes);
  const auto batch_size = read_u64(topology, topology_offset);
  if (batch_size > std::numeric_limits<std::size_t>::max()) {
    throw SerializationError("DataPipeline batch size is out of range");
  }
  state.batch_size = static_cast<std::size_t>(batch_size);
  const auto drop_last = read_byte(topology, topology_offset);
  const auto sampler_supported = read_byte(topology, topology_offset);
  if (drop_last > 1 || sampler_supported > 1) {
    throw SerializationError("DataPipeline state contains an invalid boolean");
  }
  state.drop_last = drop_last != 0;
  state.sampler_state_supported = sampler_supported != 0;
  state.sampler_state =
      read_string(topology, topology_offset, kMaxSamplerStateBytes);
  const auto count = read_u64(topology, topology_offset);
  if (count > kMaxIndexCount) {
    throw ResourceExhausted("DataPipeline index state exceeds its bound");
  }
  if (count > (topology.size() - topology_offset) / sizeof(std::uint64_t)) {
    throw SerializationError(
        "DataPipeline state ended while reading sample indices");
  }
  state.epoch_indices.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t index = 0; index < count; ++index) {
    const auto value = read_u64(topology, topology_offset);
    if (value > std::numeric_limits<std::size_t>::max()) {
      throw SerializationError("DataPipeline index is out of range");
    }
    state.epoch_indices.push_back(static_cast<std::size_t>(value));
  }
  require_end(topology, topology_offset);

  std::size_t cursor_offset = 0;
  require_magic(read_string(cursor, cursor_offset, 128), kCursorMagic);
  const auto prepared = read_byte(cursor, cursor_offset);
  if (prepared > 1) {
    throw SerializationError("DataPipeline cursor contains an invalid boolean");
  }
  state.prepared = prepared != 0;
  state.epoch = read_u64(cursor, cursor_offset);
  state.next_cursor = read_u64(cursor, cursor_offset);
  state.next_sequence = read_u64(cursor, cursor_offset);
  require_end(cursor, cursor_offset);
  state.version = DataPipelineState::format_version;
  return state;
}

void write_blob(StateWriter& writer, std::string_view name,
                const std::string& value, std::uint64_t max_bytes) {
  if (value.size() > max_bytes) {
    throw ResourceExhausted("DataPipeline state exceeds its declared bound");
  }
  writer.write_blob(name, std::as_bytes(std::span(value.data(), value.size())));
}

}  // namespace

DataPipeline::DataPipeline(std::shared_ptr<samplers::Sampler> sampler,
                           DataPipelineOptions options,
                           DataBatchFactory batch_factory)
    : sampler_(std::move(sampler)),
      options_(std::move(options)),
      batch_factory_(std::move(batch_factory)) {
  if (!sampler_) {
    throw InvalidArgument("DataPipeline sampler cannot be null");
  }
  if (!batch_factory_) {
    throw InvalidArgument("DataPipeline batch factory cannot be empty");
  }
  if (options_.pipeline_id.empty() || options_.dataset_version.empty() ||
      options_.schema_id.empty()) {
    throw InvalidArgument(
        "DataPipeline identity, dataset version, and schema ID are required");
  }
  if (options_.pipeline_id.size() > kMaxMetadataBytes ||
      options_.dataset_version.size() > kMaxMetadataBytes ||
      options_.schema_id.size() > kMaxMetadataBytes) {
    throw ResourceExhausted(
        "DataPipeline identity metadata exceeds its declared bound");
  }
  if (options_.dataset_size == 0) {
    throw InvalidArgument("DataPipeline dataset_size must be positive");
  }
  if (options_.batch_size == 0) {
    throw InvalidArgument("DataPipeline batch_size must be positive");
  }
  if (sampler_->size() == 0) {
    throw InvalidArgument("DataPipeline sampler must produce samples");
  }
  if (sampler_->size() > kMaxIndexCount) {
    throw ResourceExhausted(
        "DataPipeline sampler exceeds the checkpointable index bound");
  }
  const auto sampler_identity = sampler_->configuration_identity();
  if (sampler_identity.size() > kMaxSamplerIdentityBytes) {
    throw ResourceExhausted(
        "DataPipeline sampler identity exceeds its declared bound");
  }
  state_.pipeline_id = options_.pipeline_id;
  state_.dataset_version = options_.dataset_version;
  state_.dataset_size = options_.dataset_size;
  state_.schema_id = options_.schema_id;
  state_.batch_size = options_.batch_size;
  state_.drop_last = options_.drop_last;
  state_.sampler_state_supported = sampler_->supports_state();
  state_.sampler_identity = sampler_identity;
  if (state_.sampler_state_supported) {
    state_.sampler_state = sampler_->save_state();
    if (state_.sampler_state.size() > kMaxSamplerStateBytes) {
      throw ResourceExhausted(
          "DataPipeline sampler state exceeds its declared bound");
    }
  }
  validate_checkpoint_participant(*this);
}

DataPipeline::~DataPipeline() = default;

void DataPipeline::set_epoch(std::uint64_t epoch) {
  const auto supports_state = sampler_->supports_state();
  const auto sampler_identity = sampler_->configuration_identity();
  if (sampler_identity != state_.sampler_identity) {
    throw CompatibilityError(
        "DataPipeline sampler configuration changed after construction");
  }
  const auto previous_sampler_state =
      supports_state ? sampler_->save_state() : std::string{};
  try {
    sampler_->set_epoch(epoch);
    auto indices = sampler_->indices();
    const auto expected = sampler_->size();
    if (expected > kMaxIndexCount) {
      throw ResourceExhausted(
          "DataPipeline sampler exceeds the checkpointable index bound");
    }
    if (indices.size() != expected) {
      throw InvalidArgument(
          "DataPipeline sampler returned an unexpected epoch length",
          ErrorDetails{.component = kComponent,
                       .operation = "set_epoch",
                       .expected = std::to_string(expected),
                       .actual = std::to_string(indices.size()),
                       .hint = "Fix the sampler size contract."});
    }
    for (const auto index : indices) {
      if (index >= options_.dataset_size) {
        throw InvalidArgument(
            "DataPipeline sampler returned an out-of-range sample index",
            ErrorDetails{
                .component = kComponent,
                .operation = "set_epoch",
                .expected = "index < " + std::to_string(options_.dataset_size),
                .actual = std::to_string(index),
                .hint = "Fix the sampler dataset topology."});
      }
    }
    if (options_.drop_last) {
      indices.resize((indices.size() / options_.batch_size) *
                     options_.batch_size);
    }

    DataPipelineState candidate;
    candidate.pipeline_id = options_.pipeline_id;
    candidate.dataset_version = options_.dataset_version;
    candidate.dataset_size = options_.dataset_size;
    candidate.schema_id = options_.schema_id;
    candidate.sampler_identity = sampler_identity;
    candidate.batch_size = options_.batch_size;
    candidate.drop_last = options_.drop_last;
    candidate.epoch = epoch;
    candidate.prepared = true;
    candidate.sampler_state_supported = supports_state;
    candidate.epoch_indices = std::move(indices);
    if (supports_state) {
      candidate.sampler_state = sampler_->save_state();
      if (candidate.sampler_state.size() > kMaxSamplerStateBytes) {
        throw ResourceExhausted(
            "DataPipeline sampler state exceeds its declared bound");
      }
    }
    state_ = std::move(candidate);
  } catch (...) {
    const auto cause = std::current_exception();
    if (supports_state) {
      try {
        sampler_->load_state(previous_sampler_state);
      } catch (...) {
        std::throw_with_nested(
            InternalError("DataPipeline epoch preparation rollback failed"));
      }
    }
    std::rethrow_exception(cause);
  }
}

std::optional<BatchEnvelope> DataPipeline::next(std::stop_token stop_token) {
  if (stop_token.stop_requested()) {
    return std::nullopt;
  }
  if (!state_.prepared) {
    throw InvalidArgument(
        "DataPipeline requires set_epoch before next",
        ErrorDetails{.component = kComponent,
                     .operation = "next",
                     .expected = "a prepared epoch",
                     .actual = "no epoch",
                     .hint = "Call set_epoch exactly once per epoch."});
  }
  if (state_.next_cursor >= state_.epoch_indices.size()) {
    return std::nullopt;
  }
  const auto start = state_.next_cursor;
  const auto count =
      std::min(options_.batch_size, state_.epoch_indices.size() - start);
  const auto end = start + count;
  const std::span<const std::size_t> indices(
      state_.epoch_indices.data() + start, end - start);
  const auto value = batch_factory_(indices);
  if (state_.next_sequence == std::numeric_limits<std::uint64_t>::max()) {
    throw ResourceExhausted("DataPipeline batch sequence exhausted");
  }
  BatchEnvelope envelope;
  envelope.value = value;
  envelope.schema_id = options_.schema_id;
  envelope.dataset_version = options_.dataset_version;
  envelope.epoch = state_.epoch;
  envelope.sequence = state_.next_sequence;
  envelope.sample_ids.reserve(indices.size());
  for (const auto index : indices) {
    envelope.sample_ids.push_back(static_cast<std::uint64_t>(index));
  }
  state_.next_cursor = end;
  ++state_.next_sequence;
  return envelope;
}

DataCapability DataPipeline::capability() const {
  DataCapability capability;
  capability.deterministic_epoch = sampler_->supports_epoch_control();
  const auto has_state = sampler_->supports_state();
  const auto has_identity = !sampler_->configuration_identity().empty();
  capability.restore_level =
      has_state && has_identity ? RestoreLevel::exact : RestoreLevel::bounded;
  if (!has_state) {
    capability.degradation_reasons.emplace_back(
        "sampler does not expose exact state");
  }
  if (!has_identity) {
    capability.degradation_reasons.emplace_back(
        "sampler does not expose a stable configuration identity");
  }
  return capability;
}

DataPipelineState DataPipeline::state() const { return state_; }

std::string DataPipeline::participant_id() const {
  return options_.pipeline_id;
}

StateManifest DataPipeline::describe_state() const {
  const auto current = capability();
  return {
      .participant_id = participant_id(),
      .schema_version = DataPipelineState::format_version,
      .restore_level = current.restore_level,
      .fields = {{"topology", "application/octet-stream", kMaxTopologyBytes,
                  true},
                 {"cursor", "application/octet-stream", kMaxCursorBytes, true}},
      .degradation_reasons = current.degradation_reasons};
}

void DataPipeline::save_state(StateWriter& writer) const {
  const auto snapshot = state();
  write_blob(writer, "topology", encode_topology(snapshot), kMaxTopologyBytes);
  write_blob(writer, "cursor", encode_cursor(snapshot), kMaxCursorBytes);
}

void DataPipeline::validate_state(const DataPipelineState& state) const {
  if (state.version != DataPipelineState::format_version ||
      state.pipeline_id != options_.pipeline_id ||
      state.dataset_version != options_.dataset_version ||
      state.dataset_size != options_.dataset_size ||
      state.schema_id != options_.schema_id ||
      state.sampler_identity != sampler_->configuration_identity() ||
      state.batch_size != options_.batch_size ||
      state.drop_last != options_.drop_last) {
    throw CompatibilityError("DataPipeline state topology is incompatible");
  }
  if (state.sampler_state_supported != sampler_->supports_state()) {
    throw CompatibilityError("DataPipeline sampler state capability changed");
  }
  if (!state.prepared) {
    if (state.next_cursor != 0 || state.next_sequence != 0 ||
        !state.epoch_indices.empty()) {
      throw SerializationError(
          "Unprepared DataPipeline state contains cursor or indices");
    }
    return;
  }
  const auto expected = sampler_->size();
  const auto expected_indices =
      state.drop_last ? (expected / state.batch_size) * state.batch_size
                      : expected;
  if (state.epoch_indices.size() != expected_indices ||
      state.next_cursor > state.epoch_indices.size()) {
    throw CompatibilityError("DataPipeline state has an invalid epoch shape");
  }
  for (const auto index : state.epoch_indices) {
    if (index >= state.dataset_size) {
      throw SerializationError(
          "DataPipeline state contains an out-of-range sample index");
    }
  }
  const auto is_full_batch_boundary = state.next_cursor % state.batch_size == 0;
  const auto is_partial_final_boundary =
      state.next_cursor == state.epoch_indices.size();
  if (!is_full_batch_boundary && !is_partial_final_boundary) {
    throw SerializationError("DataPipeline cursor is not at a batch boundary");
  }
  const auto expected_sequence =
      state.next_cursor / state.batch_size + (is_full_batch_boundary ? 0 : 1);
  if (state.next_sequence != expected_sequence) {
    throw SerializationError("DataPipeline cursor is not at a batch boundary");
  }
}

void DataPipeline::load_state(const StateReader& reader, RestorePolicy policy) {
  const auto& manifest = reader.manifest();
  validate_state_manifest(manifest);
  const auto expected_manifest = describe_state();
  if (manifest.participant_id != participant_id() ||
      manifest.schema_version != expected_manifest.schema_version ||
      manifest.restore_level != expected_manifest.restore_level ||
      !restore_policy_accepts(policy, manifest.restore_level)) {
    throw CompatibilityError("DataPipeline checkpoint policy is incompatible");
  }
  const auto topology = reader.read_blob("topology", kMaxTopologyBytes);
  const auto cursor = reader.read_blob("cursor", kMaxCursorBytes);
  auto decoded = decode_state(topology, cursor);
  validate_state(decoded);

  const auto supports_state = sampler_->supports_state();
  const std::string previous_sampler_state =
      supports_state ? sampler_->save_state() : std::string{};
  const auto previous = state_;
  try {
    if (supports_state) {
      sampler_->load_state(decoded.sampler_state);
    }
    state_ = std::move(decoded);
  } catch (...) {
    const auto cause = std::current_exception();
    try {
      if (supports_state) {
        sampler_->load_state(previous_sampler_state);
      }
      state_ = previous;
    } catch (...) {
      std::throw_with_nested(
          InternalError("DataPipeline restore rollback failed"));
    }
    std::rethrow_exception(cause);
  }
}

}  // namespace jyppx::data
