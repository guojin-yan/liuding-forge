// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "batch_transform_detail.hpp"

#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

namespace jyppx::liuding::data {
namespace {

constexpr char kComponent[] = "data.pipeline";
constexpr char kCurrentTopologyMagic[] = "JYPPX-DATA-PIPELINE-TOPOLOGY/v5";
constexpr char kCurrentCursorMagic[] = "JYPPX-DATA-PIPELINE-CURSOR/v5";
constexpr char kPreviousTopologyMagic[] = "JYPPX-DATA-PIPELINE-TOPOLOGY/v4";
constexpr char kLegacyTopologyMagic[] = "JYPPX-DATA-PIPELINE-TOPOLOGY/v3";
constexpr char kPreviousCursorMagic[] = "JYPPX-DATA-PIPELINE-CURSOR/v4";
constexpr char kLegacyCursorMagic[] = "JYPPX-DATA-PIPELINE-CURSOR/v3";
constexpr std::uint32_t kPreviousSchemaVersion =
    DataPipelineState::format_version - 1U;
constexpr std::uint64_t kMaxTopologyBytes = 32U * 1024U * 1024U;
constexpr std::uint64_t kMaxCursorBytes = 128U;
constexpr std::uint64_t kMaxIndexCount = 2U * 1024U * 1024U;
constexpr std::uint64_t kMaxMetadataBytes = 4096U;
constexpr std::uint64_t kMaxSchemaDescriptorBytes = 1024U * 1024U;
constexpr std::uint64_t kMaxSamplerIdentityBytes = 4096U;
constexpr std::uint64_t kMaxSamplerStateBytes = 8U * 1024U * 1024U;
constexpr std::size_t kMaxWorkerCount = 256U;
constexpr std::size_t kMaxPrefetchBatches = 4096U;
constexpr std::uint64_t kMaxBatchTimeoutMs = 24ULL * 60ULL * 60ULL * 1000ULL;

static_assert(std::is_nothrow_move_assignable_v<DataPipelineState>);

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

std::size_t read_size(std::span<const std::byte> input, std::size_t& offset,
                      std::string_view field) {
  const auto value = read_u64(input, offset);
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw SerializationError("DataPipeline size field is out of range",
                             ErrorDetails{.component = kComponent,
                                          .operation = std::string(field),
                                          .expected = "size_t range",
                                          .actual = std::to_string(value),
                                          .hint = "Use valid current state."});
  }
  return static_cast<std::size_t>(value);
}

bool read_bool(std::span<const std::byte> input, std::size_t& offset,
               std::string_view field) {
  const auto value = read_byte(input, offset);
  if (value > 1U) {
    throw SerializationError("DataPipeline state contains an invalid boolean",
                             ErrorDetails{.component = kComponent,
                                          .operation = std::string(field),
                                          .expected = "0 or 1",
                                          .actual = std::to_string(value),
                                          .hint = "Use valid current state."});
  }
  return value != 0U;
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
                                         .hint = "Use valid current state."});
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
                     .hint = "Use state produced by a supported schema."});
  }
}

void require_end(std::span<const std::byte> input, std::size_t offset) {
  if (offset != input.size()) {
    throw SerializationError("DataPipeline state contains trailing bytes");
  }
}

std::string encode_topology(const DataPipelineState& state) {
  std::string output;
  append_string(output, kCurrentTopologyMagic);
  append_string(output, state.pipeline_id);
  append_string(output, state.dataset_version);
  append_u64(output, state.dataset_size);
  append_string(output, state.schema_descriptor);
  append_string(output, state.batch_factory_id);
  append_string(output, state.sampler_identity);
  append_u64(output, state.batch_size);
  append_byte(output, state.drop_last ? 1U : 0U);
  append_u64(output, state.num_workers);
  append_u64(output, state.prefetch_batches);
  append_byte(output, state.persistent_workers ? 1U : 0U);
  append_byte(output, state.deterministic_batch_factory ? 1U : 0U);
  append_u64(output, state.worker_seed);
  append_u64(output, state.batch_timeout_ms);
  append_byte(output, state.pin_memory ? 1U : 0U);
  append_string(output, state.target_device);
  append_byte(output, state.non_blocking_transfer ? 1U : 0U);
  append_u64(output, state.memory_stage_capacity);
  append_u64(output, state.max_batch_tensors);
  append_u64(output, state.max_batch_value_nodes);
  append_u64(output, state.max_batch_bytes);
  append_byte(output, state.sampler_state_supported ? 1U : 0U);
  append_string(output, state.sampler_state);
  append_u64(output, static_cast<std::uint64_t>(state.epoch_indices.size()));
  for (const auto index : state.epoch_indices) {
    append_u64(output, static_cast<std::uint64_t>(index));
  }
  append_string(output, state.consumer_transform_id);
  append_string(output, state.consumer_input_schema_descriptor);
  append_byte(output, state.deterministic_consumer_transform ? 1U : 0U);
  return output;
}

std::string encode_cursor(const DataPipelineState& state) {
  std::string output;
  append_string(output, kCurrentCursorMagic);
  append_byte(output, state.prepared ? 1U : 0U);
  append_u64(output, state.epoch);
  append_u64(output, state.next_cursor);
  append_u64(output, state.next_sequence);
  return output;
}

std::string migrate_schema_v3(std::string_view legacy_schema_id,
                              const BatchSchema& current_schema) {
  const auto unconstrained_descriptor =
      canonical_value_schema(ValueSchema::any("batch"));
  if (canonical_value_schema(current_schema.value) !=
      unconstrained_descriptor) {
    throw CompatibilityError(
        "DataPipeline schema-v3 cannot prove a structured batch contract",
        ErrorDetails{
            .component = kComponent,
            .operation = "migration.v3.schema",
            .expected = "an unconstrained schema matching the v3 label",
            .actual = std::string(value_kind_name(current_schema.value.kind)),
            .hint = "Restore v3 with its original unconstrained schema before "
                    "adopting a structured batch contract."});
  }
  const auto expected_legacy_id =
      current_schema.id + "/v" + std::to_string(current_schema.version);
  if (legacy_schema_id != expected_legacy_id) {
    throw CompatibilityError(
        "DataPipeline schema-v3 batch identity is incompatible",
        ErrorDetails{.component = kComponent,
                     .operation = "migration.v3.schema_id",
                     .expected = expected_legacy_id,
                     .actual = std::string(legacy_schema_id),
                     .hint = "Use the BatchSchema identity and version that "
                             "produced the v3 checkpoint."});
  }
  return canonical_batch_schema(current_schema);
}

DataPipelineState decode_state(std::span<const std::byte> topology,
                               std::span<const std::byte> cursor,
                               std::uint32_t schema_version,
                               const BatchSchema& current_schema) {
  const auto current = schema_version == DataPipelineState::format_version;
  DataPipelineState state;
  std::size_t topology_offset = 0;
  require_magic(read_string(topology, topology_offset, 128),
                current ? kCurrentTopologyMagic
                        : (schema_version == kPreviousSchemaVersion
                               ? kPreviousTopologyMagic
                               : kLegacyTopologyMagic));
  state.pipeline_id = read_string(topology, topology_offset, kMaxMetadataBytes);
  state.dataset_version =
      read_string(topology, topology_offset, kMaxMetadataBytes);
  state.dataset_size = read_size(topology, topology_offset, "dataset_size");
  if (schema_version >= kPreviousSchemaVersion) {
    state.schema_descriptor =
        read_string(topology, topology_offset, kMaxSchemaDescriptorBytes);
  } else {
    const auto legacy_schema_id =
        read_string(topology, topology_offset, kMaxMetadataBytes);
    state.schema_descriptor =
        migrate_schema_v3(legacy_schema_id, current_schema);
  }
  state.batch_factory_id =
      read_string(topology, topology_offset, kMaxMetadataBytes);
  state.sampler_identity =
      read_string(topology, topology_offset, kMaxSamplerIdentityBytes);
  state.batch_size = read_size(topology, topology_offset, "batch_size");
  state.drop_last = read_bool(topology, topology_offset, "drop_last");
  state.num_workers = read_size(topology, topology_offset, "num_workers");
  state.prefetch_batches =
      read_size(topology, topology_offset, "prefetch_batches");
  state.persistent_workers =
      read_bool(topology, topology_offset, "persistent_workers");
  state.deterministic_batch_factory =
      read_bool(topology, topology_offset, "deterministic_batch_factory");
  state.worker_seed = read_u64(topology, topology_offset);
  state.batch_timeout_ms = read_u64(topology, topology_offset);
  state.pin_memory = read_bool(topology, topology_offset, "pin_memory");
  state.target_device =
      read_string(topology, topology_offset, kMaxMetadataBytes);
  state.non_blocking_transfer =
      read_bool(topology, topology_offset, "non_blocking_transfer");
  state.memory_stage_capacity =
      read_size(topology, topology_offset, "memory_stage_capacity");
  state.max_batch_tensors =
      read_size(topology, topology_offset, "max_batch_tensors");
  state.max_batch_value_nodes =
      read_size(topology, topology_offset, "max_batch_value_nodes");
  state.max_batch_bytes = read_u64(topology, topology_offset);
  state.sampler_state_supported =
      read_bool(topology, topology_offset, "sampler_state_supported");
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
    state.epoch_indices.push_back(
        read_size(topology, topology_offset, "sample_index"));
  }
  if (current) {
    state.consumer_transform_id =
        read_string(topology, topology_offset, kMaxMetadataBytes);
    state.consumer_input_schema_descriptor =
        read_string(topology, topology_offset, kMaxSchemaDescriptorBytes);
    state.deterministic_consumer_transform = read_bool(
        topology, topology_offset, "deterministic_consumer_transform");
  }
  require_end(topology, topology_offset);

  std::size_t cursor_offset = 0;
  require_magic(read_string(cursor, cursor_offset, 128),
                current ? kCurrentCursorMagic
                        : (schema_version == kPreviousSchemaVersion
                               ? kPreviousCursorMagic
                               : kLegacyCursorMagic));
  state.prepared = read_bool(cursor, cursor_offset, "prepared");
  state.epoch = read_u64(cursor, cursor_offset);
  state.next_cursor = read_u64(cursor, cursor_offset);
  state.next_sequence = read_u64(cursor, cursor_offset);
  require_end(cursor, cursor_offset);
  state.version = DataPipelineState::format_version;
  return state;
}

bool manifest_equal(const StateManifest& left, const StateManifest& right) {
  return left.participant_id == right.participant_id &&
         left.schema_version == right.schema_version &&
         left.restore_level == right.restore_level &&
         std::is_permutation(left.degradation_reasons.begin(),
                             left.degradation_reasons.end(),
                             right.degradation_reasons.begin(),
                             right.degradation_reasons.end()) &&
         state_fields_equal(left.fields, right.fields);
}

StateManifest make_older_manifest(const StateManifest& current,
                                  std::uint32_t version) {
  StateManifest previous = current;
  previous.schema_version = version;
  validate_state_manifest(previous);
  return previous;
}

void write_blob(StateWriter& writer, std::string_view name,
                const std::string& value, std::uint64_t max_bytes) {
  if (value.size() > max_bytes) {
    throw ResourceExhausted("DataPipeline state exceeds its declared bound");
  }
  writer.write_blob(name, std::as_bytes(std::span(value.data(), value.size())));
}

std::uint64_t fnv1a(std::string_view value) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const auto character : value) {
    hash ^= static_cast<unsigned char>(character);
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::uint64_t mix64(std::uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

std::uint64_t derive_worker_seed(const DataPipelineOptions& options,
                                 std::uint64_t epoch, std::uint32_t worker_id,
                                 std::uint64_t sequence) noexcept {
  auto seed = mix64(options.worker_seed ^ fnv1a(options.dataset_version));
  seed = mix64(seed ^ epoch);
  seed = mix64(seed ^ worker_id);
  return mix64(seed ^ sequence);
}

DataWorkerContext make_worker_context(const DataPipelineOptions& options,
                                      std::uint64_t epoch,
                                      std::uint64_t sequence) noexcept {
  const auto worker_count = static_cast<std::uint32_t>(
      options.num_workers == 0 ? 1U : options.num_workers);
  const auto worker_id = static_cast<std::uint32_t>(sequence % worker_count);
  return {
      .worker_id = worker_id,
      .worker_count = worker_count,
      .shard_sequence = sequence / worker_count,
      .worker_seed = derive_worker_seed(options, epoch, worker_id, sequence),
      .epoch = epoch,
      .sequence = sequence};
}

[[noreturn]] void rethrow_factory_failure(std::string_view operation) {
  try {
    throw;
  } catch (const Error&) {
    throw;
  } catch (const std::exception& cause) {
    std::throw_with_nested(BackendError(
        "DataPipeline batch factory failed",
        ErrorDetails{.component = kComponent,
                     .operation = std::string(operation),
                     .expected = "one complete batch",
                     .actual = cause.what(),
                     .hint = "Inspect the nested factory error."}));
  } catch (...) {
    std::throw_with_nested(BackendError(
        "DataPipeline batch factory failed",
        ErrorDetails{.component = kComponent,
                     .operation = std::string(operation),
                     .expected = "one complete batch",
                     .actual = "non-standard exception",
                     .hint = "Inspect the nested factory error."}));
  }
}

c10::IValue invoke_factory(const DataBatchFactory& factory,
                           std::span<const std::size_t> indices,
                           const DataWorkerContext& context,
                           std::stop_token stop) {
  try {
    return factory(indices, context, stop);
  } catch (...) {
    rethrow_factory_failure("batch_factory");
  }
}

struct BatchWork {
  // Monotonic batch position used to restore consumer-visible ordering.
  std::uint64_t sequence = 0;
  // Cursor value to commit only after this batch is delivered successfully.
  std::size_t end_cursor = 0;
  // Stable sample selection captured when the epoch is prepared.
  std::vector<std::size_t> indices;
  // Deterministic logical-worker metadata passed to the batch factory.
  DataWorkerContext context;
};

struct BatchResult {
  // Sequence copied from the claimed work item.
  std::uint64_t sequence = 0;
  // Cursor boundary validated before the pipeline commits the result.
  std::size_t end_cursor = 0;
  // Materialized batch value when the factory succeeds.
  c10::IValue value;
  // Observable pinning and transfer work performed before publication.
  BatchMemoryMetadata memory;
  // Structured factory failure retained for ordered propagation.
  std::exception_ptr error;
};

struct EpochWork {
  // Cooperative cancellation shared by all factories serving this epoch.
  std::stop_source stop_source;
  // Immutable work plan created from the committed pipeline cursor.
  std::vector<BatchWork> batches;
  // First work-plan entry that has not yet been claimed by a worker.
  std::size_t next_claim = 0;
  // Exclusive scheduling boundary that enforces total prefetch capacity.
  std::size_t allowed = 0;
  // Number of results already delivered and committed by the consumer.
  std::size_t committed = 0;
  // Only this sequence may be observed by the consumer next.
  std::uint64_t next_commit_sequence = 0;
  // Out-of-order completions retained until their sequence becomes visible.
  std::map<std::uint64_t, std::shared_ptr<BatchResult>> results;
  // First cancellation, timeout, or factory failure terminating the epoch.
  std::exception_ptr terminal_error;
};

std::shared_ptr<EpochWork> make_epoch_work(const DataPipelineState& state,
                                           const DataPipelineOptions& options) {
  auto epoch = std::make_shared<EpochWork>();
  epoch->next_commit_sequence = state.next_sequence;
  auto cursor = static_cast<std::size_t>(state.next_cursor);
  auto sequence = state.next_sequence;
  while (cursor < state.epoch_indices.size()) {
    if (sequence == std::numeric_limits<std::uint64_t>::max()) {
      throw ResourceExhausted("DataPipeline batch sequence exhausted");
    }
    const auto end =
        std::min(cursor + options.batch_size, state.epoch_indices.size());
    BatchWork work;
    work.sequence = sequence;
    work.end_cursor = end;
    work.indices.assign(
        state.epoch_indices.begin() + static_cast<std::ptrdiff_t>(cursor),
        state.epoch_indices.begin() + static_cast<std::ptrdiff_t>(end));
    work.context = make_worker_context(options, state.epoch, sequence);
    epoch->batches.push_back(std::move(work));
    cursor = end;
    ++sequence;
  }
  auto capacity = options.prefetch_batches;
  if (batch_memory_stage_enabled(options.memory)) {
    capacity = std::min(capacity, options.memory.stage_capacity);
  }
  epoch->allowed = std::min(capacity, epoch->batches.size());
  return epoch;
}

class WorkerRuntime final {
 public:
  // Starts a fixed worker pool. Workers remain idle until activate() installs
  // an epoch plan.
  WorkerRuntime(DataBatchFactory factory, BatchMemoryOptions memory,
                std::size_t workers, std::chrono::milliseconds timeout)
      : factory_(std::move(factory)), memory_(memory), timeout_(timeout) {
    workers_.reserve(workers);
    try {
      for (std::size_t index = 0; index < workers; ++index) {
        workers_.emplace_back(
            [this](std::stop_token stop) { worker_loop(stop); });
      }
    } catch (...) {
      close();
      throw;
    }
  }

  WorkerRuntime(const WorkerRuntime&) = delete;
  WorkerRuntime& operator=(const WorkerRuntime&) = delete;

  // Cooperatively closes the pool and joins every worker.
  ~WorkerRuntime() { close(); }

  // Replaces the active epoch after cancelling any prior in-flight work.
  void activate(std::shared_ptr<EpochWork> epoch) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_) {
      active_->stop_source.request_stop();
    }
    active_ = std::move(epoch);
    condition_.notify_all();
  }

  // Waits for the next consumer-visible sequence while retaining later
  // completions in the reorder buffer.
  std::shared_ptr<const BatchResult> wait_for(std::uint64_t sequence,
                                              std::stop_token stop) {
    std::stop_callback notify_stop(stop, [this] { cancel_active(); });
    std::unique_lock<std::mutex> lock(mutex_);
    const auto deadline = std::chrono::steady_clock::now() + timeout_;
    while (true) {
      if (stop.stop_requested()) {
        return {};
      }
      if (closing_) {
        throw NotReady("DataPipeline worker runtime is closed");
      }
      if (!active_ || sequence != active_->next_commit_sequence) {
        throw InternalError("DataPipeline worker sequence is inconsistent");
      }
      if (active_->terminal_error) {
        std::rethrow_exception(active_->terminal_error);
      }
      const auto found = active_->results.find(sequence);
      if (found != active_->results.end()) {
        const auto result = found->second;
        if (result->error) {
          active_->terminal_error = result->error;
          active_->stop_source.request_stop();
          condition_.notify_all();
          std::rethrow_exception(result->error);
        }
        return result;
      }
      if (timeout_.count() == 0) {
        condition_.wait(lock);
        continue;
      }
      if (condition_.wait_until(lock, deadline) == std::cv_status::timeout) {
        if (active_->results.contains(sequence)) {
          continue;
        }
        active_->terminal_error = std::make_exception_ptr(DeadlineExceeded(
            "DataPipeline batch wait exceeded its deadline",
            ErrorDetails{
                .component = kComponent,
                .operation = "next.wait",
                .expected = "completion within " +
                            std::to_string(timeout_.count()) + " ms",
                .actual = "batch sequence " + std::to_string(sequence) +
                          " still pending",
                .hint = "Increase batch_timeout or fix the factory."}));
        active_->stop_source.request_stop();
        condition_.notify_all();
        std::rethrow_exception(active_->terminal_error);
      }
    }
  }

  // Advances the scheduling window only after successful consumer delivery.
  void commit(std::uint64_t sequence) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_ || active_->terminal_error ||
        active_->next_commit_sequence != sequence) {
      return;
    }
    active_->results.erase(sequence);
    ++active_->committed;
    ++active_->next_commit_sequence;
    if (active_->allowed < active_->batches.size()) {
      ++active_->allowed;
    }
    condition_.notify_all();
  }

  // Reports scheduled but uncommitted batches, including completed results.
  std::size_t outstanding_batches() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_ || active_->terminal_error || closing_ ||
        active_->allowed < active_->committed) {
      return 0;
    }
    return active_->allowed - active_->committed;
  }

  // Terminates the active epoch without advancing its committed cursor.
  void cancel_active() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_ || active_->terminal_error) {
      condition_.notify_all();
      return;
    }
    active_->terminal_error = std::make_exception_ptr(Cancelled(
        "DataPipeline epoch was cancelled before the next batch committed"));
    active_->stop_source.request_stop();
    condition_.notify_all();
  }

  // Requests cooperative cancellation and joins every worker exactly once.
  void close() noexcept {
    std::vector<std::jthread> workers;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closing_ && workers_.empty()) {
        return;
      }
      closing_ = true;
      if (active_) {
        active_->stop_source.request_stop();
      }
      for (auto& worker : workers_) {
        worker.request_stop();
      }
      workers.swap(workers_);
      condition_.notify_all();
    }
    workers.clear();
    std::lock_guard<std::mutex> lock(mutex_);
    active_.reset();
  }

 private:
  // Claims eligible work, invokes the factory without holding the runtime
  // mutex, and publishes completion back into the ordered result map.
  void worker_loop(std::stop_token thread_stop) {
    std::stop_callback notify_stop(thread_stop,
                                   [this] { condition_.notify_all(); });
    while (true) {
      std::shared_ptr<EpochWork> epoch;
      const BatchWork* work = nullptr;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] {
          return closing_ || thread_stop.stop_requested() ||
                 (active_ && !active_->terminal_error &&
                  active_->next_claim < active_->allowed);
        });
        if (closing_ || thread_stop.stop_requested()) {
          return;
        }
        epoch = active_;
        work = &epoch->batches[epoch->next_claim++];
      }

      auto result = std::make_shared<BatchResult>();
      result->sequence = work->sequence;
      result->end_cursor = work->end_cursor;
      try {
        if (epoch->stop_source.stop_requested()) {
          continue;
        }
        result->value = invoke_factory(factory_, work->indices, work->context,
                                       epoch->stop_source.get_token());
        if (batch_memory_stage_enabled(memory_) &&
            !epoch->stop_source.stop_requested()) {
          auto staged = prepare_batch_memory(result->value, memory_);
          result->value = std::move(staged.value);
          result->memory = staged.metadata;
        }
      } catch (...) {
        result->error = std::current_exception();
      }

      std::lock_guard<std::mutex> lock(mutex_);
      if (!closing_ && active_ == epoch && !epoch->terminal_error &&
          !epoch->stop_source.stop_requested()) {
        epoch->results.emplace(result->sequence, std::move(result));
        condition_.notify_all();
      }
    }
  }

  // User factory copied into the runtime so persistent workers can reuse it.
  DataBatchFactory factory_;
  // Immutable memory-stage policy shared by every worker invocation.
  BatchMemoryOptions memory_;
  // Zero means an unbounded wait; positive values apply per next() call.
  std::chrono::milliseconds timeout_;
  // Guards all fields below and every mutable EpochWork scheduling field.
  mutable std::mutex mutex_;
  // Wakes workers for scheduling and consumers for completion or shutdown.
  std::condition_variable condition_;
  // Join-on-destruction worker threads owned exclusively by this runtime.
  std::vector<std::jthread> workers_;
  // Current epoch plan; null while persistent workers are idle.
  std::shared_ptr<EpochWork> active_;
  // Prevents new work from being claimed once shutdown begins.
  bool closing_ = false;
};

void validate_options(const DataPipelineOptions& options) {
  validate_batch_memory_options(options.memory);
  validate_batch_schema(options.schema);
  detail::validate_consumer_transform(options.consumer_transform);
  if (options.pipeline_id.empty() || options.dataset_version.empty() ||
      options.batch_factory_id.empty()) {
    throw InvalidArgument(
        "DataPipeline identity, dataset, schema, and factory IDs are required");
  }
  if (options.pipeline_id.size() > kMaxMetadataBytes ||
      options.dataset_version.size() > kMaxMetadataBytes ||
      options.batch_factory_id.size() > kMaxMetadataBytes) {
    throw ResourceExhausted(
        "DataPipeline identity metadata exceeds its declared bound");
  }
  if (options.dataset_size == 0) {
    throw InvalidArgument("DataPipeline dataset_size must be positive");
  }
  if (options.batch_size == 0) {
    throw InvalidArgument("DataPipeline batch_size must be positive");
  }
  if (options.num_workers > kMaxWorkerCount) {
    throw ResourceExhausted("DataPipeline worker count exceeds its bound");
  }
  if (options.prefetch_batches == 0 ||
      options.prefetch_batches > kMaxPrefetchBatches) {
    throw InvalidArgument(
        "DataPipeline prefetch_batches must be within the supported bound");
  }
  if (options.num_workers == 0 && options.persistent_workers) {
    throw InvalidArgument(
        "DataPipeline persistent_workers requires background workers");
  }
  if (options.batch_timeout.count() < 0 ||
      static_cast<std::uint64_t>(options.batch_timeout.count()) >
          kMaxBatchTimeoutMs) {
    throw InvalidArgument("DataPipeline batch_timeout is out of range");
  }
  if (options.num_workers == 0 && options.batch_timeout.count() != 0) {
    throw InvalidArgument(
        "DataPipeline batch_timeout requires background workers");
  }
}

DataPipelineState make_initial_state(const DataPipelineOptions& options,
                                     const samplers::Sampler& sampler) {
  DataPipelineState state;
  state.pipeline_id = options.pipeline_id;
  state.dataset_version = options.dataset_version;
  state.dataset_size = options.dataset_size;
  state.schema_descriptor = canonical_batch_schema(options.schema);
  state.batch_factory_id = options.batch_factory_id;
  if (options.consumer_transform) {
    state.consumer_transform_id = options.consumer_transform->id;
    state.consumer_input_schema_descriptor =
        canonical_batch_schema(options.consumer_transform->input_schema);
    state.deterministic_consumer_transform =
        options.consumer_transform->deterministic;
  }
  state.batch_size = options.batch_size;
  state.drop_last = options.drop_last;
  state.num_workers = options.num_workers;
  state.prefetch_batches = options.prefetch_batches;
  state.persistent_workers = options.persistent_workers;
  state.deterministic_batch_factory = options.deterministic_batch_factory;
  state.worker_seed = options.worker_seed;
  state.batch_timeout_ms =
      static_cast<std::uint64_t>(options.batch_timeout.count());
  state.pin_memory = options.memory.pin_memory;
  state.target_device = options.memory.target_device
                            ? options.memory.target_device->str()
                            : std::string{};
  state.non_blocking_transfer = options.memory.non_blocking;
  state.memory_stage_capacity = options.memory.stage_capacity;
  state.max_batch_tensors = options.memory.max_tensors;
  state.max_batch_value_nodes = options.memory.max_value_nodes;
  state.max_batch_bytes = options.memory.max_bytes;
  state.sampler_state_supported = sampler.supports_state();
  state.sampler_identity = sampler.configuration_identity();
  if (state.sampler_state_supported) {
    state.sampler_state = sampler.save_state();
  }
  return state;
}

}  // namespace

struct DataPipeline::Impl {
  std::shared_ptr<samplers::Sampler> sampler;
  DataPipelineOptions options;
  DataBatchFactory batch_factory;
  DataPipelineState state;
  std::unique_ptr<WorkerRuntime> runtime;
  bool closed = false;
  std::exception_ptr delivery_failure;

  void fail_delivery(std::exception_ptr error) noexcept {
    delivery_failure = std::move(error);
    if (runtime) {
      runtime->close();
      runtime.reset();
    }
  }

  DataCapability capability() const {
    DataCapability result;
    result.pipeline_id = options.pipeline_id;
    result.finite_epoch = true;
    result.deterministic_epoch = sampler->supports_epoch_control();
    result.worker_model = options.num_workers == 0
                              ? DataWorkerModel::inline_execution
                              : DataWorkerModel::thread_pool;
    result.worker_shard_policy = options.num_workers == 0
                                     ? DataWorkerShardPolicy::single_shard
                                     : DataWorkerShardPolicy::round_robin_batch;
    result.worker_prefetch = options.num_workers != 0;
    result.worker_count = options.num_workers;
    result.prefetch_batches =
        options.num_workers == 0 ? 0 : options.prefetch_batches;
    result.persistent_workers = options.persistent_workers;
    result.batch_timeout = options.batch_timeout.count() != 0;
    result.memory_stage = batch_memory_stage_enabled(options.memory);
    result.pin_memory = options.memory.pin_memory;
    result.non_blocking_transfer = options.memory.non_blocking;
    result.transfer_stream_policy = BatchTransferStreamPolicy::caller_current;
    result.target_device = options.memory.target_device;
    result.memory_stage_capacity =
        result.memory_stage ? options.memory.stage_capacity : 0U;
    result.max_batch_tensors = options.memory.max_tensors;
    result.max_batch_value_nodes = options.memory.max_value_nodes;
    result.max_batch_bytes = options.memory.max_bytes;
    result.batch_schema = options.schema;
    result.batch_schema_fingerprint = batch_schema_fingerprint(options.schema);
    auto replay_capacity = options.prefetch_batches;
    if (result.memory_stage) {
      replay_capacity =
          std::min(replay_capacity, options.memory.stage_capacity);
    }
    result.max_replayed_batches =
        options.num_workers == 0 ? 0 : replay_capacity;

    const auto has_state = sampler->supports_state();
    const auto has_identity = !sampler->configuration_identity().empty();
    result.consumer_transform_id = state.consumer_transform_id;
    result.deterministic_consumer_transform =
        state.deterministic_consumer_transform;
    if (options.consumer_transform)
      result.consumer_input_schema = options.consumer_transform->input_schema;
    const auto replayable = options.deterministic_batch_factory &&
                            result.deterministic_consumer_transform;
    result.restore_level = has_state && has_identity && replayable
                               ? RestoreLevel::exact
                               : RestoreLevel::bounded;
    if (!has_state) {
      result.degradation_reasons.emplace_back(
          "sampler does not expose exact state");
    }
    if (!has_identity) {
      result.degradation_reasons.emplace_back(
          "sampler does not expose a stable configuration identity");
    }
    if (!options.deterministic_batch_factory) {
      result.degradation_reasons.emplace_back(
          "batch factory does not guarantee deterministic replay");
    }
    if (!result.deterministic_consumer_transform) {
      result.degradation_reasons.emplace_back(
          "consumer transform does not guarantee deterministic replay");
    }
    return result;
  }

  void validate_state(const DataPipelineState& candidate) const {
    if (candidate.version != DataPipelineState::format_version ||
        candidate.pipeline_id != options.pipeline_id ||
        candidate.dataset_version != options.dataset_version ||
        candidate.dataset_size != options.dataset_size ||
        candidate.schema_descriptor != canonical_batch_schema(options.schema) ||
        candidate.batch_factory_id != options.batch_factory_id ||
        candidate.consumer_transform_id != (options.consumer_transform
                                                ? options.consumer_transform->id
                                                : std::string{}) ||
        candidate.consumer_input_schema_descriptor !=
            (options.consumer_transform
                 ? canonical_batch_schema(
                       options.consumer_transform->input_schema)
                 : std::string{}) ||
        candidate.deterministic_consumer_transform !=
            (!options.consumer_transform ||
             options.consumer_transform->deterministic) ||
        candidate.sampler_identity != sampler->configuration_identity() ||
        candidate.batch_size != options.batch_size ||
        candidate.drop_last != options.drop_last ||
        candidate.num_workers != options.num_workers ||
        candidate.prefetch_batches != options.prefetch_batches ||
        candidate.persistent_workers != options.persistent_workers ||
        candidate.deterministic_batch_factory !=
            options.deterministic_batch_factory ||
        candidate.worker_seed != options.worker_seed ||
        candidate.batch_timeout_ms !=
            static_cast<std::uint64_t>(options.batch_timeout.count()) ||
        candidate.pin_memory != options.memory.pin_memory ||
        candidate.target_device != (options.memory.target_device
                                        ? options.memory.target_device->str()
                                        : std::string{}) ||
        candidate.non_blocking_transfer != options.memory.non_blocking ||
        candidate.memory_stage_capacity != options.memory.stage_capacity ||
        candidate.max_batch_tensors != options.memory.max_tensors ||
        candidate.max_batch_value_nodes != options.memory.max_value_nodes ||
        candidate.max_batch_bytes != options.memory.max_bytes) {
      throw CompatibilityError("DataPipeline state topology is incompatible");
    }
    if (candidate.sampler_state_supported != sampler->supports_state()) {
      throw CompatibilityError("DataPipeline sampler state capability changed");
    }
    if (!candidate.prepared) {
      if (candidate.next_cursor != 0 || candidate.next_sequence != 0 ||
          !candidate.epoch_indices.empty()) {
        throw SerializationError(
            "Unprepared DataPipeline state contains cursor or indices");
      }
      return;
    }
    const auto expected = sampler->size();
    const auto expected_indices =
        candidate.drop_last
            ? (expected / candidate.batch_size) * candidate.batch_size
            : expected;
    if (candidate.epoch_indices.size() != expected_indices ||
        candidate.next_cursor > candidate.epoch_indices.size()) {
      throw CompatibilityError("DataPipeline state has an invalid epoch shape");
    }
    for (const auto index : candidate.epoch_indices) {
      if (index >= candidate.dataset_size) {
        throw SerializationError(
            "DataPipeline state contains an out-of-range sample index");
      }
    }
    const auto full_boundary =
        candidate.next_cursor % candidate.batch_size == 0;
    const auto partial_final =
        candidate.next_cursor == candidate.epoch_indices.size();
    if (!full_boundary && !partial_final) {
      throw SerializationError(
          "DataPipeline cursor is not at a batch boundary");
    }
    const auto expected_sequence =
        candidate.next_cursor / candidate.batch_size +
        (full_boundary ? 0U : 1U);
    if (candidate.next_sequence != expected_sequence) {
      throw SerializationError(
          "DataPipeline cursor is not at a batch boundary");
    }
  }

  std::unique_ptr<WorkerRuntime> make_runtime() const {
    return std::make_unique<WorkerRuntime>(batch_factory, options.memory,
                                           options.num_workers,
                                           options.batch_timeout);
  }

  void install_epoch(DataPipelineState candidate,
                     std::unique_ptr<WorkerRuntime> replacement,
                     std::shared_ptr<EpochWork> epoch) noexcept {
    state = std::move(candidate);
    delivery_failure = nullptr;
    if (replacement) {
      runtime = std::move(replacement);
    }
    if (runtime) {
      runtime->activate(std::move(epoch));
    }
  }
};

DataPipeline::DataPipeline(std::shared_ptr<samplers::Sampler> sampler,
                           DataPipelineOptions options,
                           DataBatchFactory batch_factory) {
  if (!sampler) {
    throw InvalidArgument("DataPipeline sampler cannot be null");
  }
  if (!batch_factory) {
    throw InvalidArgument("DataPipeline batch factory cannot be empty");
  }
  validate_options(options);
  if (sampler->size() == 0) {
    throw InvalidArgument("DataPipeline sampler must produce samples");
  }
  if (sampler->size() > kMaxIndexCount) {
    throw ResourceExhausted(
        "DataPipeline sampler exceeds the checkpointable index bound");
  }
  const auto sampler_identity = sampler->configuration_identity();
  if (sampler_identity.size() > kMaxSamplerIdentityBytes) {
    throw ResourceExhausted(
        "DataPipeline sampler identity exceeds its declared bound");
  }
  auto state = make_initial_state(options, *sampler);
  if (state.sampler_state.size() > kMaxSamplerStateBytes) {
    throw ResourceExhausted(
        "DataPipeline sampler state exceeds its declared bound");
  }
  impl_ = std::make_unique<Impl>(Impl{.sampler = std::move(sampler),
                                      .options = std::move(options),
                                      .batch_factory = std::move(batch_factory),
                                      .state = std::move(state),
                                      .runtime = nullptr,
                                      .closed = false});
  validate_checkpoint_participant(*this);
}

DataPipeline::~DataPipeline() { close(); }

void DataPipeline::set_epoch(std::uint64_t epoch) {
  if (impl_->closed) {
    throw NotReady("DataPipeline is closed");
  }
  const auto supports_state = impl_->sampler->supports_state();
  const auto sampler_identity = impl_->sampler->configuration_identity();
  if (sampler_identity != impl_->state.sampler_identity ||
      supports_state != impl_->state.sampler_state_supported) {
    throw CompatibilityError(
        "DataPipeline sampler contract changed after construction");
  }
  const auto previous_sampler_state =
      supports_state ? impl_->sampler->save_state() : std::string{};
  try {
    impl_->sampler->set_epoch(epoch);
    auto indices = impl_->sampler->indices();
    const auto expected = impl_->sampler->size();
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
      if (index >= impl_->options.dataset_size) {
        throw InvalidArgument(
            "DataPipeline sampler returned an out-of-range sample index",
            ErrorDetails{
                .component = kComponent,
                .operation = "set_epoch",
                .expected =
                    "index < " + std::to_string(impl_->options.dataset_size),
                .actual = std::to_string(index),
                .hint = "Fix the sampler dataset topology."});
      }
    }
    if (impl_->options.drop_last) {
      indices.resize((indices.size() / impl_->options.batch_size) *
                     impl_->options.batch_size);
    }

    auto candidate = make_initial_state(impl_->options, *impl_->sampler);
    candidate.epoch = epoch;
    candidate.prepared = true;
    candidate.epoch_indices = std::move(indices);
    if (candidate.sampler_state.size() > kMaxSamplerStateBytes) {
      throw ResourceExhausted(
          "DataPipeline sampler state exceeds its declared bound");
    }

    std::unique_ptr<WorkerRuntime> replacement;
    std::shared_ptr<EpochWork> epoch_work;
    if (impl_->options.num_workers != 0) {
      epoch_work = make_epoch_work(candidate, impl_->options);
      if (!impl_->options.persistent_workers || !impl_->runtime) {
        replacement = impl_->make_runtime();
      }
    }
    impl_->install_epoch(std::move(candidate), std::move(replacement),
                         std::move(epoch_work));
  } catch (...) {
    const auto cause = std::current_exception();
    if (supports_state) {
      try {
        impl_->sampler->load_state(previous_sampler_state);
      } catch (...) {
        std::throw_with_nested(
            InternalError("DataPipeline epoch preparation rollback failed"));
      }
    }
    std::rethrow_exception(cause);
  }
}

std::optional<BatchEnvelope> DataPipeline::next(std::stop_token stop_token) {
  if (impl_->closed) throw NotReady("DataPipeline is closed");
  if (impl_->delivery_failure) std::rethrow_exception(impl_->delivery_failure);
  if (!impl_->state.prepared) {
    throw InvalidArgument("DataPipeline requires set_epoch before next");
  }
  if (stop_token.stop_requested()) {
    impl_->fail_delivery(std::make_exception_ptr(
        Cancelled("DataPipeline epoch cancelled before commit")));
    return std::nullopt;
  }
  if (impl_->state.next_cursor >= impl_->state.epoch_indices.size())
    return std::nullopt;

  try {
    const auto start = static_cast<std::size_t>(impl_->state.next_cursor);
    const auto end = std::min(start + impl_->options.batch_size,
                              impl_->state.epoch_indices.size());
    const auto sequence = impl_->state.next_sequence;
    if (sequence == std::numeric_limits<std::uint64_t>::max()) {
      throw ResourceExhausted("DataPipeline batch sequence exhausted");
    }
    BatchEnvelope envelope;
    envelope.schema = impl_->options.schema;
    envelope.dataset_version = impl_->options.dataset_version;
    envelope.epoch = impl_->state.epoch;
    envelope.sequence = sequence;
    envelope.sample_ids.reserve(end - start);
    for (auto cursor = start; cursor < end; ++cursor) {
      envelope.sample_ids.push_back(
          static_cast<std::uint64_t>(impl_->state.epoch_indices[cursor]));
    }
    if (impl_->options.num_workers == 0) {
      const std::span<const std::size_t> indices(
          impl_->state.epoch_indices.data() + start, end - start);
      const auto context =
          make_worker_context(impl_->options, impl_->state.epoch, sequence);
      envelope.value =
          invoke_factory(impl_->batch_factory, indices, context, stop_token);
      if (!stop_token.stop_requested() &&
          batch_memory_stage_enabled(impl_->options.memory)) {
        auto staged = stage_batch_memory(envelope.value, impl_->options.memory);
        envelope.value = std::move(staged.value);
        envelope.memory = staged.metadata;
      }
    } else {
      const auto result = impl_->runtime->wait_for(sequence, stop_token);
      if (!result) {
        impl_->fail_delivery(std::make_exception_ptr(
            Cancelled("DataPipeline epoch cancelled before commit")));
        return std::nullopt;
      }
      if (result->sequence != sequence || result->end_cursor != end) {
        throw InternalError(
            "DataPipeline worker result topology is inconsistent");
      }
      envelope.value = result->value;
      envelope.memory = result->memory;
      if (!stop_token.stop_requested() &&
          batch_memory_stage_enabled(impl_->options.memory)) {
        auto staged = transfer_batch_memory(
            {.value = std::move(envelope.value), .metadata = envelope.memory},
            impl_->options.memory);
        envelope.value = std::move(staged.value);
        envelope.memory = staged.metadata;
      }
    }
    if (stop_token.stop_requested())
      throw Cancelled("DataPipeline epoch cancelled before commit");
    if (impl_->options.consumer_transform) {
      envelope.consumer_transform_id = impl_->options.consumer_transform->id;
      envelope.transformed_memory = detail::run_consumer_transform(
          envelope.value, *impl_->options.consumer_transform,
          {.dataset_version = envelope.dataset_version,
           .epoch = envelope.epoch,
           .sequence = sequence,
           .sample_ids = envelope.sample_ids},
          impl_->options.memory, stop_token);
    }
    validate_batch_value(envelope.value, impl_->options.schema);
    if (stop_token.stop_requested())
      throw Cancelled("DataPipeline epoch cancelled before commit");

    impl_->state.next_cursor = end;
    ++impl_->state.next_sequence;
    if (impl_->runtime) {
      impl_->runtime->commit(sequence);
      if (end == impl_->state.epoch_indices.size() &&
          !impl_->options.persistent_workers) {
        impl_->runtime->close();
        impl_->runtime.reset();
      }
    }
    return envelope;
  } catch (...) {
    const auto error = std::current_exception();
    impl_->fail_delivery(error);
    if (stop_token.stop_requested()) return std::nullopt;
    std::rethrow_exception(error);
  }
}

void DataPipeline::close() noexcept {
  if (!impl_ || impl_->closed) {
    return;
  }
  impl_->closed = true;
  if (impl_->runtime) {
    impl_->runtime->close();
    impl_->runtime.reset();
  }
}

bool DataPipeline::closed() const noexcept { return impl_->closed; }

std::size_t DataPipeline::outstanding_batches() const noexcept {
  return impl_->runtime ? impl_->runtime->outstanding_batches() : 0U;
}

const DataPipelineOptions& DataPipeline::options() const noexcept {
  return impl_->options;
}

DataCapability DataPipeline::capability() const { return impl_->capability(); }

DataPipelineState DataPipeline::state() const { return impl_->state; }

std::string DataPipeline::participant_id() const {
  return impl_->options.pipeline_id;
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
  if (impl_->closed) {
    throw NotReady("Closed DataPipeline state cannot be saved");
  }
  const auto snapshot = impl_->state;
  write_blob(writer, "topology", encode_topology(snapshot), kMaxTopologyBytes);
  write_blob(writer, "cursor", encode_cursor(snapshot), kMaxCursorBytes);
}

void DataPipeline::load_state(const StateReader& reader, RestorePolicy policy) {
  if (impl_->closed) {
    throw NotReady("Closed DataPipeline state cannot be restored");
  }
  const auto& manifest = reader.manifest();
  validate_state_manifest(manifest);
  if (!restore_policy_accepts(policy, manifest.restore_level)) {
    throw CompatibilityError(
        "DataPipeline checkpoint restore policy is incompatible",
        ErrorDetails{.component = kComponent,
                     .operation = "restore.policy",
                     .expected = "a policy accepting the stored restore level",
                     .actual = "rejected",
                     .hint = "Use a RestorePolicy compatible with the stored "
                             "DataPipeline guarantee."});
  }

  const auto current_manifest = describe_state();
  StateManifest expected_manifest;
  if (manifest.schema_version == DataPipelineState::format_version) {
    expected_manifest = current_manifest;
  } else if (manifest.schema_version == kPreviousSchemaVersion ||
             manifest.schema_version == kPreviousSchemaVersion - 1U) {
    if (impl_->options.consumer_transform)
      throw CompatibilityError(
          "Older DataPipeline state has no consumer transform identity");
    expected_manifest =
        make_older_manifest(current_manifest, manifest.schema_version);
  } else {
    throw CompatibilityError(
        "DataPipeline checkpoint schema is unsupported",
        ErrorDetails{
            .component = kComponent,
            .operation = "restore.schema_version",
            .expected = std::to_string(kPreviousSchemaVersion - 1U) + ", " +
                        std::to_string(kPreviousSchemaVersion - 1U) + ", " +
                        std::to_string(kPreviousSchemaVersion) + " or " +
                        std::to_string(DataPipelineState::format_version),
            .actual = std::to_string(manifest.schema_version),
            .hint = "Use a supported checkpoint and its explicit migration "
                    "chain."});
  }
  if (!manifest_equal(manifest, expected_manifest)) {
    throw CompatibilityError(
        "DataPipeline checkpoint manifest is incompatible",
        ErrorDetails{.component = kComponent,
                     .operation = "restore.manifest",
                     .expected = "the exact DataPipeline schema-v" +
                                 std::to_string(manifest.schema_version) +
                                 " manifest",
                     .actual = manifest.participant_id,
                     .hint = "Do not relabel fields from another participant "
                             "or schema."});
  }
  for (const auto& field : expected_manifest.fields) {
    if (field.required && !reader.contains(field.name)) {
      throw SerializationError(
          "DataPipeline checkpoint is missing a required field",
          ErrorDetails{
              .component = kComponent,
              .operation = "restore.field",
              .expected = field.name,
              .actual = "missing",
              .hint = "Provide every field declared by the manifest."});
    }
  }
  const auto topology = reader.read_blob("topology", kMaxTopologyBytes);
  const auto cursor = reader.read_blob("cursor", kMaxCursorBytes);
  auto decoded = decode_state(topology, cursor, manifest.schema_version,
                              impl_->options.schema);
  impl_->validate_state(decoded);

  std::unique_ptr<WorkerRuntime> replacement;
  std::shared_ptr<EpochWork> epoch_work;
  if (impl_->options.num_workers != 0 && decoded.prepared &&
      decoded.next_cursor < decoded.epoch_indices.size()) {
    epoch_work = make_epoch_work(decoded, impl_->options);
    if (!impl_->options.persistent_workers || !impl_->runtime) {
      replacement = impl_->make_runtime();
    }
  }

  const auto supports_state = impl_->sampler->supports_state();
  const auto previous_sampler_state =
      supports_state ? impl_->sampler->save_state() : std::string{};
  try {
    if (supports_state) {
      impl_->sampler->load_state(decoded.sampler_state);
    }
  } catch (...) {
    const auto cause = std::current_exception();
    try {
      if (supports_state) {
        impl_->sampler->load_state(previous_sampler_state);
      }
    } catch (...) {
      std::throw_with_nested(
          InternalError("DataPipeline restore rollback failed"));
    }
    std::rethrow_exception(cause);
  }

  impl_->state = std::move(decoded);
  impl_->delivery_failure = nullptr;
  if (replacement) {
    impl_->runtime = std::move(replacement);
  }
  if (impl_->runtime) {
    impl_->runtime->activate(std::move(epoch_work));
    if (!impl_->state.prepared ||
        impl_->state.next_cursor >= impl_->state.epoch_indices.size()) {
      if (!impl_->options.persistent_workers) {
        impl_->runtime->close();
        impl_->runtime.reset();
      }
    }
  }
}

}  // namespace jyppx::liuding::data
