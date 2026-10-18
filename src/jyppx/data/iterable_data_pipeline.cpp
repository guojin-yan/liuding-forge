// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/data/iterable_data_pipeline.hpp>
#include <jyppx/foundation/error.hpp>

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

namespace jyppx::data {
namespace {

constexpr char kComponent[] = "data.iterable_pipeline";
constexpr char kTopologyMagic[] = "JYPPX-ITERABLE-PIPELINE-TOPOLOGY/v1";
constexpr char kCursorMagic[] = "JYPPX-ITERABLE-PIPELINE-CURSOR/v1";
constexpr std::uint64_t kMaxTopologyBytes = 64U * 1024U;
constexpr std::uint64_t kMaxCursorBytes = 8U * 1024U * 1024U;
constexpr std::uint64_t kMaxMetadataBytes = 4096U;
constexpr std::uint64_t kMaxSourceStateBytes = 8U * 1024U * 1024U;
constexpr std::size_t kMaxBatchSize = 1024U * 1024U;
constexpr std::size_t kMaxPrefetchBatches = 4096U;
constexpr std::uint64_t kMaxBatchTimeoutMs = 24ULL * 60ULL * 60ULL * 1000ULL;

static_assert(std::is_nothrow_move_assignable_v<IterableDataPipelineState>);

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

void append_optional_u64(std::string& output,
                         const std::optional<std::uint64_t>& value) {
  append_byte(output, value.has_value() ? 1U : 0U);
  if (value) {
    append_u64(output, *value);
  }
}

std::uint8_t read_byte(std::span<const std::byte> input, std::size_t& offset) {
  if (offset >= input.size()) {
    throw SerializationError(
        "IterableDataPipeline state ended while reading a byte");
  }
  return std::to_integer<std::uint8_t>(input[offset++]);
}

std::uint64_t read_u64(std::span<const std::byte> input, std::size_t& offset) {
  if (input.size() - offset < sizeof(std::uint64_t)) {
    throw SerializationError(
        "IterableDataPipeline state ended while reading an integer");
  }
  std::uint64_t value = 0;
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    value |= static_cast<std::uint64_t>(read_byte(input, offset)) << shift;
  }
  return value;
}

bool read_bool(std::span<const std::byte> input, std::size_t& offset,
               std::string_view field) {
  const auto value = read_byte(input, offset);
  if (value > 1U) {
    throw SerializationError(
        "IterableDataPipeline state contains an invalid boolean",
        ErrorDetails{.component = kComponent,
                     .operation = std::string(field),
                     .expected = "0 or 1",
                     .actual = std::to_string(value),
                     .hint = "Use valid current state."});
  }
  return value != 0U;
}

std::size_t read_size(std::span<const std::byte> input, std::size_t& offset,
                      std::string_view field) {
  const auto value = read_u64(input, offset);
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw SerializationError("IterableDataPipeline size field is out of range",
                             ErrorDetails{.component = kComponent,
                                          .operation = std::string(field),
                                          .expected = "size_t range",
                                          .actual = std::to_string(value),
                                          .hint = "Use valid current state."});
  }
  return static_cast<std::size_t>(value);
}

std::string read_string(std::span<const std::byte> input, std::size_t& offset,
                        std::uint64_t max_bytes) {
  const auto size = read_u64(input, offset);
  if (size > max_bytes) {
    throw ResourceExhausted(
        "IterableDataPipeline state string exceeds its bound");
  }
  if (size > input.size() - offset) {
    throw SerializationError(
        "IterableDataPipeline state ended while reading a string");
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

std::optional<std::uint64_t> read_optional_u64(std::span<const std::byte> input,
                                               std::size_t& offset,
                                               std::string_view field) {
  if (!read_bool(input, offset, field)) {
    return std::nullopt;
  }
  return read_u64(input, offset);
}

RestoreLevel read_restore_level(std::span<const std::byte> input,
                                std::size_t& offset) {
  const auto value = read_byte(input, offset);
  switch (static_cast<RestoreLevel>(value)) {
    case RestoreLevel::exact:
    case RestoreLevel::bounded:
    case RestoreLevel::best_effort:
      return static_cast<RestoreLevel>(value);
  }
  throw SerializationError(
      "IterableDataPipeline state contains an invalid restore level");
}

void require_magic(const std::string& actual, const char* expected) {
  if (actual != expected) {
    throw SerializationError(
        "IterableDataPipeline state has an unknown encoding");
  }
}

void require_end(std::span<const std::byte> input, std::size_t offset) {
  if (offset != input.size()) {
    throw SerializationError(
        "IterableDataPipeline state contains trailing bytes");
  }
}

std::string encode_topology(const IterableDataPipelineState& state) {
  std::string output;
  append_string(output, kTopologyMagic);
  append_string(output, state.pipeline_id);
  append_string(output, state.dataset_version);
  append_string(output, state.schema_id);
  append_string(output, state.batch_factory_id);
  append_string(output, state.source_id);
  append_byte(output, state.finite_epoch ? 1U : 0U);
  append_byte(output, state.deterministic_epoch ? 1U : 0U);
  append_byte(output, state.checkpointable ? 1U : 0U);
  append_byte(output, static_cast<std::uint8_t>(state.source_restore_level));
  append_optional_u64(output, state.max_replayed_samples);
  append_optional_u64(output, state.max_omitted_samples);
  append_u64(output, static_cast<std::uint64_t>(state.batch_size));
  append_byte(output, state.drop_last ? 1U : 0U);
  append_u64(output, static_cast<std::uint64_t>(state.prefetch_batches));
  append_u64(output, state.batch_timeout_ms);
  return output;
}

std::string encode_cursor(const IterableDataPipelineState& state) {
  std::string output;
  append_string(output, kCursorMagic);
  append_byte(output, state.prepared ? 1U : 0U);
  append_byte(output, state.exhausted ? 1U : 0U);
  append_u64(output, state.epoch);
  append_u64(output, state.next_sequence);
  append_u64(output, state.committed_samples);
  append_string(output, state.source_watermark);
  append_string(output, state.source_state);
  return output;
}

IterableDataPipelineState decode_state(std::span<const std::byte> topology,
                                       std::span<const std::byte> cursor) {
  IterableDataPipelineState state;
  std::size_t topology_offset = 0;
  require_magic(read_string(topology, topology_offset, 128), kTopologyMagic);
  state.pipeline_id = read_string(topology, topology_offset, kMaxMetadataBytes);
  state.dataset_version =
      read_string(topology, topology_offset, kMaxMetadataBytes);
  state.schema_id = read_string(topology, topology_offset, kMaxMetadataBytes);
  state.batch_factory_id =
      read_string(topology, topology_offset, kMaxMetadataBytes);
  state.source_id = read_string(topology, topology_offset, kMaxMetadataBytes);
  state.finite_epoch = read_bool(topology, topology_offset, "finite_epoch");
  state.deterministic_epoch =
      read_bool(topology, topology_offset, "deterministic_epoch");
  state.checkpointable = read_bool(topology, topology_offset, "checkpointable");
  state.source_restore_level = read_restore_level(topology, topology_offset);
  state.max_replayed_samples =
      read_optional_u64(topology, topology_offset, "max_replayed_samples");
  state.max_omitted_samples =
      read_optional_u64(topology, topology_offset, "max_omitted_samples");
  state.batch_size = read_size(topology, topology_offset, "batch_size");
  state.drop_last = read_bool(topology, topology_offset, "drop_last");
  state.prefetch_batches =
      read_size(topology, topology_offset, "prefetch_batches");
  state.batch_timeout_ms = read_u64(topology, topology_offset);
  require_end(topology, topology_offset);

  std::size_t cursor_offset = 0;
  require_magic(read_string(cursor, cursor_offset, 128), kCursorMagic);
  state.prepared = read_bool(cursor, cursor_offset, "prepared");
  state.exhausted = read_bool(cursor, cursor_offset, "exhausted");
  state.epoch = read_u64(cursor, cursor_offset);
  state.next_sequence = read_u64(cursor, cursor_offset);
  state.committed_samples = read_u64(cursor, cursor_offset);
  state.source_watermark =
      read_string(cursor, cursor_offset, kMaxMetadataBytes);
  state.source_state = read_string(cursor, cursor_offset, kMaxSourceStateBytes);
  require_end(cursor, cursor_offset);
  state.version = IterableDataPipelineState::format_version;
  return state;
}

void write_blob(StateWriter& writer, std::string_view name,
                const std::string& value, std::uint64_t max_bytes) {
  if (value.size() > max_bytes) {
    throw ResourceExhausted(
        "IterableDataPipeline state exceeds its declared bound");
  }
  writer.write_blob(name, std::as_bytes(std::span(value.data(), value.size())));
}

bool descriptor_equal(const IterableSourceDescriptor& left,
                      const IterableSourceDescriptor& right) {
  return left.source_id == right.source_id &&
         left.finite_epoch == right.finite_epoch &&
         left.deterministic_epoch == right.deterministic_epoch &&
         left.checkpointable == right.checkpointable &&
         left.restore_level == right.restore_level &&
         left.max_replayed_samples == right.max_replayed_samples &&
         left.max_omitted_samples == right.max_omitted_samples &&
         left.degradation_reasons == right.degradation_reasons;
}

bool manifest_equal(const StateManifest& left, const StateManifest& right) {
  if (left.participant_id != right.participant_id ||
      left.schema_version != right.schema_version ||
      left.restore_level != right.restore_level ||
      left.degradation_reasons != right.degradation_reasons ||
      left.fields.size() != right.fields.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.fields.size(); ++index) {
    const auto& left_field = left.fields[index];
    const auto& right_field = right.fields[index];
    if (left_field.name != right_field.name ||
        left_field.media_type != right_field.media_type ||
        left_field.max_bytes != right_field.max_bytes ||
        left_field.required != right_field.required) {
      return false;
    }
  }
  return true;
}

void validate_options(const IterableDataPipelineOptions& options) {
  if (options.pipeline_id.empty() || options.dataset_version.empty() ||
      options.schema_id.empty() || options.batch_factory_id.empty()) {
    throw InvalidArgument(
        "IterableDataPipeline identity, dataset, schema, and factory IDs are "
        "required");
  }
  if (options.pipeline_id.size() > kMaxMetadataBytes ||
      options.dataset_version.size() > kMaxMetadataBytes ||
      options.schema_id.size() > kMaxMetadataBytes ||
      options.batch_factory_id.size() > kMaxMetadataBytes) {
    throw ResourceExhausted(
        "IterableDataPipeline metadata exceeds its declared bound");
  }
  if (options.batch_size == 0 || options.batch_size > kMaxBatchSize) {
    throw InvalidArgument(
        "IterableDataPipeline batch_size is outside the supported bound");
  }
  if (options.prefetch_batches == 0 ||
      options.prefetch_batches > kMaxPrefetchBatches) {
    throw InvalidArgument(
        "IterableDataPipeline prefetch capacity is outside the supported "
        "bound");
  }
  if (options.batch_timeout.count() < 0 ||
      static_cast<std::uint64_t>(options.batch_timeout.count()) >
          kMaxBatchTimeoutMs) {
    throw InvalidArgument(
        "IterableDataPipeline batch_timeout is outside the supported bound");
  }
}

[[noreturn]] void rethrow_user_failure(std::string_view operation,
                                       std::string_view subject) {
  try {
    throw;
  } catch (const Error&) {
    throw;
  } catch (const std::exception& cause) {
    std::throw_with_nested(BackendError(
        "IterableDataPipeline user operation failed",
        ErrorDetails{.component = kComponent,
                     .operation = std::string(operation),
                     .expected = std::string(subject),
                     .actual = cause.what(),
                     .hint = "Inspect the nested source or factory error."}));
  } catch (...) {
    std::throw_with_nested(BackendError(
        "IterableDataPipeline user operation failed",
        ErrorDetails{.component = kComponent,
                     .operation = std::string(operation),
                     .expected = std::string(subject),
                     .actual = "non-standard exception",
                     .hint = "Inspect the source or factory implementation."}));
  }
}

std::unique_ptr<IterableSourceCursor> open_cursor(const IterableSource& source,
                                                  std::uint64_t epoch) {
  try {
    auto cursor = source.open_epoch(epoch);
    if (!cursor) {
      throw InvalidArgument("IterableSource returned a null epoch cursor");
    }
    return cursor;
  } catch (...) {
    rethrow_user_failure("source.open_epoch", "one non-null isolated cursor");
  }
}

std::string save_cursor_state(const IterableSourceCursor& cursor) {
  try {
    auto state = cursor.save_state();
    if (state.size() > kMaxSourceStateBytes) {
      throw ResourceExhausted(
          "IterableSource cursor state exceeds its declared bound");
    }
    return state;
  } catch (...) {
    rethrow_user_failure("source.save_state", "bounded cursor state");
  }
}

void load_cursor_state(IterableSourceCursor& cursor, std::string_view state) {
  try {
    cursor.load_state(state);
  } catch (...) {
    rethrow_user_failure("source.load_state", "compatible cursor state");
  }
}

struct ProducedBatch {
  std::uint64_t sequence = 0;  ///< Ordered consumer-visible batch sequence.
  std::optional<BatchEnvelope> envelope;  ///< Empty for exhaustion marker.
  std::string source_state;       ///< Cursor position after this batch or tail.
  std::string source_watermark;   ///< Last pulled source watermark.
  bool source_exhausted = false;  ///< Whether no later source item exists.
  std::exception_ptr error;       ///< Ordered source or collator failure.
};

class IterableRuntime final {
 public:
  IterableRuntime(std::unique_ptr<IterableSourceCursor> cursor,
                  IterableSourceDescriptor descriptor,
                  IterableDataPipelineOptions options,
                  IterableBatchFactory factory, std::uint64_t epoch,
                  std::uint64_t next_sequence, std::string source_watermark)
      : cursor_(std::move(cursor)),
        descriptor_(std::move(descriptor)),
        options_(std::move(options)),
        factory_(std::move(factory)),
        epoch_(epoch),
        next_sequence_(next_sequence),
        source_watermark_(std::move(source_watermark)),
        producer_([this](std::stop_token stop) { producer_loop(stop); }) {}

  IterableRuntime(const IterableRuntime&) = delete;
  IterableRuntime& operator=(const IterableRuntime&) = delete;

  ~IterableRuntime() { close(); }

  std::shared_ptr<const ProducedBatch> wait_next(std::stop_token stop) {
    std::stop_callback notify_stop(stop, [this] { condition_.notify_all(); });
    std::unique_lock<std::mutex> lock(mutex_);
    const auto deadline =
        std::chrono::steady_clock::now() + options_.batch_timeout;
    while (true) {
      if (stop.stop_requested()) {
        lock.unlock();
        cancel();
        return {};
      }
      if (terminal_error_) {
        std::rethrow_exception(terminal_error_);
      }
      if (closing_) {
        throw NotReady("IterableDataPipeline producer runtime is closed");
      }
      if (!queue_.empty()) {
        const auto result = queue_.front();
        if (result->error) {
          terminal_error_ = result->error;
          epoch_stop_.request_stop();
          condition_.notify_all();
          std::rethrow_exception(result->error);
        }
        return result;
      }
      if (producer_done_) {
        throw InternalError(
            "IterableDataPipeline producer stopped without a terminal record");
      }
      if (options_.batch_timeout.count() == 0) {
        condition_.wait(lock);
        continue;
      }
      if (condition_.wait_until(lock, deadline) == std::cv_status::timeout &&
          queue_.empty()) {
        terminal_error_ = std::make_exception_ptr(DeadlineExceeded(
            "IterableDataPipeline wait exceeded its deadline",
            ErrorDetails{
                .component = kComponent,
                .operation = "next.wait",
                .expected = "completion within " +
                            std::to_string(options_.batch_timeout.count()) +
                            " ms",
                .actual = "next source batch still pending",
                .hint = "Increase batch_timeout or fix the source."}));
        const auto failure = terminal_error_;
        epoch_stop_.request_stop();
        auto* cursor = cursor_.get();
        condition_.notify_all();
        lock.unlock();
        if (cursor) {
          cursor->close();
        }
        std::rethrow_exception(failure);
      }
    }
  }

  void commit(std::uint64_t sequence) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty() || queue_.front()->sequence != sequence) {
      return;
    }
    queue_.pop_front();
    condition_.notify_all();
  }

  std::size_t outstanding_batches() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closing_ || terminal_error_) {
      return 0;
    }
    return queue_.size() + (in_flight_ ? 1U : 0U);
  }

  void cancel() noexcept {
    IterableSourceCursor* cursor = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!terminal_error_) {
        terminal_error_ = std::make_exception_ptr(Cancelled(
            "IterableDataPipeline epoch was cancelled before commit"));
      }
      epoch_stop_.request_stop();
      cursor = cursor_.get();
      condition_.notify_all();
    }
    if (cursor) {
      cursor->close();
    }
  }

  void close() noexcept {
    IterableSourceCursor* cursor = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closing_ && !producer_.joinable()) {
        return;
      }
      closing_ = true;
      epoch_stop_.request_stop();
      producer_.request_stop();
      cursor = cursor_.get();
      condition_.notify_all();
    }
    if (cursor) {
      cursor->close();
    }
    if (producer_.joinable()) {
      producer_.join();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.clear();
    cursor_.reset();
  }

 private:
  std::optional<IterableSample> pull_next(std::stop_token stop) {
    try {
      return cursor_->next(stop);
    } catch (...) {
      rethrow_user_failure("source.next", "one item or end-of-source");
    }
  }

  c10::IValue collate(std::span<const IterableSample> samples,
                      std::uint64_t sequence, std::stop_token stop) {
    try {
      return factory_(samples, epoch_, sequence, stop);
    } catch (...) {
      rethrow_user_failure("batch_factory", "one complete batch value");
    }
  }

  std::shared_ptr<ProducedBatch> produce_one(std::uint64_t sequence,
                                             std::stop_token stop) {
    auto result = std::make_shared<ProducedBatch>();
    result->sequence = sequence;
    result->source_watermark = source_watermark_;
    std::vector<IterableSample> samples;
    samples.reserve(options_.batch_size);
    while (samples.size() < options_.batch_size && !stop.stop_requested()) {
      auto sample = pull_next(stop);
      if (!sample) {
        result->source_exhausted = true;
        break;
      }
      if (sample->watermark.size() > kMaxMetadataBytes) {
        throw ResourceExhausted(
            "IterableSource watermark exceeds its declared bound");
      }
      if (descriptor_.restore_level != RestoreLevel::best_effort &&
          sample->watermark.empty()) {
        throw InvalidArgument(
            "Checkpointable iterable source returned an empty watermark");
      }
      result->source_watermark = sample->watermark;
      source_watermark_ = sample->watermark;
      samples.push_back(std::move(*sample));
    }
    if (stop.stop_requested()) {
      return {};
    }
    if (descriptor_.checkpointable) {
      result->source_state = save_cursor_state(*cursor_);
    }
    if (samples.empty() || (result->source_exhausted && options_.drop_last &&
                            samples.size() < options_.batch_size)) {
      return result;
    }

    BatchEnvelope envelope;
    envelope.value = collate(samples, sequence, stop);
    if (stop.stop_requested()) {
      return {};
    }
    envelope.schema_id = options_.schema_id;
    envelope.dataset_version = options_.dataset_version;
    envelope.epoch = epoch_;
    envelope.sequence = sequence;
    envelope.source_watermark = result->source_watermark;
    envelope.sample_ids.reserve(samples.size());
    for (const auto& sample : samples) {
      envelope.sample_ids.push_back(sample.sample_id);
    }
    result->envelope = std::move(envelope);
    return result;
  }

  void producer_loop(std::stop_token thread_stop) {
    std::stop_callback notify_stop(thread_stop,
                                   [this] { condition_.notify_all(); });
    while (!thread_stop.stop_requested() && !epoch_stop_.stop_requested()) {
      std::uint64_t sequence = 0;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] {
          return closing_ || terminal_error_ || thread_stop.stop_requested() ||
                 epoch_stop_.stop_requested() ||
                 queue_.size() < options_.prefetch_batches;
        });
        if (closing_ || terminal_error_ || thread_stop.stop_requested() ||
            epoch_stop_.stop_requested()) {
          return;
        }
        if (next_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
          terminal_error_ = std::make_exception_ptr(ResourceExhausted(
              "IterableDataPipeline batch sequence exhausted"));
          epoch_stop_.request_stop();
          condition_.notify_all();
          return;
        }
        in_flight_ = true;
        sequence = next_sequence_++;
      }

      std::shared_ptr<ProducedBatch> result;
      try {
        result = produce_one(sequence, epoch_stop_.get_token());
      } catch (...) {
        result = std::make_shared<ProducedBatch>();
        result->sequence = sequence;
        result->error = std::current_exception();
      }

      bool done = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        in_flight_ = false;
        if (!closing_ && !terminal_error_ && result &&
            !epoch_stop_.stop_requested()) {
          done = result->source_exhausted || result->error != nullptr;
          queue_.push_back(std::move(result));
          producer_done_ = done;
        }
        condition_.notify_all();
      }
      if (done) {
        return;
      }
    }
  }

  std::unique_ptr<IterableSourceCursor> cursor_;  ///< Active epoch cursor.
  IterableSourceDescriptor descriptor_;           ///< Frozen source behavior.
  IterableDataPipelineOptions options_;  ///< Queue and batching limits.
  IterableBatchFactory factory_;         ///< Ordered sample collator.
  std::uint64_t epoch_ = 0;              ///< Active logical epoch.
  std::uint64_t next_sequence_ = 0;      ///< Next producer sequence.
  std::string source_watermark_;       ///< Last watermark pulled by the cursor.
  std::stop_source epoch_stop_;        ///< Cooperative epoch cancellation.
  mutable std::mutex mutex_;           ///< Guards all runtime state below.
  std::condition_variable condition_;  ///< Queue/backpressure notification.
  std::deque<std::shared_ptr<ProducedBatch>> queue_;  ///< Ordered batch queue.
  std::exception_ptr terminal_error_;  ///< Cancellation or observed failure.
  bool in_flight_ = false;             ///< Producer owns one capacity slot.
  bool producer_done_ = false;         ///< Terminal record has been queued.
  bool closing_ = false;               ///< Permanent shutdown has begun.
  std::jthread producer_;              ///< Single cursor-owner thread.
};

IterableDataPipelineState make_initial_state(
    const IterableDataPipelineOptions& options,
    const IterableSourceDescriptor& descriptor) {
  IterableDataPipelineState state;
  state.pipeline_id = options.pipeline_id;
  state.dataset_version = options.dataset_version;
  state.schema_id = options.schema_id;
  state.batch_factory_id = options.batch_factory_id;
  state.source_id = descriptor.source_id;
  state.finite_epoch = descriptor.finite_epoch;
  state.deterministic_epoch = descriptor.deterministic_epoch;
  state.checkpointable = descriptor.checkpointable;
  state.source_restore_level = descriptor.restore_level;
  state.max_replayed_samples = descriptor.max_replayed_samples;
  state.max_omitted_samples = descriptor.max_omitted_samples;
  state.batch_size = options.batch_size;
  state.drop_last = options.drop_last;
  state.prefetch_batches = options.prefetch_batches;
  state.batch_timeout_ms =
      static_cast<std::uint64_t>(options.batch_timeout.count());
  return state;
}

}  // namespace

void validate_iterable_source_descriptor(
    const IterableSourceDescriptor& descriptor) {
  if (descriptor.source_id.empty()) {
    throw InvalidArgument("IterableSource source_id is required");
  }
  if (descriptor.source_id.size() > kMaxMetadataBytes) {
    throw ResourceExhausted("IterableSource source_id exceeds its bound");
  }
  switch (descriptor.restore_level) {
    case RestoreLevel::exact:
      if (!descriptor.checkpointable || !descriptor.deterministic_epoch ||
          descriptor.max_replayed_samples != 0U ||
          descriptor.max_omitted_samples != 0U ||
          !descriptor.degradation_reasons.empty()) {
        throw InvalidArgument(
            "Exact IterableSource requires deterministic checkpoint state, "
            "zero replay/omission bounds, and no degradation reasons");
      }
      break;
    case RestoreLevel::bounded:
      if (!descriptor.checkpointable ||
          !descriptor.max_replayed_samples.has_value() ||
          !descriptor.max_omitted_samples.has_value() ||
          descriptor.degradation_reasons.empty()) {
        throw InvalidArgument(
            "Bounded IterableSource requires state, finite bounds, and a "
            "degradation reason");
      }
      break;
    case RestoreLevel::best_effort:
      if (descriptor.max_replayed_samples.has_value() ||
          descriptor.max_omitted_samples.has_value() ||
          descriptor.degradation_reasons.empty()) {
        throw InvalidArgument(
            "Best-effort IterableSource requires unbounded replay/omission "
            "and a degradation reason");
      }
      break;
    default:
      throw InvalidArgument("IterableSource restore level is invalid");
  }
  for (const auto& reason : descriptor.degradation_reasons) {
    if (reason.empty() || reason.size() > kMaxMetadataBytes) {
      throw InvalidArgument(
          "IterableSource degradation reasons must be non-empty and bounded");
    }
  }
}

struct IterableDataPipeline::Impl {
  std::shared_ptr<const IterableSource> source;
  IterableSourceDescriptor descriptor;
  IterableDataPipelineOptions options;
  IterableBatchFactory batch_factory;
  IterableDataPipelineState state;
  std::unique_ptr<IterableRuntime> runtime;
  bool closed = false;

  void validate_source_contract() const {
    const auto current = source->descriptor();
    validate_iterable_source_descriptor(current);
    if (!descriptor_equal(current, descriptor)) {
      throw CompatibilityError(
          "IterableSource descriptor changed after pipeline construction");
    }
  }

  void validate_state(const IterableDataPipelineState& candidate) const {
    if (candidate.version != IterableDataPipelineState::format_version ||
        candidate.pipeline_id != options.pipeline_id ||
        candidate.dataset_version != options.dataset_version ||
        candidate.schema_id != options.schema_id ||
        candidate.batch_factory_id != options.batch_factory_id ||
        candidate.source_id != descriptor.source_id ||
        candidate.finite_epoch != descriptor.finite_epoch ||
        candidate.deterministic_epoch != descriptor.deterministic_epoch ||
        candidate.checkpointable != descriptor.checkpointable ||
        candidate.source_restore_level != descriptor.restore_level ||
        candidate.max_replayed_samples != descriptor.max_replayed_samples ||
        candidate.max_omitted_samples != descriptor.max_omitted_samples ||
        candidate.batch_size != options.batch_size ||
        candidate.drop_last != options.drop_last ||
        candidate.prefetch_batches != options.prefetch_batches ||
        candidate.batch_timeout_ms !=
            static_cast<std::uint64_t>(options.batch_timeout.count())) {
      throw CompatibilityError(
          "IterableDataPipeline state topology is incompatible");
    }
    if (!candidate.prepared &&
        (candidate.exhausted || candidate.epoch != 0 ||
         candidate.next_sequence != 0 || candidate.committed_samples != 0 ||
         !candidate.source_watermark.empty() ||
         !candidate.source_state.empty())) {
      throw SerializationError(
          "Unprepared IterableDataPipeline state contains cursor data");
    }
    if (!candidate.checkpointable && !candidate.source_state.empty()) {
      throw SerializationError(
          "Non-checkpointable IterableSource state contains cursor bytes");
    }
    if (candidate.prepared && candidate.checkpointable &&
        candidate.source_state.size() > kMaxSourceStateBytes) {
      throw ResourceExhausted(
          "IterableDataPipeline source state exceeds its bound");
    }
    if (candidate.source_watermark.size() > kMaxMetadataBytes) {
      throw ResourceExhausted(
          "IterableDataPipeline watermark exceeds its bound");
    }
    if (candidate.committed_samples > 0 &&
        candidate.source_restore_level != RestoreLevel::best_effort &&
        candidate.source_watermark.empty()) {
      throw SerializationError(
          "Checkpointable IterableDataPipeline state lacks a watermark");
    }
    if (candidate.next_sequence > candidate.committed_samples) {
      throw SerializationError(
          "IterableDataPipeline state has more batches than samples");
    }
    const auto max_sequence_without_overflow =
        std::numeric_limits<std::uint64_t>::max() / candidate.batch_size;
    if (candidate.next_sequence <= max_sequence_without_overflow &&
        candidate.committed_samples >
            candidate.next_sequence * candidate.batch_size) {
      throw SerializationError(
          "IterableDataPipeline committed sample count is inconsistent");
    }
  }

  std::unique_ptr<IterableRuntime> make_runtime(
      std::unique_ptr<IterableSourceCursor> cursor, std::uint64_t epoch,
      std::uint64_t next_sequence, std::string source_watermark) const {
    return std::make_unique<IterableRuntime>(
        std::move(cursor), descriptor, options, batch_factory, epoch,
        next_sequence, std::move(source_watermark));
  }
};

IterableDataPipeline::IterableDataPipeline(
    std::shared_ptr<const IterableSource> source,
    IterableDataPipelineOptions options, IterableBatchFactory batch_factory) {
  if (!source) {
    throw InvalidArgument("IterableDataPipeline source cannot be null");
  }
  if (!batch_factory) {
    throw InvalidArgument("IterableDataPipeline batch factory cannot be empty");
  }
  validate_options(options);
  auto descriptor = source->descriptor();
  validate_iterable_source_descriptor(descriptor);
  auto state = make_initial_state(options, descriptor);
  impl_ = std::make_unique<Impl>(Impl{.source = std::move(source),
                                      .descriptor = std::move(descriptor),
                                      .options = std::move(options),
                                      .batch_factory = std::move(batch_factory),
                                      .state = std::move(state),
                                      .runtime = nullptr,
                                      .closed = false});
  validate_checkpoint_participant(*this);
}

IterableDataPipeline::~IterableDataPipeline() { close(); }

void IterableDataPipeline::set_epoch(std::uint64_t epoch) {
  if (impl_->closed) {
    throw NotReady("IterableDataPipeline is closed");
  }
  impl_->validate_source_contract();
  auto cursor = open_cursor(*impl_->source, epoch);
  auto candidate = make_initial_state(impl_->options, impl_->descriptor);
  candidate.prepared = true;
  candidate.epoch = epoch;
  if (impl_->descriptor.checkpointable) {
    candidate.source_state = save_cursor_state(*cursor);
  }
  auto replacement = impl_->make_runtime(std::move(cursor), epoch, 0, {});
  if (impl_->runtime) {
    impl_->runtime->close();
  }
  impl_->state = std::move(candidate);
  impl_->runtime = std::move(replacement);
}

std::optional<BatchEnvelope> IterableDataPipeline::next(
    std::stop_token stop_token) {
  if (impl_->closed) {
    throw NotReady("IterableDataPipeline is closed");
  }
  if (!impl_->state.prepared) {
    throw InvalidArgument(
        "IterableDataPipeline requires set_epoch before next");
  }
  if (impl_->state.exhausted) {
    return std::nullopt;
  }
  if (stop_token.stop_requested()) {
    impl_->runtime->cancel();
    return std::nullopt;
  }
  const auto result = impl_->runtime->wait_next(stop_token);
  if (!result) {
    return std::nullopt;
  }
  if (result->sequence != impl_->state.next_sequence) {
    throw InternalError(
        "IterableDataPipeline producer sequence is inconsistent");
  }

  auto candidate = impl_->state;
  candidate.source_state = result->source_state;
  candidate.source_watermark = result->source_watermark;
  candidate.exhausted = result->source_exhausted;
  if (!result->envelope) {
    impl_->state = std::move(candidate);
    impl_->runtime->commit(result->sequence);
    impl_->runtime->close();
    impl_->runtime.reset();
    return std::nullopt;
  }

  auto envelope = *result->envelope;
  if (envelope.sample_ids.size() >
      std::numeric_limits<std::uint64_t>::max() - candidate.committed_samples) {
    throw ResourceExhausted(
        "IterableDataPipeline committed sample counter exhausted");
  }
  if (candidate.next_sequence == std::numeric_limits<std::uint64_t>::max()) {
    throw ResourceExhausted(
        "IterableDataPipeline committed sequence counter exhausted");
  }
  candidate.committed_samples += envelope.sample_ids.size();
  ++candidate.next_sequence;
  impl_->state = std::move(candidate);
  impl_->runtime->commit(result->sequence);
  if (result->source_exhausted) {
    impl_->runtime->close();
    impl_->runtime.reset();
  }
  return envelope;
}

void IterableDataPipeline::close() noexcept {
  if (!impl_ || impl_->closed) {
    return;
  }
  impl_->closed = true;
  if (impl_->runtime) {
    impl_->runtime->close();
    impl_->runtime.reset();
  }
}

bool IterableDataPipeline::closed() const noexcept { return impl_->closed; }

std::size_t IterableDataPipeline::outstanding_batches() const noexcept {
  return impl_->runtime ? impl_->runtime->outstanding_batches() : 0U;
}

const IterableDataPipelineOptions& IterableDataPipeline::options()
    const noexcept {
  return impl_->options;
}

DataCapability IterableDataPipeline::capability() const {
  DataCapability result;
  result.map_style = false;
  result.iterable_style = true;
  result.deterministic_epoch = impl_->descriptor.deterministic_epoch;
  result.in_epoch_cursor = impl_->descriptor.checkpointable;
  result.worker_prefetch = true;
  result.worker_count = 1;
  result.prefetch_batches = impl_->options.prefetch_batches;
  result.batch_timeout = impl_->options.batch_timeout.count() != 0;
  result.restore_level = impl_->descriptor.restore_level;
  result.max_replayed_samples = impl_->descriptor.max_replayed_samples;
  result.max_omitted_samples = impl_->descriptor.max_omitted_samples;
  result.max_replayed_batches = impl_->options.prefetch_batches;
  result.degradation_reasons = impl_->descriptor.degradation_reasons;
  return result;
}

IterableDataPipelineState IterableDataPipeline::state() const {
  return impl_->state;
}

std::string IterableDataPipeline::participant_id() const {
  return impl_->options.pipeline_id;
}

StateManifest IterableDataPipeline::describe_state() const {
  return {
      .participant_id = participant_id(),
      .schema_version = IterableDataPipelineState::format_version,
      .restore_level = impl_->descriptor.restore_level,
      .fields = {{"topology", "application/octet-stream", kMaxTopologyBytes,
                  true},
                 {"cursor", "application/octet-stream", kMaxCursorBytes, true}},
      .degradation_reasons = impl_->descriptor.degradation_reasons};
}

void IterableDataPipeline::save_state(StateWriter& writer) const {
  if (impl_->closed) {
    throw NotReady("Closed IterableDataPipeline state cannot be saved");
  }
  const auto snapshot = impl_->state;
  write_blob(writer, "topology", encode_topology(snapshot), kMaxTopologyBytes);
  write_blob(writer, "cursor", encode_cursor(snapshot), kMaxCursorBytes);
}

void IterableDataPipeline::load_state(const StateReader& reader,
                                      RestorePolicy policy) {
  if (impl_->closed) {
    throw NotReady("Closed IterableDataPipeline state cannot be restored");
  }
  impl_->validate_source_contract();
  const auto& manifest = reader.manifest();
  validate_state_manifest(manifest);
  if (!manifest_equal(manifest, describe_state()) ||
      !restore_policy_accepts(policy, manifest.restore_level)) {
    throw CompatibilityError(
        "IterableDataPipeline checkpoint policy is incompatible");
  }
  auto decoded = decode_state(reader.read_blob("topology", kMaxTopologyBytes),
                              reader.read_blob("cursor", kMaxCursorBytes));
  impl_->validate_state(decoded);

  std::unique_ptr<IterableRuntime> replacement;
  if (decoded.prepared && !decoded.exhausted) {
    auto cursor = open_cursor(*impl_->source, decoded.epoch);
    if (impl_->descriptor.checkpointable) {
      load_cursor_state(*cursor, decoded.source_state);
    }
    replacement =
        impl_->make_runtime(std::move(cursor), decoded.epoch,
                            decoded.next_sequence, decoded.source_watermark);
  }
  if (impl_->runtime) {
    impl_->runtime->close();
  }
  impl_->state = std::move(decoded);
  impl_->runtime = std::move(replacement);
}

}  // namespace jyppx::data
