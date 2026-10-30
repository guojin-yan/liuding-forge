// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "detail/telemetry_proto.hpp"
#include "detail/telemetry_proto/tensorboard/compat/proto/event.pb.h"

#include <jyppx/liuding/experiment/telemetry_exporters.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/telemetry_native/tensorboard.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string_view>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace jyppx::liuding::experiment {
namespace {
using Json = nlohmann::json;

[[noreturn]] void io_failure(std::string operation) {
  throw IoError(
      "Native TensorBoard observer I/O failed",
      ErrorDetails{.component = "experiment.tensorboard",
                   .operation = std::move(operation),
                   .expected = "complete exclusive observer I/O",
                   .actual = "destination unavailable; path omitted",
                   .hint =
                       "Retain durable events and replay into a new private "
                       "directory; do not retry a poisoned writer."});
}

void capacity(std::uint64_t bytes, std::uint64_t maximum) {
  if (bytes > maximum)
    throw ResourceExhausted("Native TensorBoard observer capacity exceeded");
}

// Bounded TFRecord framing storage; protobuf bytes come exclusively from
// the official nanopb runtime and generated schema descriptors below.
class BoundedBytes {
 public:
  explicit BoundedBytes(std::size_t maximum) : maximum_(maximum) {}
  void raw(std::string_view value) {
    capacity(value.size(), maximum_ - bytes_.size());
    bytes_.append(value);
  }
  const std::string& bytes() const noexcept { return bytes_; }

 private:
  std::size_t maximum_;
  std::string bytes_;
};

// Exact integer division rounded once to binary64, including on MSVC where
// long double has double precision. This matches Python int/int wall seconds.
double wall_seconds(std::int64_t nanoseconds) {
  constexpr std::uint64_t denominator = 1000000000U;
  const bool negative = nanoseconds < 0;
  const auto magnitude =
      negative ? std::uint64_t{0U} - static_cast<std::uint64_t>(nanoseconds)
               : static_cast<std::uint64_t>(nanoseconds);
  if (magnitude == 0U) return 0.0;
  auto significand = magnitude / denominator;
  auto remainder = magnitude % denominator;
  int exponent = 0;
  unsigned digits = 0U;
  if (significand != 0U) {
    digits = static_cast<unsigned>(std::bit_width(significand));
    exponent = static_cast<int>(digits) - 1;
  } else {
    do {
      remainder *= 2U;
      --exponent;
    } while (remainder < denominator);
    significand = 1U;
    remainder -= denominator;
    digits = 1U;
  }
  while (digits < 53U) {
    remainder *= 2U;
    significand = (significand << 1U) | (remainder >= denominator ? 1U : 0U);
    if (remainder >= denominator) remainder -= denominator;
    ++digits;
  }
  const auto twice = remainder * 2U;
  if (twice > denominator || (twice == denominator && (significand & 1U)))
    ++significand;
  const auto value =
      std::ldexp(static_cast<double>(significand), exponent - 52);
  return negative ? -value : value;
}

constexpr auto kCrc32cTable = [] {
  std::array<std::uint32_t, 256U> table{};
  for (std::size_t index = 0U; index < table.size(); ++index) {
    auto value = static_cast<std::uint32_t>(index);
    for (unsigned bit = 0U; bit < 8U; ++bit)
      value = (value >> 1U) ^ (0x82f63b78U & (0U - (value & 1U)));
    table[index] = value;
  }
  return table;
}();

std::uint32_t crc32c(std::string_view bytes) noexcept {
  auto crc = std::uint32_t{0xffffffffU};
  for (const unsigned char byte : bytes)
    crc = (crc >> 8U) ^ kCrc32cTable[(crc ^ byte) & 255U];
  return ~crc;
}

std::uint32_t masked_crc32c(std::string_view bytes) noexcept {
  const auto crc = crc32c(bytes);
  return ((crc >> 15U) | (crc << 17U)) + 0xa282ead8U;
}

void little_endian(BoundedBytes& output, std::uint64_t value, unsigned width) {
  for (unsigned index = 0U; index < width; ++index) {
    const char byte = static_cast<char>(value & 255U);
    output.raw({&byte, 1U});
    value >>= 8U;
  }
}

void record(BoundedBytes& batch, std::string_view event) {
  BoundedBytes length(8U);
  little_endian(length, event.size(), 8U);
  batch.raw(length.bytes());
  little_endian(batch, masked_crc32c(length.bytes()), 4U);
  batch.raw(event);
  little_endian(batch, masked_crc32c(event), 4U);
}

std::string hash_json(const Json& value) {
  const auto text = value.dump();
  return sha256_hex(
      {reinterpret_cast<const std::byte*>(text.data()), text.size()});
}

std::string summary_event(double wall, std::uint64_t step, std::string_view tag,
                          std::string_view description, bool text,
                          std::string_view body, double scalar,
                          std::size_t maximum) {
  namespace encoding = detail::telemetry_proto;
  using Event = jyppx_liuding_telemetry_tensorboard_Event;
  using Value = jyppx_liuding_telemetry_tensorboard_Summary_Value;
  Event event{};
  Value value{};
  encoding::Bytes tag_bytes{tag}, description_bytes{description};
  encoding::Bytes plugin_name{text ? "text" : "scalars"}, body_bytes{body};
  encoding::bind_bytes(value.tag, tag_bytes);
  value.which_value =
      jyppx_liuding_telemetry_tensorboard_Summary_Value_tensor_tag;
  auto& tensor = value.value.tensor;
  tensor = {};
  tensor.dtype = text ? jyppx_liuding_telemetry_tensorboard_DataType_DT_STRING
                      : jyppx_liuding_telemetry_tensorboard_DataType_DT_DOUBLE;
  if (text) {
    tensor.has_tensor_shape = true;  // Present rank-0 shape, like text_pb.
    encoding::bind_bytes(tensor.string_val, body_bytes);
  } else {
    tensor.double_val_count = 1U;
    tensor.double_val[0] = scalar;
  }
  value.has_metadata = true;
  value.metadata.has_plugin_data = true;
  encoding::bind_bytes(value.metadata.plugin_data.plugin_name, plugin_name);
  encoding::bind_bytes(value.metadata.summary_description, description_bytes);
  const encoding::Message item{
      &jyppx_liuding_telemetry_tensorboard_Summary_Value_msg, &value};
  event.wall_time = wall;
  event.step = static_cast<std::int64_t>(step);
  event.which_what = jyppx_liuding_telemetry_tensorboard_Event_summary_tag;
  event.what.summary = {};
  encoding::bind_message(event.what.summary.value, item);
  return encoding::encode(&jyppx_liuding_telemetry_tensorboard_Event_msg,
                          &event, maximum);
}

double symbolic(const Json& point) {
  switch (point.at("status").get<unsigned>()) {
    case 1U:
      return point.at("value").get<double>();
    case 2U:
      return std::numeric_limits<double>::quiet_NaN();
    case 3U:
      return std::numeric_limits<double>::infinity();
    case 4U:
      return -std::numeric_limits<double>::infinity();
    default:
      throw InternalError("Missing metric cannot have a scalar tensor");
  }
}

bool prepare(const TelemetryEnvelope& envelope,
             const TensorBoardTelemetryOptions& options, BoundedBytes& batch,
             std::stop_token stop, TelemetryDeadline deadline) {
  const auto kind = *telemetry_payload_kind(envelope);
  if (!envelope.time.wall_time_unix_ns ||
      (kind != TelemetryPayloadKind::lifecycle &&
       kind != TelemetryPayloadKind::metric_batch &&
       kind != TelemetryPayloadKind::resource_sample))
    return false;
  const auto projected = telemetry_projection_json(envelope, options.limits);
  const auto value = Json::parse(projected);
  const auto& payload = value.at("payload");
  const bool lifecycle = kind == TelemetryPayloadKind::lifecycle;
  const auto& points =
      payload.at(lifecycle                                    ? "metrics"
                 : kind == TelemetryPayloadKind::metric_batch ? "points"
                                                              : "measurements");
  constexpr auto maximum_step =
      static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)());
  if ((lifecycle && envelope.identity.sequence > maximum_step) ||
      (!lifecycle && points.empty()))
    return false;
  for (const auto& point : points) {
    check_telemetry_operation(stop, deadline);
    if (point.at("step").at("value").get<std::uint64_t>() > maximum_step ||
        (!lifecycle && point.at("status") == 5U))
      return false;
  }
  const auto wall = wall_seconds(*envelope.time.wall_time_unix_ns);
  if (lifecycle) {
    const Json series{{"stream_id", envelope.identity.stream_id},
                      {"project_id", envelope.correlation.project_id},
                      {"run_id", envelope.correlation.run_id},
                      {"attempt_id", envelope.correlation.attempt_id},
                      {"lifecycle_name", payload.at("name")}};
    const auto description =
        Json{{"schema", "jyppx.tensorboard.lifecycle/v1"},
             {"series", series},
             {"step_domain", "telemetry.sequence"},
             {"encoding", "complete-envelope-json"},
             {"missing_metric_projection", "text-only; no scalar value exists"}}
            .dump();
    record(batch, summary_event(wall, envelope.identity.sequence,
                                "jyppx/liuding/lifecycle/" + hash_json(series),
                                description, true, projected, 0.0,
                                options.max_record_bytes));
  }
  for (const auto& point : points) {
    check_telemetry_operation(stop, deadline);
    if (point.at("status") == 5U) continue;
    const auto& definition = point.at("definition");
    const Json series{
        {"definition", definition},
        {"context", point.at("context")},
        {"step_kind", point.at("step").at("kind")},
        {"resource_id", payload.value("resource_id", std::string{})}};
    const auto description =
        Json{{"series", series},
             {"wall_time_unix_ns", *envelope.time.wall_time_unix_ns},
             {"original_metric_id", definition.at("metric_id")}}
            .dump();
    record(batch, summary_event(
                      wall, point.at("step").at("value").get<std::uint64_t>(),
                      "jyppx/liuding/scalar/" + hash_json(series), description,
                      false, {}, symbolic(point), options.max_record_bytes));
  }
  return true;
}

class NativeEventFile {
 public:
  ~NativeEventFile() { release(); }
  NativeEventFile() = default;
  NativeEventFile(const NativeEventFile&) = delete;
  NativeEventFile& operator=(const NativeEventFile&) = delete;
  void open(const std::filesystem::path& path) {
#ifdef _WIN32
    handle_ = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                          CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE) io_failure("open");
#else
    handle_ =
        ::open(path.c_str(),
               O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (handle_ < 0) io_failure("open");
#endif
  }
  void write(std::string_view bytes, std::stop_token stop,
             TelemetryDeadline deadline) {
    while (!bytes.empty()) {
      check_telemetry_operation(stop, deadline);
      const auto count = (std::min)(bytes.size(), std::size_t{64U * 1024U});
#ifdef _WIN32
      DWORD written = 0U;
      if (!WriteFile(handle_, bytes.data(), static_cast<DWORD>(count), &written,
                     nullptr) ||
          written == 0U)
        io_failure("write");
#else
      const auto written = ::write(handle_, bytes.data(), count);
      if (written < 0 && errno == EINTR) continue;
      if (written <= 0) io_failure("write");
#endif
      bytes.remove_prefix(static_cast<std::size_t>(written));
      check_telemetry_operation(stop, deadline);
    }
  }
  void close() {
#ifdef _WIN32
    const auto handle = std::exchange(handle_, INVALID_HANDLE_VALUE);
    if (handle != INVALID_HANDLE_VALUE && !CloseHandle(handle))
      io_failure("close");
#else
    const auto handle = std::exchange(handle_, -1);
    // POSIX close(EINTR) may already have released/reused the descriptor.
    if (handle >= 0 && ::close(handle) != 0) io_failure("close");
#endif
  }

 private:
  void release() noexcept {
#ifdef _WIN32
    if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
    handle_ = INVALID_HANDLE_VALUE;
#else
    if (handle_ >= 0) ::close(handle_);
    handle_ = -1;
#endif
  }
#ifdef _WIN32
  HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
  int handle_ = -1;
#endif
};

struct Seen {
  std::string digest;
  TelemetryDisposition disposition;
};
using SeenIndex = std::map<std::string, Seen>;
using EventIndex = std::map<std::pair<std::string, std::string>, std::string>;
using SequenceIndex =
    std::map<std::pair<std::string, std::uint64_t>, std::string>;

template <typename Index, typename Key>
void check_coordinate(const Index& current, Index& staged, Key key,
                      const std::string& operation) {
  const auto existing = current.find(key);
  if (existing != current.end() && existing->second != operation)
    throw CompatibilityError("Conflicting native TensorBoard event coordinate");
  const auto [position, added] = staged.emplace(std::move(key), operation);
  if (!added && position->second != operation)
    throw CompatibilityError("Conflicting native TensorBoard event coordinate");
}
}  // namespace

class TensorBoardTelemetrySink::Impl {
 public:
  explicit Impl(TensorBoardTelemetryOptions configured)
      : options(std::move(configured)) {
    validate_telemetry_limits(options.limits);
    descriptor = {.type_id = "jyppx.telemetry-sink.tensorboard-native/v1",
                  .sink_id = options.sink_id,
                  .destination_id = options.destination_id,
                  .role = TelemetrySinkRole::observer};
    validate_telemetry_sink_descriptor(descriptor, options.limits);
    if (options.directory.empty() || options.max_file_bytes == 0U ||
        options.max_record_bytes == 0U ||
        options.max_batch_encoded_bytes < 16U || options.max_seen == 0U ||
        options.max_seen_bytes == 0U)
      throw InvalidArgument("Invalid native TensorBoard observer options");
    namespace encoding = detail::telemetry_proto;
    jyppx_liuding_telemetry_tensorboard_Event header{};
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
    header.wall_time = wall_seconds(now);
    header.which_what =
        jyppx_liuding_telemetry_tensorboard_Event_file_version_tag;
    const encoding::Bytes version{"brain.Event:2"};
    const encoding::Bytes writer{"liuding-forge.native-tensorboard/v1"};
    encoding::bind_bytes(header.what.file_version, version);
    header.has_source_metadata = true;
    encoding::bind_bytes(header.source_metadata.writer, writer);
    const auto header_bytes =
        encoding::encode(&jyppx_liuding_telemetry_tensorboard_Event_msg,
                         &header, options.max_record_bytes);
    BoundedBytes framed(options.max_batch_encoded_bytes);
    record(framed, header_bytes);
    capacity(framed.bytes().size(), options.max_file_bytes);
    std::error_code error;
    if (!std::filesystem::create_directory(options.directory, error) || error)
      io_failure("exclusive_directory");
    path = options.directory / "events.out.tfevents.liuding";
    file.open(path);
    file.write(framed.bytes(), {}, std::nullopt);
    file_bytes = framed.bytes().size();
  }
  void ready() const {
    if (closed || poisoned)
      throw NotReady("Native TensorBoard observer is closed or poisoned");
  }
  TensorBoardTelemetryOptions options;
  TelemetrySinkDescriptor descriptor;
  std::filesystem::path path;
  NativeEventFile file;
  SeenIndex seen;
  EventIndex events;
  SequenceIndex sequences;
  TelemetryFlushResult counters;
  std::uint64_t file_bytes = 0U, seen_bytes = 0U;
  bool closed = false, poisoned = false;
};

TensorBoardTelemetrySink::TensorBoardTelemetrySink(
    TensorBoardTelemetryOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
TensorBoardTelemetrySink::~TensorBoardTelemetrySink() = default;
const TelemetrySinkDescriptor& TensorBoardTelemetrySink::descriptor()
    const noexcept {
  return impl_->descriptor;
}
const std::filesystem::path& TensorBoardTelemetrySink::event_file()
    const noexcept {
  return impl_->path;
}

TelemetryEmitResult TensorBoardTelemetrySink::emit(
    std::span<const TelemetryEnvelope> envelopes, std::stop_token stop,
    TelemetryDeadline deadline) {
  check_telemetry_operation(stop, deadline);
  impl_->ready();
  validate_telemetry_batch(envelopes, impl_->options.limits);
  BoundedBytes encoded(impl_->options.max_batch_encoded_bytes);
  SeenIndex staged;
  EventIndex events;
  SequenceIndex sequences;
  auto next = impl_->counters;
  auto next_bytes = impl_->seen_bytes;
  TelemetryEmitResult result;
  result.dispositions.reserve(envelopes.size());
  for (const auto& envelope : envelopes) {
    check_telemetry_operation(stop, deadline);
    const auto key =
        telemetry_idempotency_digest(envelope.identity, impl_->options.limits)
            .hex;
    const auto digest =
        telemetry_content_digest(envelope, impl_->options.limits).hex;
    const auto existing = impl_->seen.find(key);
    const auto pending = staged.find(key);
    const auto* known = existing != impl_->seen.end() ? &existing->second
                        : pending != staged.end()     ? &pending->second
                                                      : nullptr;
    if (known) {
      if (known->digest != digest)
        throw CompatibilityError(
            "Conflicting native TensorBoard operation identity");
      result.dispositions.push_back(known->disposition ==
                                            TelemetryDisposition::accepted
                                        ? TelemetryDisposition::duplicate
                                        : known->disposition);
      continue;
    }
    capacity(impl_->seen.size() + staged.size() + 1U, impl_->options.max_seen);
    const auto index_bytes =
        std::uint64_t{320U} +
        2U * static_cast<std::uint64_t>(envelope.identity.stream_id.size()) +
        envelope.identity.event_id.size();
    capacity(index_bytes, impl_->options.max_seen_bytes - next_bytes);
    next_bytes += index_bytes;
    check_coordinate(
        impl_->events, events,
        std::pair{envelope.identity.stream_id, envelope.identity.event_id},
        key);
    check_coordinate(
        impl_->sequences, sequences,
        std::pair{envelope.identity.stream_id, envelope.identity.sequence},
        key);
    const auto disposition =
        prepare(envelope, impl_->options, encoded, stop, deadline)
            ? TelemetryDisposition::accepted
            : TelemetryDisposition::unsupported;
    if (disposition == TelemetryDisposition::accepted)
      ++next.exported_events;
    else
      ++next.unsupported_events;
    staged.emplace(key, Seen{digest, disposition});
    result.dispositions.push_back(disposition);
  }
  capacity(encoded.bytes().size(),
           impl_->options.max_file_bytes - impl_->file_bytes);
  validate_telemetry_emit_result(impl_->descriptor, envelopes, result,
                                 impl_->options.limits);
  check_telemetry_operation(stop, deadline);
  if (!encoded.bytes().empty()) {
    try {
      impl_->file.write(encoded.bytes(), stop, deadline);
      check_telemetry_operation(stop, deadline);
    } catch (...) {
      impl_->poisoned = true;
      throw;
    }
  }
  impl_->seen.merge(staged);
  impl_->events.merge(events);
  impl_->sequences.merge(sequences);
  impl_->file_bytes += encoded.bytes().size();
  impl_->seen_bytes = next_bytes;
  impl_->counters = next;
  return result;
}

TelemetryFlushResult TensorBoardTelemetrySink::flush(
    std::stop_token stop, TelemetryDeadline deadline) {
  if (impl_->closed) return impl_->counters;
  check_telemetry_operation(stop, deadline);
  impl_->ready();
  // No user-space buffer: successful native writes are already at the declared
  // observer boundary. FlushFileBuffers/fsync would assert a different
  // contract.
  return impl_->counters;
}
TelemetryFlushResult TensorBoardTelemetrySink::close(
    std::stop_token stop, TelemetryDeadline deadline) {
  if (impl_->closed) return impl_->counters;
  static_cast<void>(flush(stop, deadline));
  try {
    impl_->file.close();
    check_telemetry_operation(stop, deadline);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
  impl_->closed = true;
  return impl_->counters;
}

}  // namespace jyppx::liuding::experiment
