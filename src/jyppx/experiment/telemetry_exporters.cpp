// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/experiment/telemetry_exporters.hpp>
#include <jyppx/foundation/error.hpp>
#include <jyppx/foundation/sha256.hpp>

#include <fstream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <type_traits>
#include <utility>

namespace jyppx::experiment {
namespace {
using Json = nlohmann::json;

[[noreturn]] void protocol_error(std::string operation) {
  throw SerializationError(
      "Invalid telemetry export/protocol data",
      ErrorDetails{
          .component = "experiment.telemetry_exporters",
          .operation = std::move(operation),
          .expected = "complete bounded schema-v1 data and matching identity",
          .actual = "invalid input; child/file content omitted",
          .hint = "Retain canonical durable events and repair the observer "
                  "before replaying original immutable envelopes."});
}

[[noreturn]] void io_error(std::string operation) {
  throw IoError(
      "Telemetry observer I/O failed",
      ErrorDetails{.component = "experiment.telemetry_exporters",
                   .operation = std::move(operation),
                   .hint = "Reopen and recover the observer before retrying; "
                           "the durable RunSink remains the source of truth."});
}

void bound(std::uint64_t size, std::uint64_t maximum) {
  if (size > maximum)
    throw ResourceExhausted("Telemetry observer capacity exceeded");
}

Json parse(std::string_view input, std::size_t maximum) {
  bound(input.size(), maximum);
  std::vector<std::set<std::string>> keys;
  try {
    return Json::parse(
        input.begin(), input.end(),
        [&](int depth, Json::parse_event_t event, Json& value) {
          if (depth > 32) protocol_error("json.depth");
          if (event == Json::parse_event_t::object_start) keys.emplace_back();
          if (event == Json::parse_event_t::key &&
              (keys.empty() ||
               !keys.back().insert(value.get<std::string>()).second)) {
            protocol_error("json.duplicate_key");
          }
          if (event == Json::parse_event_t::object_end) keys.pop_back();
          return true;
        });
  } catch (const Json::exception&) {
    protocol_error("json.syntax");
  }
}

Json definition(const TelemetryMetricDefinition& value,
                const TelemetryLimits& limits) {
  return {{"schema_version", value.schema_version},
          {"metric_id", value.metric_id},
          {"version", value.version},
          {"unit", static_cast<unsigned>(value.unit)},
          {"aggregation", static_cast<unsigned>(value.aggregation)},
          {"direction", static_cast<unsigned>(value.direction)},
          {"parameters", value.parameters},
          {"context_keys", value.context_keys},
          {"definition_digest",
           telemetry_metric_definition_digest(value, limits).hex}};
}

Json points(const std::vector<TelemetryMetricPoint>& values,
            const TelemetryLimits& limits) {
  auto result = Json::array();
  for (const auto& value : values) {
    result.push_back(
        {{"definition", definition(value.definition, limits)},
         {"step",
          {{"kind", static_cast<unsigned>(value.step.kind)},
           {"value", value.step.value}}},
         {"status", static_cast<unsigned>(value.observation.status)},
         {"value", value.observation.value ? Json(*value.observation.value)
                                           : Json(nullptr)},
         {"reason", value.observation.reason},
         {"context", value.context},
         {"sample_count",
          value.sample_count ? Json(*value.sample_count) : Json(nullptr)}});
  }
  return result;
}

Json projection(const TelemetryEnvelope& value, const TelemetryLimits& limits) {
  Json result{{"schema_version", value.schema_version},
              {"identity",
               {{"stream_id", value.identity.stream_id},
                {"event_id", value.identity.event_id},
                {"idempotency_key", value.identity.idempotency_key},
                {"sequence", value.identity.sequence}}},
              {"time",
               {{"wall_time_unix_ns", value.time.wall_time_unix_ns
                                          ? Json(*value.time.wall_time_unix_ns)
                                          : Json(nullptr)},
                {"monotonic_time_ns", value.time.monotonic_time_ns
                                          ? Json(*value.time.monotonic_time_ns)
                                          : Json(nullptr)},
                {"clock_id", value.time.clock_id}}},
              {"correlation",
               {{"project_id", value.correlation.project_id},
                {"run_id", value.correlation.run_id},
                {"attempt_id", value.correlation.attempt_id},
                {"trace_id", value.correlation.trace_id},
                {"span_id", value.correlation.span_id},
                {"parent_span_id", value.correlation.parent_span_id}}},
              {"reliability", static_cast<unsigned>(value.reliability)},
              {"attributes", value.attributes}};
  Json payload{{"kind", static_cast<unsigned>(*telemetry_payload_kind(value))}};
  std::visit(
      [&](const auto& data) {
        using T = std::decay_t<decltype(data)>;
        if constexpr (std::is_same_v<T, TelemetryLifecycle>) {
          payload["name"] = data.name;
          payload["terminal"] = data.terminal;
          payload["metrics"] = points(data.metrics, limits);
          payload["decisions"] = Json::array();
          for (const auto& decision : data.decisions)
            payload["decisions"].push_back({{"actor_id", decision.actor_id},
                                            {"action", decision.action},
                                            {"reason", decision.reason},
                                            {"effective", decision.effective}});
        } else if constexpr (std::is_same_v<T, TelemetryMetricBatch>) {
          payload["points"] = points(data.points, limits);
        } else if constexpr (std::is_same_v<T, TelemetryLog>) {
          payload["severity"] = static_cast<unsigned>(data.severity);
          payload["body"] = data.body;
        } else if constexpr (std::is_same_v<T, TelemetrySpan>) {
          payload["name"] = data.name;
          payload["start_monotonic_ns"] = data.start_monotonic_ns;
          payload["end_monotonic_ns"] = data.end_monotonic_ns;
          payload["status"] = static_cast<unsigned>(data.status);
          payload["reason"] = data.reason;
        } else if constexpr (std::is_same_v<T, TelemetryResourceSample>) {
          payload["resource_id"] = data.resource_id;
          payload["measurements"] = points(data.measurements, limits);
        } else if constexpr (std::is_same_v<T, TelemetryArtifactReference>) {
          payload["role"] = data.role;
          payload["artifact"] = {{"algorithm", "sha256"},
                                 {"digest", data.artifact.digest.hex},
                                 {"size_bytes", data.artifact.size_bytes},
                                 {"logical_name", data.artifact.logical_name},
                                 {"media_type", data.artifact.media_type},
                                 {"producer", data.artifact.producer}};
        }
      },
      value.payload);
  result["payload"] = std::move(payload);
  return result;
}

std::string hex(std::span<const std::byte> bytes) {
  constexpr char alphabet[] = "0123456789abcdef";
  std::string output;
  if (bytes.size() > output.max_size() / 2U)
    throw ResourceExhausted("Hex encoding overflow");
  output.reserve(bytes.size() * 2U);
  for (const auto byte : bytes) {
    const auto value = std::to_integer<unsigned>(byte);
    output.push_back(alphabet[value >> 4U]);
    output.push_back(alphabet[value & 15U]);
  }
  return output;
}

std::vector<std::byte> unhex(std::string_view text, std::uint64_t maximum) {
  if (text.size() % 2U != 0U) protocol_error("canonical_hex.length");
  bound(text.size() / 2U, maximum);
  std::vector<std::byte> result;
  result.reserve(text.size() / 2U);
  const auto digit = [](char c) -> unsigned {
    if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<unsigned>(c - 'a' + 10);
    protocol_error("canonical_hex.digit");
  };
  for (std::size_t index = 0U; index < text.size(); index += 2U) {
    result.push_back(static_cast<std::byte>((digit(text[index]) << 4U) |
                                            digit(text[index + 1U])));
  }
  return result;
}

std::string operation_key(const TelemetryEnvelope& value,
                          const TelemetryLimits& limits) {
  return telemetry_idempotency_digest(value.identity, limits).hex;
}

}  // namespace

std::string telemetry_projection_json(const TelemetryEnvelope& envelope,
                                      const TelemetryLimits& limits) {
  validate_telemetry_envelope(envelope, limits);
  return projection(envelope, limits).dump();
}

class JsonlTelemetrySink::Impl {
 public:
  explicit Impl(JsonlTelemetryOptions configured)
      : options(std::move(configured)) {
    descriptor = {.type_id = "jyppx.telemetry-sink.jsonl/v1",
                  .sink_id = options.sink_id,
                  .destination_id = options.destination_id,
                  .role = TelemetrySinkRole::observer};
    validate_telemetry_sink_descriptor(descriptor, options.limits);
    if (options.path.empty() || options.max_line_bytes == 0U ||
        options.max_records == 0U || options.max_file_bytes == 0U)
      throw InvalidArgument("Invalid JSONL observer options");
    std::error_code error;
    const auto status = std::filesystem::symlink_status(options.path, error);
    if (error && error != std::errc::no_such_file_or_directory)
      io_error("jsonl.status");
    if (!error && std::filesystem::exists(status)) {
      if (!std::filesystem::is_regular_file(status))
        io_error("jsonl.nonregular");
      file_bytes = std::filesystem::file_size(options.path, error);
      if (error) io_error("jsonl.size");
      bound(file_bytes, options.max_file_bytes);
      std::ifstream input(options.path, std::ios::binary);
      if (!input) io_error("jsonl.recover.open");
      std::string line;
      std::uint64_t offset = 0U;
      std::uint64_t complete = 0U;
      char character = 0;
      while (input.get(character)) {
        ++offset;
        if (character != '\n') {
          if (line.size() == options.max_line_bytes)
            throw ResourceExhausted("JSONL line bound exceeded");
          line.push_back(character);
          continue;
        }
        bound(recovered.complete_records + 1U, options.max_records);
        const auto record = parse(line, options.max_line_bytes);
        try {
          if (!record.is_object() || record.size() != 4U ||
              record.at("schema_version") != 1U) {
            protocol_error("jsonl.record.schema");
          }
          const auto bytes =
              unhex(record.at("canonical_hex").get<std::string>(),
                    options.limits.max_envelope_bytes);
          const auto envelope =
              decode_telemetry_envelope(bytes, options.limits);
          const auto digest = sha256_hex(bytes);
          if (record.at("content_digest") != digest ||
              record.at("event") != projection(envelope, options.limits)) {
            protocol_error("jsonl.record.integrity");
          }
          const auto [known, inserted] =
              accepted.emplace(operation_key(envelope, options.limits), digest);
          if (!inserted && known->second != digest)
            protocol_error("jsonl.record.conflict");
        } catch (const Json::exception&) {
          protocol_error("jsonl.record.fields");
        }
        ++recovered.complete_records;
        complete = offset;
        line.clear();
      }
      if (!input.eof()) io_error("jsonl.recover.read");
      input.close();
      recovered.truncated_tail_bytes = offset - complete;
      if (recovered.truncated_tail_bytes != 0U) {
        std::filesystem::resize_file(options.path, complete, error);
        if (error) io_error("jsonl.recover.truncate");
      }
      file_bytes = complete;
      counters.exported_events = accepted.size();
      record_count = recovered.complete_records;
    }
    output.open(options.path, std::ios::binary | std::ios::app);
    if (!output) io_error("jsonl.append.open");
  }

  void ready() const {
    if (closed || poisoned)
      throw NotReady("JSONL observer is closed or requires recovery");
  }

  JsonlTelemetryOptions options;
  TelemetrySinkDescriptor descriptor;
  JsonlTelemetryRecovery recovered;
  std::ofstream output;
  std::map<std::string, std::string> accepted;
  TelemetryFlushResult counters;
  std::uint64_t file_bytes = 0U;
  std::size_t record_count = 0U;
  bool closed = false;
  bool poisoned = false;
};

JsonlTelemetrySink::JsonlTelemetrySink(JsonlTelemetryOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
JsonlTelemetrySink::~JsonlTelemetrySink() = default;
const TelemetrySinkDescriptor& JsonlTelemetrySink::descriptor() const noexcept {
  return impl_->descriptor;
}
JsonlTelemetryRecovery JsonlTelemetrySink::recovery() const noexcept {
  return impl_->recovered;
}

TelemetryEmitResult JsonlTelemetrySink::emit(
    std::span<const TelemetryEnvelope> envelopes, std::stop_token stop,
    TelemetryDeadline deadline) {
  check_telemetry_operation(stop, deadline);
  impl_->ready();
  validate_telemetry_batch(envelopes, impl_->options.limits);
  struct Record {
    std::string key, digest, line;
  };
  std::vector<Record> records;
  std::map<std::string, std::string> additions;
  auto next_bytes = impl_->file_bytes;
  auto next_count = impl_->record_count;
  TelemetryEmitResult result;
  for (const auto& envelope : envelopes) {
    const auto key = operation_key(envelope, impl_->options.limits);
    const auto digest =
        telemetry_content_digest(envelope, impl_->options.limits).hex;
    const auto existing = impl_->accepted.find(key);
    if (existing != impl_->accepted.end()) {
      if (existing->second != digest)
        throw CompatibilityError("Conflicting JSONL operation identity");
      result.dispositions.push_back(TelemetryDisposition::duplicate);
      continue;
    }
    const auto [known, inserted] = additions.emplace(key, digest);
    if (!inserted) {
      if (known->second != digest)
        throw CompatibilityError("Conflicting JSONL operation identity");
      result.dispositions.push_back(TelemetryDisposition::duplicate);
      continue;
    }
    const auto bytes =
        encode_telemetry_envelope(envelope, impl_->options.limits);
    const auto line =
        Json{{"schema_version", 1U},
             {"content_digest", digest},
             {"canonical_hex", hex(bytes)},
             {"event", projection(envelope, impl_->options.limits)}}
            .dump();
    bound(line.size(), impl_->options.max_line_bytes);
    bound(++next_count, impl_->options.max_records);
    if (line.size() + 1U > impl_->options.max_file_bytes - next_bytes) {
      throw ResourceExhausted("JSONL file capacity exceeded");
    }
    next_bytes += line.size() + 1U;
    records.push_back({key, digest, line});
    result.dispositions.push_back(TelemetryDisposition::accepted);
  }
  try {
    for (const auto& record : records) {
      check_telemetry_operation(stop, deadline);
      impl_->output << record.line << '\n';
      if (!impl_->output) io_error("jsonl.append.write");
    }
    impl_->output.flush();
    if (!impl_->output) io_error("jsonl.append.flush");
    check_telemetry_operation(stop, deadline);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
  impl_->accepted.merge(additions);
  impl_->file_bytes = next_bytes;
  impl_->record_count = next_count;
  impl_->counters.exported_events += records.size();
  validate_telemetry_emit_result(impl_->descriptor, envelopes, result,
                                 impl_->options.limits);
  return result;
}

TelemetryFlushResult JsonlTelemetrySink::flush(std::stop_token stop,
                                               TelemetryDeadline deadline) {
  check_telemetry_operation(stop, deadline);
  if (impl_->closed) return impl_->counters;
  impl_->ready();
  impl_->output.flush();
  if (!impl_->output) {
    impl_->poisoned = true;
    io_error("jsonl.flush");
  }
  check_telemetry_operation(stop, deadline);
  return impl_->counters;
}

TelemetryFlushResult JsonlTelemetrySink::close(std::stop_token stop,
                                               TelemetryDeadline deadline) {
  if (impl_->closed) return impl_->counters;
  const auto result = flush(stop, deadline);
  impl_->output.close();
  if (impl_->output.fail()) {
    impl_->poisoned = true;
    io_error("jsonl.close");
  }
  impl_->closed = true;
  return result;
}

class OfficialTelemetrySink::Impl {
 public:
  Impl(std::shared_ptr<TelemetrySidecarTransport> channel,
       OfficialTelemetryOptions configured)
      : transport(std::move(channel)), options(std::move(configured)) {
    if (!transport || options.max_request_bytes == 0U ||
        options.max_reply_bytes == 0U ||
        (options.backend != OfficialTelemetryBackend::tensorboard &&
         options.backend != OfficialTelemetryBackend::opentelemetry)) {
      throw InvalidArgument("Invalid official telemetry sidecar options");
    }
    backend = options.backend == OfficialTelemetryBackend::tensorboard
                  ? "tensorboard"
                  : "opentelemetry";
    descriptor = {.type_id = "jyppx.telemetry-sink." + backend + "/v1",
                  .sink_id = options.sink_id,
                  .destination_id = options.destination_id,
                  .role = TelemetrySinkRole::observer};
    validate_telemetry_sink_descriptor(descriptor, options.limits);
  }
  Json call(Json request, std::stop_token stop, TelemetryDeadline deadline) {
    check_telemetry_operation(stop, deadline);
    request["protocol_version"] = 1U;
    request["backend"] = backend;
    const auto text = request.dump();
    bound(text.size(), options.max_request_bytes);
    const auto response = parse(transport->transact(text, stop, deadline),
                                options.max_reply_bytes);
    check_telemetry_operation(stop, deadline);
    try {
      if (!response.is_object() ||
          !response.at("protocol_version").is_number_unsigned() ||
          response.at("protocol_version") != 1U ||
          response.at("request_id") != request.at("request_id")) {
        protocol_error("sidecar.reply.identity");
      }
      if (response.contains("error")) {
        if (!response.at("error").is_string() ||
            response.at("error").get_ref<const std::string&>().size() > 4096U) {
          protocol_error("sidecar.error.type");
        }
        throw BackendError("Official telemetry sidecar rejected a request");
      }
      const auto& counters_json = response.at("counters");
      const auto read = [&](const char* name) {
        if (!counters_json.at(name).is_number_unsigned())
          protocol_error("sidecar.counter.type");
        return counters_json.at(name).get<std::uint64_t>();
      };
      TelemetryFlushResult next{read("exported_events"),
                                read("sampled_out_events"),
                                read("unsupported_events")};
      if (next.exported_events < counters.exported_events ||
          next.sampled_out_events < counters.sampled_out_events ||
          next.unsupported_events < counters.unsupported_events)
        protocol_error("sidecar.counter.regression");
      return response;
    } catch (const Json::exception&) {
      protocol_error("sidecar.reply.fields");
    }
  }
  void commit_counters(const Json& response) {
    const auto& value = response.at("counters");
    counters = {value.at("exported_events").get<std::uint64_t>(),
                value.at("sampled_out_events").get<std::uint64_t>(),
                value.at("unsupported_events").get<std::uint64_t>()};
  }
  std::shared_ptr<TelemetrySidecarTransport> transport;
  OfficialTelemetryOptions options;
  TelemetrySinkDescriptor descriptor;
  TelemetryFlushResult counters;
  std::string backend;
  bool closed = false;
};

OfficialTelemetrySink::OfficialTelemetrySink(
    std::shared_ptr<TelemetrySidecarTransport> transport,
    OfficialTelemetryOptions options)
    : impl_(std::make_unique<Impl>(std::move(transport), std::move(options))) {}
OfficialTelemetrySink::~OfficialTelemetrySink() = default;
const TelemetrySinkDescriptor& OfficialTelemetrySink::descriptor()
    const noexcept {
  return impl_->descriptor;
}

TelemetryEmitResult OfficialTelemetrySink::emit(
    std::span<const TelemetryEnvelope> envelopes, std::stop_token stop,
    TelemetryDeadline deadline) {
  if (impl_->closed) throw NotReady("Official telemetry sidecar is closed");
  check_telemetry_operation(stop, deadline);
  validate_telemetry_batch(envelopes, impl_->options.limits);
  if (envelopes.empty()) return {};
  auto events = Json::array();
  std::string identity = "jyppx.telemetry.sidecar-batch/v1";
  for (const auto& event : envelopes) {
    const auto digest =
        telemetry_content_digest(event, impl_->options.limits).hex;
    identity += digest;
    events.push_back({{"content_digest", digest},
                      {"envelope", projection(event, impl_->options.limits)}});
  }
  const auto request_id = sha256_hex(
      {reinterpret_cast<const std::byte*>(identity.data()), identity.size()});
  const auto response = impl_->call(
      {{"request_id", request_id}, {"operation", "emit"}, {"events", events}},
      stop, deadline);
  TelemetryEmitResult result;
  try {
    const auto& receipts = response.at("receipts");
    if (!receipts.is_array() || receipts.size() != envelopes.size())
      protocol_error("sidecar.receipts.count");
    for (const auto& receipt : receipts) {
      if (!receipt.is_number_unsigned() || receipt.get<std::uint64_t>() < 1U ||
          receipt.get<std::uint64_t>() > 4U) {
        protocol_error("sidecar.receipt.type");
      }
      result.dispositions.push_back(
          static_cast<TelemetryDisposition>(receipt.get<std::uint64_t>()));
    }
  } catch (const Json::exception&) {
    protocol_error("sidecar.receipts.fields");
  }
  validate_telemetry_emit_result(impl_->descriptor, envelopes, result,
                                 impl_->options.limits);
  impl_->commit_counters(response);
  return result;
}

TelemetryFlushResult OfficialTelemetrySink::flush(std::stop_token stop,
                                                  TelemetryDeadline deadline) {
  if (impl_->closed) return impl_->counters;
  const auto response = impl_->call(
      {{"request_id", "flush"}, {"operation", "flush"}}, stop, deadline);
  impl_->commit_counters(response);
  return impl_->counters;
}
TelemetryFlushResult OfficialTelemetrySink::close(std::stop_token stop,
                                                  TelemetryDeadline deadline) {
  if (impl_->closed) return impl_->counters;
  const auto response = impl_->call(
      {{"request_id", "close"}, {"operation", "close"}}, stop, deadline);
  impl_->commit_counters(response);
  impl_->closed = true;
  return impl_->counters;
}

}  // namespace jyppx::experiment
