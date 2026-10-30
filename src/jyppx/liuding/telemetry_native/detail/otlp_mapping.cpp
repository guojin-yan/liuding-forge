// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "otlp_mapping.hpp"

#include "telemetry_proto.hpp"
#include "telemetry_proto/opentelemetry/proto/collector/logs/v1/logs_service.pb.h"
#include "telemetry_proto/opentelemetry/proto/collector/metrics/v1/metrics_service.pb.h"
#include "telemetry_proto/opentelemetry/proto/collector/trace/v1/trace_service.pb.h"

#include <jyppx/liuding/experiment/telemetry_exporters.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <pb_common.h>
#include <utility>

namespace jyppx::liuding::experiment::detail {
namespace {
using Json = nlohmann::json;
namespace encoding = telemetry_proto;
using KeyValue = jyppx_liuding_telemetry_opentelemetry_proto_common_v1_KeyValue;
using Scope =
    jyppx_liuding_telemetry_opentelemetry_proto_common_v1_InstrumentationScope;
using MetricsRequest =
    jyppx_liuding_telemetry_opentelemetry_proto_collector_metrics_v1_ExportMetricsServiceRequest;
using MetricsResponse =
    jyppx_liuding_telemetry_opentelemetry_proto_collector_metrics_v1_ExportMetricsServiceResponse;
using ResourceMetrics =
    jyppx_liuding_telemetry_opentelemetry_proto_metrics_v1_ResourceMetrics;
using ScopeMetrics =
    jyppx_liuding_telemetry_opentelemetry_proto_metrics_v1_ScopeMetrics;
using Metric = jyppx_liuding_telemetry_opentelemetry_proto_metrics_v1_Metric;
using NumberDataPoint =
    jyppx_liuding_telemetry_opentelemetry_proto_metrics_v1_NumberDataPoint;
using LogsRequest =
    jyppx_liuding_telemetry_opentelemetry_proto_collector_logs_v1_ExportLogsServiceRequest;
using LogsResponse =
    jyppx_liuding_telemetry_opentelemetry_proto_collector_logs_v1_ExportLogsServiceResponse;
using ResourceLogs =
    jyppx_liuding_telemetry_opentelemetry_proto_logs_v1_ResourceLogs;
using ScopeLogs = jyppx_liuding_telemetry_opentelemetry_proto_logs_v1_ScopeLogs;
using LogRecord = jyppx_liuding_telemetry_opentelemetry_proto_logs_v1_LogRecord;
using TracesRequest =
    jyppx_liuding_telemetry_opentelemetry_proto_collector_trace_v1_ExportTraceServiceRequest;
using TracesResponse =
    jyppx_liuding_telemetry_opentelemetry_proto_collector_trace_v1_ExportTraceServiceResponse;
using ResourceSpans =
    jyppx_liuding_telemetry_opentelemetry_proto_trace_v1_ResourceSpans;
using ScopeSpans =
    jyppx_liuding_telemetry_opentelemetry_proto_trace_v1_ScopeSpans;
using Span = jyppx_liuding_telemetry_opentelemetry_proto_trace_v1_Span;

struct Attribute {
  std::string key;
  enum class Kind { text, boolean, integer } kind = Kind::text;
  std::string text;
  bool boolean = false;
  std::int64_t integer = 0;
};
using Attributes = std::vector<Attribute>;

Attributes attributes(const Json& value) {
  Attributes result;
  result.reserve(value.size());
  for (const auto& [key, item] : value.items()) {
    Attribute attribute;
    attribute.key = key;
    if (item.is_boolean()) {
      attribute.kind = Attribute::Kind::boolean;
      attribute.boolean = item.get<bool>();
    } else if (item.is_number_integer() &&
               (!item.is_number_unsigned() ||
                item.get<std::uint64_t>() <=
                    static_cast<std::uint64_t>(
                        (std::numeric_limits<std::int64_t>::max)()))) {
      attribute.kind = Attribute::Kind::integer;
      attribute.integer = item.get<std::int64_t>();
    } else {
      attribute.text = item.is_string() ? item.get<std::string>() : item.dump();
    }
    result.push_back(std::move(attribute));
  }
  return result;
}

bool encode_attributes(pb_ostream_t* stream, const pb_field_t* field,
                       void* const* argument) noexcept {
  const auto& source = *static_cast<const Attributes*>(*argument);
  for (const auto& attribute : source) {
    KeyValue value{};
    const encoding::Bytes key{attribute.key}, text{attribute.text};
    encoding::bind_bytes(value.key, key);
    value.has_value = true;
    switch (attribute.kind) {
      case Attribute::Kind::boolean:
        value.value.which_value =
            jyppx_liuding_telemetry_opentelemetry_proto_common_v1_AnyValue_bool_value_tag;
        value.value.value.bool_value = attribute.boolean;
        break;
      case Attribute::Kind::integer:
        value.value.which_value =
            jyppx_liuding_telemetry_opentelemetry_proto_common_v1_AnyValue_int_value_tag;
        value.value.value.int_value = attribute.integer;
        break;
      case Attribute::Kind::text:
        value.value.which_value =
            jyppx_liuding_telemetry_opentelemetry_proto_common_v1_AnyValue_string_value_tag;
        encoding::bind_bytes(value.value.value.string_value, text);
        break;
    }
    if (!pb_encode_tag_for_field(stream, field) ||
        !pb_encode_submessage(
            stream,
            &jyppx_liuding_telemetry_opentelemetry_proto_common_v1_KeyValue_msg,
            &value))
      return false;
  }
  return true;
}

void bind_attributes(pb_callback_t& field, const Attributes& source) noexcept {
  field.funcs.encode = &encode_attributes;
  field.arg = const_cast<Attributes*>(&source);
}

Json base_attributes(const Json& value) {
  const auto& wall = value.at("time").at("wall_time_unix_ns");
  return {{"jyppx.identity", value.at("identity").dump()},
          {"jyppx.correlation", value.at("correlation").dump()},
          {"jyppx.attributes", value.at("attributes").dump()},
          {"jyppx.reliability", value.at("reliability")},
          {"jyppx.source_schema", 1},
          {"jyppx.wall_time_unix_ns", wall.is_null() ? "None" : wall.dump()}};
}

Attributes resource_attributes(const Json& value,
                               const OtlpMappingOptions& options,
                               bool correlation) {
  Json result{{"service.name", "liuding-forge"},
              {"service.instance.id", options.resource_instance_id}};
  if (correlation) {
    for (const auto* key : {"project_id", "run_id", "attempt_id"})
      result[std::string("jyppx.") + key] = value.at("correlation").at(key);
  }
  return attributes(result);
}

void bind_scope(Scope& scope, const encoding::Bytes& name,
                const encoding::Bytes& version) noexcept {
  encoding::bind_bytes(scope.name, name);
  encoding::bind_bytes(scope.version, version);
}

template <std::size_t Size>
std::array<std::byte, Size> native_id(std::string_view domain,
                                      std::string_view value) {
  Sha256 hash;
  hash.update(
      {reinterpret_cast<const std::byte*>(domain.data()), domain.size()});
  constexpr std::array zero{std::byte{0U}};
  hash.update(zero);
  hash.update({reinterpret_cast<const std::byte*>(value.data()), value.size()});
  const auto complete = hash.finish();
  std::array<std::byte, Size> result{};
  std::copy_n(complete.begin(), Size, result.begin());
  if (std::all_of(result.begin(), result.end(),
                  [](std::byte byte) { return byte == std::byte{0U}; }))
    result.back() = std::byte{1U};
  return result;
}

template <std::size_t Size>
std::string_view bytes_view(const std::array<std::byte, Size>& bytes) noexcept {
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
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
      throw InternalError("Missing OTLP point has no scalar value");
  }
}

void validate_projection_tree(const Json& value, unsigned depth,
                              std::size_t& nodes) {
  if (depth > 32U || ++nodes > 200000U)
    throw ResourceExhausted("Native OTLP projection tree capacity exceeded");
  if (value.is_object()) {
    for (const auto& [key, child] : value.items()) {
      static_cast<void>(key);
      if (depth + 1U > 32U || ++nodes > 200000U)
        throw ResourceExhausted(
            "Native OTLP projection tree capacity exceeded");
      validate_projection_tree(child, depth + 1U, nodes);
    }
  } else if (value.is_array()) {
    for (const auto& child : value)
      validate_projection_tree(child, depth + 1U, nodes);
  }
}

struct MetricInput {
  std::string name;
  std::string description;
  std::string unit;
  Attributes attributes;
  std::uint64_t wall = 0U;
  bool missing = false;
  double value = 0.0;
};

bool encode_metrics(pb_ostream_t* stream, const pb_field_t* field,
                    void* const* argument) noexcept {
  const auto& points = *static_cast<const std::vector<MetricInput>*>(*argument);
  for (const auto& point : points) {
    Metric metric{};
    NumberDataPoint sample{};
    const encoding::Bytes name{point.name}, description{point.description},
        unit{point.unit};
    encoding::bind_bytes(metric.name, name);
    encoding::bind_bytes(metric.description, description);
    encoding::bind_bytes(metric.unit, unit);
    metric.which_data =
        jyppx_liuding_telemetry_opentelemetry_proto_metrics_v1_Metric_gauge_tag;
    metric.data.gauge = {};
    sample.time_unix_nano = point.wall;
    bind_attributes(sample.attributes, point.attributes);
    if (point.missing) {
      sample.flags =
          jyppx_liuding_telemetry_opentelemetry_proto_metrics_v1_DataPointFlags_DATA_POINT_FLAGS_NO_RECORDED_VALUE_MASK;
    } else {
      sample.which_value =
          jyppx_liuding_telemetry_opentelemetry_proto_metrics_v1_NumberDataPoint_as_double_tag;
      sample.value.as_double = point.value;
    }
    const encoding::Message message{
        &jyppx_liuding_telemetry_opentelemetry_proto_metrics_v1_NumberDataPoint_msg,
        &sample};
    encoding::bind_message(metric.data.gauge.data_points, message);
    if (!pb_encode_tag_for_field(stream, field) ||
        !pb_encode_submessage(
            stream,
            &jyppx_liuding_telemetry_opentelemetry_proto_metrics_v1_Metric_msg,
            &metric))
      return false;
  }
  return true;
}

std::optional<PreparedOtlpRequest> metrics_request(
    const Json& value, const OtlpMappingOptions& options, std::stop_token stop,
    TelemetryDeadline deadline) {
  const auto& wall = value.at("time").at("wall_time_unix_ns");
  if (wall.is_null() || wall.get<std::int64_t>() < 0) return std::nullopt;
  const auto& payload = value.at("payload");
  const auto kind = payload.at("kind").get<unsigned>();
  const auto& points = payload.at(kind == 1U   ? "metrics"
                                  : kind == 2U ? "points"
                                               : "measurements");
  if (points.empty()) return std::nullopt;
  constexpr std::array<std::string_view, 10> units{
      "", "1", "{count}", "s", "By", "1", "%", "{count}/s", "By/s", ""};
  std::vector<MetricInput> inputs;
  inputs.reserve(points.size());
  for (const auto& point : points) {
    check_telemetry_operation(stop, deadline);
    const auto& definition = point.at("definition");
    MetricInput input;
    input.name =
        "jyppx.scalar." + definition.at("definition_digest").get<std::string>();
    input.description = definition.at("metric_id").get<std::string>();
    input.unit = units.at(definition.at("unit").get<std::size_t>());
    input.attributes = attributes(
        {{"jyppx.metric_definition", definition.dump()},
         {"jyppx.metric_context", point.at("context").dump()},
         {"jyppx.step_kind", point.at("step").at("kind")},
         {"jyppx.resource_id", payload.value("resource_id", std::string{})}});
    input.wall = wall.get<std::uint64_t>();
    input.missing = point.at("status") == 5U;
    if (!input.missing) input.value = symbolic(point);
    inputs.push_back(std::move(input));
  }
  MetricsRequest request{};
  ResourceMetrics resource{};
  ScopeMetrics scope{};
  const auto resource_fields = resource_attributes(value, options, false);
  resource.has_resource = true;
  bind_attributes(resource.resource.attributes, resource_fields);
  scope.has_scope = true;
  const encoding::Bytes scope_name{"jyppx.telemetry"}, scope_version{"1"};
  bind_scope(scope.scope, scope_name, scope_version);
  scope.metrics.funcs.encode = &encode_metrics;
  scope.metrics.arg = &inputs;
  const encoding::Message scope_message{
      &jyppx_liuding_telemetry_opentelemetry_proto_metrics_v1_ScopeMetrics_msg,
      &scope};
  encoding::bind_message(resource.scope_metrics, scope_message);
  const encoding::Message resource_message{
      &jyppx_liuding_telemetry_opentelemetry_proto_metrics_v1_ResourceMetrics_msg,
      &resource};
  encoding::bind_message(request.resource_metrics, resource_message);
  auto body = encoding::encode(
      &jyppx_liuding_telemetry_opentelemetry_proto_collector_metrics_v1_ExportMetricsServiceRequest_msg,
      &request, options.max_request_bytes);
  check_telemetry_operation(stop, deadline);
  return PreparedOtlpRequest{OtlpSignal::metrics, std::move(body)};
}

std::optional<PreparedOtlpRequest> logs_request(
    const Json& value, const OtlpMappingOptions& options, std::stop_token stop,
    TelemetryDeadline deadline) {
  const auto& wall = value.at("time").at("wall_time_unix_ns");
  if (!wall.is_null() && wall.get<std::int64_t>() < 0) return std::nullopt;
  LogsRequest request{};
  ResourceLogs resource{};
  ScopeLogs scope{};
  LogRecord record{};
  const auto resource_fields = resource_attributes(value, options, true);
  resource.has_resource = true;
  bind_attributes(resource.resource.attributes, resource_fields);
  scope.has_scope = true;
  const encoding::Bytes scope_name{"jyppx.telemetry"}, scope_version{"1"};
  bind_scope(scope.scope, scope_name, scope_version);
  if (!wall.is_null()) record.time_unix_nano = wall.get<std::uint64_t>();
  const auto& payload = value.at("payload");
  const auto severity =
      payload.at("kind") == 3U ? payload.at("severity").get<unsigned>() : 2U;
  constexpr std::array severities{
      jyppx_liuding_telemetry_opentelemetry_proto_logs_v1_SeverityNumber_SEVERITY_NUMBER_UNSPECIFIED,
      jyppx_liuding_telemetry_opentelemetry_proto_logs_v1_SeverityNumber_SEVERITY_NUMBER_DEBUG,
      jyppx_liuding_telemetry_opentelemetry_proto_logs_v1_SeverityNumber_SEVERITY_NUMBER_INFO,
      jyppx_liuding_telemetry_opentelemetry_proto_logs_v1_SeverityNumber_SEVERITY_NUMBER_WARN,
      jyppx_liuding_telemetry_opentelemetry_proto_logs_v1_SeverityNumber_SEVERITY_NUMBER_ERROR};
  constexpr std::array<std::string_view, 5> severity_texts{"", "DEBUG", "INFO",
                                                           "WARN", "ERROR"};
  record.severity_number = severities.at(severity);
  const encoding::Bytes severity_text{severity_texts.at(severity)};
  encoding::bind_bytes(record.severity_text, severity_text);
  const auto body = payload.dump();
  const encoding::Bytes body_bytes{body};
  record.has_body = true;
  record.body.which_value =
      jyppx_liuding_telemetry_opentelemetry_proto_common_v1_AnyValue_string_value_tag;
  encoding::bind_bytes(record.body.value.string_value, body_bytes);
  auto base = base_attributes(value);
  base["jyppx.timestamp_missing"] = wall.is_null();
  base["jyppx.envelope"] = value.dump();
  const auto record_fields = attributes(base);
  bind_attributes(record.attributes, record_fields);
  const auto& correlation = value.at("correlation");
  const auto trace =
      native_id<16U>("jyppx.trace/v1",
                     correlation.at("trace_id").get_ref<const std::string&>());
  const auto span = native_id<8U>(
      "jyppx.span/v1", correlation.at("span_id").get_ref<const std::string&>());
  const encoding::Bytes trace_bytes{bytes_view(trace)},
      span_bytes{bytes_view(span)};
  if (!correlation.at("trace_id").get_ref<const std::string&>().empty())
    encoding::bind_bytes(record.trace_id, trace_bytes);
  if (!correlation.at("span_id").get_ref<const std::string&>().empty())
    encoding::bind_bytes(record.span_id, span_bytes);
  const encoding::Message record_message{
      &jyppx_liuding_telemetry_opentelemetry_proto_logs_v1_LogRecord_msg,
      &record};
  encoding::bind_message(scope.log_records, record_message);
  const encoding::Message scope_message{
      &jyppx_liuding_telemetry_opentelemetry_proto_logs_v1_ScopeLogs_msg,
      &scope};
  encoding::bind_message(resource.scope_logs, scope_message);
  const encoding::Message resource_message{
      &jyppx_liuding_telemetry_opentelemetry_proto_logs_v1_ResourceLogs_msg,
      &resource};
  encoding::bind_message(request.resource_logs, resource_message);
  auto encoded = encoding::encode(
      &jyppx_liuding_telemetry_opentelemetry_proto_collector_logs_v1_ExportLogsServiceRequest_msg,
      &request, options.max_request_bytes);
  check_telemetry_operation(stop, deadline);
  return PreparedOtlpRequest{OtlpSignal::logs, std::move(encoded)};
}

std::optional<std::uint64_t> anchored(std::uint64_t monotonic,
                                      const OtlpClockAnchor& anchor) noexcept {
  if (monotonic >= anchor.monotonic_ns) {
    const auto delta = monotonic - anchor.monotonic_ns;
    if (delta > (std::numeric_limits<std::uint64_t>::max)() - anchor.unix_ns)
      return std::nullopt;
    return anchor.unix_ns + delta;
  }
  const auto delta = anchor.monotonic_ns - monotonic;
  if (delta > anchor.unix_ns) return std::nullopt;
  return anchor.unix_ns - delta;
}

std::optional<PreparedOtlpRequest> traces_request(
    const Json& value, const OtlpMappingOptions& options, std::stop_token stop,
    TelemetryDeadline deadline) {
  const auto& clock =
      value.at("time").at("clock_id").get_ref<const std::string&>();
  const auto anchor = options.clock_mappings.find(clock);
  if (anchor == options.clock_mappings.end()) return std::nullopt;
  const auto& payload = value.at("payload");
  const auto start = anchored(
      payload.at("start_monotonic_ns").get<std::uint64_t>(), anchor->second);
  const auto end = anchored(payload.at("end_monotonic_ns").get<std::uint64_t>(),
                            anchor->second);
  if (!start || !end || *start > *end) return std::nullopt;
  TracesRequest request{};
  ResourceSpans resource{};
  ScopeSpans scope{};
  Span span{};
  const auto resource_fields = resource_attributes(value, options, true);
  resource.has_resource = true;
  bind_attributes(resource.resource.attributes, resource_fields);
  scope.has_scope = true;
  const encoding::Bytes scope_name{"jyppx.telemetry"}, scope_version{"1"};
  bind_scope(scope.scope, scope_name, scope_version);
  const auto& correlation = value.at("correlation");
  const auto trace =
      native_id<16U>("jyppx.trace/v1",
                     correlation.at("trace_id").get_ref<const std::string&>());
  const auto id = native_id<8U>(
      "jyppx.span/v1", correlation.at("span_id").get_ref<const std::string&>());
  const auto parent = native_id<8U>(
      "jyppx.span/v1",
      correlation.at("parent_span_id").get_ref<const std::string&>());
  const encoding::Bytes trace_bytes{bytes_view(trace)},
      id_bytes{bytes_view(id)}, parent_bytes{bytes_view(parent)};
  encoding::bind_bytes(span.trace_id, trace_bytes);
  encoding::bind_bytes(span.span_id, id_bytes);
  if (!correlation.at("parent_span_id").get_ref<const std::string&>().empty())
    encoding::bind_bytes(span.parent_span_id, parent_bytes);
  const encoding::Bytes name{payload.at("name").get_ref<const std::string&>()};
  encoding::bind_bytes(span.name, name);
  span.kind =
      jyppx_liuding_telemetry_opentelemetry_proto_trace_v1_Span_SpanKind_SPAN_KIND_INTERNAL;
  span.start_time_unix_nano = *start;
  span.end_time_unix_nano = *end;
  span.has_status = true;
  constexpr std::array statuses{
      jyppx_liuding_telemetry_opentelemetry_proto_trace_v1_Status_StatusCode_STATUS_CODE_UNSET,
      jyppx_liuding_telemetry_opentelemetry_proto_trace_v1_Status_StatusCode_STATUS_CODE_UNSET,
      jyppx_liuding_telemetry_opentelemetry_proto_trace_v1_Status_StatusCode_STATUS_CODE_OK,
      jyppx_liuding_telemetry_opentelemetry_proto_trace_v1_Status_StatusCode_STATUS_CODE_ERROR,
      jyppx_liuding_telemetry_opentelemetry_proto_trace_v1_Status_StatusCode_STATUS_CODE_ERROR};
  span.status.code = statuses.at(payload.at("status").get<std::size_t>());
  const encoding::Bytes reason{
      payload.at("reason").get_ref<const std::string&>()};
  encoding::bind_bytes(span.status.message, reason);
  auto base = base_attributes(value);
  base["jyppx.clock_id"] = clock;
  base["jyppx.clock_mapping"] = Json{
      {"monotonic_ns", anchor->second.monotonic_ns},
      {"unix_ns",
       anchor->second.unix_ns}}.dump();
  base["jyppx.span_status"] = payload.at("status");
  const auto span_fields = attributes(base);
  bind_attributes(span.attributes, span_fields);
  const encoding::Message span_message{
      &jyppx_liuding_telemetry_opentelemetry_proto_trace_v1_Span_msg, &span};
  encoding::bind_message(scope.spans, span_message);
  const encoding::Message scope_message{
      &jyppx_liuding_telemetry_opentelemetry_proto_trace_v1_ScopeSpans_msg,
      &scope};
  encoding::bind_message(resource.scope_spans, scope_message);
  const encoding::Message resource_message{
      &jyppx_liuding_telemetry_opentelemetry_proto_trace_v1_ResourceSpans_msg,
      &resource};
  encoding::bind_message(request.resource_spans, resource_message);
  auto encoded = encoding::encode(
      &jyppx_liuding_telemetry_opentelemetry_proto_collector_trace_v1_ExportTraceServiceRequest_msg,
      &request, options.max_request_bytes);
  check_telemetry_operation(stop, deadline);
  return PreparedOtlpRequest{OtlpSignal::traces, std::move(encoded)};
}

bool read_error_message(pb_istream_t* stream, const pb_field_t*,
                        void** argument) noexcept {
  try {
    // Last occurrence wins, like the official singular protobuf string. Check
    // every occurrence's UTF-8, including portions after embedded NUL bytes.
    std::string text(stream->bytes_left, '\0');
    if (!pb_read(stream, reinterpret_cast<pb_byte_t*>(text.data()),
                 text.size()))
      return false;
    std::size_t offset = 0U;
    for (;;) {
      if (!pb_validate_utf8(text.c_str() + offset)) return false;
      const auto end = text.find('\0', offset);
      if (end == std::string::npos) break;
      offset = end + 1U;
    }
    *static_cast<bool*>(*argument) = !text.empty();
    return true;
  } catch (...) {
    return false;  // Never let a C callback unwind across the mature runtime.
  }
}

template <typename Response, typename Count>
void decode_response(const pb_msgdesc_t* descriptor,
                     std::span<const std::byte> bytes, Count rejected) {
  Response reply{};
  bool error_message = false;
  reply.partial_success.error_message.funcs.decode = &read_error_message;
  reply.partial_success.error_message.arg = &error_message;
  auto stream = pb_istream_from_buffer(
      reinterpret_cast<const pb_byte_t*>(bytes.data()), bytes.size());
  if (!pb_decode(&stream, descriptor, &reply) || stream.bytes_left != 0U)
    throw SerializationError("OTLP collector returned malformed protobuf");
  if (error_message || rejected(reply.partial_success) != 0)
    throw IoError("OTLP collector reported partial success");
}
}  // namespace

void validate_otlp_mapping_options(const OtlpMappingOptions& options) {
  validate_telemetry_limits(options.limits);
  const TelemetryLimits reference;
  const auto& limits = options.limits;
  if (limits.max_identifier_bytes > reference.max_identifier_bytes ||
      limits.max_text_bytes > reference.max_text_bytes ||
      limits.max_attributes > reference.max_attributes ||
      limits.max_context_fields > reference.max_context_fields ||
      limits.max_metrics_per_envelope > reference.max_metrics_per_envelope ||
      limits.max_decisions_per_envelope >
          reference.max_decisions_per_envelope ||
      limits.max_batch_events > reference.max_batch_events ||
      limits.max_envelope_bytes > reference.max_envelope_bytes ||
      limits.max_batch_bytes > reference.max_batch_bytes ||
      limits.max_artifact_reference_bytes >
          reference.max_artifact_reference_bytes)
    throw InvalidArgument(
        "Native OTLP cannot widen the fixed neutral mapping bounds");
  if (options.resource_instance_id.empty() ||
      options.resource_instance_id.size() > 256U ||
      options.resource_instance_id.find('\0') != std::string::npos ||
      options.clock_mappings.size() > 64U || options.max_request_bytes == 0U ||
      options.max_request_bytes > 16U * 1024U * 1024U ||
      options.max_reply_bytes == 0U ||
      options.max_reply_bytes > 1024U * 1024U ||
      options.max_prepared_bytes == 0U ||
      options.max_prepared_bytes > 32U * 1024U * 1024U ||
      options.max_request_bytes > options.max_prepared_bytes)
    throw InvalidArgument("Invalid native OTLP mapping options");
  // Mature runtime UTF-8 validator, also used for explicit clock/resource IDs.
  if (!pb_validate_utf8(options.resource_instance_id.c_str()))
    throw InvalidArgument("Native OTLP instance identity is not valid UTF-8");
  for (const auto& [clock, anchor] : options.clock_mappings) {
    static_cast<void>(anchor);
    if (clock.empty() || clock.size() > 256U ||
        clock.find('\0') != std::string::npos ||
        !pb_validate_utf8(clock.c_str()))
      throw InvalidArgument("Invalid native OTLP clock identity");
  }
}

std::vector<PreparedOtlpRequest> prepare_otlp_requests(
    const TelemetryEnvelope& envelope, const OtlpMappingOptions& options,
    std::stop_token stop, TelemetryDeadline deadline) {
  check_telemetry_operation(stop, deadline);
  validate_otlp_mapping_options(options);
  const auto value =
      Json::parse(telemetry_projection_json(envelope, options.limits));
  std::size_t nodes = 0U;
  validate_projection_tree(value, 0U, nodes);
  const auto kind = value.at("payload").at("kind").get<unsigned>();
  std::vector<PreparedOtlpRequest> result;
  result.reserve(2U);
  if (kind == 2U || kind == 5U ||
      (kind == 1U && !value.at("payload").at("metrics").empty())) {
    auto measured = metrics_request(value, options, stop, deadline);
    if (!measured) return {};
    auto audit_options = options;
    const auto remaining = options.max_prepared_bytes - measured->body.size();
    if (remaining == 0U)
      throw ResourceExhausted("Native OTLP prepared request capacity exceeded");
    audit_options.max_request_bytes =
        (std::min)(audit_options.max_request_bytes, remaining);
    auto audit = logs_request(value, audit_options, stop, deadline);
    if (!audit) return {};
    result.push_back(std::move(*measured));
    result.push_back(std::move(*audit));
  } else {
    auto prepared = kind == 4U ? traces_request(value, options, stop, deadline)
                               : logs_request(value, options, stop, deadline);
    if (prepared) result.push_back(std::move(*prepared));
  }
  check_telemetry_operation(stop, deadline);
  return result;
}

void validate_otlp_reply(OtlpSignal signal, std::span<const std::byte> bytes,
                         std::size_t maximum) {
  if (maximum == 0U || maximum > 1024U * 1024U)
    throw InvalidArgument("Invalid OTLP reply capacity");
  if (bytes.size() > maximum)
    throw ResourceExhausted("OTLP collector reply capacity exceeded");
  switch (signal) {
    case OtlpSignal::metrics:
      decode_response<MetricsResponse>(
          &jyppx_liuding_telemetry_opentelemetry_proto_collector_metrics_v1_ExportMetricsServiceResponse_msg,
          bytes,
          [](const auto& partial) { return partial.rejected_data_points; });
      return;
    case OtlpSignal::logs:
      decode_response<LogsResponse>(
          &jyppx_liuding_telemetry_opentelemetry_proto_collector_logs_v1_ExportLogsServiceResponse_msg,
          bytes,
          [](const auto& partial) { return partial.rejected_log_records; });
      return;
    case OtlpSignal::traces:
      decode_response<TracesResponse>(
          &jyppx_liuding_telemetry_opentelemetry_proto_collector_trace_v1_ExportTraceServiceResponse_msg,
          bytes, [](const auto& partial) { return partial.rejected_spans; });
      return;
  }
  throw InvalidArgument("Unknown OTLP signal");
}

std::string_view otlp_signal_path(OtlpSignal signal) {
  switch (signal) {
    case OtlpSignal::metrics:
      return "/v1/metrics";
    case OtlpSignal::logs:
      return "/v1/logs";
    case OtlpSignal::traces:
      return "/v1/traces";
  }
  throw InvalidArgument("Unknown OTLP signal");
}

}  // namespace jyppx::liuding::experiment::detail
