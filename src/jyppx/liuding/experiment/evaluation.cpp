// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "jyppx/liuding/experiment/evaluation.hpp"

#include "jyppx/liuding/foundation/error.hpp"
#include "jyppx/liuding/foundation/sha256.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <tuple>
#include <utility>

namespace jyppx::liuding::experiment {
namespace {
using Json = nlohmann::json;

[[noreturn]] void reject(std::string operation, std::string actual,
                         ErrorCode code = ErrorCode::invalid_argument) {
  ErrorDetails details{
      .component = "experiment.evaluation",
      .operation = std::move(operation),
      .expected = "bounded compatible schema-v1 evidence",
      .actual = std::move(actual),
      .hint = "Use the immutable suite and complete producer provenance."};
  const std::string message = "Invalid local evaluation evidence";
  switch (code) {
    case ErrorCode::invalid_argument:
      throw InvalidArgument(message, std::move(details));
    case ErrorCode::serialization:
      throw SerializationError(message, std::move(details));
    case ErrorCode::compatibility:
      throw CompatibilityError(message, std::move(details));
    case ErrorCode::resource_exhausted:
      throw ResourceExhausted(message, std::move(details));
    default:
      throw Error(code, message, std::move(details));
  }
}

void bound(std::size_t count, std::size_t maximum, std::string_view field) {
  if (count > maximum) {
    reject(std::string(field), "configured limit exceeded",
           ErrorCode::resource_exhausted);
  }
}

void text(std::string_view value, std::string_view field,
          const EvaluationLimits &limits, bool required = true) {
  bound(value.size(), limits.max_text_bytes, field);
  if ((required && value.empty()) ||
      std::any_of(value.begin(), value.end(),
                  [](unsigned char c) { return c < 0x20U || c == 0x7fU; })) {
    reject(std::string(field), "empty or control-bearing text");
  }
  try {
    static_cast<void>(Json(value).dump());
  } catch (const Json::exception &) {
    reject(std::string(field), "invalid UTF-8");
  }
}

void validate_limits(const EvaluationLimits &limits) {
  if (limits.max_rules == 0U || limits.max_text_bytes == 0U ||
      limits.max_json_bytes == 0U || limits.max_json_nodes == 0U ||
      limits.max_json_depth == 0U || limits.max_data_rows == 0U ||
      limits.max_data_fields == 0U || limits.max_data_text_bytes == 0U) {
    reject("limits", "zero bound");
  }
  validate_telemetry_limits(limits.telemetry);
}

void version(std::uint32_t value, std::string_view field) {
  if (value != 1U) {
    reject(std::string(field), "unsupported schema", ErrorCode::compatibility);
  }
}

void fields(const Json &value, std::initializer_list<std::string_view> names) {
  if (!value.is_object() || value.size() != names.size()) {
    reject("json.fields", "missing or unknown fields",
           ErrorCode::serialization);
  }
  for (auto name : names) {
    if (!value.contains(std::string(name))) {
      reject("json.fields", "missing or unknown fields",
             ErrorCode::serialization);
    }
  }
}

template <typename T>
T unsigned_number(const Json &value) {
  if (!(value.is_number_unsigned() ||
        (value.is_number_integer() && value.get<std::int64_t>() >= 0))) {
    reject("json.integer", "expected nonnegative integer",
           ErrorCode::serialization);
  }
  const auto number = value.get<std::uint64_t>();
  if (number > std::numeric_limits<T>::max()) {
    reject("json.integer", "integer overflow", ErrorCode::serialization);
  }
  return static_cast<T>(number);
}

std::optional<double> optional_number(const Json &value) {
  if (value.is_null()) return std::nullopt;
  if (!value.is_number()) {
    reject("json.number", "expected finite number or null",
           ErrorCode::serialization);
  }
  const auto number = value.get<double>();
  if (!std::isfinite(number)) reject("json.number", "nonfinite number");
  return number == 0.0 ? 0.0 : number;
}

Json number_json(const std::optional<double> &number) {
  return number ? Json(*number == 0.0 ? 0.0 : *number) : Json(nullptr);
}

Json digest_json(const ArtifactDigest &digest) {
  return Json{{"algorithm", "sha256"}, {"hex", digest.hex}};
}

ArtifactDigest read_digest(const Json &value) {
  fields(value, {"algorithm", "hex"});
  if (value.at("algorithm") != "sha256") {
    reject("digest.algorithm", "unsupported digest", ErrorCode::compatibility);
  }
  ArtifactDigest digest{ArtifactDigestAlgorithm::sha256,
                        value.at("hex").get<std::string>()};
  validate_artifact_digest(digest);
  return digest;
}

void consume_json_node(std::size_t &nodes, const EvaluationLimits &limits) {
  if (nodes == limits.max_json_nodes) {
    reject("json.nodes", "configured node limit exceeded",
           ErrorCode::resource_exhausted);
  }
  ++nodes;
}

void validate_json_budget(const Json &value, const EvaluationLimits &limits,
                          std::size_t depth, std::size_t &nodes) {
  consume_json_node(nodes, limits);
  if (value.is_string()) {
    bound(value.get_ref<const std::string &>().size(), limits.max_text_bytes,
          "json.string");
  }
  if (value.is_object() || value.is_array()) {
    bound(depth + 1U, limits.max_json_depth, "json.depth");
    for (auto it = value.begin(); it != value.end(); ++it) {
      if (value.is_object()) {
        consume_json_node(nodes, limits);
        bound(it.key().size(), limits.max_text_bytes, "json.string");
      }
      validate_json_budget(it.value(), limits, depth + 1U, nodes);
    }
  }
}

std::string finish_json(const Json &value, const EvaluationLimits &limits) {
  try {
    std::size_t nodes = 0U;
    validate_json_budget(value, limits, 0U, nodes);
    auto result = value.dump();
    bound(result.size(), limits.max_json_bytes, "json.bytes");
    return result;
  } catch (const Json::exception &) {
    reject("json.encode", "malformed object text or value",
           ErrorCode::serialization);
  }
}

Json parse_json(std::string_view value, const EvaluationLimits &limits) {
  validate_limits(limits);
  bound(value.size(), limits.max_json_bytes, "json.bytes");
  std::vector<std::set<std::string>> keys;
  std::size_t nodes = 0U;
  auto callback = [&](int depth, Json::parse_event_t event, Json &parsed) {
    if (event == Json::parse_event_t::object_start ||
        event == Json::parse_event_t::array_start) {
      bound(static_cast<std::size_t>(depth + 1), limits.max_json_depth,
            "json.depth");
    }
    if (event == Json::parse_event_t::object_start) keys.emplace_back();
    if (event == Json::parse_event_t::key) {
      if (keys.empty() ||
          !keys.back().insert(parsed.get<std::string>()).second) {
        reject("json.key", "duplicate field", ErrorCode::serialization);
      }
    }
    if (event == Json::parse_event_t::object_end && !keys.empty())
      keys.pop_back();
    if (event == Json::parse_event_t::object_start ||
        event == Json::parse_event_t::array_start ||
        event == Json::parse_event_t::key ||
        event == Json::parse_event_t::value) {
      consume_json_node(nodes, limits);
      if (parsed.is_string()) {
        bound(parsed.get_ref<const std::string &>().size(),
              limits.max_text_bytes, "json.string");
      }
    }
    return true;
  };
  try {
    return Json::parse(value.begin(), value.end(), callback, true, false);
  } catch (const Json::exception &) {
    reject("json.syntax", "malformed JSON or UTF-8", ErrorCode::serialization);
  }
}

template <typename Callable>
auto decode(Callable &&callable) {
  try {
    return callable();
  } catch (const Json::exception &) {
    reject("json.value", "wrong field type or numeric range",
           ErrorCode::serialization);
  } catch (const std::bad_alloc &) {
    reject("json.memory", "allocation failed", ErrorCode::resource_exhausted);
  }
}

Json definition_json(const TelemetryMetricDefinition &value) {
  return Json{{"schema_version", value.schema_version},
              {"metric_id", value.metric_id},
              {"version", value.version},
              {"unit", static_cast<unsigned>(value.unit)},
              {"aggregation", static_cast<unsigned>(value.aggregation)},
              {"direction", static_cast<unsigned>(value.direction)},
              {"parameters", value.parameters},
              {"context_keys", value.context_keys}};
}

TelemetryMetricDefinition read_definition(const Json &value) {
  fields(value, {"schema_version", "metric_id", "version", "unit",
                 "aggregation", "direction", "parameters", "context_keys"});
  TelemetryMetricDefinition result;
  result.schema_version =
      unsigned_number<std::uint32_t>(value.at("schema_version"));
  result.metric_id = value.at("metric_id").get<std::string>();
  result.version = unsigned_number<std::uint32_t>(value.at("version"));
  result.unit = static_cast<TelemetryMetricUnit>(
      unsigned_number<std::uint8_t>(value.at("unit")));
  result.aggregation = static_cast<TelemetryAggregation>(
      unsigned_number<std::uint8_t>(value.at("aggregation")));
  result.direction = static_cast<TelemetryMetricDirection>(
      unsigned_number<std::uint8_t>(value.at("direction")));
  result.parameters = value.at("parameters").get<TelemetryAttributes>();
  const auto context_keys =
      value.at("context_keys").get<std::vector<std::string>>();
  result.context_keys =
      std::set<std::string>(context_keys.begin(), context_keys.end());
  if (context_keys.size() != result.context_keys.size()) {
    reject("metric.context_keys", "duplicate context key",
           ErrorCode::serialization);
  }
  return result;
}

Json point_json(const TelemetryMetricPoint &value) {
  return Json{{"definition", definition_json(value.definition)},
              {"step",
               {{"kind", static_cast<unsigned>(value.step.kind)},
                {"value", value.step.value}}},
              {"observation",
               {{"status", static_cast<unsigned>(value.observation.status)},
                {"value", number_json(value.observation.value)},
                {"reason", value.observation.reason}}},
              {"context", value.context},
              {"sample_count",
               value.sample_count ? Json(*value.sample_count) : Json(nullptr)}};
}

TelemetryMetricPoint read_point(const Json &value) {
  fields(value,
         {"definition", "step", "observation", "context", "sample_count"});
  const auto &step = value.at("step");
  const auto &observation = value.at("observation");
  fields(step, {"kind", "value"});
  fields(observation, {"status", "value", "reason"});
  return TelemetryMetricPoint{
      .definition = read_definition(value.at("definition")),
      .step = {.kind = static_cast<TelemetryStepKind>(
                   unsigned_number<std::uint8_t>(step.at("kind"))),
               .value = unsigned_number<std::uint64_t>(step.at("value"))},
      .observation = {.status = static_cast<TelemetryValueStatus>(
                          unsigned_number<std::uint8_t>(
                              observation.at("status"))),
                      .value = optional_number(observation.at("value")),
                      .reason = observation.at("reason").get<std::string>()},
      .context = value.at("context").get<TelemetryAttributes>(),
      .sample_count = value.at("sample_count").is_null()
                          ? std::nullopt
                          : std::optional(unsigned_number<std::uint64_t>(
                                value.at("sample_count")))};
}

void validate_point(const TelemetryMetricPoint &point,
                    const EvaluationLimits &limits) {
  TelemetryEnvelope envelope;
  envelope.identity = {"evaluation.validation", "point", "point", 1U};
  envelope.payload = TelemetryMetricBatch{{point}};
  validate_telemetry_envelope(envelope, limits.telemetry);
}

std::string metric_key(const TelemetryMetricDefinition &definition,
                       const TelemetryAttributes &context,
                       const EvaluationLimits &limits) {
  return telemetry_metric_definition_digest(definition, limits.telemetry).hex +
         Json(context).dump();
}

void interval(const std::optional<double> &minimum,
              const std::optional<double> &maximum) {
  if ((minimum && !std::isfinite(*minimum)) ||
      (maximum && !std::isfinite(*maximum)) ||
      (minimum && maximum && *minimum > *maximum)) {
    reject("threshold", "nonfinite or reversed interval");
  }
}

void validate_identity(const EvaluationIdentity &identity,
                       const EvaluationLimits &limits) {
  text(identity.evaluation_id, "evaluation_id", limits);
  text(identity.run_id, "run_id", limits);
  text(identity.build_fingerprint, "build_fingerprint", limits);
  text(identity.adapter_id, "adapter_id", limits);
  if (identity.adapter_version == 0U) reject("adapter_version", "zero version");
  validate_artifact_digest(identity.model_digest);
  validate_artifact_digest(identity.config_digest);
}

Json suite_json(const EvaluationSuite &suite) {
  Json rules = Json::array();
  for (const auto &rule : suite.rules) {
    rules.push_back(
        {{"rule_id", rule.rule_id},
         {"definition", definition_json(rule.definition)},
         {"context", rule.context},
         {"minimum_samples", rule.minimum_samples},
         {"minimum", number_json(rule.minimum)},
         {"maximum", number_json(rule.maximum)},
         {"max_absolute_regression", number_json(rule.max_absolute_regression)},
         {"max_relative_regression",
          number_json(rule.max_relative_regression)}});
  }
  return Json{{"schema_version", suite.schema_version},
              {"suite_id", suite.suite_id},
              {"version", suite.version},
              {"task", suite.task},
              {"dataset_digest", digest_json(suite.dataset_digest)},
              {"environment_id", suite.environment_id},
              {"seed", suite.seed},
              {"rules", rules}};
}

EvaluationSuite read_suite(const Json &value, const EvaluationLimits &limits) {
  fields(value, {"schema_version", "suite_id", "version", "task",
                 "dataset_digest", "environment_id", "seed", "rules"});
  EvaluationSuite suite;
  suite.schema_version =
      unsigned_number<std::uint32_t>(value.at("schema_version"));
  suite.suite_id = value.at("suite_id").get<std::string>();
  suite.version = unsigned_number<std::uint32_t>(value.at("version"));
  suite.task = value.at("task").get<std::string>();
  suite.dataset_digest = read_digest(value.at("dataset_digest"));
  suite.environment_id = value.at("environment_id").get<std::string>();
  suite.seed = unsigned_number<std::uint64_t>(value.at("seed"));
  const auto &rules = value.at("rules");
  if (!rules.is_array())
    reject("rules", "expected array", ErrorCode::serialization);
  bound(rules.size(), limits.max_rules, "rules");
  for (const auto &rule : rules) {
    fields(rule,
           {"rule_id", "definition", "context", "minimum_samples", "minimum",
            "maximum", "max_absolute_regression", "max_relative_regression"});
    suite.rules.push_back(EvaluationRule{
        .rule_id = rule.at("rule_id").get<std::string>(),
        .definition = read_definition(rule.at("definition")),
        .context = rule.at("context").get<TelemetryAttributes>(),
        .minimum_samples =
            unsigned_number<std::uint64_t>(rule.at("minimum_samples")),
        .minimum = optional_number(rule.at("minimum")),
        .maximum = optional_number(rule.at("maximum")),
        .max_absolute_regression =
            optional_number(rule.at("max_absolute_regression")),
        .max_relative_regression =
            optional_number(rule.at("max_relative_regression"))});
  }
  validate_evaluation_suite(suite, limits);
  return suite;
}

int priority(QualityVerdict verdict) {
  switch (verdict) {
    case QualityVerdict::pass:
      return 0;
    case QualityVerdict::warn:
      return 1;
    case QualityVerdict::incomplete:
      return 2;
    case QualityVerdict::fail:
      return 3;
    case QualityVerdict::invalid:
      return 4;
  }
  reject("verdict", "unknown verdict");
}

QualityVerdict combine(QualityVerdict left, QualityVerdict right) {
  return priority(left) >= priority(right) ? left : right;
}

QualityRuleResult apply_rule(const EvaluationRule &rule,
                             const std::optional<TelemetryMetricPoint> &point) {
  QualityRuleResult result{.rule_id = rule.rule_id,
                           .verdict = QualityVerdict::incomplete,
                           .reason = {},
                           .observation = point};
  if (!point)
    result.reason = "required observation missing";
  else if (point->observation.status != TelemetryValueStatus::finite) {
    result.reason = "observation is nonfinite or missing";
  } else if (!point->sample_count ||
             *point->sample_count < rule.minimum_samples) {
    result.reason = "known sample count below suite minimum";
  } else {
    const auto value = *point->observation.value;
    if ((rule.minimum && value < *rule.minimum) ||
        (rule.maximum && value > *rule.maximum)) {
      result.verdict = QualityVerdict::fail;
      result.reason = "observation outside inclusive threshold";
    } else
      result.verdict = QualityVerdict::pass;
  }
  return result;
}

Json identity_json(const EvaluationIdentity &identity) {
  return Json{{"evaluation_id", identity.evaluation_id},
              {"run_id", identity.run_id},
              {"model_digest", digest_json(identity.model_digest)},
              {"build_fingerprint", identity.build_fingerprint},
              {"adapter_id", identity.adapter_id},
              {"adapter_version", identity.adapter_version},
              {"config_digest", digest_json(identity.config_digest)}};
}

EvaluationIdentity read_identity(const Json &source,
                                 const EvaluationLimits &limits) {
  fields(source,
         {"evaluation_id", "run_id", "model_digest", "build_fingerprint",
          "adapter_id", "adapter_version", "config_digest"});
  EvaluationIdentity result{
      .evaluation_id = source.at("evaluation_id").get<std::string>(),
      .run_id = source.at("run_id").get<std::string>(),
      .model_digest = read_digest(source.at("model_digest")),
      .build_fingerprint = source.at("build_fingerprint").get<std::string>(),
      .adapter_id = source.at("adapter_id").get<std::string>(),
      .adapter_version =
          unsigned_number<std::uint32_t>(source.at("adapter_version")),
      .config_digest = read_digest(source.at("config_digest"))};
  validate_identity(result, limits);
  return result;
}

void validate_input(const EvaluationInput &input,
                    const EvaluationLimits &limits) {
  validate_limits(limits);
  version(input.schema_version, "input.schema_version");
  validate_identity(input.identity, limits);
  bound(input.observations.size(), limits.max_rules, "observations");
  std::set<std::string> keys;
  for (const auto &point : input.observations) {
    validate_point(point, limits);
    if (!keys.insert(metric_key(point.definition, point.context, limits))
             .second) {
      reject("observations", "duplicate final metric/context result");
    }
  }
}

Json report_json(const QualityReport &report) {
  Json results = Json::array();
  for (const auto &result : report.results) {
    results.push_back(
        {{"rule_id", result.rule_id},
         {"verdict", static_cast<unsigned>(result.verdict)},
         {"reason", result.reason},
         {"observation", result.observation ? point_json(*result.observation)
                                            : Json(nullptr)}});
  }
  return Json{{"schema_version", report.schema_version},
              {"suite", suite_json(report.suite)},
              {"identity", identity_json(report.identity)},
              {"verdict", static_cast<unsigned>(report.verdict)},
              {"results", results}};
}

void validate_report(const QualityReport &report,
                     const EvaluationLimits &limits) {
  version(report.schema_version, "report.schema_version");
  validate_evaluation_suite(report.suite, limits);
  validate_identity(report.identity, limits);
  if (report.results.size() != report.suite.rules.size())
    reject("report.results", "wrong result count");
  auto overall = QualityVerdict::pass;
  for (std::size_t i = 0U; i < report.results.size(); ++i) {
    const auto &result = report.results[i];
    const auto &rule = report.suite.rules[i];
    text(result.reason, "report.reason", limits, false);
    if (result.observation) {
      validate_point(*result.observation, limits);
      if (metric_key(result.observation->definition,
                     result.observation->context, limits) !=
          metric_key(rule.definition, rule.context, limits))
        reject("report.observation", "wrong metric definition or context");
    }
    const auto expected = apply_rule(rule, result.observation);
    if (result.rule_id != expected.rule_id ||
        result.verdict != expected.verdict ||
        result.reason != expected.reason) {
      reject("report.verdict", "rule verdict disagrees with evidence",
             ErrorCode::serialization);
    }
    overall = combine(overall, result.verdict);
  }
  if (overall != report.verdict)
    reject("report.verdict", "aggregate verdict disagrees with evidence",
           ErrorCode::serialization);
}

std::string html_escape(std::string_view value) {
  std::string result;
  for (char c : value) {
    switch (c) {
      case '&':
        result += "&amp;";
        break;
      case '<':
        result += "&lt;";
        break;
      case '>':
        result += "&gt;";
        break;
      case '"':
        result += "&quot;";
        break;
      case '\'':
        result += "&#39;";
        break;
      default:
        result += c;
        break;
    }
  }
  return result;
}
}  // namespace

std::string_view quality_verdict_name(QualityVerdict verdict) noexcept {
  switch (verdict) {
    case QualityVerdict::pass:
      return "pass";
    case QualityVerdict::fail:
      return "fail";
    case QualityVerdict::warn:
      return "warn";
    case QualityVerdict::incomplete:
      return "incomplete";
    case QualityVerdict::invalid:
      return "invalid";
  }
  return "unknown";
}

void validate_evaluation_suite(const EvaluationSuite &suite,
                               const EvaluationLimits &limits) {
  validate_limits(limits);
  version(suite.schema_version, "suite.schema_version");
  text(suite.suite_id, "suite_id", limits);
  text(suite.task, "task", limits);
  text(suite.environment_id, "environment_id", limits);
  validate_artifact_digest(suite.dataset_digest);
  if (suite.version == 0U || suite.rules.empty())
    reject("suite", "zero version or empty rules");
  bound(suite.rules.size(), limits.max_rules, "suite.rules");
  std::set<std::string> ids;
  std::set<std::string> keys;
  for (const auto &rule : suite.rules) {
    text(rule.rule_id, "rule_id", limits);
    validate_telemetry_metric_definition(rule.definition, limits.telemetry);
    if (rule.definition.unit == TelemetryMetricUnit::unspecified ||
        rule.definition.aggregation == TelemetryAggregation::unspecified) {
      reject("rule.definition",
             "quality rules require declared metric semantics");
    }
    validate_point(
        TelemetryMetricPoint{.definition = rule.definition,
                             .step = {},
                             .observation = make_telemetry_metric_value(0.0),
                             .context = rule.context,
                             .sample_count = rule.minimum_samples},
        limits);
    if (!ids.insert(rule.rule_id).second ||
        !keys.insert(metric_key(rule.definition, rule.context, limits))
             .second) {
      reject("suite.rules", "duplicate rule or metric/context key");
    }
    if (rule.minimum_samples == 0U)
      reject("rule.minimum_samples", "zero sample minimum");
    interval(rule.minimum, rule.maximum);
    for (const auto &regression :
         {rule.max_absolute_regression, rule.max_relative_regression}) {
      if (regression && (!std::isfinite(*regression) || *regression < 0.0)) {
        reject("rule.regression", "negative or nonfinite tolerance");
      }
    }
    if ((rule.max_absolute_regression || rule.max_relative_regression) &&
        rule.definition.direction == TelemetryMetricDirection::none) {
      reject("rule.direction",
             "regression tolerance requires an explicit direction");
    }
  }
}

std::string evaluation_input_json(const EvaluationInput &input,
                                  const EvaluationLimits &limits) {
  validate_input(input, limits);
  Json observations = Json::array();
  for (const auto &point : input.observations)
    observations.push_back(point_json(point));
  return finish_json(Json{{"schema_version", input.schema_version},
                          {"identity", identity_json(input.identity)},
                          {"observations", observations}},
                     limits);
}

EvaluationInput parse_evaluation_input_json(std::string_view text,
                                            const EvaluationLimits &limits) {
  return decode([&] {
    const auto root = parse_json(text, limits);
    fields(root, {"schema_version", "identity", "observations"});
    EvaluationInput input;
    input.schema_version =
        unsigned_number<std::uint32_t>(root.at("schema_version"));
    version(input.schema_version, "input.schema_version");
    input.identity = read_identity(root.at("identity"), limits);
    const auto &observations = root.at("observations");
    if (!observations.is_array())
      reject("observations", "expected array", ErrorCode::serialization);
    bound(observations.size(), limits.max_rules, "observations");
    input.observations.reserve(observations.size());
    for (const auto &point : observations)
      input.observations.push_back(read_point(point));
    validate_input(input, limits);
    return input;
  });
}

std::string evaluation_suite_json(const EvaluationSuite &suite,
                                  const EvaluationLimits &limits) {
  validate_evaluation_suite(suite, limits);
  return finish_json(suite_json(suite), limits);
}

EvaluationSuite parse_evaluation_suite_json(std::string_view value,
                                            const EvaluationLimits &limits) {
  return decode([&] { return read_suite(parse_json(value, limits), limits); });
}

ArtifactDigest evaluation_suite_digest(const EvaluationSuite &suite,
                                       const EvaluationLimits &limits) {
  const auto encoded = evaluation_suite_json(suite, limits);
  return {.hex = sha256_hex(std::as_bytes(std::span(encoded)))};
}

QualityReport evaluate_observations(
    const EvaluationSuite &suite, const EvaluationIdentity &identity,
    std::span<const TelemetryMetricPoint> observations, std::stop_token stop,
    TelemetryDeadline deadline, const EvaluationLimits &limits) {
  check_telemetry_operation(stop, deadline);
  validate_evaluation_suite(suite, limits);
  validate_identity(identity, limits);
  bound(observations.size(), limits.max_rules, "observations");
  std::map<std::string, std::size_t> rules;
  for (std::size_t i = 0U; i < suite.rules.size(); ++i) {
    rules.emplace(
        metric_key(suite.rules[i].definition, suite.rules[i].context, limits),
        i);
  }
  std::vector<std::optional<TelemetryMetricPoint>> points(suite.rules.size());
  for (const auto &point : observations) {
    check_telemetry_operation(stop, deadline);
    validate_point(point, limits);
    const auto found =
        rules.find(metric_key(point.definition, point.context, limits));
    if (found == rules.end())
      reject("observation", "unknown metric definition or context",
             ErrorCode::compatibility);
    if (points[found->second])
      reject("observation", "duplicate final metric/context result");
    points[found->second] = point;
  }
  QualityReport report{.schema_version = 1U,
                       .suite = suite,
                       .identity = identity,
                       .verdict = QualityVerdict::pass,
                       .results = {}};
  for (std::size_t i = 0U; i < suite.rules.size(); ++i) {
    report.results.push_back(apply_rule(suite.rules[i], points[i]));
    report.verdict = combine(report.verdict, report.results.back().verdict);
  }
  static_cast<void>(finish_json(report_json(report), limits));
  check_telemetry_operation(stop, deadline);
  return report;
}

QualityReport run_evaluation(
    const EvaluationSuite &suite, const EvaluationIdentity &identity,
    const std::function<std::vector<TelemetryMetricPoint>(
        const EvaluationSuite &, std::stop_token, TelemetryDeadline)>
        &evaluator,
    std::stop_token stop, TelemetryDeadline deadline,
    const EvaluationLimits &limits) {
  check_telemetry_operation(stop, deadline);
  validate_evaluation_suite(suite, limits);
  validate_identity(identity, limits);
  if (!evaluator) reject("evaluator", "empty trusted task adapter");
  const auto observations = evaluator(suite, stop, deadline);
  return evaluate_observations(suite, identity, observations, stop, deadline,
                               limits);
}

std::string quality_report_json(const QualityReport &report,
                                const EvaluationLimits &limits) {
  validate_report(report, limits);
  return finish_json(report_json(report), limits);
}

QualityReport parse_quality_report_json(std::string_view value,
                                        const EvaluationLimits &limits) {
  return decode([&] {
    const auto root = parse_json(value, limits);
    fields(root, {"schema_version", "suite", "identity", "verdict", "results"});
    QualityReport report;
    report.schema_version =
        unsigned_number<std::uint32_t>(root.at("schema_version"));
    report.suite = read_suite(root.at("suite"), limits);
    report.identity = read_identity(root.at("identity"), limits);
    report.verdict = static_cast<QualityVerdict>(
        unsigned_number<std::uint8_t>(root.at("verdict")));
    const auto &results = root.at("results");
    if (!results.is_array())
      reject("results", "expected array", ErrorCode::serialization);
    bound(results.size(), limits.max_rules, "results");
    for (const auto &result : results) {
      fields(result, {"rule_id", "verdict", "reason", "observation"});
      report.results.push_back(QualityRuleResult{
          .rule_id = result.at("rule_id").get<std::string>(),
          .verdict = static_cast<QualityVerdict>(
              unsigned_number<std::uint8_t>(result.at("verdict"))),
          .reason = result.at("reason").get<std::string>(),
          .observation =
              result.at("observation").is_null()
                  ? std::nullopt
                  : std::optional(read_point(result.at("observation")))});
    }
    validate_report(report, limits);
    return report;
  });
}

std::string quality_report_html(const QualityReport &report,
                                const EvaluationLimits &limits) {
  const auto canonical = quality_report_json(report, limits);
  std::string html =
      "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
      "<meta name=\"viewport\" content=\"width=device-width\"><title>Local "
      "quality report</title>"
      "<body><h1>" +
      html_escape(report.suite.suite_id) + "</h1><p>" +
      html_escape(quality_verdict_name(report.verdict)) + "</p><pre>" +
      html_escape(canonical) + "</pre></body></html>";
  bound(html.size(), limits.max_json_bytes, "html.bytes");
  return html;
}

QualityComparison compare_quality_reports(const QualityReport &baseline,
                                          const QualityReport &candidate,
                                          const EvaluationLimits &limits) {
  const auto left_json = quality_report_json(baseline, limits);
  const auto right_json = quality_report_json(candidate, limits);
  if (evaluation_suite_digest(baseline.suite, limits) !=
          evaluation_suite_digest(candidate.suite, limits) ||
      baseline.identity.adapter_id != candidate.identity.adapter_id ||
      baseline.identity.adapter_version != candidate.identity.adapter_version) {
    reject("compare.contract",
           "suite/dataset/environment/adapter definitions differ",
           ErrorCode::compatibility);
  }
  QualityComparison report{
      .baseline_report_digest = {.hex = sha256_hex(
                                     std::as_bytes(std::span(left_json)))},
      .candidate_report_digest = {.hex = sha256_hex(
                                      std::as_bytes(std::span(right_json)))},
      .verdict = QualityVerdict::pass,
      .results = {}};
  for (std::size_t i = 0U; i < baseline.results.size(); ++i) {
    const auto &left = baseline.results[i];
    const auto &right = candidate.results[i];
    const auto &rule = baseline.suite.rules[i];
    QualityComparisonResult result{.rule_id = rule.rule_id,
                                   .verdict = QualityVerdict::incomplete,
                                   .candidate_minus_baseline = std::nullopt,
                                   .regression = std::nullopt,
                                   .reason = {}};
    if (left.verdict == QualityVerdict::incomplete ||
        right.verdict == QualityVerdict::incomplete) {
      result.reason =
          "finite observations with sufficient samples required on both sides";
    } else {
      if (left.observation->step.kind != right.observation->step.kind) {
        reject("compare.step", "metric counter meanings differ",
               ErrorCode::compatibility);
      }
      const auto difference = *right.observation->observation.value -
                              *left.observation->observation.value;
      if (!std::isfinite(difference)) {
        result.reason = "difference exceeds finite binary64 range";
      } else {
        result.candidate_minus_baseline = difference == 0.0 ? 0.0 : difference;
        result.verdict = right.verdict;
        if (rule.definition.direction != TelemetryMetricDirection::none) {
          const auto regression =
              std::max(0.0, rule.definition.direction ==
                                    TelemetryMetricDirection::minimize
                                ? difference
                                : -difference);
          result.regression = regression;
          if (rule.max_absolute_regression &&
              regression > *rule.max_absolute_regression) {
            result.verdict = QualityVerdict::fail;
            result.reason = "absolute regression tolerance exceeded";
          }
          if (rule.max_relative_regression) {
            const auto magnitude =
                std::abs(*left.observation->observation.value);
            if (magnitude == 0.0 && regression > 0.0) {
              if (result.verdict != QualityVerdict::fail)
                result.verdict = QualityVerdict::incomplete;
              result.reason = "relative regression undefined for zero baseline";
            } else if (magnitude > 0.0 &&
                       regression / magnitude > *rule.max_relative_regression) {
              result.verdict = QualityVerdict::fail;
              result.reason = "relative regression tolerance exceeded";
            }
          }
        }
        if (result.verdict == QualityVerdict::fail && result.reason.empty()) {
          result.reason = "candidate violates an absolute suite threshold";
        }
      }
    }
    report.verdict = combine(report.verdict, result.verdict);
    report.results.push_back(std::move(result));
  }
  return report;
}

DataQualityReport inspect_data_quality(
    const std::map<std::string, DataQualityField> &schema,
    std::span<const DataQualityRow> rows, std::stop_token stop,
    TelemetryDeadline deadline, const EvaluationLimits &limits) {
  validate_limits(limits);
  check_telemetry_operation(stop, deadline);
  bound(rows.size(), limits.max_data_rows, "data.rows");
  bound(schema.size(), limits.max_data_fields, "data.schema");
  for (const auto &[name, field] : schema) {
    text(name, "data.field", limits);
    interval(field.minimum, field.maximum);
  }
  DataQualityReport report;
  std::set<std::pair<std::string, std::string>> ids;
  std::map<std::string, std::set<std::string>> content_splits;
  std::map<std::string, std::set<std::string>> content_labels;
  std::size_t text_bytes = 0U;
  const auto count_text = [&](std::size_t count) {
    if (count > limits.max_data_text_bytes - text_bytes) {
      reject("data.text_bytes", "aggregate row text bound exceeded",
             ErrorCode::resource_exhausted);
    }
    text_bytes += count;
  };
  for (const auto &row : rows) {
    check_telemetry_operation(stop, deadline);
    text(row.sample_id, "data.sample_id", limits);
    text(row.split, "data.split", limits);
    validate_artifact_digest(row.content_digest);
    bound(row.fields.size(), limits.max_data_fields, "data.row_fields");
    if (row.label) text(*row.label, "data.label", limits);
    count_text(row.sample_id.size());
    count_text(row.split.size());
    count_text(row.content_digest.hex.size());
    if (row.label) count_text(row.label->size());
    for (const auto &[name, value] : row.fields) {
      static_cast<void>(value);
      count_text(name.size());
    }
    ++report.checked_rows;
    if (!ids.emplace(row.split, row.sample_id).second) ++report.duplicate_ids;
    auto &splits = content_splits[row.content_digest.hex];
    if (splits.contains(row.split)) ++report.duplicate_contents;
    if (std::any_of(splits.begin(), splits.end(),
                    [&](const auto &split) { return split != row.split; })) {
      ++report.cross_split_leaks;
    }
    splits.insert(row.split);
    if (row.label) {
      auto &labels = content_labels[row.content_digest.hex];
      if (std::any_of(labels.begin(), labels.end(),
                      [&](const auto &label) { return label != *row.label; })) {
        ++report.conflicting_labels;
      }
      labels.insert(*row.label);
    }
    for (const auto &[name, value] : row.fields) {
      text(name, "data.field", limits);
      if (!schema.contains(name)) ++report.unknown_fields;
      if (value && !std::isfinite(*value)) ++report.nonfinite_values;
    }
    for (const auto &[name, field] : schema) {
      const auto found = row.fields.find(name);
      if (found == row.fields.end() || !found->second) {
        if (field.required) ++report.missing_values;
      } else if (std::isfinite(*found->second) &&
                 ((field.minimum && *found->second < *field.minimum) ||
                  (field.maximum && *found->second > *field.maximum))) {
        ++report.out_of_range_values;
      }
    }
  }
  if (!rows.empty()) {
    report.verdict =
        report.missing_values != 0U || report.nonfinite_values != 0U ||
                report.out_of_range_values != 0U ||
                report.unknown_fields != 0U || report.duplicate_ids != 0U ||
                report.duplicate_contents != 0U ||
                report.cross_split_leaks != 0U ||
                report.conflicting_labels != 0U
            ? QualityVerdict::fail
            : QualityVerdict::pass;
  }
  check_telemetry_operation(stop, deadline);
  return report;
}

}  // namespace jyppx::liuding::experiment
