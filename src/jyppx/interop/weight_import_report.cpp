// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "metadata_validation.hpp"

#include <jyppx/foundation/error.hpp>
#include <jyppx/interop/weight_import_report.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace jyppx::interop {
namespace {

namespace md = metadata_detail;
using md::Json;
constexpr std::string_view kComponent = "interop.weight_import";

struct Entry {
  std::size_t index = 0U;
  std::size_t count = 0U;
  std::uint64_t elements = 0U;
};

using Index = std::map<std::string_view, Entry>;

std::string_view ModeName(WeightImportMode mode) noexcept {
  return mode == WeightImportMode::strict ? "strict" : "partial";
}

std::string_view ReasonName(WeightSkipReason reason) noexcept {
  switch (reason) {
    case WeightSkipReason::missing:
      return "missing";
    case WeightSkipReason::shape_mismatch:
      return "shape_mismatch";
    case WeightSkipReason::dtype_mismatch:
      return "dtype_mismatch";
    case WeightSkipReason::duplicate_source:
      return "duplicate_source";
    case WeightSkipReason::duplicate_target:
      return "duplicate_target";
  }
  return "unknown";
}

void ValidateLimits(const WeightImportLimits& limits) {
  if (limits.max_tensors == 0U || limits.max_rank == 0U ||
      limits.max_dimension == 0U || limits.max_tensor_elements == 0U ||
      limits.max_total_elements == 0U || limits.max_string_bytes == 0U ||
      limits.max_total_string_bytes == 0U || limits.max_report_bytes == 0U) {
    md::Invalid(kComponent, "limits", "strictly positive metadata bounds");
  }
}

void CountString(std::string_view value, const WeightImportLimits& limits,
                 std::uint64_t& strings, std::string_view field) {
  md::Text(value, limits.max_string_bytes, kComponent, field);
  md::Add(strings, value.size(), limits.max_total_string_bytes, kComponent,
          "strings.total");
}

Index BuildIndex(std::span<const WeightTensorDescriptor> entries,
                 const WeightImportLimits& limits, std::uint64_t& strings,
                 std::uint64_t& total_elements) {
  if (entries.size() > limits.max_tensors)
    md::Resource(kComponent, "tensors.count", limits.max_tensors);
  Index result;
  for (std::size_t index = 0U; index < entries.size(); ++index) {
    const auto& entry = entries[index];
    CountString(entry.name, limits, strings, "tensor.name");
    const auto width = safetensors_dtype_size(entry.dtype);
    if (width == 0U)
      md::Invalid(kComponent, "tensor.dtype", "a supported scalar encoding");
    if (entry.shape.size() > limits.max_rank)
      md::Resource(kComponent, "shape.rank", limits.max_rank);
    std::uint64_t elements = 1U;
    for (const auto dimension : entry.shape) {
      if (dimension > limits.max_dimension)
        md::Resource(kComponent, "shape.dimension", limits.max_dimension);
      if (dimension != 0U &&
          elements > (std::numeric_limits<std::uint64_t>::max)() / dimension)
        md::Invalid(kComponent, "shape.product", "a uint64 shape product");
      elements *= dimension;
      if (elements > limits.max_tensor_elements)
        md::Resource(kComponent, "tensor.elements", limits.max_tensor_elements);
    }
    if (elements > (std::numeric_limits<std::uint64_t>::max)() / width)
      md::Invalid(kComponent, "tensor.bytes", "a uint64 encoded byte count");
    md::Add(total_elements, elements, limits.max_total_elements, kComponent,
            "tensors.elements");
    const auto [position, inserted] = result.try_emplace(
        entry.name, Entry{.index = index, .count = 1U, .elements = elements});
    if (!inserted) ++position->second.count;
  }
  return result;
}

std::map<std::string_view, const WeightConversionRule*> ValidateRules(
    const WeightImportOptions& options,
    std::span<const WeightTensorDescriptor> source,
    std::span<const WeightTensorDescriptor> target, const Index& sources,
    const Index& targets, const WeightImportLimits& limits,
    std::uint64_t& strings) {
  if (options.mode != WeightImportMode::strict &&
      options.mode != WeightImportMode::partial)
    md::Invalid(kComponent, "mode", "explicit strict or partial import");
  if (!std::isfinite(options.minimum_tensor_coverage) ||
      options.minimum_tensor_coverage < 0.0 ||
      options.minimum_tensor_coverage > 1.0)
    md::Invalid(kComponent, "minimum_coverage", "a finite value in [0,1]");
  if (options.mode == WeightImportMode::strict &&
      (!options.conversions.empty() || options.minimum_tensor_coverage != 0.0))
    md::Invalid(kComponent, "strict.options", "no partial-only import options");
  if (options.conversions.size() > limits.max_tensors)
    md::Resource(kComponent, "conversions.count", limits.max_tensors);
  std::map<std::string_view, const WeightConversionRule*> result;
  for (const auto& rule : options.conversions) {
    CountString(rule.key, limits, strings, "conversion.key");
    CountString(rule.rule_id, limits, strings, "conversion.rule");
    md::Token(rule.rule_id, kComponent, "conversion.rule");
    if (safetensors_dtype_size(rule.source_dtype) == 0U ||
        safetensors_dtype_size(rule.target_dtype) == 0U ||
        rule.source_dtype == rule.target_dtype ||
        !result.emplace(rule.key, &rule).second)
      md::Invalid(kComponent, "conversion", "unique non-identity dtype rules");
    const auto from = sources.find(rule.key);
    const auto to = targets.find(rule.key);
    if (from == sources.end() || to == targets.end() ||
        from->second.count != 1U || to->second.count != 1U ||
        source[from->second.index].shape != target[to->second.index].shape ||
        source[from->second.index].dtype != rule.source_dtype ||
        target[to->second.index].dtype != rule.target_dtype)
      md::Invalid(kComponent, "conversion.mapping",
                  "a rule matching one complete compatible-shape key pair");
  }
  return result;
}

Json ReportJson(const WeightImportReport& report) {
  Json shapes = Json::array();
  for (const auto& item : report.shape_mismatches) {
    shapes.push_back(Json{{"key", item.key},
                          {"source_shape", item.source_shape},
                          {"target_shape", item.target_shape}});
  }
  Json dtypes = Json::array();
  for (const auto& item : report.dtype_mismatches) {
    dtypes.push_back(
        Json{{"key", item.key},
             {"source_dtype", safetensors_dtype_name(item.source_dtype)},
             {"target_dtype", safetensors_dtype_name(item.target_dtype)},
             {"conversion_rule", item.conversion_rule}});
  }
  Json conversions = Json::array();
  for (const auto& item : report.conversions) {
    conversions.push_back(
        Json{{"key", item.key},
             {"source_dtype", safetensors_dtype_name(item.source_dtype)},
             {"target_dtype", safetensors_dtype_name(item.target_dtype)},
             {"rule_id", item.rule_id}});
  }
  Json retained = Json::array();
  for (const auto& item : report.retained_targets) {
    retained.push_back(
        Json{{"key", item.key}, {"reason", ReasonName(item.reason)}});
  }
  return Json{{"schema_version", report.schema_version},
              {"phase", "preflight"},
              {"status", report.ready ? "ready" : "rejected"},
              {"source_digest", md::DigestJson(report.source_digest)},
              {"mode", ModeName(report.mode)},
              {"coverage_satisfied", report.coverage_satisfied},
              {"source_tensor_count", report.source_tensor_count},
              {"target_tensor_count", report.target_tensor_count},
              {"source_element_count", report.source_element_count},
              {"target_element_count", report.target_element_count},
              {"matched_element_count", report.matched_element_count},
              {"tensor_coverage", report.tensor_coverage},
              {"element_coverage", report.element_coverage},
              {"minimum_tensor_coverage", report.minimum_tensor_coverage},
              {"matched", report.matched},
              {"missing", report.missing},
              {"unexpected", report.unexpected},
              {"duplicate_source", report.duplicate_source},
              {"duplicate_target", report.duplicate_target},
              {"shape_mismatches", std::move(shapes)},
              {"dtype_mismatches", std::move(dtypes)},
              {"conversions", std::move(conversions)},
              {"unassigned_destination_action", "retain-existing"},
              {"retained_targets", std::move(retained)}};
}

}  // namespace

WeightImportPlan::WeightImportPlan(WeightImportReport report,
                                   std::vector<WeightAssignment> assignments,
                                   std::size_t maximum_report_bytes)
    : report_(std::move(report)),
      assignments_(std::move(assignments)),
      maximum_report_bytes_(maximum_report_bytes) {}

const WeightImportReport& WeightImportPlan::report() const noexcept {
  return report_;
}

std::span<const WeightAssignment> WeightImportPlan::assignments() const {
  if (!report_.ready) {
    throw CompatibilityError("Rejected weight import plan cannot be staged",
                             md::Details(kComponent, "assignments",
                                         "a complete accepted preflight"));
  }
  return assignments_;
}

std::string WeightImportPlan::report_json() const {
  try {
    auto encoded = ReportJson(report_).dump();
    if (encoded.size() > maximum_report_bytes_)
      md::Resource(kComponent, "report.bytes", maximum_report_bytes_);
    return encoded;
  } catch (const std::bad_alloc&) {
    md::Resource(kComponent, "memory", maximum_report_bytes_);
  } catch (const std::length_error&) {
    md::Resource(kComponent, "memory", maximum_report_bytes_);
  }
}

WeightImportPlan plan_weight_import(
    const ArtifactDigest& source_digest,
    std::span<const WeightTensorDescriptor> source,
    std::span<const WeightTensorDescriptor> target,
    const WeightImportOptions& options, WeightImportLimits limits) {
  try {
    ValidateLimits(limits);
    validate_artifact_digest(source_digest);
    WeightImportReport report;
    report.source_digest = source_digest;
    report.mode = options.mode;
    report.minimum_tensor_coverage = options.minimum_tensor_coverage;
    report.source_tensor_count = source.size();
    report.target_tensor_count = target.size();
    std::uint64_t strings = 0U;
    const auto sources =
        BuildIndex(source, limits, strings, report.source_element_count);
    const auto targets =
        BuildIndex(target, limits, strings, report.target_element_count);
    const auto rules = ValidateRules(options, source, target, sources, targets,
                                     limits, strings);
    for (const auto& [name, entry] : sources) {
      if (entry.count > 1U) report.duplicate_source.emplace_back(name);
      if (!targets.contains(name)) report.unexpected.emplace_back(name);
    }
    std::vector<WeightAssignment> assignments;
    assignments.reserve(targets.size());
    for (const auto& [name, to] : targets) {
      const auto from = sources.find(name);
      if (to.count > 1U) report.duplicate_target.emplace_back(name);
      if (from == sources.end()) report.missing.emplace_back(name);
      const auto retain = [&, target_name = name](WeightSkipReason reason) {
        report.retained_targets.push_back(
            {.key = std::string(target_name), .reason = reason});
      };
      if (to.count > 1U) {
        retain(WeightSkipReason::duplicate_target);
        continue;
      }
      if (from == sources.end()) {
        retain(WeightSkipReason::missing);
        continue;
      }
      if (from->second.count > 1U) {
        retain(WeightSkipReason::duplicate_source);
        continue;
      }
      const auto& input = source[from->second.index];
      const auto& output = target[to.index];
      const bool shape_matches = input.shape == output.shape;
      if (!shape_matches)
        report.shape_mismatches.push_back({.key = std::string(name),
                                           .source_shape = input.shape,
                                           .target_shape = output.shape});
      const auto conversion = rules.find(name);
      const std::string rule = conversion == rules.end()
                                   ? std::string{}
                                   : conversion->second->rule_id;
      if (input.dtype != output.dtype)
        report.dtype_mismatches.push_back({.key = std::string(name),
                                           .source_dtype = input.dtype,
                                           .target_dtype = output.dtype,
                                           .conversion_rule = rule});
      if (!shape_matches) {
        retain(WeightSkipReason::shape_mismatch);
        continue;
      }
      if (input.dtype != output.dtype && conversion == rules.end()) {
        retain(WeightSkipReason::dtype_mismatch);
        continue;
      }
      if (conversion != rules.end())
        report.conversions.push_back(*conversion->second);
      report.matched.emplace_back(name);
      md::Add(report.matched_element_count, to.elements,
              limits.max_total_elements, kComponent, "matched.elements");
      assignments.push_back({.key = std::string(name),
                             .source_index = from->second.index,
                             .target_index = to.index,
                             .conversion_rule = rule});
    }
    report.tensor_coverage = target.empty()
                                 ? 1.0
                                 : static_cast<double>(report.matched.size()) /
                                       static_cast<double>(target.size());
    report.element_coverage =
        report.target_element_count == 0U
            ? 1.0
            : static_cast<double>(report.matched_element_count) /
                  static_cast<double>(report.target_element_count);
    report.coverage_satisfied =
        report.tensor_coverage >= options.minimum_tensor_coverage;
    const bool unambiguous =
        report.duplicate_source.empty() && report.duplicate_target.empty();
    report.ready =
        unambiguous && report.coverage_satisfied &&
        (options.mode == WeightImportMode::partial ||
         (report.missing.empty() && report.unexpected.empty() &&
          report.shape_mismatches.empty() && report.dtype_mismatches.empty()));
    if (!report.ready) assignments.clear();
    return WeightImportPlan(std::move(report), std::move(assignments),
                            limits.max_report_bytes);
  } catch (const std::bad_alloc&) {
    md::Resource(kComponent, "memory", limits.max_report_bytes);
  } catch (const std::length_error&) {
    md::Resource(kComponent, "memory", limits.max_report_bytes);
  }
}

}  // namespace jyppx::interop
