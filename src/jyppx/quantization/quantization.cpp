// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/quantization/quantization.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace jyppx::quantization {
namespace {

constexpr std::string_view kComponent = "quantization";
constexpr std::string_view kFormatId = "jyppx.qint8.symmetric.per_tensor/v1";
constexpr std::string_view kMagic{"JYPPXQ1\0", 8U};
constexpr std::uint32_t kSchemaVersion = 1U;
constexpr std::size_t kObserverStateBytes = 40U;
constexpr std::size_t kMaxJsonNumberBytes = 128U;

[[noreturn]] void invalid(std::string operation, std::string expected,
                          std::string actual, std::string hint) {
  throw InvalidArgument("Invalid quantization input",
                        ErrorDetails{.component = std::string(kComponent),
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void serialization(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw SerializationError("Malformed quantization artifact",
                           ErrorDetails{.component = std::string(kComponent),
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void compatibility(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw CompatibilityError("Incompatible quantization state",
                           ErrorDetails{.component = std::string(kComponent),
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void resource(std::string operation, std::size_t limit,
                           std::size_t actual, std::string hint) {
  throw ResourceExhausted(
      "Quantization resource limit exceeded",
      ErrorDetails{.component = std::string(kComponent),
                   .operation = std::move(operation),
                   .expected = "at most " + std::to_string(limit),
                   .actual = std::to_string(actual),
                   .hint = std::move(hint)});
}

[[noreturn]] void numerical(std::string operation, std::string expected,
                            std::string actual, std::string hint) {
  throw NumericalError("Invalid quantization numerical state",
                       ErrorDetails{.component = std::string(kComponent),
                                    .operation = std::move(operation),
                                    .expected = std::move(expected),
                                    .actual = std::move(actual),
                                    .hint = std::move(hint)});
}

bool checked_add(std::size_t left, std::size_t right, std::size_t* result) {
  if (right > std::numeric_limits<std::size_t>::max() - left) return false;
  *result = left + right;
  return true;
}

bool checked_mul(std::size_t left, std::size_t right, std::size_t* result) {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left)
    return false;
  *result = left * right;
  return true;
}

bool valid_utf8(std::string_view text) {
  std::size_t offset = 0U;
  while (offset < text.size()) {
    const auto first = static_cast<std::uint8_t>(text[offset++]);
    if (first <= 0x7fU) continue;
    std::size_t continuation = 0U;
    std::uint32_t scalar = 0U;
    std::uint32_t minimum = 0U;
    if (first >= 0xc2U && first <= 0xdfU) {
      continuation = 1U;
      scalar = first & 0x1fU;
      minimum = 0x80U;
    } else if (first >= 0xe0U && first <= 0xefU) {
      continuation = 2U;
      scalar = first & 0x0fU;
      minimum = 0x800U;
    } else if (first >= 0xf0U && first <= 0xf4U) {
      continuation = 3U;
      scalar = first & 0x07U;
      minimum = 0x10000U;
    } else {
      return false;
    }
    if (continuation > text.size() - offset) return false;
    for (std::size_t i = 0U; i < continuation; ++i) {
      const auto next = static_cast<std::uint8_t>(text[offset++]);
      if ((next & 0xc0U) != 0x80U) return false;
      scalar = (scalar << 6U) | (next & 0x3fU);
    }
    if (scalar < minimum || scalar > 0x10ffffU ||
        (scalar >= 0xd800U && scalar <= 0xdfffU))
      return false;
  }
  return true;
}

void validate_qparams(const QuantizationParams& params) {
  if (!std::isfinite(params.scale) || params.scale <= 0.0 ||
      params.scale >
          static_cast<double>(std::numeric_limits<float>::max()) / 127.0 ||
      params.zero_point != 0 || params.quantized_min != -127 ||
      params.quantized_max != 127 ||
      params.quantized_min >= params.quantized_max) {
    invalid("qparams", "finite positive scale, zero point 0, range [-127,127]",
            "invalid qparams", "Use derive_symmetric_params for v1 data.");
  }
}

void validate_lineage(const CalibrationLineage& lineage,
                      const QuantizationLimits& limits) {
  validate_artifact_digest(lineage.dataset_digest);
  if (lineage.dataset_version.empty() ||
      lineage.dataset_version.size() > limits.max_lineage_text_bytes ||
      !valid_utf8(lineage.dataset_version)) {
    invalid("calibration.dataset_version", "non-empty bounded text",
            std::to_string(lineage.dataset_version.size()),
            "Use a stable dataset version within the configured limit.");
  }
  if (lineage.preprocessing_fingerprint.size() >
      limits.max_lineage_text_bytes) {
    resource("calibration.preprocessing_fingerprint",
             limits.max_lineage_text_bytes,
             lineage.preprocessing_fingerprint.size(),
             "Shorten the preprocessing fingerprint metadata.");
  }
  if (lineage.preprocessing_fingerprint.empty() ||
      !valid_utf8(lineage.preprocessing_fingerprint))
    invalid("calibration.preprocessing_fingerprint", "non-empty valid UTF-8",
            "empty or invalid bytes",
            "Use a stable UTF-8 preprocessing fingerprint.");
}

void validate_observer_state(const MinMaxObserverState& state) {
  if ((state.sample_count == 0U &&
       (state.value_count != 0U || state.has_finite_value)) ||
      (state.has_finite_value && state.value_count == 0U)) {
    serialization("observer.counts", "consistent sample/value counters",
                  "inconsistent finite state",
                  "Regenerate the observer checkpoint.");
  }
  if (state.has_finite_value) {
    if (!std::isfinite(state.minimum) || !std::isfinite(state.maximum) ||
        state.minimum > state.maximum) {
      serialization("observer.values", "finite minimum <= maximum",
                    "invalid range", "Regenerate the observer checkpoint.");
    }
  } else if (state.minimum != 0.0F || state.maximum != 0.0F) {
    serialization("observer.empty", "zero extrema for empty observer",
                  "non-zero extrema", "Regenerate the observer checkpoint.");
  }
}

void validate_calibration(const CalibrationResult& calibration,
                          const QuantizationLimits& limits) {
  validate_lineage(calibration.lineage, limits);
  validate_observer_state(calibration.activation_state);
  validate_qparams(calibration.activation_params);
  if (calibration.sample_ids.size() > limits.max_samples) {
    resource("calibration.sample_ids", limits.max_samples,
             calibration.sample_ids.size(), "Reduce calibration samples.");
  }
  std::unordered_set<std::string> unique;
  unique.reserve(calibration.sample_ids.size());
  for (const auto& sample_id : calibration.sample_ids) {
    if (sample_id.empty() || sample_id.size() > limits.max_sample_id_bytes ||
        !valid_utf8(sample_id)) {
      invalid("calibration.sample_id", "non-empty bounded sample ID",
              std::to_string(sample_id.size()), "Use stable short sample IDs.");
    }
    if (!unique.emplace(sample_id).second) {
      invalid("calibration.sample_id", "unique sample IDs", sample_id,
              "Deduplicate calibration samples before serialization.");
    }
  }
  if (calibration.activation_state.sample_count !=
      calibration.sample_ids.size()) {
    invalid("calibration.sample_count", "one observer sample per sample ID",
            std::to_string(calibration.activation_state.sample_count),
            "Finalize the same CalibrationSession that collected sample IDs.");
  }
  if (calibration.activation_state.has_finite_value) {
    const std::array<float, 2U> extrema{calibration.activation_state.minimum,
                                        calibration.activation_state.maximum};
    if (derive_symmetric_params(extrema).scale !=
        calibration.activation_params.scale)
      invalid("calibration.activation_params",
              "qparams derived from observer extrema", "qparams disagree",
              "Finalize the same calibration observer.");
  }
}

void validate_size_limit(std::size_t value, std::size_t limit,
                         std::string_view operation) {
  if (limit == 0U || value > limit) {
    resource(std::string(operation), limit, value,
             "Increase the configured bounded resource limit.");
  }
}

void add_to_artifact_budget(std::size_t& total, std::size_t bytes,
                            std::size_t limit) {
  if (total > limit || bytes > limit - total)
    resource(
        "artifact", limit,
        limit == std::numeric_limits<std::size_t>::max() ? limit : limit + 1U,
        "Reduce tensor dimensions or calibration metadata.");
  total += bytes;
}

class Encoder {
 public:
  explicit Encoder(std::size_t limit) : limit_(limit) {
    if (limit_ == 0U)
      resource("artifact", 0U, 1U, "Set a positive artifact limit.");
  }

  void raw(std::span<const std::byte> bytes) {
    std::size_t next = 0U;
    if (!checked_add(data_.size(), bytes.size(), &next) || next > limit_)
      resource("artifact", limit_, next,
               "Reduce metadata or tensor dimensions.");
    data_.insert(data_.end(), bytes.begin(), bytes.end());
  }

  void u8(std::uint8_t value) { raw(std::as_bytes(std::span(&value, 1U))); }

  void u16(std::uint16_t value) {
    std::array<std::byte, 2U> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = static_cast<std::byte>((value >> (8U * i)) & 0xffU);
    raw(bytes);
  }

  void u32(std::uint32_t value) {
    std::array<std::byte, 4U> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = static_cast<std::byte>((value >> (8U * i)) & 0xffU);
    raw(bytes);
  }

  void u64(std::uint64_t value) {
    std::array<std::byte, 8U> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = static_cast<std::byte>((value >> (8U * i)) & 0xffU);
    raw(bytes);
  }

  void f32(float value) {
    u32(std::bit_cast<std::uint32_t>(value == 0.0F ? 0.0F : value));
  }
  void f64(double value) {
    u64(std::bit_cast<std::uint64_t>(value == 0.0 ? 0.0 : value));
  }

  void string(std::string_view value, std::size_t max_bytes,
              std::string_view field) {
    if (value.size() > max_bytes ||
        value.size() > std::numeric_limits<std::uint32_t>::max()) {
      resource(std::string(field), max_bytes, value.size(),
               "Reduce the bounded metadata field.");
    }
    u32(static_cast<std::uint32_t>(value.size()));
    raw(std::as_bytes(std::span(value.data(), value.size())));
  }

  std::vector<std::byte> finish() && { return std::move(data_); }

 private:
  std::size_t limit_;
  std::vector<std::byte> data_;
};

class Reader {
 public:
  explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

  std::span<const std::byte> raw(std::size_t count, std::string_view field) {
    if (offset_ > bytes_.size() || count > bytes_.size() - offset_)
      serialization(std::string(field), std::to_string(count) + " bytes",
                    std::to_string(bytes_.size() - offset_) + " bytes",
                    "Reject truncated quantization artifact.");
    const auto result = bytes_.subspan(offset_, count);
    offset_ += count;
    return result;
  }

  std::uint8_t u8(std::string_view field) {
    return std::to_integer<std::uint8_t>(raw(1U, field).front());
  }

  std::uint16_t u16(std::string_view field) {
    return static_cast<std::uint16_t>(read_integer(2U, field));
  }

  std::uint32_t u32(std::string_view field) {
    return static_cast<std::uint32_t>(read_integer(4U, field));
  }

  std::uint64_t u64(std::string_view field) { return read_integer(8U, field); }

  float f32(std::string_view field) { return std::bit_cast<float>(u32(field)); }
  double f64(std::string_view field) {
    return std::bit_cast<double>(u64(field));
  }

  std::string string(std::size_t max_bytes, std::string_view field) {
    const auto length = u32(std::string(field) + ".length");
    if (length > max_bytes)
      resource(std::string(field), max_bytes, length,
               "Reject oversized quantization metadata.");
    const auto bytes = raw(length, field);
    return std::string(reinterpret_cast<const char*>(bytes.data()),
                       bytes.size());
  }

  bool empty() const noexcept { return offset_ == bytes_.size(); }
  std::size_t remaining() const noexcept { return bytes_.size() - offset_; }

  std::size_t size(std::uint64_t value, std::size_t limit,
                   std::string_view field) const {
    if (value > static_cast<std::uint64_t>(limit) ||
        value > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
      resource(
          std::string(field), limit,
          limit == std::numeric_limits<std::size_t>::max() ? limit : limit + 1U,
          "Reject dimensions beyond the configured platform limit.");
    }
    return static_cast<std::size_t>(value);
  }

 private:
  std::uint64_t read_integer(std::size_t width, std::string_view field) {
    const auto bytes = raw(width, field);
    std::uint64_t value = 0U;
    for (std::size_t i = 0; i < width; ++i)
      value |=
          static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(bytes[i]))
          << (8U * i);
    return value;
  }

  std::span<const std::byte> bytes_;
  std::size_t offset_ = 0U;
};

void encode_qparams(Encoder& encoder, const QuantizationParams& params) {
  validate_qparams(params);
  encoder.f64(params.scale);
  encoder.u32(static_cast<std::uint32_t>(params.zero_point));
  encoder.u8(static_cast<std::uint8_t>(params.quantized_min));
  encoder.u8(static_cast<std::uint8_t>(params.quantized_max));
  encoder.u16(0U);
}

QuantizationParams decode_qparams(Reader& reader, std::string_view field) {
  QuantizationParams params;
  params.scale = reader.f64(std::string(field) + ".scale");
  params.zero_point =
      static_cast<std::int32_t>(reader.u32(std::string(field) + ".zero_point"));
  params.quantized_min =
      static_cast<std::int8_t>(reader.u8(std::string(field) + ".min"));
  params.quantized_max =
      static_cast<std::int8_t>(reader.u8(std::string(field) + ".max"));
  if (reader.u16(std::string(field) + ".reserved") != 0U)
    serialization(std::string(field) + ".reserved", "zero", "non-zero",
                  "Reject unknown v1 extensions.");
  validate_qparams(params);
  return params;
}

void encode_observer_state(Encoder& encoder, const MinMaxObserverState& state) {
  validate_observer_state(state);
  encoder.u32(0x4f51594aU);  // JYQO, observer state magic.
  encoder.u32(kSchemaVersion);
  encoder.u64(state.sample_count);
  encoder.u64(state.value_count);
  encoder.u8(state.has_finite_value ? 1U : 0U);
  for (std::size_t i = 0; i < 7U; ++i) encoder.u8(0U);
  encoder.f32(state.minimum);
  encoder.f32(state.maximum);
}

MinMaxObserverState decode_observer_state(Reader& reader) {
  if (reader.u32("observer.magic") != 0x4f51594aU ||
      reader.u32("observer.schema") != kSchemaVersion) {
    serialization("observer.header", "schema-version-1 observer state",
                  "unknown header", "Use a compatible observer checkpoint.");
  }
  MinMaxObserverState state;
  state.sample_count = reader.u64("observer.sample_count");
  state.value_count = reader.u64("observer.value_count");
  const auto present = reader.u8("observer.has_finite_value");
  if (present > 1U)
    serialization("observer.has_finite_value", "0 or 1",
                  std::to_string(present), "Reject malformed state.");
  state.has_finite_value = present != 0U;
  for (std::size_t i = 0; i < 7U; ++i) {
    if (reader.u8("observer.padding") != 0U)
      serialization("observer.padding", "zero", "non-zero",
                    "Reject unknown state extensions.");
  }
  state.minimum = reader.f32("observer.minimum");
  state.maximum = reader.f32("observer.maximum");
  validate_observer_state(state);
  return state;
}

void encode_descriptor(Encoder& encoder,
                       const QuantizationFormatDescriptor& descriptor,
                       const QuantizationLimits& limits) {
  encoder.string(descriptor.format_id, limits.max_lineage_text_bytes,
                 "format_id");
  encoder.u32(descriptor.format_version);
  encoder.u8(static_cast<std::uint8_t>(descriptor.qscheme));
  encoder.u8(static_cast<std::uint8_t>(descriptor.granularity));
  encoder.u8(static_cast<std::uint8_t>(descriptor.execution_plane));
  encoder.u8(descriptor.little_endian ? 1U : 0U);
  encoder.u8(descriptor.compute_quantized ? 1U : 0U);
  encoder.u8(descriptor.qat_supported ? 1U : 0U);
  encoder.u16(0U);
  encoder.u32(descriptor.block_size);
  encoder.u32(descriptor.alignment_bytes);
}

QuantizationFormatDescriptor decode_descriptor(
    Reader& reader, const QuantizationLimits& limits) {
  QuantizationFormatDescriptor descriptor;
  descriptor.format_id =
      reader.string(limits.max_lineage_text_bytes, "format_id");
  descriptor.format_version = reader.u32("format_version");
  descriptor.qscheme = static_cast<QuantizationQScheme>(reader.u8("qscheme"));
  descriptor.granularity =
      static_cast<QuantizationGranularity>(reader.u8("granularity"));
  descriptor.execution_plane =
      static_cast<QuantizationExecutionPlane>(reader.u8("execution_plane"));
  const auto little = reader.u8("little_endian");
  const auto compute = reader.u8("compute_quantized");
  const auto qat = reader.u8("qat_supported");
  if (little > 1U || compute > 1U || qat > 1U)
    serialization("descriptor.flags", "boolean flags", "invalid flag",
                  "Reject malformed format claims.");
  descriptor.little_endian = little != 0U;
  descriptor.compute_quantized = compute != 0U;
  descriptor.qat_supported = qat != 0U;
  if (reader.u16("descriptor.reserved") != 0U)
    serialization("descriptor.reserved", "zero", "non-zero",
                  "Reject unknown descriptor extensions.");
  descriptor.block_size = reader.u32("block_size");
  descriptor.alignment_bytes = reader.u32("alignment_bytes");
  validate_format_descriptor(descriptor);
  return descriptor;
}

void encode_lineage(Encoder& encoder, const CalibrationLineage& lineage,
                    const QuantizationLimits& limits) {
  validate_lineage(lineage, limits);
  encoder.string(lineage.dataset_version, limits.max_lineage_text_bytes,
                 "dataset_version");
  encoder.u8(static_cast<std::uint8_t>(lineage.dataset_digest.algorithm));
  encoder.string(lineage.dataset_digest.hex, 64U, "dataset_digest.hex");
  encoder.string(lineage.preprocessing_fingerprint,
                 limits.max_lineage_text_bytes, "preprocessing_fingerprint");
  encoder.u64(lineage.seed);
}

CalibrationLineage decode_lineage(Reader& reader,
                                  const QuantizationLimits& limits) {
  CalibrationLineage lineage;
  lineage.dataset_version =
      reader.string(limits.max_lineage_text_bytes, "dataset_version");
  lineage.dataset_digest.algorithm = static_cast<ArtifactDigestAlgorithm>(
      reader.u8("dataset_digest.algorithm"));
  lineage.dataset_digest.hex = reader.string(64U, "dataset_digest.hex");
  lineage.preprocessing_fingerprint =
      reader.string(limits.max_lineage_text_bytes, "preprocessing_fingerprint");
  lineage.seed = reader.u64("calibration.seed");
  validate_lineage(lineage, limits);
  return lineage;
}

void encode_calibration(Encoder& encoder, const CalibrationResult& calibration,
                        const QuantizationLimits& limits) {
  validate_calibration(calibration, limits);
  encode_lineage(encoder, calibration.lineage, limits);
  encoder.u64(static_cast<std::uint64_t>(calibration.sample_ids.size()));
  for (const auto& sample_id : calibration.sample_ids)
    encoder.string(sample_id, limits.max_sample_id_bytes, "sample_id");
  encode_observer_state(encoder, calibration.activation_state);
  encode_qparams(encoder, calibration.activation_params);
}

CalibrationResult decode_calibration(Reader& reader,
                                     const QuantizationLimits& limits) {
  CalibrationResult result;
  result.lineage = decode_lineage(reader, limits);
  const auto count = reader.u64("sample_ids.count");
  const auto sample_count =
      reader.size(count, limits.max_samples, "sample_ids");
  constexpr std::size_t kCalibrationTailBytes = kObserverStateBytes + 16U;
  if (reader.remaining() < kCalibrationTailBytes ||
      sample_count > (reader.remaining() - kCalibrationTailBytes) / 5U)
    serialization("sample_ids.count", "complete IDs and observer state",
                  "truncated payload",
                  "Reject oversized sample counts before allocation.");
  result.sample_ids.reserve(sample_count);
  for (std::size_t i = 0; i < sample_count; ++i)
    result.sample_ids.push_back(
        reader.string(limits.max_sample_id_bytes, "sample_id"));
  result.activation_state = decode_observer_state(reader);
  result.activation_params = decode_qparams(reader, "activation_params");
  validate_calibration(result, limits);
  return result;
}

bool manifests_equal(const StateManifest& left, const StateManifest& right) {
  return left.participant_id == right.participant_id &&
         left.schema_version == right.schema_version &&
         left.restore_level == right.restore_level &&
         state_fields_equal(left.fields, right.fields) &&
         std::is_permutation(left.degradation_reasons.begin(),
                             left.degradation_reasons.end(),
                             right.degradation_reasons.begin(),
                             right.degradation_reasons.end());
}

std::string json_escape(std::string_view value) {
  std::string result;
  result.reserve(value.size() + 2U);
  for (const char ch : value) {
    switch (ch) {
      case '"':
        result += "\\\"";
        break;
      case '\\':
        result += "\\\\";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20U) {
          std::ostringstream escaped;
          escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                  << static_cast<unsigned int>(static_cast<unsigned char>(ch));
          result += escaped.str();
        } else {
          result.push_back(ch);
        }
    }
  }
  return result;
}

std::string number(double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::setprecision(17) << value;
  auto result = stream.str();
  if (result.size() > kMaxJsonNumberBytes) return "null";
  return result;
}

}  // namespace

void validate_format_descriptor(
    const QuantizationFormatDescriptor& descriptor) {
  if (descriptor.format_id != kFormatId || descriptor.format_version != 1U ||
      descriptor.dtype != "int8" ||
      descriptor.qscheme != QuantizationQScheme::symmetric ||
      descriptor.granularity != QuantizationGranularity::per_tensor ||
      descriptor.block_size != 0U || descriptor.alignment_bytes != 1U ||
      !descriptor.little_endian ||
      descriptor.execution_plane != QuantizationExecutionPlane::storage_only ||
      descriptor.compute_quantized || descriptor.qat_supported) {
    invalid("format_descriptor", std::string(kFormatId), descriptor.format_id,
            "Use the v1 storage-only symmetric per-tensor int8 descriptor.");
  }
}

void validate_limits(const QuantizationLimits& limits) {
  if (limits.max_artifact_bytes < 128U || limits.max_features == 0U ||
      limits.max_samples == 0U || limits.max_sample_id_bytes == 0U ||
      limits.max_lineage_text_bytes == 0U) {
    invalid("limits", "positive limits and max_artifact_bytes >= 128",
            "invalid resource limits", "Configure explicit bounded limits.");
  }
}

QuantizationParams derive_symmetric_params(std::span<const float> values) {
  double maximum = 0.0;
  bool has_finite = false;
  for (const float value : values) {
    if (!std::isfinite(value)) continue;
    has_finite = true;
    maximum = std::max(maximum, std::abs(static_cast<double>(value)));
  }
  if (!has_finite) {
    throw NotReady(
        "Quantization parameters are unavailable",
        ErrorDetails{.component = std::string(kComponent),
                     .operation = "derive_symmetric_params",
                     .expected = "at least one finite value",
                     .actual = "empty or all-nonfinite input",
                     .hint = "Collect a finite calibration sample first."});
  }
  const double scale = maximum == 0.0 ? 1.0 : maximum / 127.0;
  if (!std::isfinite(scale) || scale <= 0.0)
    numerical("derive_symmetric_params", "finite positive scale",
              std::to_string(scale), "Use values within FP32 range.");
  return {.scale = scale,
          .zero_point = 0,
          .quantized_min = static_cast<std::int8_t>(-127),
          .quantized_max = static_cast<std::int8_t>(127)};
}

std::vector<std::int8_t> quantize_symmetric(std::span<const float> values,
                                            QuantizationParams params) {
  validate_qparams(params);
  std::vector<std::int8_t> result;
  result.reserve(values.size());
  for (const float value : values) {
    if (!std::isfinite(value)) {
      numerical("quantize_symmetric", "finite input values", "NaN or infinity",
                "Filter non-finite values before quantization.");
    }
    const double scaled = static_cast<double>(value) / params.scale;
    if (!std::isfinite(scaled))
      numerical("quantize_symmetric", "finite scaled value", "overflow",
                "Use a positive scale covering the input range.");
    const double lower = std::floor(scaled);
    const double fraction = scaled - lower;
    const double rounded =
        fraction < 0.5
            ? lower
            : (fraction > 0.5
                   ? lower + 1.0
                   : (std::fmod(lower, 2.0) == 0.0 ? lower : lower + 1.0));
    const double clamped =
        std::clamp(rounded, static_cast<double>(params.quantized_min),
                   static_cast<double>(params.quantized_max));
    result.push_back(static_cast<std::int8_t>(clamped));
  }
  return result;
}

std::vector<float> dequantize_symmetric(std::span<const std::int8_t> values,
                                        QuantizationParams params) {
  validate_qparams(params);
  std::vector<float> result;
  result.reserve(values.size());
  for (const auto value : values) {
    if (value < params.quantized_min || value > params.quantized_max) {
      invalid("dequantize_symmetric", "int8 value in declared range",
              std::to_string(static_cast<int>(value)),
              "Use a canonical symmetric int8 payload.");
    }
    const double reconstructed =
        static_cast<double>(value - params.zero_point) * params.scale;
    if (!std::isfinite(reconstructed) ||
        std::abs(reconstructed) > std::numeric_limits<float>::max()) {
      numerical("dequantize_symmetric", "finite FP32 result", "overflow",
                "Use a representable qparam scale.");
    }
    result.push_back(static_cast<float>(reconstructed));
  }
  return result;
}

QuantizationErrorReport compare_reference(
    std::span<const float> original, std::span<const float> reconstructed) {
  if (original.size() != reconstructed.size()) {
    invalid("compare_reference", "equal vector lengths",
            std::to_string(original.size()) + " vs " +
                std::to_string(reconstructed.size()),
            "Compare tensors with the same shape.");
  }
  QuantizationErrorReport report;
  report.count = static_cast<std::uint64_t>(original.size());
  double squared_error = 0.0;
  double absolute_error = 0.0;
  double signal = 0.0;
  double original_norm = 0.0;
  double reconstructed_norm = 0.0;
  double dot = 0.0;
  constexpr double kRelativeEpsilon = 1.0e-12;
  for (std::size_t i = 0; i < original.size(); ++i) {
    const double lhs = original[i];
    const double rhs = reconstructed[i];
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
      ++report.nonfinite_inputs;
      continue;
    }
    const double difference = rhs - lhs;
    squared_error += difference * difference;
    absolute_error += std::abs(difference);
    signal += lhs * lhs;
    original_norm += lhs * lhs;
    reconstructed_norm += rhs * rhs;
    dot += lhs * rhs;
    report.max_absolute_error =
        std::max(report.max_absolute_error, std::abs(difference));
    report.max_relative_error = std::max(
        report.max_relative_error,
        std::abs(difference) / std::max(std::abs(lhs), kRelativeEpsilon));
  }
  const auto finite_count = report.count - report.nonfinite_inputs;
  if (finite_count != 0U) {
    report.rmse = std::sqrt(squared_error / static_cast<double>(finite_count));
    report.mae = absolute_error / static_cast<double>(finite_count);
  }
  if (squared_error == 0.0) {
    report.sqnr_db =
        signal == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
  } else if (signal != 0.0) {
    report.sqnr_db = 10.0 * std::log10(signal / squared_error);
  }
  const double norm_product = std::sqrt(original_norm * reconstructed_norm);
  if (norm_product == 0.0) {
    report.cosine_similarity = original_norm == reconstructed_norm ? 1.0 : 0.0;
  } else {
    report.cosine_similarity = std::clamp(dot / norm_product, -1.0, 1.0);
  }
  return report;
}

MinMaxObserver::MinMaxObserver(std::string participant)
    : participant_(std::move(participant)) {
  if (participant_.empty())
    invalid("observer.participant", "non-empty participant identity", "empty",
            "Use a stable checkpoint participant identifier.");
  static_cast<void>(describe_state());
}

void MinMaxObserver::observe(std::span<const float> values) {
  if (state_.sample_count == std::numeric_limits<std::uint64_t>::max() ||
      static_cast<std::uint64_t>(values.size()) >
          std::numeric_limits<std::uint64_t>::max() - state_.value_count) {
    throw ResourceExhausted("Observer counter overflow");
  }
  ++state_.sample_count;
  state_.value_count += static_cast<std::uint64_t>(values.size());
  for (const float value : values) {
    if (!std::isfinite(value)) continue;
    if (!state_.has_finite_value) {
      state_.minimum = value;
      state_.maximum = value;
      state_.has_finite_value = true;
    } else {
      state_.minimum = std::min(state_.minimum, value);
      state_.maximum = std::max(state_.maximum, value);
    }
  }
}

void MinMaxObserver::merge(const MinMaxObserver& other) {
  if (state_.sample_count > std::numeric_limits<std::uint64_t>::max() -
                                other.state_.sample_count ||
      state_.value_count > std::numeric_limits<std::uint64_t>::max() -
                               other.state_.value_count) {
    throw ResourceExhausted("Observer counter overflow");
  }
  state_.sample_count += other.state_.sample_count;
  state_.value_count += other.state_.value_count;
  if (!other.state_.has_finite_value) return;
  if (!state_.has_finite_value) {
    state_.minimum = other.state_.minimum;
    state_.maximum = other.state_.maximum;
    state_.has_finite_value = true;
  } else {
    state_.minimum = std::min(state_.minimum, other.state_.minimum);
    state_.maximum = std::max(state_.maximum, other.state_.maximum);
  }
}

MinMaxObserverState MinMaxObserver::state() const noexcept { return state_; }

QuantizationParams MinMaxObserver::finalize() const {
  if (!state_.has_finite_value)
    throw NotReady("Observer has no finite values",
                   "Observe at least one finite calibration value.");
  const std::array<float, 2U> extrema{state_.minimum, state_.maximum};
  return derive_symmetric_params(extrema);
}

std::string MinMaxObserver::participant_id() const { return participant_; }

StateManifest MinMaxObserver::describe_state() const {
  StateManifest manifest{
      .participant_id = participant_,
      .schema_version = kSchemaVersion,
      .restore_level = RestoreLevel::exact,
      .fields = {{.name = "state",
                  .media_type =
                      "application/vnd.jyppx.quantization-observer-v1",
                  .max_bytes = kObserverStateBytes}},
      .degradation_reasons = {}};
  validate_state_manifest(manifest);
  return manifest;
}

void MinMaxObserver::save_state(StateWriter& writer) const {
  const auto manifest = describe_state();
  static_cast<void>(manifest);
  Encoder encoder(kObserverStateBytes);
  encode_observer_state(encoder, state_);
  auto bytes = std::move(encoder).finish();
  if (bytes.size() != kObserverStateBytes)
    throw InternalError("Unexpected observer state encoding size");
  writer.write_blob("state", bytes);
}

void MinMaxObserver::load_state(const StateReader& reader,
                                RestorePolicy policy) {
  if (!restore_policy_accepts(policy, RestoreLevel::exact))
    compatibility("restore.policy", "policy accepting exact state",
                  std::string(restore_policy_name(policy)),
                  "Use a valid RestorePolicy value.");
  const auto current = describe_state();
  const auto& stored = reader.manifest();
  validate_state_manifest(stored);
  if (!manifests_equal(current, stored))
    compatibility("restore.manifest", current.participant_id,
                  stored.participant_id, "Use state from the same observer.");
  if (!reader.contains("state"))
    serialization("restore.state", "required state field", "missing",
                  "Provide the observer state blob.");
  const auto encoded = reader.read_blob("state", kObserverStateBytes);
  if (encoded.size() != kObserverStateBytes)
    serialization("restore.state",
                  std::to_string(kObserverStateBytes) + " bytes",
                  std::to_string(encoded.size()) + " bytes",
                  "Provide a complete observer state blob.");
  Reader decoder(encoded);
  const auto candidate = decode_observer_state(decoder);
  if (!decoder.empty())
    serialization("restore.state.trailing", "no trailing bytes", "present",
                  "Reject ambiguous observer state extensions.");
  state_ = candidate;
}

CalibrationSession::CalibrationSession(CalibrationLineage lineage,
                                       QuantizationLimits limits)
    : limits_(limits), lineage_(std::move(lineage)) {
  validate_limits(limits_);
  validate_lineage(lineage_, limits_);
}

void CalibrationSession::observe(std::string_view sample_id,
                                 std::span<const float> values) {
  if (sample_id.empty() || sample_id.size() > limits_.max_sample_id_bytes ||
      !valid_utf8(sample_id))
    invalid("calibration.sample_id", "non-empty bounded sample ID",
            std::to_string(sample_id.size()), "Use a stable short sample ID.");
  if (sample_ids_.size() >= limits_.max_samples)
    resource("calibration.sample_ids", limits_.max_samples,
             sample_ids_.size() + 1U, "Reduce calibration samples.");
  auto [index, inserted] = sample_id_index_.emplace(sample_id);
  if (!inserted)
    invalid("calibration.sample_id", "unique sample IDs",
            std::string(sample_id),
            "Deduplicate calibration samples before observation.");
  // Insert before observer mutation so allocation failure cannot skew lineage.
  try {
    sample_ids_.emplace_back(sample_id);
  } catch (...) {
    sample_id_index_.erase(index);
    throw;
  }
  try {
    observer_.observe(values);
  } catch (...) {
    sample_ids_.pop_back();
    sample_id_index_.erase(index);
    throw;
  }
}

CalibrationResult CalibrationSession::finalize() const {
  if (sample_ids_.empty())
    throw NotReady("Calibration session is empty",
                   "Observe at least one finite calibration sample.");
  CalibrationResult result{.lineage = lineage_,
                           .sample_ids = sample_ids_,
                           .activation_state = observer_.state(),
                           .activation_params = observer_.finalize()};
  validate_calibration(result, limits_);
  return result;
}

MinMaxObserver& CalibrationSession::observer() noexcept { return observer_; }

const CalibrationLineage& CalibrationSession::lineage() const noexcept {
  return lineage_;
}

void validate_linear_artifact(const QuantizedLinearArtifact& artifact,
                              QuantizationLimits limits) {
  validate_limits(limits);
  validate_format_descriptor(artifact.format);
  validate_qparams(artifact.weight_params);
  validate_calibration(artifact.calibration, limits);
  if (artifact.artifact_version != kSchemaVersion)
    compatibility("artifact_version", "1",
                  std::to_string(artifact.artifact_version),
                  "Use an explicit version migration.");
  if (artifact.in_features == 0U || artifact.out_features == 0U ||
      artifact.in_features > limits.max_features ||
      artifact.out_features > limits.max_features)
    invalid("linear.dimensions", "positive dimensions within max_features",
            std::to_string(artifact.in_features) + "x" +
                std::to_string(artifact.out_features),
            "Use bounded non-empty Linear dimensions.");
  std::size_t weight_count = 0U;
  if (!checked_mul(artifact.in_features, artifact.out_features,
                   &weight_count) ||
      weight_count > limits.max_artifact_bytes)
    resource("linear.weights", limits.max_artifact_bytes,
             limits.max_artifact_bytes + 1U, "Reduce the Linear dimensions.");
  if (artifact.quantized_weights.size() != weight_count)
    invalid("linear.weights", std::to_string(weight_count) + " elements",
            std::to_string(artifact.quantized_weights.size()) + " elements",
            "Provide one row-major int8 weight per Linear coefficient.");
  if (!artifact.bias.empty() && artifact.bias.size() != artifact.out_features)
    invalid("linear.bias", "empty or out_features elements",
            std::to_string(artifact.bias.size()),
            "Provide one FP32 bias per output feature.");
  if (artifact.calibration.sample_ids.empty() ||
      !artifact.calibration.activation_state.has_finite_value)
    invalid("linear.calibration", "completed finite calibration session",
            "empty or non-finite calibration",
            "Finalize calibration before conversion.");
  for (const auto value : artifact.quantized_weights) {
    if (value < artifact.weight_params.quantized_min ||
        value > artifact.weight_params.quantized_max)
      invalid("linear.weights", "canonical range [-127,127]",
              std::to_string(static_cast<int>(value)),
              "Use a payload produced by quantize_symmetric.");
  }
  for (const float value : artifact.bias) {
    if (!std::isfinite(value))
      numerical("linear.bias", "finite values", "NaN or infinity",
                "Use finite FP32 bias values.");
  }
  // Fixed-width fields and length prefixes excluding per-sample ID prefixes.
  std::size_t encoded_size = 233U;
  add_to_artifact_budget(encoded_size, artifact.format.format_id.size(),
                         limits.max_artifact_bytes);
  add_to_artifact_budget(encoded_size, artifact.quantized_weights.size(),
                         limits.max_artifact_bytes);
  std::size_t bias_bytes = 0U;
  if (!checked_mul(artifact.bias.size(), sizeof(float), &bias_bytes))
    throw ResourceExhausted("Linear bias byte count overflow");
  add_to_artifact_budget(encoded_size, bias_bytes, limits.max_artifact_bytes);
  add_to_artifact_budget(encoded_size,
                         artifact.calibration.lineage.dataset_version.size(),
                         limits.max_artifact_bytes);
  add_to_artifact_budget(
      encoded_size,
      artifact.calibration.lineage.preprocessing_fingerprint.size(),
      limits.max_artifact_bytes);
  for (const auto& sample_id : artifact.calibration.sample_ids) {
    add_to_artifact_budget(encoded_size, 4U, limits.max_artifact_bytes);
    add_to_artifact_budget(encoded_size, sample_id.size(),
                           limits.max_artifact_bytes);
  }
}

QuantizedLinearArtifact quantize_linear(const FloatLinearModel& model,
                                        const CalibrationResult& calibration,
                                        QuantizationLimits limits) {
  validate_limits(limits);
  if (model.in_features == 0U || model.out_features == 0U ||
      model.in_features > limits.max_features ||
      model.out_features > limits.max_features)
    invalid("linear.dimensions", "positive dimensions within max_features",
            std::to_string(model.in_features) + "x" +
                std::to_string(model.out_features),
            "Use bounded non-empty Linear dimensions.");
  std::size_t weight_count = 0U;
  if (!checked_mul(model.in_features, model.out_features, &weight_count) ||
      weight_count > limits.max_artifact_bytes)
    resource("linear.weights", limits.max_artifact_bytes,
             limits.max_artifact_bytes + 1U, "Reduce the Linear dimensions.");
  if (weight_count != model.weights.size())
    invalid("linear.weights", std::to_string(weight_count) + " coefficients",
            std::to_string(model.weights.size()),
            "Use row-major [out,in] weights.");
  const auto params = derive_symmetric_params(model.weights);
  QuantizedLinearArtifact artifact{
      .artifact_version = kSchemaVersion,
      .format = {},
      .in_features = model.in_features,
      .out_features = model.out_features,
      .weight_params = params,
      .quantized_weights = quantize_symmetric(model.weights, params),
      .bias = std::vector<float>(model.bias.begin(), model.bias.end()),
      .calibration = calibration};
  validate_linear_artifact(artifact, limits);
  // Encoding is the exact size gate; it includes metadata as well as tensors.
  static_cast<void>(serialize_linear_artifact(artifact, limits));
  return artifact;
}

std::vector<std::byte> serialize_linear_artifact(
    const QuantizedLinearArtifact& artifact, QuantizationLimits limits) {
  validate_linear_artifact(artifact, limits);
  Encoder encoder(limits.max_artifact_bytes);
  encoder.raw(std::as_bytes(std::span(kMagic.data(), kMagic.size())));
  encoder.u32(artifact.artifact_version);
  encode_descriptor(encoder, artifact.format, limits);
  encoder.u64(static_cast<std::uint64_t>(artifact.in_features));
  encoder.u64(static_cast<std::uint64_t>(artifact.out_features));
  encode_qparams(encoder, artifact.weight_params);
  encoder.u64(static_cast<std::uint64_t>(artifact.quantized_weights.size()));
  encoder.raw(std::as_bytes(std::span(artifact.quantized_weights)));
  encoder.u64(static_cast<std::uint64_t>(artifact.bias.size()));
  for (const float value : artifact.bias) encoder.f32(value);
  encode_calibration(encoder, artifact.calibration, limits);
  return std::move(encoder).finish();
}

QuantizedLinearArtifact deserialize_linear_artifact(
    std::span<const std::byte> bytes, QuantizationLimits limits) {
  validate_limits(limits);
  validate_size_limit(bytes.size(), limits.max_artifact_bytes, "artifact");
  Reader reader(bytes);
  const auto magic = reader.raw(kMagic.size(), "magic");
  if (!std::equal(
          magic.begin(), magic.end(),
          std::as_bytes(std::span(kMagic.data(), kMagic.size())).begin()))
    serialization("magic", "JYPPXQ1 binary header", "unknown header",
                  "Use the quantization artifact codec.");
  QuantizedLinearArtifact artifact;
  artifact.artifact_version = reader.u32("artifact_version");
  if (artifact.artifact_version != kSchemaVersion)
    compatibility("artifact_version", "1",
                  std::to_string(artifact.artifact_version),
                  "Apply an explicit quantization artifact migration.");
  artifact.format = decode_descriptor(reader, limits);
  artifact.in_features = reader.size(reader.u64("in_features"),
                                     limits.max_features, "in_features");
  artifact.out_features = reader.size(reader.u64("out_features"),
                                      limits.max_features, "out_features");
  if (artifact.in_features == 0U || artifact.out_features == 0U)
    serialization("linear.dimensions", "positive dimensions", "zero dimension",
                  "Reject empty Linear artifacts.");
  std::size_t weight_count = 0U;
  if (!checked_mul(artifact.in_features, artifact.out_features,
                   &weight_count) ||
      weight_count > limits.max_artifact_bytes)
    resource("linear.weights", limits.max_artifact_bytes,
             limits.max_artifact_bytes + 1U,
             "Reject oversized Linear weights.");
  artifact.weight_params = decode_qparams(reader, "weight_params");
  const auto encoded_weight_count = reader.u64("weights.count");
  if (encoded_weight_count != static_cast<std::uint64_t>(weight_count))
    serialization("weights.count", std::to_string(weight_count),
                  std::to_string(encoded_weight_count),
                  "Reject shape/payload mismatches before allocation.");
  const auto weight_bytes = reader.raw(weight_count, "weights");
  artifact.quantized_weights.reserve(weight_count);
  for (const auto value : weight_bytes)
    artifact.quantized_weights.push_back(
        static_cast<std::int8_t>(std::to_integer<std::uint8_t>(value)));
  const auto bias_count = reader.u64("bias.count");
  if (bias_count != 0U && bias_count != artifact.out_features)
    serialization("bias.count", "zero or out_features",
                  std::to_string(bias_count),
                  "Reject incompatible bias dimensions.");
  const auto bias_size =
      reader.size(bias_count, limits.max_features, "bias.count");
  if (bias_size > reader.remaining() / sizeof(float))
    serialization("bias", "complete FP32 bias payload", "truncated payload",
                  "Reject truncated bias before allocation.");
  artifact.bias.reserve(bias_size);
  for (std::size_t i = 0; i < bias_size; ++i)
    artifact.bias.push_back(reader.f32("bias"));
  artifact.calibration = decode_calibration(reader, limits);
  if (!reader.empty())
    serialization("artifact.trailing", "no trailing bytes", "present",
                  "Reject ambiguous artifact extensions.");
  validate_linear_artifact(artifact, limits);
  const auto canonical = serialize_linear_artifact(artifact, limits);
  if (canonical.size() != bytes.size() ||
      !std::equal(canonical.begin(), canonical.end(), bytes.begin()))
    serialization("artifact.canonical", "canonical bytes", "different bytes",
                  "Regenerate the artifact using the v1 canonical encoder.");
  return artifact;
}

StorageQuantizedLinearRuntime::StorageQuantizedLinearRuntime(
    QuantizedLinearArtifact artifact, QuantizationLimits limits)
    : artifact_(std::move(artifact)) {
  validate_linear_artifact(artifact_, limits);
  dequantized_weights_ = dequantize_symmetric(artifact_.quantized_weights,
                                              artifact_.weight_params);
}

std::vector<float> StorageQuantizedLinearRuntime::forward(
    std::span<const float> input, std::size_t batch) const {
  std::size_t input_count = 0U;
  std::size_t output_count = 0U;
  if (batch == 0U || !checked_mul(batch, artifact_.in_features, &input_count) ||
      input_count != input.size())
    invalid("runtime.input", "batch x in_features finite values",
            std::to_string(input.size()), "Provide complete FP32 input rows.");
  if (!checked_mul(batch, artifact_.out_features, &output_count))
    throw ResourceExhausted("Runtime output dimension overflow");
  for (const float value : input) {
    if (!std::isfinite(value))
      numerical("runtime.input", "finite FP32 values", "NaN or infinity",
                "Validate input before inference.");
  }
  std::vector<float> result(output_count, 0.0F);
  for (std::size_t row = 0; row < batch; ++row) {
    for (std::size_t out = 0; out < artifact_.out_features; ++out) {
      float value = artifact_.bias.empty() ? 0.0F : artifact_.bias[out];
      for (std::size_t in = 0; in < artifact_.in_features; ++in)
        value += input[row * artifact_.in_features + in] *
                 dequantized_weights_[out * artifact_.in_features + in];
      if (!std::isfinite(value))
        numerical("runtime.output", "finite FP32 result", "overflow",
                  "Use numerically bounded weights and inputs.");
      result[row * artifact_.out_features + out] = value;
    }
  }
  return result;
}

std::string_view StorageQuantizedLinearRuntime::execution_path()
    const noexcept {
  return "storage-quantized-dequantize-to-fp32-linear";
}

bool StorageQuantizedLinearRuntime::compute_quantized() const noexcept {
  return false;
}

bool StorageQuantizedLinearRuntime::qat_supported() const noexcept {
  return false;
}

QuantizationReport StorageQuantizedLinearRuntime::report(
    std::span<const float> original_weights, QuantizationLimits limits) const {
  QuantizationReport result{
      .schema_version = kSchemaVersion,
      .format_id = artifact_.format.format_id,
      .execution_claim = "storage-quantized",
      .compute_quantized = false,
      .qat = false,
      .artifact_bytes = serialize_linear_artifact(artifact_, limits).size(),
      .fallback_modules = 1U,
      .unsupported_modules = 0U,
      .weight_error = compare_reference(original_weights, dequantized_weights_),
      .calibration = artifact_.calibration.lineage};
  return result;
}

std::string quantization_report_json(const QuantizationReport& report,
                                     QuantizationLimits limits) {
  validate_limits(limits);
  validate_lineage(report.calibration, limits);
  if (report.schema_version != kSchemaVersion ||
      report.format_id != kFormatId ||
      report.execution_claim != "storage-quantized" ||
      report.compute_quantized || report.qat)
    invalid("report.claims", "v1 storage-only claims", report.execution_claim,
            "Report the execution plane actually implemented.");
  std::string result =
      "{\"schema_version\":" + std::to_string(report.schema_version) +
      ",\"format_id\":\"" + json_escape(report.format_id) +
      "\",\"execution_claim\":\"storage-quantized\",\"compute_quantized\":"
      "false,\"qat\":false" +
      ",\"artifact_bytes\":" + std::to_string(report.artifact_bytes) +
      ",\"fallback_modules\":" + std::to_string(report.fallback_modules) +
      ",\"unsupported_modules\":" + std::to_string(report.unsupported_modules) +
      ",\"weight_error\":{\"count\":" +
      std::to_string(report.weight_error.count) + ",\"nonfinite_inputs\":" +
      std::to_string(report.weight_error.nonfinite_inputs) +
      ",\"rmse\":" + number(report.weight_error.rmse) +
      ",\"mae\":" + number(report.weight_error.mae) +
      ",\"max_absolute_error\":" +
      number(report.weight_error.max_absolute_error) +
      ",\"max_relative_error\":" +
      number(report.weight_error.max_relative_error) +
      ",\"sqnr_db\":" + number(report.weight_error.sqnr_db) +
      ",\"cosine_similarity\":" +
      number(report.weight_error.cosine_similarity) +
      "},\"calibration\":{\"dataset_version\":\"" +
      json_escape(report.calibration.dataset_version) +
      "\",\"dataset_digest\":{\"algorithm\":\"sha256\",\"hex\":\"" +
      report.calibration.dataset_digest.hex +
      "\"},\"preprocessing_fingerprint\":\"" +
      json_escape(report.calibration.preprocessing_fingerprint) +
      "\",\"seed\":" + std::to_string(report.calibration.seed) + "}}";
  validate_size_limit(result.size(), limits.max_artifact_bytes, "report.json");
  return result;
}

}  // namespace jyppx::quantization
