// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DEPLOYMENT_ONNX_EXPORT_REPORT_HPP
#define JYPPX_LIUDING_DEPLOYMENT_ONNX_EXPORT_REPORT_HPP
/** @file report.hpp
 * @brief Bounded native export observations and caller-declared source lineage.
 */

#include <jyppx/liuding/foundation/artifact.hpp>
#include <jyppx/liuding/foundation/version.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::onnx_export {

/** @brief One CPU contiguous tensor observed before lowering. */
struct ExportTensorRecord {
  std::string name;   ///< Public tensor or registered state key.
  std::string dtype;  ///< Explicit portable scalar token such as float32.
  std::vector<std::int64_t> shape = {};  ///< Concrete example; scalar is empty.
  ArtifactDigest digest;  ///< SHA-256 of native contiguous tensor bytes.
  std::uint64_t size_bytes = 0U;  ///< Exact tensor payload byte count.
  std::string role;               ///< input, output, parameter or buffer.
};

/** @brief Caller assertions; the exporter does not authenticate source code. */
struct ExportSource {
  std::optional<std::string> run_id;  ///< Original training run, if known.
  std::optional<std::string> source_digest;  ///< Run's exact source identity.
  std::optional<std::string> config_digest;  ///< Resolved config SHA-256.
  std::optional<std::string> data_manifest_digest;  ///< Dataset SHA-256.
};

/** @brief Immutable-by-convention observations of one successful native trace.
 * @note The official serializer/checker ran. No runtime, numerical, signature
 * or arbitrary-control-flow qualification is implied. Digests identify bytes,
 * not a trusted producer. Caller must prevent concurrent model/input mutation.
 */
struct ExportReport {
  static constexpr std::uint32_t format_version = 1U;  ///< Closed wire schema.
  ArtifactDigest model_digest;  ///< Complete self-contained ONNX SHA-256.
  std::uint64_t model_size_bytes = 0U;  ///< Exact encoded model size.
  BuildFingerprint build;     ///< Exporter binary and dependency-lock identity.
  std::uint32_t opset = 17U;  ///< Actual qualified serializer opset.
  std::int64_t minimum_batch = 1;     ///< Caller policy, not encoded in ONNX.
  std::int64_t maximum_batch = 1024;  ///< Finite caller batch policy.
  std::map<std::string, std::map<std::int64_t, std::string>> dynamic_axes;
  ///< Actual resolved leading axes, including automatic batch names.
  std::vector<ExportTensorRecord> inputs;   ///< Ordered example inputs.
  std::vector<ExportTensorRecord> outputs;  ///< Ordered captured outputs.
  std::vector<ExportTensorRecord> state;  ///< All parameters/buffers, by name.
  std::vector<std::string>
      operators;        ///< Sorted unique lowered ONNX operators.
  ExportSource source;  ///< Optional caller-supplied original run lineage.
};

/** @brief Serializes validated native observations as deterministic schema v1.
 * @param report Successful export observations and optional source assertions.
 * @param maximum_bytes Positive bound, at most 64 MiB; default 4 MiB.
 * @return Compact UTF-8 JSON with explicit unchecked runtime/numerical status.
 * @throws Error Invalid metadata, unsupported profile or exceeded bounds.
 */
std::string serialize_export_report(const ExportReport& report,
                                    std::size_t maximum_bytes = 4U << 20U);

/** @brief Reads strict bounded metadata without trusting producer assertions.
 * @param json Complete report; duplicate/unknown fields and coercions fail.
 * @param maximum_bytes Positive encoded-size bound, at most 64 MiB.
 * @return Validated observations; no model parsing or execution is performed.
 * @throws Error Invalid schema, malformed content or exceeded bounds.
 */
ExportReport deserialize_export_report(std::string_view json,
                                       std::size_t maximum_bytes = 4U << 20U);

/** @brief Checks exact model identity and calls LibTorch's official checker.
 * @param report Valid native observations for this model.
 * @param onnx Complete immutable self-contained ONNX bytes.
 * @throws Error Invalid metadata, model size/digest mismatch or checker error.
 * @note This checks integrity and schema; it does not authenticate assertions
 * about source, tensor snapshots or execution, and does not run ONNX Runtime.
 */
void verify_export_report_model(const ExportReport& report,
                                std::span<const std::byte> onnx);

}  // namespace jyppx::liuding::onnx_export

#endif  // JYPPX_LIUDING_DEPLOYMENT_ONNX_EXPORT_REPORT_HPP
