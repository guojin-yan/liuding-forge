// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_MODELS_WEIGHTS_HPP
#define JYPPX_LIUDING_MODELS_WEIGHTS_HPP
/** @file weights.hpp
 * @brief Auditable local safetensors loading and digest-addressed weight cache.
 */
#include <jyppx/liuding/interop/model_weight_import.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>
namespace jyppx::liuding::models {
/** @brief Explicit acquisition state; missing hashes never authorize fetching.
 */
enum class WeightAvailability : std::uint8_t {
  not_provisioned = 1,  ///< Source candidate only; no approved exact bytes.
  pinned_external = 2,  ///< Caller supplied complete byte identity and license.
};
/** @brief Exact auditable artifact metadata, separate from model code license.
 */
struct WeightRecord {
  std::string id;          ///< Unique producer-selected weight identity.
  std::string model_name;  ///< Architecture association, not topology proof.
  std::string source_uri;  ///< Provenance/transport reference; never executed.
  std::string
      license_spdx;         ///< Weight license or NOASSERTION when unavailable.
  std::string license_uri;  ///< Source of the artifact license determination.
  std::string format =
      "safetensors";  ///< Executable/pickle formats not loaded.
  WeightAvailability availability =
      WeightAvailability::not_provisioned;  ///< Explicit acquisition state.
  std::optional<ArtifactRef>
      artifact;               ///< Required complete hash/size if pinned.
  std::string status_reason;  ///< Explicit missing/qualification explanation.
};
/** @brief Returns upstream candidates with missing hash/license status
 * explicit.
 * @param model_name Exact built-in architecture name.
 * @return Unprovisioned upstream candidates; the feature encoder has none.
 * @note No upstream artifact is downloaded, converted, licensed or qualified.
 */
std::vector<WeightRecord> available_weights(std::string_view model_name);
/** @brief Produces a versioned truthful catalog of code/reference/weight
 * status.
 * @return Deterministic UTF-8 JSON describing the preview qualification.
 */
std::string models_preview_manifest_json();
/** @brief Validates metadata; a pinned record needs full SHA256,size and
 * license.
 * @param record Complete acquisition and artifact-license provenance.
 */
void validate_weight_record(const WeightRecord& record);
/** @brief Verifies exact bytes and bounded data-only tensor structure.
 * @param record Independently approved exact artifact identity and provenance.
 * @param bytes Stable complete encoded safetensors artifact.
 * @param max_bytes Maximum accepted encoded artifact size, at most 16 GiB.
 * @throws NotReady If no pinned verified artifact identity is available.
 */
void verify_weights(const WeightRecord& record,
                    std::span<const std::byte> bytes,
                    std::uint64_t max_bytes = 512ULL << 20U);
/** @brief Applies verified safetensors by the strict staged nn import path.
 * @param model Exclusively owned quiescent model state to update.
 * @param record Independently approved exact artifact identity and provenance.
 * @param bytes Stable complete encoded safetensors artifact.
 * @param options Explicit strict/partial mapping and resource limits. Encoded
 * artifacts are capped at 16 GiB; staging has its own independent byte budget.
 * @return Full import mapping and actual application result.
 * @note Caller must exclusively own quiescent model state as documented by
 * import_model_weights. Optimizer/storage handles need rebuilding after import.
 */
interop::ModelWeightImportResult load_weights(
    nn::ModelAdapter& model, const WeightRecord& record,
    std::span<const std::byte> bytes,
    const interop::ModelWeightImportOptions& options = {});
/** @brief Explicit transport supplied by caller, without implicit networking.
 * @note Callback owns its allocation/time/transport bounds; the returned bytes
 * are rechecked before any cache write. No package imports or shell execution.
 */
using WeightFetcher = std::function<std::vector<std::byte>(
    const WeightRecord&, std::uint64_t max_bytes)>;
/** @brief Bound for exact local cache read and transport response acceptance.
 */
struct WeightCacheOptions {
  std::uint64_t max_bytes = 512ULL
                            << 20U;  ///< Accepted complete artifact bytes.
};
/** @brief Acquires or reuses hash-addressed, structurally validated weights.
 * @param record Independently approved pinned external safetensors metadata.
 * @param directory Caller-owned trusted cache directory.
 * @param fetch Explicit caller transport, used only on a cache miss.
 * @param options Encoded response/read bound.
 * @return Stable digest-addressed safetensors path, after revalidation.
 * @note Publication uses a no-replace hard link from private staging. Corrupt
 * cache hits fail and are never silently replaced. Cache must remain
 * exclusively controlled by the caller; this is not a hostile-directory
 * sandbox. A filesystem without hard links reports IoError rather than weaker
 * publication.
 */
std::filesystem::path cache_weights(const WeightRecord& record,
                                    const std::filesystem::path& directory,
                                    const WeightFetcher& fetch,
                                    const WeightCacheOptions& options = {});
/** @brief Removes only old canonical hash.safetensors regular cache entries.
 * @param directory Trusted caller-owned cache root.
 * @param keep_age Minimum last-write age to keep; must be nonnegative.
 * @param max_entries Maximum directory entries inspected before refusing work.
 * @return Number of removed canonical entries; other content is untouched.
 * @note Exclude concurrent cache/model consumers while cleaning.
 */
std::size_t clean_weight_cache(const std::filesystem::path& directory,
                               std::chrono::hours keep_age,
                               std::size_t max_entries = 4096U);
}  // namespace jyppx::liuding::models
#endif
