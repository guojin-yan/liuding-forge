// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_TRAINING_TRANSFER_HPP
#define JYPPX_LIUDING_VISION_TASKS_TRAINING_TRANSFER_HPP
/** @file transfer.hpp @brief Local native weights, explicit mappings and static
 * freezing. */
#include <jyppx/liuding/nn/model_adapter.hpp>
#include <jyppx/liuding/vision/tasks/training/strategy.hpp>

#include <filesystem>
#include <map>
namespace jyppx::liuding::vision::tasks {
struct ParameterRateRule {
  std::string prefix;  ///< Exact module subtree; longest matching prefix wins.
  double multiplier = 1.;
};
struct InitializationOptions {
  std::filesystem::path
      weights;  ///< Safetensors; no pickle execution or downloads.
  std::string
      sha256;  ///< Independently supplied digest required for any import.
  std::map<std::string, std::string>
      key_mapping;  ///< Source key -> native key, injective.
  std::vector<std::string>
      ignored_source_keys;  ///< Exact extra keys, explicitly acknowledged.
  std::vector<std::string> retain_buffer_keys;  ///< Exact derived buffers kept
                                                ///< from the model definition.
  std::vector<std::string>
      replace_head_prefixes;  ///< Only these subtrees may be omitted.
  std::vector<std::string>
      freeze_prefixes;  ///< Freeze parameters and persistent buffers.
  std::vector<ParameterRateRule> learning_rates;
  std::uint64_t max_bytes = 1ULL << 30U;
};
/** @brief Loads verified local state before optimizer/EMA creation and returns
 * JSON. Strict by default. Replacing a head permits missing/shape/dtype
 * differences only inside explicitly named subtrees, preserving their fresh
 * initialization. Every other source/target entry must match exactly. No
 * partial success is reported.
 */
std::string initialize_vision(nn::ModelAdapter& model,
                              const InitializationOptions& options,
                              std::string_view architecture,
                              std::stop_token stop = {});
/** @brief Proposes exact-name/declared-alias mappings after digest and shape
 * checks. Returns mapping, shape mismatches and unmatched source/target
 * entries. Never loads weights, guesses by tensor shape, ignores keys, or
 * qualifies a weight.
 */
std::string propose_vision_weight_mapping(const nn::ModelAdapter& model,
                                          const std::filesystem::path& source,
                                          std::string_view sha256,
                                          std::string_view architecture,
                                          std::uint64_t max_bytes = 1ULL
                                                                    << 30U);
/** @brief Saves parameters and buffers as native safetensors with source
 * identity. Caller excludes concurrent forwards/updates; CPU copies are
 * bounded. Role must explicitly name raw, ema or fused state; saving never
 * converts training state.
 */
std::string save_vision_weights(
    const nn::ModelAdapter& model, const std::filesystem::path& path,
    std::map<std::string, std::string> metadata = {},
    std::uint64_t max_bytes = 1ULL << 30U);
/** @brief Saves only a declared SimMIM encoder subtree as a strict native
 * initialization artifact. Reconstruction head, mask token and training state
 * are excluded. Downstream classifier/neck replacement remains explicit. */
std::string save_simmim_encoder(const nn::ModelAdapter& model,
    const std::filesystem::path& path,std::string_view architecture,
    std::uint64_t max_bytes=1ULL<<30U);
/** @brief Reapplies eval to frozen subtrees after Trainer selects training
 * mode. */
void enforce_frozen_modules(nn::ModelAdapter& model,
                            const InitializationOptions& options);
/** @brief Splits existing decay groups into stable differential learning-rate
 * groups. */
void apply_parameter_rates(DetectionSgdGroups& groups,
                           std::span<const ParameterRateRule> rules);
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_TRAINING_TRANSFER_HPP
