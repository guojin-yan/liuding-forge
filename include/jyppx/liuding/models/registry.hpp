// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_MODELS_REGISTRY_HPP
#define JYPPX_LIUDING_MODELS_REGISTRY_HPP
/** @file registry.hpp
 * @brief Explicit owning model catalog; no static plugin registration.
 */
#include <jyppx/liuding/models/transformer.hpp>
#include <jyppx/liuding/models/vision.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>
namespace jyppx::liuding::models {
/** @brief Model catalog metadata; default count excludes buffers. */
struct ModelInfo {
  std::string name;       ///< Unique lowercase architecture name.
  std::string family;     ///< resnet/mobilenet/transformer.
  std::string task;       ///< image_classification or feature_encoding.
  std::string reference;  ///< Exact authoritative algorithm/source family.
  std::uint64_t default_parameters = 0;  ///< Default options only.
  bool preview = true;  ///< Independent optional preview, not core GA.
};
/** @brief Typed architecture options for the selected catalog family. */
struct ModelOptions {
  ResNetOptions resnet;                   ///< Applied only to residual models.
  MobileNetV3Options mobile;              ///< Applied only to mobile models.
  TransformerEncoderOptions transformer;  ///< Applied only to feature encoder.
};
/** @brief An explicit factory retaining model state through its adapter. */
using ModelFactory = std::function<nn::ModelAdapter(const ModelOptions&)>;
/** @brief Local extensible catalog; modification needs external exclusion. */
class ModelRegistry final {
 public:
  /** @brief Adds one nonempty factory and validated unique name.
   * @param info Complete metadata identifying the architecture.
   * @param factory Owning factory to invoke for this registry entry.
   * @throws InvalidArgument For invalid metadata, duplicates or an empty
   * factory.
   */
  void add(ModelInfo info, ModelFactory factory);
  /** @brief Returns model metadata or an actionable unknown-model error.
   * @param name Exact registered architecture name.
   * @return Copy of the selected architecture metadata.
   */
  ModelInfo info(std::string_view name) const;
  /** @brief Lists the deterministic name order, with optional name substring.
   * @param filter Optional substring matched against architecture names.
   * @return Metadata copies in lexical name order.
   */
  std::vector<ModelInfo> list(std::string_view filter = {}) const;
  /** @brief Creates the selected concrete architecture and schema boundary.
   * @param name Exact registered architecture name.
   * @param options Typed options consumed by the selected family.
   * @return Adapter owning a newly constructed randomly initialized module.
   */
  nn::ModelAdapter create(std::string_view name,
                          const ModelOptions& options = {}) const;

 private:
  /** @brief Metadata and factory retained by one explicit registry entry. */
  struct Entry {
    ModelInfo info;        ///< Validated public architecture metadata.
    ModelFactory factory;  ///< Nonempty owning constructor callback.
  };
  std::map<std::string, Entry, std::less<>>
      entries_;  ///< Explicit lexical catalog.
};
/** @brief Builds the five explicit built-in entries, without global mutation.
 * @return Independent caller-owned registry with the five preview factories.
 */
ModelRegistry default_model_registry();
/** @brief Lists built-in architectures, optionally filtered by name substring.
 * @param filter Optional substring matched against architecture names.
 * @return Built-in metadata copies in lexical name order.
 */
std::vector<ModelInfo> list_models(std::string_view filter = {});
/** @brief Looks up one built-in architecture's default metadata.
 * @param name Exact built-in architecture name.
 * @return Copy of the selected built-in metadata.
 */
ModelInfo model_info(std::string_view name);
/** @brief Returns a real architecture through the common ModelAdapter boundary.
 * @param name Exact built-in architecture name.
 * @param options Typed options consumed by the selected family.
 * @return Adapter owning a newly constructed module.
 * @note Construction is random initialization. No implicit pretrained loading.
 */
nn::ModelAdapter create_model(std::string_view name,
                              const ModelOptions& options = {});
/** @brief Counts registered parameter elements with overflow checks.
 * @param module Concrete module to inspect without modifying its state.
 * @return Total registered parameter elements, excluding buffers.
 */
std::uint64_t parameter_count(const torch::nn::Module& module);
}  // namespace jyppx::liuding::models
#endif
