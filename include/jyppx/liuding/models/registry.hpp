// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_MODELS_REGISTRY_HPP
#define JYPPX_LIUDING_MODELS_REGISTRY_HPP
/** @file registry.hpp
 * @brief Explicit owning model catalog; no static plugin registration.
 */
#include <jyppx/liuding/models/convnext.hpp>
#include <jyppx/liuding/models/densenet.hpp>
#include <jyppx/liuding/models/dino.hpp>
#include <jyppx/liuding/models/dinov3_convnext.hpp>
#include <jyppx/liuding/models/efficientnet.hpp>
#include <jyppx/liuding/models/efficientnetv2.hpp>
#include <jyppx/liuding/models/features.hpp>
#include <jyppx/liuding/models/mobilenetv2.hpp>
#include <jyppx/liuding/models/swin.hpp>
#include <jyppx/liuding/models/transformer.hpp>
#include <jyppx/liuding/models/vision.hpp>
#include <jyppx/liuding/models/vit.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
namespace jyppx::liuding::models {
/** @brief Model catalog metadata; default count excludes buffers. */
struct ModelInfo {
  std::string name;       ///< Unique lowercase architecture name.
  std::string family;     ///< Model family, such as resnet or efficientnet.
  std::string task;       ///< image_classification or feature_encoding.
  std::string reference;  ///< Exact authoritative algorithm/source family.
  std::uint64_t default_parameters = 0;  ///< Default options only.
  bool preview = true;  ///< Independent optional preview, not core GA.
  std::vector<FeatureInfo> features = {};
  ///< Ordered feature outputs; empty means no registered feature factory.
  std::optional<FeatureInputSpec> feature_input = {};
  ///< Spatial feature-input rule; present only with a feature factory.
};
/** @brief Typed architecture options for the selected catalog family. */
struct ModelOptions {
  ConvNeXtOptions convnext;          ///< Applied only to ConvNeXt models.
  DenseNetOptions densenet;          ///< Applied only to DenseNet models.
  EfficientNetOptions efficientnet;  ///< Applied only to EfficientNet.
  EfficientNetV2Options
      efficientnet_v2;           ///< Applied only to TF EfficientNetV2.
  ResNetOptions resnet;          ///< Applied only to residual models.
  MobileNetV2Options mobile_v2;  ///< Applied only to MobileNetV2.
  MobileNetV3Options mobile;     ///< Applied only to mobile models.
  TransformerEncoderOptions transformer;  ///< Applied only to feature encoder.
  SwinOptions swin;                       ///< Applied only to Swin-T.
  ViTOptions vit;                         ///< Applied only to ViT/16 models.
  DinoV2Options dinov2;                   ///< Applied only to DINOv2 models.
  DinoV3Options dinov3;                   ///< Applied only to DINOv3 models.
  DinoV3ConvNeXtOptions dinov3_convnext;  ///< Applied only to ConvNeXt models.
};
/** @brief An explicit factory retaining model state through its adapter. */
using ModelFactory = std::function<nn::ModelAdapter(const ModelOptions&)>;
/** @brief Local extensible catalog; modification needs external exclusion. */
class ModelRegistry final {
 public:
  /** @brief Adds one nonempty factory and validated unique name.
   * @param info Complete metadata identifying the architecture.
   * @param factory Owning factory to invoke for this registry entry.
   * @param feature_factory Optional owning factory producing all features as
   * an ordered tensor tuple from a first positional NCHW tensor. Required
   * exactly when feature metadata is present.
   * @throws InvalidArgument For invalid metadata, duplicates or an empty
   * factory.
   */
  void add(ModelInfo info, ModelFactory factory,
           ModelFactory feature_factory = {});
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

  /** @brief Creates an owning extractor with ordered, schema-checked features.
   * @param name Exact registered architecture name.
   * @param options Architecture settings consumed by the selected family.
   * @param out_indices Unique zero-based feature indices in requested order;
   * empty selects all. Invalid selections fail before invoking the factory.
   * @return ModelAdapter retaining the original module and a tensor tuple.
   * Outputs preserve autograd, device and dtype; no detach or tensor copying.
   * @throws NotImplemented If the entry has no explicit feature factory.
   * @throws InvalidArgument For invalid indices or factory output contracts.
   * @note Mode/state mutation and invocation need caller synchronization.
   * Metadata describes architecture semantics, not device qualification.
   */
  nn::ModelAdapter create_features(
      std::string_view name, const ModelOptions& options = {},
      const std::vector<std::size_t>& out_indices = {}) const;

 private:
  /** @brief Metadata and factory retained by one explicit registry entry. */
  struct Entry {
    ModelInfo info;                ///< Validated public architecture metadata.
    ModelFactory factory;          ///< Nonempty owning constructor callback.
    ModelFactory feature_factory;  ///< Optional ordered-feature constructor.
  };
  std::map<std::string, Entry, std::less<>>
      entries_;  ///< Explicit lexical catalog.
};
/** @brief Builds the explicit built-in entries, without global mutation.
 * @return Independent caller-owned registry with built-in preview factories.
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
/** @brief Creates a built-in feature extractor through ModelAdapter.
 * @param name Exact built-in architecture name with feature metadata.
 * @param options Explicit architecture options; no implicit preprocessing.
 * @param out_indices Unique metadata indices in requested order; empty is all.
 * @return Owning extractor returning a tensor tuple with matching schema.
 * @note ModelInfo::features names each output and its layout. The extractor
 * runs one complete feature forward and retains selected tensor handles.
 * Cropping follows ModelInfo::feature_input; no geometry is inferred.
 * @throws NotImplemented If the architecture has no feature factory.
 */
nn::ModelAdapter create_feature_extractor(
    std::string_view name, const ModelOptions& options = {},
    const std::vector<std::size_t>& out_indices = {});
/** @brief Counts registered parameter elements with overflow checks.
 * @param module Concrete module to inspect without modifying its state.
 * @return Total registered parameter elements, excluding buffers.
 */
std::uint64_t parameter_count(const torch::nn::Module& module);
}  // namespace jyppx::liuding::models
#endif
