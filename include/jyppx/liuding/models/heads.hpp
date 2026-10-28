// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_MODELS_HEADS_HPP
#define JYPPX_LIUDING_MODELS_HEADS_HPP
/** @file heads.hpp
 * @brief Explicit classifier metadata and replacement for built-in vision
 * models.
 */
#include <jyppx/liuding/nn/model_adapter.hpp>

#include <cstdint>
#include <string>

namespace jyppx::liuding::models {
/** @brief Structure replaced by reset_classifier, separate from the backbone.
 */
enum class ClassifierKind : std::uint8_t {
  linear = 1,      ///< One affine projection of the architecture's embedding.
  mobile_mlp = 2,  ///< MobileNetV3 projection, hard-swish, dropout and output.
};
/** @brief Meaning of the headless NC output; normalization remains
 * model-specific. */
enum class EmbeddingKind : std::uint8_t {
  pooled =
      1,  ///< Global spatial average, with architecture-specific normalization.
  cls_token = 2,  ///< The final normalized CLS token.
};
/** @brief Current classifier configuration, queried without model construction.
 */
struct ClassifierInfo {
  ClassifierKind kind =
      ClassifierKind::linear;  ///< Classification-head topology.
  EmbeddingKind embedding =
      EmbeddingKind::pooled;  ///< Headless output semantics.
  std::int64_t embedding_features =
      0;  ///< Width returned when classes is zero.
  std::int64_t num_classes =
      0;  ///< Zero removes the head; positive yields logits.
  std::string parameter_prefix;  ///< State subtree to replace; absent from
                                 ///< headless state.
};
/** @brief Queries the concrete built-in vision module retained by an adapter.
 * @param model Adapter retaining a supported concrete vision module.
 * @return Current head topology, class count, state prefix and embedding width.
 * @throws NotImplemented For the generic encoder or an unknown concrete module.
 * @note A feature extractor may be queried; this reports its underlying head.
 */
ClassifierInfo classifier_info(const nn::ModelAdapter& model);
/** @brief Replaces/removes a classifier and refreshes this adapter's schema.
 * @param model Exclusively owned quiescent classification/embedding adapter.
 * @param num_classes Integer in [0,1000000]; zero selects the model's
 * embedding.
 * @throws InvalidArgument For invalid class counts or an incompatible schema.
 * @throws NotImplemented For unsupported concrete modules.
 * @note The backbone retains tensor identity, device, dtype, mode and gradient
 * flags. A positive head is freshly initialized, trainable and follows the
 * current mode/device/dtype. MobileNetV3 replaces its complete classifier MLP;
 * zero returns the pooled last convolution before that MLP. Rebuild optimizers,
 * parameter groups, checkpoint topology and any other adapter aliases
 * afterward. The refreshed invocation calls the concrete module's forward
 * method. Do not mutate during forward/backward or concurrent state access.
 * Replacement is prepared before changing registered state; no implicit weight
 * import occurs.
 */
void reset_classifier(nn::ModelAdapter& model, std::int64_t num_classes);
}  // namespace jyppx::liuding::models
#endif
