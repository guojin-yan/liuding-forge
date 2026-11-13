// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_TRAINING_WEIGHT_CONVERSION_HPP
#define JYPPX_LIUDING_VISION_TASKS_TRAINING_WEIGHT_CONVERSION_HPP
/** @file weight_conversion.hpp @brief Offline native PyTorch ZIP tensor
 * extraction. */
#include <jyppx/liuding/vision/tasks/training/transfer.hpp>
namespace jyppx::liuding::vision::tasks {
struct WeightConversionOptions {
  std::filesystem::path source, output;
  std::string sha256;  ///< Required digest of the complete local input.
  std::string root =
      "model";  ///< model/ema/state_dict/root; no implicit fallback.
  std::map<std::string, std::string>
      key_mapping;  ///< Explicit source -> target renames.
  std::map<std::string, std::string>
      metadata;  ///< Include upstream revision/license/source.
  std::uint64_t max_bytes = 2ULL << 30U;
  std::size_t max_nodes = 2000000, max_tensors = 65536;
  bool floating_to_float32 = true;
};
/** @brief Extracts tensors without Python or executing serialized
 * functions/classes. Accepts modern PyTorch ZIP checkpoints and protocols 2..5.
 * Tensor/Parameter, OrderedDict and inert nn.Module state are supported;
 * arbitrary reductions, tensor subclasses, sparse/quantized and legacy non-ZIP
 * archives are rejected. Output keeps source keys unless explicitly mapped.
 * Structural compatibility with a particular model is checked separately by
 * initialize_vision().
 */
std::string convert_vision_weights(const WeightConversionOptions& options,
                                   std::stop_token stop = {});
}  // namespace jyppx::liuding::vision::tasks
#endif
