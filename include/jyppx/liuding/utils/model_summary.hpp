// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
#ifndef JYPPX_LIUDING_UTILS_MODEL_SUMMARY_HPP
#define JYPPX_LIUDING_UTILS_MODEL_SUMMARY_HPP
#include <jyppx/liuding/core/types.hpp>

#include <torch/nn/module.h>

#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>

namespace jyppx::liuding {
namespace utils {

/** @brief Aggregate parameter, buffer, and storage statistics for a module. */
struct ModelStats {
  /** Number of elements in all recursively registered parameters. */
  int64_t total_params = 0;

  /** Number of parameter elements for which gradient tracking is enabled. */
  int64_t trainable_params = 0;

  /** Number of elements in all recursively registered buffers. */
  int64_t total_buffers = 0;

  /** Estimated bytes occupied by parameter and buffer element storage. */
  int64_t memory_bytes = 0;
};

/**
 * @brief Computes recursive parameter and buffer statistics for a module.
 * @param module Module whose registered tensors are inspected.
 * @return Aggregate counts and estimated tensor storage size.
 */
inline ModelStats calculate_model_stats(const torch::nn::Module& module) {
  ModelStats stats;
  for (const auto& parameter : module.parameters(/*recurse=*/true)) {
    if (!parameter.defined()) {
      continue;
    }
    const int64_t elements = parameter.numel();
    stats.total_params += elements;
    if (parameter.requires_grad()) {
      stats.trainable_params += elements;
    }
    stats.memory_bytes += elements * parameter.element_size();
  }
  for (const auto& buffer : module.buffers(/*recurse=*/true)) {
    if (!buffer.defined()) {
      continue;
    }
    stats.total_buffers += buffer.numel();
    stats.memory_bytes += buffer.numel() * buffer.element_size();
  }
  return stats;
}

/**
 * @brief Formats a byte count using binary KB, MB, or GB units.
 * @param bytes Byte count to format.
 * @return Human-readable value with one selected unit suffix.
 */
inline std::string format_bytes(int64_t bytes) {
  std::ostringstream oss;
  if (bytes >= 1024 * 1024 * 1024) {
    oss << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) << " GB";
  } else if (bytes >= 1024 * 1024) {
    oss << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MB";
  } else if (bytes >= 1024) {
    oss << (static_cast<double>(bytes) / 1024.0) << " KB";
  } else {
    oss << bytes << " B";
  }
  return oss.str();
}

/**
 * @brief Produces a Keras-style summary of direct submodules and total storage.
 * @param module Module whose direct children and recursive totals are reported.
 * @param name Display name placed in the summary heading.
 * @return Multi-line plain-text report.
 */
inline std::string model_summary(const torch::nn::Module& module,
                                 const std::string& name = "Model") {
  const ModelStats stats = calculate_model_stats(module);

  std::ostringstream oss;
  oss << name << " summary\n";
  oss << "----------------------------------------------------------\n";
  // Keep the returned container alive while its items() view is iterated.
  const auto children = module.named_children();
  for (const auto& item : children.items()) {
    const auto& child_name = item.key();
    const auto& child = item.value();
    const ModelStats child_stats = calculate_model_stats(*child);
    std::string type = "Module";
    try {
      type = child->name();
    } catch (...) {
      // Modules without a registered name keep the placeholder.
    }
    char line[128];
    std::snprintf(line, sizeof(line), "%-28s %-18s %12lld params",
                  child_name.c_str(), type.c_str(),
                  static_cast<long long>(child_stats.total_params));
    oss << line << "\n";
  }
  oss << "----------------------------------------------------------\n";
  oss << "Total params:        " << stats.total_params << "\n";
  oss << "Trainable params:    " << stats.trainable_params << "\n";
  oss << "Non-trainable:       "
      << (stats.total_params - stats.trainable_params) << "\n";
  oss << "Buffer elements:     " << stats.total_buffers << "\n";
  oss << "Estimated size:      " << format_bytes(stats.memory_bytes) << "\n";
  return oss.str();
}

}  // namespace utils
}  // namespace jyppx::liuding

#endif  // JYPPX_LIUDING_UTILS_MODEL_SUMMARY_HPP