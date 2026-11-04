// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TOOLS_IMAGE_SEARCH_BRIDGE_HPP
#define JYPPX_LIUDING_TOOLS_IMAGE_SEARCH_BRIDGE_HPP

#include "cli.hpp"

#include <jyppx/liuding/foundation/telemetry.hpp>

#include <cstddef>
#include <string>

namespace jyppx::liuding::cli {
// Pure inspection uses exactly the local-run resolver and bounded data readers.
Json inspect_image_search_inputs(const std::filesystem::path &path);
inline TelemetryMetricDefinition image_search_objective(std::size_t classes) {
  return {.metric_id = "classification.accuracy",
          .version = 1,
          .unit = TelemetryMetricUnit::ratio,
          .aggregation = TelemetryAggregation::mean,
          .direction = TelemetryMetricDirection::maximize,
          .parameters = {{"classes", std::to_string(classes)},
                         {"ignore_index", "none"},
                         {"zero_division", "zero"},
                         {"ties", "lowest_class_index"},
                         {"macro_classes", "all"}},
          .context_keys = {"slice", "source"}};
}
}  // namespace jyppx::liuding::cli
#endif  // JYPPX_LIUDING_TOOLS_IMAGE_SEARCH_BRIDGE_HPP
