// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "telemetry.hpp"

#include <jyppx/liuding/foundation/error.hpp>
namespace jyppx::liuding::vision::tasks::detail {
MetricObserver tensorboard_observer(const std::filesystem::path&,
                                    bool enabled) {
  if (enabled)
    throw CompatibilityError(
        "TensorBoard requires the telemetry_native component");
  return {};
}
}  // namespace jyppx::liuding::vision::tasks::detail
