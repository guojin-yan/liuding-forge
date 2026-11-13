// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_PRIVATE_TELEMETRY_HPP
#define JYPPX_LIUDING_VISION_TASKS_PRIVATE_TELEMETRY_HPP
#include <jyppx/liuding/training/callback_dispatcher.hpp>

#include <filesystem>
#include <functional>
#include <map>
namespace jyppx::liuding::vision::tasks::detail {
using MetricObserver = std::function<void(
    const training::CallbackEvent&, const std::map<std::string, double>&)>;
MetricObserver tensorboard_observer(const std::filesystem::path& output,
                                    bool enabled);
}  // namespace jyppx::liuding::vision::tasks::detail
#endif
