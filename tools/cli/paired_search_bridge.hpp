// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TOOLS_PAIRED_SEARCH_BRIDGE_HPP
#define JYPPX_LIUDING_TOOLS_PAIRED_SEARCH_BRIDGE_HPP

#include "cli.hpp"

#include <jyppx/liuding/foundation/telemetry.hpp>

namespace jyppx::liuding::cli {
// Pure recipe inspection, with the same bounded resolver/readers as local-run.
Json inspect_paired_search_inputs(const std::filesystem::path &path);
TelemetryMetricDefinition paired_search_objective();
}  // namespace jyppx::liuding::cli
#endif  // JYPPX_LIUDING_TOOLS_PAIRED_SEARCH_BRIDGE_HPP
