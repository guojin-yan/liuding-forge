// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TOOLS_CLI_SIDECAR_PROCESS_HPP
#define JYPPX_TOOLS_CLI_SIDECAR_PROCESS_HPP

#include <jyppx/liuding/experiment/local_execution/json_lines.hpp>

namespace jyppx::liuding::cli {
using SidecarProcessOptions = experiment::JsonLineProcessOptions;
using SidecarProcessStats = experiment::JsonLineProcessStats;
using SidecarProcessTransport = experiment::JsonLineProcessTransport;
}  // namespace jyppx::liuding::cli

#endif  // JYPPX_TOOLS_CLI_SIDECAR_PROCESS_HPP
