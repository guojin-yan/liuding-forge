// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_TOOLS_CLI_NATIVE_OTLP_WORKER_HPP
#define JYPPX_TOOLS_CLI_NATIVE_OTLP_WORKER_HPP

#include "cli.hpp"
#include "sidecar_process.hpp"

#include <cstdint>
#include <span>
#include <string_view>

namespace jyppx::liuding::cli {

// The executable is always this CLI; no config/event may select an executable.
SidecarProcessOptions native_otlp_worker_options(const Arguments& arguments,
                                                 std::uint64_t snapshot_events);

// Intercept before normal CLI rendering: stdout belongs solely to protocol v1.
int native_otlp_worker_entry(std::span<const std::string_view> values) noexcept;

}  // namespace jyppx::liuding::cli
#endif
