// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_TOOLS_CLI_NATIVE_TENSORBOARD_WORKER_HPP
#define JYPPX_TOOLS_CLI_NATIVE_TENSORBOARD_WORKER_HPP

#include "cli.hpp"
#include "sidecar_process.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>

namespace jyppx::liuding::cli {

// Fixed self-executable, bounded immutable configuration and a fresh output
// outside the managed store; configuration/events cannot choose an executable.
SidecarProcessOptions native_tensorboard_worker_options(
    const Arguments& arguments, std::uint64_t snapshot_events,
    const std::filesystem::path& managed_root);

// Stdout is protocol-only, intercepted before ordinary CLI rendering.
int native_tensorboard_worker_entry(
    std::span<const std::string_view> values) noexcept;

}  // namespace jyppx::liuding::cli
#endif
