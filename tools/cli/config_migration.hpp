// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TOOLS_CONFIG_MIGRATION_HPP
#define JYPPX_TOOLS_CONFIG_MIGRATION_HPP

#include "cli.hpp"

#include <jyppx/liuding/experiment/config.hpp>

namespace jyppx::liuding::cli {

// Shared CLI schema conversion; the SDK remains the sole type/validation owner.
experiment::config::ConfigSchema decode_config_schema(const Json& encoded);
void validate_migration_values(const Json& values,
                               const experiment::config::ConfigSchema& schema);
Result config_migration_command(const Arguments& arguments);

}  // namespace jyppx::liuding::cli

#endif  // JYPPX_TOOLS_CONFIG_MIGRATION_HPP
