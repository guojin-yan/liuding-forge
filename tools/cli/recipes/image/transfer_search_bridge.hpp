// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TOOLS_TRANSFER_SEARCH_BRIDGE_HPP
#define JYPPX_LIUDING_TOOLS_TRANSFER_SEARCH_BRIDGE_HPP

#include "cli.hpp"

namespace jyppx::liuding::cli {
// Inspect with the local-run resolver and freeze its full weight descriptor.
Json inspect_transfer_search_inputs(const std::filesystem::path &path);

inline Json transfer_search_configuration(const Json &configuration,
                                          const Json &transfer) {
  return {{"format", "jyppx.liuding.local_transfer_configuration"},
          {"schema_version", 1},
          {"configuration", configuration},
          {"transfer", transfer}};
}
}  // namespace jyppx::liuding::cli
#endif  // JYPPX_LIUDING_TOOLS_TRANSFER_SEARCH_BRIDGE_HPP
