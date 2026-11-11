// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_LOCAL_EXECUTION_LOG_RETENTION_HPP
#define JYPPX_LIUDING_LOCAL_EXECUTION_LOG_RETENTION_HPP

#include <jyppx/liuding/foundation/artifact.hpp>

#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <string>

namespace jyppx::liuding::experiment::local_execution_detail {

enum class LogRetention { always, on_failure, never };

struct LocalLogRetention {
  LogRetention stdout_policy = LogRetention::always;
  LogRetention stderr_policy = LogRetention::always;
};

LocalLogRetention read_log_retention(const nlohmann::json& value);

// Retention runs after the final typed-output and cleanup verdict. It never
// changes capture limits, process outcomes or already committed artifacts.
nlohmann::json retain_local_log(LogRetention policy, bool successful,
                                std::string& capture);

// Replay binds each receipt to the frozen policy and the actual CAS reference.
// A dropped capture's digest describes the bounded prefix, not unobserved
// bytes.
void validate_log_retention(const nlohmann::json& receipt, LogRetention policy,
                            bool successful, std::uint64_t observed_bytes,
                            std::uint64_t capture_limit,
                            const ArtifactRef& stored);

}  // namespace jyppx::liuding::experiment::local_execution_detail
#endif
