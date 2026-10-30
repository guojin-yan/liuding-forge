// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_TOOLS_CLI_LOCAL_ATTEMPT_HPP
#define JYPPX_LIUDING_TOOLS_CLI_LOCAL_ATTEMPT_HPP

#include "cli.hpp"

namespace jyppx::liuding::cli {

/** @brief Durable bounded local execution composed over run/CAS and OS locks.
 * @param arguments local execute/show/recover/cancel, or the private worker.
 * @return Detached job/attempt state; unresolved ownership never relaunches.
 */
Result local_attempt_command(const Arguments& arguments);

/** @brief Validates a frozen request, optionally checking executable and cwd.
 * Optional typed training inputs are structurally checked here; their run/CAS
 * bindings are checked by execute before admission and by the owned worker.
 */
void validate_local_execution_spec(const Json& spec, bool inspect_files = true);

/** @brief Executes/replays one frozen job with an optional run parent. */
Result execute_local_job(
    const std::filesystem::path& root, std::string_view job_id,
    const Json& spec, std::optional<std::string> parent_run_id = std::nullopt);

/** @brief Reads, recovers or cancels a job without parsing CLI output. */
Result control_local_job(std::string_view action,
                         const std::filesystem::path& root,
                         std::string_view job_id);

}  // namespace jyppx::liuding::cli

#endif  // JYPPX_LIUDING_TOOLS_CLI_LOCAL_ATTEMPT_HPP
