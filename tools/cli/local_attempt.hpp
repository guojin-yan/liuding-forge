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

}  // namespace jyppx::liuding::cli

#endif  // JYPPX_LIUDING_TOOLS_CLI_LOCAL_ATTEMPT_HPP
