// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_TOOLS_CLI_LOCAL_PROCESS_HPP
#define JYPPX_LIUDING_TOOLS_CLI_LOCAL_PROCESS_HPP

/** @file local_process.hpp
 * @brief Private tool-layer lifecycle primitive for bounded local attempts.
 */

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>

namespace jyppx::liuding::cli {

/** @brief OS memory boundary for one native process and its descendants. */
struct LocalProcessMemoryLimit {
  std::uint64_t bytes = 0;  ///< Positive whole MiB; backend-specific charges.
  std::filesystem::path linux_cgroup_parent;  ///< Delegated cgroup v2 parent.
};

/** @brief Confirmed OS policy and final counters, separate from direct RSS. */
struct LocalProcessMemoryBoundary {
  std::string source;             ///< windows_job_commit or linux_cgroup_v2.
  std::uint64_t limit_bytes = 0;  ///< Configured enforced cap.
  std::optional<std::uint64_t> peak_bytes;  ///< Job commit or cgroup charges.
  std::optional<std::uint64_t> oom_kills;   ///< Linux memory.events only.
};

/** @brief Immutable launch and finite resource policy for one invocation. */
struct LocalProcessOptions {
  std::filesystem::path
      executable;  ///< Absolute executable; never PATH lookup.
  std::vector<std::string>
      arguments;  ///< NUL-free UTF-8 arguments after argv[0].
  std::filesystem::path
      working_directory;  ///< Absolute existing caller-owned cwd.
  std::map<std::string, std::string>
      environment;  ///< Child-only UTF-8 overrides.
  bool inherit_environment =
      false;  ///< Explicit opt-in to the parent snapshot.
  std::size_t max_launch_bytes =
      256U * 1024U;  ///< Combined native argv/env cap.
  std::size_t max_stdout_bytes =
      1024U * 1024U;  ///< Total output and capture cap.
  std::size_t max_stderr_bytes =
      1024U * 1024U;  ///< Total diagnostic/capture cap.
  std::chrono::milliseconds timeout{
      3600000};  ///< Launch through exit/drain, <=7d.
  std::chrono::milliseconds startup_timeout{10000};  ///< Launch upper bound.
  std::chrono::milliseconds cleanup_timeout{2000};   ///< Extra kill/reap grace.
  // Linux opt-in for synchronous ownership. Ordinary same-credential exec
  // preserves this fence; privilege transitions or child code can clear it.
  // This protects the direct child, not arbitrary forked descendants.
  bool kill_on_parent_death =
      false;  ///< SIGKILL if the Linux launching thread dies.
  std::optional<LocalProcessMemoryLimit>
      memory_limit;  ///< Fail closed if the requested OS boundary is
                     ///< unavailable.
};

/** @brief A child failure is a result; invalid launch and I/O throw Error. */
enum class LocalProcessStatus : std::uint8_t {
  exited,       ///< Includes zero and nonzero native exit codes.
  signalled,    ///< POSIX signal termination; no fabricated exit code.
  cancelled,    ///< Caller requested cancellation.
  timed_out,    ///< Finite wall-clock budget expired.
  output_limit  ///< Either stream exceeded its cumulative byte quota.
};

/** @brief OS accounting scope; neither source measures concurrent tree RSS. */
enum class LocalProcessResourceSource : std::uint8_t {
  windows_process,  ///< Direct process times and peak working set only.
  linux_wait4       ///< Kernel wait4 accounting, including waited descendants.
};

/** @brief Final OS counters; unavailable individual measurements stay absent.
 */
struct LocalProcessResourceUsage {
  LocalProcessResourceSource source;
  std::optional<std::uint64_t> user_cpu_time_us;
  std::optional<std::uint64_t> system_cpu_time_us;
  std::optional<std::uint64_t> peak_resident_memory_bytes;
};

/** @brief Detached bounded output and status; never owns a running child. */
struct LocalProcessResult {
  LocalProcessStatus status = LocalProcessStatus::exited;
  std::optional<std::uint32_t>
      exit_code;              ///< Native code only on ordinary exit.
  std::optional<int> signal;  ///< Native signal only for signalled POSIX exit.
  std::uint64_t process_id =
      0;  ///< Zero if cancelled before successful launch.
  std::uint64_t stdout_bytes =
      0;  ///< Drained bytes; may exceed cap by one chunk.
  std::uint64_t stderr_bytes = 0;  ///< Drained diagnostic bytes.
  std::string stdout_capture;      ///< Raw bytes; at most max_stdout_bytes.
  std::string stderr_capture;      ///< Raw bytes; at most max_stderr_bytes.
  std::chrono::milliseconds elapsed{};  ///< Includes launch, drain and cleanup.
  bool process_tree_cleanup_confirmed = false;  ///< Conservative native proof.
  std::optional<LocalProcessResourceUsage>
      resource_usage;  ///< Absent before launch or if final accounting failed.
  std::optional<LocalProcessMemoryBoundary>
      memory_boundary;  ///< Present only after an OS-limited native launch.
};

/**
 * @brief Runs one local command without a shell or changes to parent state.
 *
 * stdin is EOF. Both streams are drained in fixed chunks and captured only up
 * to the configured caps. Exit, failure, cancellation and quota/timeout paths
 * terminate remaining descendants before releasing ownership. Windows uses a
 * suspended launch in a non-breakaway kill-on-close Job Object; POSIX uses a
 * distinct process group and reaps its direct child. POSIX descendants must
 * cooperate with group ownership: setsid()/setpgid() escape is not sandboxed.
 *
 * This primitive does not claim durable attempt ownership, retry, run storage,
 * recipe recovery or search. Those compose above it using existing run/CAS.
 */
class LocalProcessRunner final {
 public:
  /**
   * @brief Executes exactly once under the supplied finite policy.
   * @param options Explicit launch, capture and timeout constraints.
   * @param stop Cancellation checked during startup and execution.
   * @return Native termination or explicit cancellation/timeout/quota result.
   * @throws InvalidArgument Invalid paths, text, quotas or time constraints.
   * @throws Error Launch, platform compatibility or pipe failures.
   */
  LocalProcessResult run(const LocalProcessOptions& options,
                         std::stop_token stop = {}) const;
};

}  // namespace jyppx::liuding::cli

#endif  // JYPPX_LIUDING_TOOLS_CLI_LOCAL_PROCESS_HPP
