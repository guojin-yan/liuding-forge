// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TOOLS_CLI_SIDECAR_PROCESS_HPP
#define JYPPX_TOOLS_CLI_SIDECAR_PROCESS_HPP

/** @file sidecar_process.hpp
 * @brief Tool-owned, bounded argv subprocess transport for telemetry observers.
 * @note This is a private CLI implementation, not an installed component API.
 */

#include <jyppx/liuding/experiment/telemetry_exporters.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::cli {

/** @brief Fixed launch configuration; no field is populated from an event. */
struct SidecarProcessOptions {
  std::filesystem::path
      executable;  ///< Absolute native executable; no PATH search.
  std::vector<std::string>
      arguments;  ///< UTF-8 argv after argv[0]; no shell syntax.
  std::filesystem::path
      working_directory;  ///< Existing, caller-owned isolated cwd.
  std::map<std::string, std::string>
      environment;  ///< UTF-8 child-only overrides.
  bool inherit_environment =
      false;  ///< Explicitly opt in to a parent snapshot.
  std::size_t max_request_bytes = 16U * 1024U * 1024U;  ///< Excludes final LF.
  std::size_t max_reply_bytes = 128U * 1024U;           ///< Excludes final LF.
  std::size_t max_stderr_bytes = 64U * 1024U;  ///< Cumulative discard quota.
  std::size_t max_launch_bytes =
      256U * 1024U;  ///< argv/environment native bytes.
  std::chrono::milliseconds operation_timeout{
      10000};  ///< Upper bound per exchange.
  std::chrono::milliseconds startup_timeout{
      10000};  ///< Upper bound on launch/exec.
  std::chrono::milliseconds cleanup_timeout{
      2000};  ///< Bounded termination/reap grace.
  std::chrono::milliseconds reply_quiet_period{
      2};  ///< Drain immediate extra output.
};

/** @brief Detached counters; no child diagnostics or launch secrets are
 * exposed. */
struct SidecarProcessStats {
  std::uint64_t completed_requests =
      0U;  ///< Fully framed successful exchanges.
  std::uint64_t stdout_bytes =
      0U;  ///< Total drained stdout bytes, including LF.
  std::uint64_t stderr_bytes = 0U;  ///< Total discarded diagnostic bytes.
};

/** @brief Single-owner persistent subprocess with bounded serial JSON lines.
 *
 * The caller serializes all methods and reserves the working directory. The
 * executable receives a fixed argv vector, and only three standard handles.
 * This class never invokes a shell, changes the parent's cwd/environment, or
 * exposes stderr or launch data in exceptions. It bounds request/reply memory
 * and stderr output, checks stop and the earlier of the supplied absolute
 * deadline/default operation timeout, and rejects EOF, crash, extra stdout,
 * malformed UTF-8 or incomplete frames. OfficialTelemetrySink validates JSON
 * structure and request/reply identity above this byte transport.
 *
 * Windows starts suspended, assigns a non-breakaway kill-on-close Job Object,
 * and only then resumes. Its parent pipe handles use synchronous PIPE_NOWAIT
 * polling; child standard handles retain blocking mode. No pending asynchronous
 * I/O or background thread survives cleanup. POSIX starts a distinct process
 * group; its members are
 * killed on failure/destruction. POSIX process groups contain cooperative
 * official sidecars, not a sandbox for hostile children that call setsid().
 * Any failed transact poisons and terminates the child tree. Reuse then throws
 * NotReady. Restart requires constructing a fresh transport and replaying the
 * original durable envelopes according to at-least-once delivery semantics.
 * Cleanup can extend a reached operation deadline by one cleanup_timeout;
 * Windows post-exit draining and repeated cleanup share that first budget.
 * It never performs a protocol flush or other fallible exporter finalization.
 * These are application wait bounds, not an absolute wall-clock guarantee for
 * arbitrary Win32 kernel calls or OS scheduling.
 */
class SidecarProcessTransport final
    : public experiment::TelemetrySidecarTransport {
 public:
  /** @brief Validates fixed configuration and starts the isolated child.
   * @param options Absolute launch paths, argv, child environment and quotas.
   * @param stop Cancellation observed before and during process launch.
   * @param deadline Optional absolute steady-clock launch deadline.
   * @throws Error For invalid configuration, launch, cancellation or deadline.
   */
  explicit SidecarProcessTransport(SidecarProcessOptions options,
                                   std::stop_token stop = {},
                                   TelemetryDeadline deadline = std::nullopt);
  /** @brief Terminates/reaps owned processes and releases handles without
   * throwing. */
  ~SidecarProcessTransport() noexcept override;
  /** @brief A running process and its stream ownership cannot be copied. */
  SidecarProcessTransport(const SidecarProcessTransport&) = delete;
  /** @brief A running process cannot be replaced by assignment. */
  SidecarProcessTransport& operator=(const SidecarProcessTransport&) = delete;
  /** @brief Exchanges one UTF-8 line; any failure permanently poisons this
   * object.
   * @param request_json Bounded JSON bytes without LF, CR or embedded NUL.
   * @param stop Cooperative cancellation propagated by terminating the child
   * tree.
   * @param deadline Optional absolute steady-clock operation deadline.
   * @return One complete UTF-8 response, without LF; stdout extras fail
   * explicitly.
   * @throws Error For protocol/I/O/crash/quota/cancellation/deadline failures.
   */
  std::string transact(std::string_view request_json, std::stop_token stop = {},
                       TelemetryDeadline deadline = std::nullopt) override;
  /** @brief Explicitly terminates/reaps ownership; no exporter flush is
   * implied. */
  void terminate() noexcept;
  /** @brief Reports whether termination/failure has made reuse invalid.
   * @return True after terminate or any failed exchange.
   */
  bool poisoned() const noexcept;
  /** @brief Returns detached bounded-I/O counters without child content.
   * @return A snapshot valid when called by this object's serialized owner.
   */
  SidecarProcessStats stats() const noexcept;

 private:
  /** @brief Owns platform handles, frame state and termination policy. */
  class Impl;
  std::unique_ptr<Impl> impl_;  ///< Exclusive subprocess/pipe ownership.
};

}  // namespace jyppx::liuding::cli

#endif  // JYPPX_TOOLS_CLI_SIDECAR_PROCESS_HPP
