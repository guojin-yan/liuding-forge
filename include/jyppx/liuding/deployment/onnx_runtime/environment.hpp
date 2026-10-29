// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#ifndef JYPPX_LIUDING_DEPLOYMENT_ONNX_RUNTIME_ENVIRONMENT_HPP
#define JYPPX_LIUDING_DEPLOYMENT_ONNX_RUNTIME_ENVIRONMENT_HPP
/** @file environment.hpp @brief Bounded environment locks shared by deployment
 * and serving, without trusting producer declarations as execution evidence. */

#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::onnx_runtime {
/** @brief Existing version-one environment-lock wire schema identity. */
inline constexpr std::string_view kEnvironmentLockSchemaId =
    "jyppx.liuding.onnx_runtime_lock";
/** @brief Explicit adapter identity used in ONNX bundle runtime requirements.
 */
inline constexpr std::string_view kEnvironmentBackendId =
    "jyppx.liuding.onnx_runtime";
/** @brief Maximum complete UTF-8 lock payload, checked before JSON parsing. */
inline constexpr std::size_t kEnvironmentLockMaxBytes = 64U << 10U;

/** @brief One exact package version, never a wildcard or resolver request. */
struct EnvironmentDependency {
  std::string name;     ///< Unique nonempty dependency identity.
  std::string version;  ///< Exact nonempty package version.
};

/** @brief Runtime declaration projected independently of a bundle API.
 * @note Dependency order is insignificant; all entries and other fields must
 * match. Strings/lists are bounded and duplicate identities are rejected.
 */
struct EnvironmentRequirements {
  std::string backend_id;       ///< Exact ONNX Runtime adapter identity.
  std::string backend_version;  ///< Exact linked Liuding Forge version.
  std::vector<EnvironmentDependency>
      dependencies;  ///< Exact LibTorch, Liuding Forge and loaded ORT versions.
  std::vector<std::string>
      devices;                      ///< Provider execution domain declarations.
  std::vector<std::string> dtypes;  ///< Exact boundary dtype declarations.
  std::vector<std::string>
      os_arch;                  ///< Exact normalized build OS/architecture.
  bool dynamic_shapes = false;  ///< Whether any graph I/O uses dynamic batch.
};

/** @brief Owned environment-lock fields, without model or producer trust.
 * @note Provider identity/options record an explicit session request. They do
 * not prove actual kernel selection, driver qualification or allocation bounds.
 * Tensor I/O remains in host memory independently of the provider domain.
 */
struct EnvironmentLock {
  EnvironmentRequirements runtime;  ///< Complete versioned runtime declaration.
  ExecutionProvider provider;       ///< Exact requested provider and options.
  bool allow_cpu_fallback = false;  ///< Explicit partition-fallback policy.
};

/** @brief Observes the real loaded session and this library's build identity.
 * @param session Live session with validated graph signatures and provider.
 * @return Environment declaration using the loaded ORT version, foundation
 * build fingerprint, actual graph dynamic-batch metadata and retained options.
 * @throws Error If build observation fails or the metadata exceeds its bounds.
 * @note No caller-supplied build fingerprint can substitute for observation.
 * No package resolution, network, code loading or hardware benchmark occurs.
 */
EnvironmentLock observe_environment(const Session& session);

/** @brief Parses the existing strict version-one lock without running a model.
 * @param encoded Complete stable UTF-8 JSON payload, at most 64 KiB.
 * @return Owned bounded fields, including all exact provider options.
 * @throws SerializationError For unknown/duplicate/missing fields, incorrect
 * JSON types, unsupported schema/version, duplicate declarations or bad text.
 * @throws ResourceExhausted For byte, depth, node or collection bounds.
 * @note Parsing and a SHA-256 digest establish no producer authentication.
 */
EnvironmentLock parse_environment_lock(std::string_view encoded);

/** @brief Serializes a bounded lock using the existing version-one wire schema.
 * @param environment Complete immutable declaration to encode.
 * @return Deterministic compact UTF-8 JSON; dependency entries sort by name.
 * @throws InvalidArgument For invalid declaration fields or duplicate entries.
 * @throws ResourceExhausted For lock size or collection bounds.
 * @note Serialization does not certify that supplied fields were observed.
 */
std::string serialize_environment_lock(const EnvironmentLock& environment);

/** @brief Verifies exact bytes, the bundle declaration and real session state.
 * @param session Live session to observe independently of the lock payload.
 * @param encoded Complete stable version-one environment-lock bytes.
 * @param expected Independently supplied expected identity of those exact
 * bytes.
 * @param declared Runtime requirements projected from the verified bundle.
 * @return Parsed lock only after all complete comparisons succeed.
 * @throws CompatibilityError For digest, observed environment, dependency,
 * provider/options/fallback, platform, dtype or declared-runtime mismatch.
 * @throws InvalidArgument For a malformed expected digest or declaration.
 * @throws SerializationError For an invalid lock schema or payload.
 * @throws ResourceExhausted For metadata/resource bounds.
 * @throws Error If actual build observation fails.
 * @note Does not execute reference vectors, authenticate source/model owners,
 * verify a signature or qualify hardware. Keep those independent checks.
 */
EnvironmentLock verify_environment_lock(
    const Session& session, std::string_view encoded,
    const ArtifactDigest& expected, const EnvironmentRequirements& declared);
}  // namespace jyppx::liuding::onnx_runtime
#endif
