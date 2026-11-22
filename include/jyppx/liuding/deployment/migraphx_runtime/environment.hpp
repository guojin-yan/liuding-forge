// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#ifndef JYPPX_LIUDING_DEPLOYMENT_MIGRAPHX_RUNTIME_ENVIRONMENT_HPP
#define JYPPX_LIUDING_DEPLOYMENT_MIGRAPHX_RUNTIME_ENVIRONMENT_HPP
/** @file environment.hpp @brief Bounded identities for one native GPU session.
 */
#include <jyppx/liuding/deployment/migraphx_runtime/session.hpp>

#include <string_view>

namespace jyppx::liuding::migraphx_runtime {
/** @brief Wire identity, separate from ORT environment locks. */
inline constexpr std::string_view kEnvironmentLockSchemaId =
    "jyppx.liuding.migraphx_runtime_lock";
/** @brief Complete encoded metadata ceiling, checked before JSON parsing. */
inline constexpr std::size_t kEnvironmentLockMaxBytes = 16U << 10U;
/** @brief Maximum size of each observed library backing file. */
inline constexpr std::uint64_t kEnvironmentModuleMaxBytes = 512ULL << 20U;

/** @brief SHA-256 and size of a loaded API module's backing file.
 * Paths are deliberately absent from the portable lock. This is a file-content
 * observation, not a measurement of mapped memory or a transitive SDK closure.
 */
struct EnvironmentModule {
  std::string sha256;  ///< Complete canonical lowercase file SHA-256.
  std::uint64_t size_bytes =
      0;  ///< Positive encoded file size, at most 512 MiB.
};
/** @brief Owned exact build, SDK, device and fixed specialization metadata.
 * Header versions describe compilation; MIGraphX 2.18.0 has no runtime version
 * query in its public C API. Loaded C API/HIP backing-file identities therefore
 * remain separate. No model/producer authentication or numerical qualification
 * follows from these declarations.
 */
struct EnvironmentLock {
  std::string adapter_version;  ///< Actual foundation build version.
  std::string
      libtorch_version;  ///< Foundation's configured dependency version.
  std::string os_arch;  ///< Normalized build OS/architecture, not caller input.
  std::string migraphx_header_version;  ///< Compile-time MIGraphX headers.
  std::string hip_header_version;       ///< Compile-time HIP headers.
  EnvironmentModule migraphx_c;   ///< Module exporting migraphx_program_create.
  EnvironmentModule hip_runtime;  ///< Module exporting hipRuntimeGetVersion.
  std::int32_t hip_runtime_version = 0;  ///< Actual HIP runtime API value.
  std::int32_t hip_driver_version = 0;   ///< Actual HIP driver API value.
  std::int32_t device_id = 0;  ///< Session's explicit visible HIP ordinal.
  std::string
      device_architecture;  ///< Actual gcnArchName including qualifiers.
  std::string device_uuid;  ///< Actual nonzero 16-byte UUID as 32 hex digits.
  std::string device_pci_bus_id;          ///< Actual HIP PCI bus identity.
  std::string input_name;                 ///< Exact single graph input name.
  std::vector<std::size_t> input_shape;   ///< Fixed complete dense FP32 input.
  std::vector<std::size_t> output_shape;  ///< Fixed complete dense FP32 output.
};
/** @brief Observes the selected session, actual HIP metadata and two modules.
 * @param session Live immutable specialization; destruction must not race.
 * @return Owning metadata without absolute machine paths.
 * @throws DeviceError For unavailable device/identity/version queries.
 * @throws CompatibilityError For an unresolvable or changing backing file.
 * @throws ResourceExhausted For metadata, file size or host allocation bounds.
 * @note Does not change the thread's selected device, compile/run a graph,
 * resolve packages, load extra libraries or access the network. Backing files
 * must remain local, readable and unchanged during observation. Other loaded
 * GPU/math libraries, code objects, driver binaries and environment variables
 * are outside this lock; qualify them independently where required.
 */
EnvironmentLock observe_environment(const Session& session);
/** @brief Parses strict version-one JSON without device or SDK queries.
 * @param encoded Complete immutable UTF-8 payload, at most 16 KiB.
 * @return Owned validated metadata; fields are declarations until verified.
 * @throws SerializationError For duplicate/unknown/missing fields, bad types,
 * invalid declarations or unsupported schema/version/program options.
 * @throws ResourceExhausted For byte, depth, node or metadata bounds.
 */
EnvironmentLock parse_environment_lock(std::string_view encoded);
/** @brief Deterministically encodes a declaration, without certifying its
 * origin.
 * @param environment Complete positive bounded metadata.
 * @return Compact version-one JSON including fixed GPU/FP32 compile options.
 * @throws InvalidArgument For invalid fields, shape, digest or UUID.
 * @throws ResourceExhausted For encoded or metadata bounds/allocation failure.
 */
std::string serialize_environment_lock(const EnvironmentLock& environment);
/** @brief Checks exact lock bytes against an independently observed session.
 * @param session Live session to query, never replaced by producer metadata.
 * @param encoded Complete stable JSON bytes.
 * @param expected Independently trusted identity of the complete lock bytes.
 * @return Parsed lock only after every observed field matches exactly.
 * @throws CompatibilityError For digest or any environment/specialization
 * drift.
 * @throws InvalidArgument For a malformed expected digest.
 * @throws Error For parsing, resource or actual observation failure.
 * @note Checks the digest/schema before device queries. Does not authenticate
 * a producer, verify bundle declarations/signatures or run known vectors.
 */
EnvironmentLock verify_environment_lock(const Session& session,
                                        std::string_view encoded,
                                        const ArtifactDigest& expected);
}  // namespace jyppx::liuding::migraphx_runtime
#endif  // JYPPX_LIUDING_DEPLOYMENT_MIGRAPHX_RUNTIME_ENVIRONMENT_HPP
