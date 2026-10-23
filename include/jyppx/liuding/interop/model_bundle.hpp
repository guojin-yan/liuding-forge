// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_INTEROP_MODEL_BUNDLE_HPP
#define JYPPX_LIUDING_INTEROP_MODEL_BUNDLE_HPP
/** @file model_bundle.hpp
 * @brief Defines immutable, bounded, data-only model bundle directories.
 *
 * Only foundation and the tensor-neutral safetensors codec are required. Schema
 * owners supply independently versioned descriptor artifacts; no core, nn,
 * training, experiment, Python, or model runtime is imported or executed.
 */

#include <jyppx/liuding/foundation/artifact.hpp>
#include <jyppx/liuding/foundation/version.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::interop {

/** @brief The one reserved, case-sensitive manifest filename. */
inline constexpr std::string_view kModelBundleManifest = "manifest.json";

/** @brief Closed data roles; executable/plugin roles are deliberately absent.
 */
enum class BundleFileRole : std::uint8_t {
  weights = 1,               ///< One complete safetensors shard.
  model = 2,                 ///< One ONNX model; official checking is external.
  schema = 3,                ///< Owner-versioned input/output data schema.
  prepost = 4,               ///< Declarative preprocessing/postprocessing data.
  labels = 5,                ///< Optional labels; absence is explicit.
  evaluation = 6,            ///< Evaluation evidence, never an implied pass.
  sbom = 7,                  ///< Software bill of materials data.
  provenance = 8,            ///< Code/config/dataset/run lineage data.
  weight_import_report = 9,  ///< Explicit full/partial weight-import evidence.
};

/** @brief Formats admitted by the data-only v1 envelope. */
enum class BundleModelFormat : std::uint8_t {
  safetensors =
      1,     ///< Structurally verified including every tensor association.
  onnx = 2,  ///< Integrity only; no protobuf/runtime verification claim.
};

/** @brief One uniquely owned local file and its foundation content identity. */
struct BundleFile {
  std::string id;             ///< Unique lowercase machine token.
  std::string relative_path;  ///< Portable lowercase path with '/' separators.
  BundleFileRole role = BundleFileRole::schema;  ///< Closed data role.
  ArtifactRef artifact;  ///< Exact complete-file digest, size, and provenance.
};

/** @brief An owner-versioned descriptor, resolved through the file inventory.
 */
struct BundleDescriptorRef {
  std::string schema_id;              ///< Stable schema owner/type identity.
  std::uint32_t schema_version = 1U;  ///< Positive owner codec version.
  std::string file_id;                ///< Exactly one matching inventory entry.
};

/** @brief Complete tensor association; names are local to a model variant. */
struct BundleTensor {
  std::string name;                             ///< Unique logical tensor name.
  SafeTensorDtype dtype = SafeTensorDtype::u8;  ///< Exact on-disk dtype.
  std::vector<std::uint64_t> shape;             ///< Exact C-order dimensions.
  std::string device_intent = "cpu";  ///< cpu/cuda/any; never allocation.
  std::string shard_file_id;          ///< One model-owned weights file.
};

/** @brief An explicit model variant and all its data files. */
struct BundleModel {
  std::string id;  ///< Unique model variant identity.
  BundleModelFormat format = BundleModelFormat::safetensors;  ///< Explicit.
  std::vector<std::string>
      file_ids;  ///< Complete, uniquely owned shards/files.
  std::vector<BundleTensor> tensors;  ///< Complete safetensors inventory only.
};

/** @brief A pinned package requirement, never a wildcard/version range. */
struct BundleDependency {
  std::string name;     ///< Stable package/runtime identity.
  std::string version;  ///< Exact required version.
};

/** @brief Declared requirements; directory verification does not probe them. */
struct BundleRuntimeRequirements {
  std::string backend_id;           ///< Explicit runtime adapter identity.
  std::string backend_version;      ///< Exact adapter version.
  ArtifactDigest environment_lock;  ///< Identity of the pinned environment.
  std::vector<BundleDependency> dependencies;  ///< Exact package versions.
  std::vector<std::string> devices;            ///< Explicit cpu/cuda families.
  std::vector<std::string> dtypes;   ///< Explicit scalar encoding tokens.
  std::vector<std::string> os_arch;  ///< Explicit supported OS/architectures.
  bool dynamic_shapes = false;       ///< Declaration, not execution evidence.
};

/** @brief Detached signature metadata; key material is supplied by the caller.
 */
struct BundleSignature {
  std::string algorithm;      ///< Registered algorithm understood by verifier.
  std::string key_id;         ///< Key identity resolved by a trusted callback.
  std::string signature_hex;  ///< Nonempty lowercase hex, bounded before use.
};

/** @brief Strict schema-v1 metadata; every inventory file has exactly one
 * owner.
 *
 * Compression and encryption are explicitly none in v1. Future versions or
 * unknown fields fail; no undocumented historical schema is accepted. Identity
 * pre/postprocessing still has an explicit declarative descriptor artifact.
 */
struct ModelBundleManifest {
  static constexpr std::uint32_t format_version = 1U;  ///< Current wire schema.
  std::uint32_t schema_version = format_version;       ///< Exact wire version.
  std::string bundle_id;                               ///< Producer identity.
  BuildFingerprint build;                              ///< Producer build.
  std::string compression = "none";                    ///< No decompression.
  std::string encryption = "none";                     ///< No custom crypto.
  std::vector<BundleFile> files;              ///< Exhaustive inventory.
  std::vector<BundleModel> models;            ///< At least one variant.
  BundleDescriptorRef input_schema;           ///< Required owner schema.
  BundleDescriptorRef output_schema;          ///< Required owner schema.
  BundleDescriptorRef preprocessing;          ///< Declarative data.
  BundleDescriptorRef postprocessing;         ///< Declarative data.
  std::optional<BundleDescriptorRef> labels;  ///< null for no labels.
  BundleDescriptorRef evaluation;             ///< Evidence data.
  BundleDescriptorRef sbom;                   ///< Required SBOM data.
  BundleDescriptorRef provenance;             ///< Required lineage data.
  std::vector<BundleDescriptorRef>
      weight_import_reports;                ///< Optional evidence.
  BundleRuntimeRequirements runtime;        ///< Required metadata.
  std::vector<BundleSignature> signatures;  ///< May be empty.
};

/** @brief Independent positive limits applied before allocations and reads.
 *
 * One safetensors shard is buffered for structure verification. Other files are
 * streamed. Bounds constrain encoded bytes/counts, not total process RSS.
 */
struct ModelBundleLimits {
  std::size_t max_manifest_bytes = 16U * 1024U * 1024U;   ///< Hard cap 64 MiB.
  std::size_t max_files = 1024U;                          ///< Inventory bound.
  std::size_t max_models = 32U;                           ///< Variant bound.
  std::size_t max_tensors = 65536U;                       ///< All variants.
  std::size_t max_rank = 32U;                             ///< Tensor rank.
  std::uint64_t max_dimension = 1ULL << 32U;              ///< One dimension.
  std::uint64_t max_tensor_elements = 1ULL << 32U;        ///< Scalar count 1.
  std::uint64_t max_file_bytes = 512ULL * 1024U * 1024U;  ///< One file/shard.
  std::uint64_t max_total_bytes = 4ULL * 1024U * 1024U * 1024U;  ///< All files.
  std::size_t max_string_bytes = 4096U;     ///< One decoded text.
  std::size_t max_path_bytes = 1024U;       ///< Relative path.
  std::size_t max_signatures = 16U;         ///< Detached signatures.
  std::size_t max_signature_bytes = 2048U;  ///< Decoded signature.
};

/** @brief Authentication state independent of successful digest verification.
 */
enum class BundleSignatureStatus : std::uint8_t {
  unsigned_bundle = 1,     ///< No signatures and no authentication claim.
  present_unverified = 2,  ///< Metadata present; no verifier was supplied.
  trusted = 3,             ///< Every signature accepted by trusted callback.
};

/** @brief Verifies actual canonical signing bytes using externally trusted
 * keys.
 *
 * Return true only after cryptographic verification AND trust-policy checks.
 * The framework supplies no fake/default signature algorithm or key lookup.
 */
using BundleSignatureVerifier = std::function<bool(
    const BundleSignature&, std::span<const std::byte> signing_payload)>;

/** @brief Local integrity/authentication policy; no code execution is enabled.
 */
struct ModelBundleVerifyOptions {
  ModelBundleLimits limits;                    ///< Bounded resources.
  bool require_trusted_signature = false;      ///< Reject unsigned.
  BundleSignatureVerifier signature_verifier;  ///< Caller trust boundary.
};

/** @brief Verified directory metadata, without runtime compatibility claims. */
struct ModelBundleVerification {
  ModelBundleManifest manifest;    ///< Validated metadata snapshot.
  ArtifactDigest manifest_digest;  ///< Digest of original manifest bytes.
  std::uint64_t verified_file_bytes = 0U;  ///< Exact aggregate data-file bytes.
  std::size_t verified_tensors = 0U;  ///< Complete safetensors tensor count.
  std::size_t integrity_only_models =
      0U;  ///< ONNX models without parser checks.
  BundleSignatureStatus signature_status =
      BundleSignatureStatus::unsigned_bundle;  ///< Explicit trust result.
};

/** @brief One source for publication; caller retains immutable source content.
 */
struct BundlePublishSource {
  std::string file_id;                ///< Exactly one manifest file identity.
  std::filesystem::path source_path;  ///< Local file; no links in its ancestry.
};

/** @brief An encoded shard and its complete manifest associations. */
struct EncodedBundleShard {
  BundleFile file;                    ///< Digest/size/portable path populated.
  std::vector<BundleTensor> tensors;  ///< Complete tensor-to-shard mapping.
  std::vector<std::byte> bytes;       ///< Complete official safetensors file.
};

/** @brief Validates strict v1 metadata and all cross references, without I/O.
 * @param manifest Metadata and tensor/shard references to validate.
 * @param limits Resource limits applied before expansion.
 * @throws InvalidArgument For invalid metadata; CompatibilityError for
 * versions.
 * @throws ResourceExhausted When any configured resource limit is exceeded.
 */
void validate_model_bundle_manifest(const ModelBundleManifest& manifest,
                                    ModelBundleLimits limits = {});

/** @brief Returns deterministic UTF-8 JSON, sorted by stable identities.
 * @param manifest Validated model bundle metadata.
 * @param limits Serialization resource limits.
 * @return Canonical UTF-8 manifest JSON.
 */
std::string serialize_model_bundle_manifest(const ModelBundleManifest& manifest,
                                            ModelBundleLimits limits = {});

/** @brief Rejects duplicate/unknown fields, malformed UTF-8 and unsupported v1.
 * @param json Canonical model bundle manifest JSON.
 * @param limits Parsing resource limits.
 * @return Parsed and validated manifest.
 * @throws SerializationError For malformed encoded data or cross references.
 */
ModelBundleManifest deserialize_model_bundle_manifest(
    std::string_view json, ModelBundleLimits limits = {});

/** @brief Returns domain prefix + canonical JSON with signatures omitted.
 * @param manifest Manifest to encode.
 * @param limits Serialization resource limits.
 * @return Signature payload bytes represented as UTF-8 text.
 *
 * Prefix is exactly "JYPPX-MODEL-BUNDLE-SIGNATURE-V1\n". The complete payload,
 * not just a user-declared digest, is passed to every signature trust callback.
 */
std::string model_bundle_signing_payload(const ModelBundleManifest& manifest,
                                         ModelBundleLimits limits = {});

/** @brief Checks exhaustive local files, hashes, sizes and every tensor
 * mapping.
 * @param directory Bundle directory to verify.
 * @param options Verification limits and policy.
 * @return Verified snapshot and file/tensor evidence.
 *
 * Rejects undeclared files/directories, symlinks/reparse points, hard-linked
 * files and unsafe/aliased paths. Native handles pin traversal and file reads.
 * The result describes the verification snapshot, not future path reads.
 */
ModelBundleVerification verify_model_bundle_directory(
    const std::filesystem::path& directory,
    const ModelBundleVerifyOptions& options = {});

/** @brief Stages, hashes, validates, flushes and atomically publishes a new
 * dir.
 * @param destination New bundle destination.
 * @param manifest Manifest describing the published files.
 * @param sources Source files associated with manifest identities.
 * @param options Verification and publication limits.
 * @return Published bundle verification snapshot.
 *
 * Windows and Linux support atomic no-replace publication. The destination's
 * parent must already exist. Existing/racing destinations are never replaced.
 * A failed precommit leaves any previous bundle intact and attempts cleanup of
 * private staging. A post-rename durability error may leave the complete
 * bundle.
 */
ModelBundleVerification publish_model_bundle(
    const std::filesystem::path& destination,
    const ModelBundleManifest& manifest,
    std::span<const BundlePublishSource> sources,
    const ModelBundleVerifyOptions& options = {});

/** @brief Deterministically packs whole tensors into bounded safetensors
 * shards.
 * @param tensors Tensor byte sources in stable order.
 * @param max_shard_bytes Maximum encoded shard size.
 * @param name_prefix Prefix for generated shard names.
 * @param device_intent Declared device intent in each tensor descriptor.
 * @param limits Header and metadata resource limits.
 * @return Encoded shards with complete manifest associations.
 *
 * No tensor is split. A conservative exact-JSON-header upper bound selects
 * groups; every encoded shard must fit max_shard_bytes, including its header.
 * Caller bytes must remain alive/unchanged. No filesystem I/O occurs.
 */
std::vector<EncodedBundleShard> shard_safetensors_for_bundle(
    std::span<const SafeTensorSource> tensors, std::uint64_t max_shard_bytes,
    std::string_view name_prefix = "weights",
    std::string_view device_intent = "cpu", ModelBundleLimits limits = {});

}  // namespace jyppx::liuding::interop

#endif  // JYPPX_LIUDING_INTEROP_MODEL_BUNDLE_HPP