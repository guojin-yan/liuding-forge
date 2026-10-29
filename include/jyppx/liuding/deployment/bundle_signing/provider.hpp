// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DEPLOYMENT_BUNDLE_SIGNING_PROVIDER_HPP
#define JYPPX_LIUDING_DEPLOYMENT_BUNDLE_SIGNING_PROVIDER_HPP
/** @file provider.hpp
 * @brief Native Ed25519 signing with an externally authorized public key.
 *
 * This optional component needs OpenSSL EVP 3.0 APIs. The supported exact SDK
 * version is determined by the dependency lock, not by that API minimum.
 * Public interfaces do not expose OpenSSL objects or export private keys.
 * Operations use a private library context and the explicit built-in default
 * provider. A loaded runtime whose full numeric version differs from the build
 * SDK is rejected before algorithm/key initialization. This is an identity
 * check, not a claim of dependency review or FIPS/HSM qualification.
 */

#include <jyppx/liuding/interop/model_bundle.hpp>

#include <array>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::bundle_signing {

/** @brief An Ed25519 public key in RFC 8032's 32-byte raw encoding. */
using Ed25519PublicKey = std::array<std::byte, 32U>;

/** @brief Supplies a PKCS#8 passphrase without interactive prompting.
 *
 * Return 1 to 1024 bytes. The component erases the returned allocation when
 * it has been passed to OpenSSL; the caller owns any earlier copies. Exceptions
 * are caught at the C callback boundary and reported without their text.
 */
using PassphraseCallback = std::function<std::vector<std::byte>()>;

/** @brief Returns `ed25519-sha256-` plus SHA-256 of the raw public key.
 * @param public_key Public key bytes; no bundle metadata is trusted.
 * @return Deterministic lowercase key identity, not an authorization grant.
 */
std::string ed25519_key_id(const Ed25519PublicKey& public_key);

/** @brief An immutable, externally authorized single-key trust policy.
 *
 * Parse/load always bind the exact policy bytes to an independently supplied
 * SHA-256. Possession of a policy file or its digest alone is not
 * authorization; the application must obtain that expected digest through a
 * trusted channel.
 */
class Ed25519TrustPolicy final {
 public:
  /** @brief Copies the immutable authorized policy. */
  Ed25519TrustPolicy(const Ed25519TrustPolicy&) noexcept = default;
  /** @brief Rebinds this value to another immutable authorized policy.
   * @return This policy after sharing the source's immutable state.
   */
  Ed25519TrustPolicy& operator=(const Ed25519TrustPolicy&) noexcept = default;
  /** @brief Copies shared ownership, preserving a valid source value.
   * @param other Policy whose immutable state is shared with this value.
   */
  Ed25519TrustPolicy(Ed25519TrustPolicy&& other) noexcept;
  /** @brief Copies shared ownership, preserving a valid source value.
   * @param other Policy whose immutable state is shared with this value.
   * @return This policy after sharing the source's immutable state.
   */
  Ed25519TrustPolicy& operator=(Ed25519TrustPolicy&& other) noexcept;
  /** @brief Parses at most 4096 strict JSON bytes, rejecting unknown fields.
   * @param bytes Exact original policy bytes, including any trailing newline.
   * @param expected_digest Independently authorized digest of those bytes.
   * @return Immutable key, revocation state and checked policy identity.
   */
  static Ed25519TrustPolicy parse(std::string_view bytes,
                                  const ArtifactDigest& expected_digest);
  /** @brief Reads a bounded non-linked ordinary file and binds its bytes.
   * @param path Explicit local policy path; no link in the ancestry is allowed.
   * @param expected_digest Independently authorized original-byte digest.
   * @return Immutable checked policy.
   */
  static Ed25519TrustPolicy load(const std::filesystem::path& path,
                                 const ArtifactDigest& expected_digest);
  /** @brief Returns the public key authorized by this exact policy.
   * @return Reference to the policy's raw public key.
   */
  const Ed25519PublicKey& public_key() const noexcept;
  /** @brief Returns the deterministically validated key identity.
   * @return Reference to the public key's deterministic fingerprint.
   */
  const std::string& key_id() const noexcept;
  /** @brief Returns the independently checked original-byte policy digest.
   * @return Reference to the independently checked policy digest.
   */
  const ArtifactDigest& digest() const noexcept;
  /** @brief Returns whether this policy explicitly revokes its key.
   * @return True when the authorized policy revokes the public key.
   */
  bool revoked() const noexcept;

 private:
  /** @brief Shared immutable policy data and the private EVP backend. */
  struct State;
  /** @brief Wraps an already checked immutable policy state.
   * @param state Validated policy state retained by this value.
   */
  explicit Ed25519TrustPolicy(std::shared_ptr<const State> state);
  /** @brief Shared ownership of the checked policy and its backend lifetime. */
  std::shared_ptr<const State> state_;
};

/** @brief Encodes one public policy; the output must be authorized externally.
 * @param public_key The key to authorize through a separate trust channel.
 * @param revoked Explicit revocation state.
 * @return Strict version-1 JSON; contains public data only.
 */
std::string serialize_ed25519_trust_policy(const Ed25519PublicKey& public_key,
                                           bool revoked = false);

/** @brief A move-only EVP Ed25519 signer; no private-key extraction API. */
class Ed25519Signer final {
 public:
  /** @brief Generates an in-memory key, suitable for ephemeral examples.
   * @return A signer owning the new private key.
   * @note Generation does not provide persistence or production key governance.
   */
  static Ed25519Signer generate();
  /** @brief Imports one PKCS#8 Ed25519 private key, at most 64 KiB PEM.
   * @param pem Exact unencrypted/encrypted PKCS#8 PEM, with whitespace tail
   * only.
   * @param passphrase Callback for encrypted PEM; no implicit prompt is
   * allowed.
   * @return A signer owning the parsed key. Caller retains its input bytes.
   */
  static Ed25519Signer from_pkcs8_pem(std::span<const std::byte> pem,
                                      PassphraseCallback passphrase = {});
  /** @brief Imports an explicit bounded, non-linked ordinary PKCS#8 PEM file.
   * @param path Private local file path; never included in diagnostics.
   * @param passphrase Callback for encrypted PEM, never command-line text.
   * @return Parsed signer; the temporary file buffer is securely erased.
   */
  static Ed25519Signer from_pkcs8_file(const std::filesystem::path& path,
                                       PassphraseCallback passphrase = {});
  /** @brief Releases and erases the EVP-owned private key. */
  ~Ed25519Signer();
  Ed25519Signer(const Ed25519Signer&) = delete;
  Ed25519Signer& operator=(const Ed25519Signer&) = delete;
  /** @brief Transfers key ownership; other operations on the source fail. */
  Ed25519Signer(Ed25519Signer&&) noexcept;
  /** @brief Releases the old key and transfers key ownership.
   * @return This signer after taking ownership of the source's private key.
   */
  Ed25519Signer& operator=(Ed25519Signer&&) noexcept;
  /** @brief Returns public-only raw key bytes; rejects a moved-from signer.
   * @return The signer's raw 32-byte Ed25519 public key.
   */
  Ed25519PublicKey public_key() const;
  /** @brief Returns the deterministic public-key fingerprint identity.
   * @return Deterministic lowercase fingerprint of the signer's public key.
   */
  std::string key_id() const;
  /** @brief Signs complete bytes using Ed25519 one-shot; no prehash.
   * @param payload Complete message, at most 64 MiB plus 128 prefix bytes;
   * empty messages are valid RFC 8032 inputs.
   * @return Exact `ed25519` metadata and 128 lowercase signature hex digits.
   * @note This low-level primitive does not authorize a key or create a bundle.
   */
  interop::BundleSignature sign_payload(
      std::span<const std::byte> payload) const;
  /** @brief Signs the existing versioned bundle canonical payload unchanged.
   * @param manifest Validated bundle metadata; signatures are excluded by wire.
   * @param limits Existing bundle resource limits.
   * @return Detached signature over the complete domain-separated payload.
   */
  interop::BundleSignature sign_manifest(
      const interop::ModelBundleManifest& manifest,
      interop::ModelBundleLimits limits = {}) const;

 private:
  /** @brief Owned private key and its isolated EVP backend. */
  struct State;
  /** @brief Takes ownership of an initialized private key state.
   * @param state Valid private key state exclusively owned by this signer.
   */
  explicit Ed25519Signer(std::unique_ptr<State> state);
  /** @brief Exclusive private key ownership; empty after a move. */
  std::unique_ptr<State> state_;
};

/** @brief Verifies raw Ed25519 bytes, independently of trust authorization.
 * @param public_key Raw public key, from an independently trusted source.
 * @param payload Complete message, without digest substitution.
 * @param signature Exact 64-byte RFC 8032 signature; other sizes return false.
 * @return Cryptographic validity only; backend unavailability throws an Error.
 */
bool verify_ed25519_signature(const Ed25519PublicKey& public_key,
                              std::span<const std::byte> payload,
                              std::span<const std::byte> signature);

/** @brief Adapts one immutable authorized key to the existing all-of verifier.
 * @param policy Independently bound single-key policy; revoked keys reject all.
 * @param max_payload_bytes Positive bound, at most 64 MiB plus domain prefix.
 * @return Callback rejecting every other algorithm, key ID or signature shape.
 */
interop::BundleSignatureVerifier make_ed25519_verifier(
    Ed25519TrustPolicy policy,
    std::size_t max_payload_bytes = (64U << 20U) + 128U);

/** @brief Requires at least one signature and every declaration from this key.
 * @param policy Externally authorized policy, never extracted from the bundle.
 * @param limits Existing independent integrity/resource limits.
 * @return Options reusable by bundle readers and native/ORT Serving loaders.
 */
interop::ModelBundleVerifyOptions trusted_ed25519_options(
    Ed25519TrustPolicy policy, interop::ModelBundleLimits limits = {});

/** @brief Public evidence of a newly signed immutable bundle directory. */
struct SignedBundlePublication {
  ArtifactDigest source_manifest_digest;  ///< Exact checked source manifest.
  ArtifactDigest trust_policy_digest;     ///< Externally bound policy bytes.
  std::string key_id;                     ///< Public signer fingerprint.
  interop::ModelBundleVerification published;  ///< Final trusted snapshot.
};

/** @brief Fully checks a source, signs, then atomically publishes a new bundle.
 * @param source Existing directory; the full inventory is integrity checked.
 * @param expected_manifest Independently expected original manifest SHA-256.
 * @param destination New directory, outside source, with an existing parent.
 * @param signer Signer matching the one non-revoked authorized public key.
 * @param policy Independently bound authorization used before and after
 * signing.
 * @param limits Existing bundle limits, including manifest/signature bounds.
 * @return Source/policy identities and the final verified new bundle.
 *
 * Existing source signatures must all be valid under the same single-key
 * policy. An existing valid signature from that key is preserved. Source files
 * are streamed through the existing no-follow publisher, never loaded again
 * through a weaker path. Existing destinations are never replaced. Precommit
 * failure attempts staging cleanup; a post-rename durability error may leave
 * the complete signed directory. The source is never overwritten.
 */
SignedBundlePublication sign_model_bundle_directory(
    const std::filesystem::path& source,
    const ArtifactDigest& expected_manifest,
    const std::filesystem::path& destination, const Ed25519Signer& signer,
    Ed25519TrustPolicy policy, interop::ModelBundleLimits limits = {});

}  // namespace jyppx::liuding::bundle_signing

#endif  // JYPPX_LIUDING_DEPLOYMENT_BUNDLE_SIGNING_PROVIDER_HPP
