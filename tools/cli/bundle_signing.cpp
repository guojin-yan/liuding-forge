// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "bundle_signing.hpp"

#if defined(JYPPX_LIUDING_CLI_BUNDLE_SIGNING) && \
    JYPPX_LIUDING_CLI_BUNDLE_SIGNING
#include <jyppx/liuding/deployment/bundle_signing/provider.hpp>
#endif

#include <utility>

namespace jyppx::liuding::cli {
namespace {

#if !defined(JYPPX_LIUDING_CLI_BUNDLE_SIGNING) || \
    !JYPPX_LIUDING_CLI_BUNDLE_SIGNING
[[noreturn]] void signing_disabled() {
  throw NotImplemented(
      "Native model-bundle signing is disabled in this CLI",
      "Use a CLI explicitly built with the bundle_signing component; no "
      "cryptographic backend is enabled automatically.");
}
#endif

}  // namespace

BundleTrust bundle_trust_options(const Arguments& arguments,
                                 interop::ModelBundleLimits limits) {
  const auto path = arguments.optional("trust-policy");
  const auto digest = arguments.optional("trust-policy-sha256");
  const auto required = arguments.optional("require-trusted");
  if (required && *required != "true" && *required != "false")
    usage("model.bundle.trust", "--require-trusted accepts true or false.");
  if (path.has_value() != digest.has_value())
    usage("model.bundle.trust",
          "Supply both --trust-policy and its independently authorized "
          "--trust-policy-sha256.");
  if (!path) {
    if (required == "true")
      throw CompatibilityError(
          "Trusted bundle verification requires an external trust policy",
          "Supply --trust-policy and --trust-policy-sha256 from a trusted "
          "channel.");
    BundleTrust result;
    result.options.limits = limits;
    return result;
  }
#if defined(JYPPX_LIUDING_CLI_BUNDLE_SIGNING) && \
    JYPPX_LIUDING_CLI_BUNDLE_SIGNING
  const ArtifactDigest expected{.hex = *digest};
  validate_artifact_digest(expected);
  const auto policy = bundle_signing::Ed25519TrustPolicy::load(
      std::filesystem::u8path(*path), expected);
  auto options = bundle_signing::trusted_ed25519_options(policy, limits);
  // Explicit false only admits unsigned bundles. The installed callback still
  // checks every declared signature, including unknown algorithms and keys.
  options.require_trusted_signature = required != "false";
  return {.options = std::move(options),
          .key_id = policy.key_id(),
          .policy_sha256 = policy.digest().hex};
#else
  signing_disabled();
#endif
}

Json bundle_trust_document(const BundleTrust& trust) {
  if (trust.key_id.empty()) return nullptr;
  return {
      {"algorithm", "ed25519"},
      {"key_id", trust.key_id},
      {"policy_sha256", trust.policy_sha256},
      {"require_trusted_signature", trust.options.require_trusted_signature},
      {"signature_rule", "all_declared_signatures_must_verify"}};
}

Json bundle_signing_capabilities() {
#if defined(JYPPX_LIUDING_CLI_BUNDLE_SIGNING) && \
    JYPPX_LIUDING_CLI_BUNDLE_SIGNING
  constexpr bool enabled = true;
#else
  constexpr bool enabled = false;
#endif
  return {{"enabled", enabled},
          {"algorithm", enabled ? Json("ed25519") : Json(nullptr)},
          {"private_key_input", "unencrypted_pkcs8_file_only"},
          {"encrypted_private_key_cli", false},
          {"interactive_prompt", false},
          {"trust_policy", "external_sha256_bound_single_public_key"},
          {"signature_rule", "all_declared_signatures_must_verify"}};
}

Result bundle_sign_command(const Arguments& arguments) {
  arguments.allow({"source", "manifest-sha256", "output", "private-key-file",
                   "trust-policy", "trust-policy-sha256"});
#if defined(JYPPX_LIUDING_CLI_BUNDLE_SIGNING) && \
    JYPPX_LIUDING_CLI_BUNDLE_SIGNING
  const auto source = std::filesystem::u8path(arguments.require("source"));
  const ArtifactDigest manifest{.hex = arguments.require("manifest-sha256")};
  validate_artifact_digest(manifest);
  const auto output = std::filesystem::u8path(arguments.require("output"));
  const auto key =
      std::filesystem::u8path(arguments.require("private-key-file"));
  const ArtifactDigest expected{.hex =
                                    arguments.require("trust-policy-sha256")};
  validate_artifact_digest(expected);
  const auto policy = bundle_signing::Ed25519TrustPolicy::load(
      std::filesystem::u8path(arguments.require("trust-policy")), expected);
  // No callback means encrypted PKCS#8 is rejected before PEM decryption.
  // The provider reads/erases the bounded key buffer; it never prompts or
  // renders private bytes, filenames, passphrases or OpenSSL error stacks.
  const auto signer = bundle_signing::Ed25519Signer::from_pkcs8_file(key);
  const auto publication = bundle_signing::sign_model_bundle_directory(
      source, manifest, output, signer, policy);
  return {
      Json{
          {"operation", "model.bundle.sign"},
          {"published", true},
          {"source_manifest_sha256", publication.source_manifest_digest.hex},
          {"manifest_sha256", publication.published.manifest_digest.hex},
          {"trust_policy_sha256", publication.trust_policy_digest.hex},
          {"key_id", publication.key_id},
          {"algorithm", "ed25519"},
          {"signature_status", "trusted"},
          {"authenticated", true},
          {"signature_count", publication.published.manifest.signatures.size()},
          {"runtime_executed", false}},
      0};
#else
  signing_disabled();
#endif
}

}  // namespace jyppx::liuding::cli
