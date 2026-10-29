// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/deployment/bundle_signing/provider.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
namespace signing = jyppx::liuding::bundle_signing;

void Require(bool accepted, std::string_view operation) {
  if (!accepted) throw std::runtime_error(std::string(operation));
}

std::span<const std::byte> Bytes(std::string_view value) {
  return {reinterpret_cast<const std::byte*>(value.data()), value.size()};
}

jyppx::liuding::ArtifactDigest Digest(std::string_view bytes) {
  return {.hex = jyppx::liuding::sha256_hex(Bytes(bytes))};
}

std::vector<std::byte> Decode(std::string_view hex) {
  Require(hex.size() % 2U == 0U, "fixed fixture hex length");
  const auto digit = [](char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    throw std::runtime_error("fixed fixture hex digit");
  };
  std::vector<std::byte> bytes(hex.size() / 2U);
  for (std::size_t index = 0U; index < bytes.size(); ++index)
    bytes[index] = static_cast<std::byte>(16 * digit(hex[index * 2U]) +
                                          digit(hex[index * 2U + 1U]));
  return bytes;
}

std::string Base64(std::span<const std::byte> bytes) {
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string encoded;
  for (std::size_t offset = 0U; offset < bytes.size(); offset += 3U) {
    const auto count = std::min<std::size_t>(3U, bytes.size() - offset);
    const auto a = std::to_integer<unsigned int>(bytes[offset]);
    const auto b =
        count > 1U ? std::to_integer<unsigned int>(bytes[offset + 1U]) : 0U;
    const auto c =
        count > 2U ? std::to_integer<unsigned int>(bytes[offset + 2U]) : 0U;
    encoded += alphabet[a >> 2U];
    encoded += alphabet[((a & 3U) << 4U) | (b >> 4U)];
    encoded += count > 1U ? alphabet[((b & 15U) << 2U) | (c >> 6U)] : '=';
    encoded += count > 2U ? alphabet[c & 63U] : '=';
  }
  return encoded;
}

void Rfc8032Oracle() {
  // Public TEST 1 from RFC 8032 section 7.1, encoded as RFC 8410 PKCS#8.
  // No user secret is discovered, generated on disk, or printed.
  const auto der = Decode(
      "302e020100300506032b657004220420"
      "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
  const auto pem = "-----BEGIN PRIVATE KEY-----\n" + Base64(der) +
                   "\n-----END PRIVATE KEY-----\n";
  const auto signer = signing::Ed25519Signer::from_pkcs8_pem(Bytes(pem));
  const auto raw_public = Decode(
      "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
  const auto public_key = signer.public_key();
  Require(std::equal(public_key.begin(), public_key.end(), raw_public.begin(),
                     raw_public.end()),
          "installed RFC public key");
  const auto signature = signer.sign_payload({});
  Require(signature.algorithm == "ed25519", "installed RFC algorithm");
  Require(signature.key_id == signing::ed25519_key_id(public_key),
          "installed RFC key identity");
  Require(
      signature.signature_hex ==
          "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
          "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
      "installed signature matches independent RFC oracle");
  auto raw_signature = Decode(signature.signature_hex);
  Require(signing::verify_ed25519_signature(public_key, {}, raw_signature),
          "installed RFC verification");
  raw_signature[0] ^= std::byte{1U};
  Require(!signing::verify_ed25519_signature(public_key, {}, raw_signature),
          "installed RFC tamper rejection");
}

void GeneratedPolicyAndTamper() {
  const auto signer = signing::Ed25519Signer::generate();
  const auto public_key = signer.public_key();
  const auto policy_json = signing::serialize_ed25519_trust_policy(public_key);
  const auto policy =
      signing::Ed25519TrustPolicy::parse(policy_json, Digest(policy_json));
  Require(policy.public_key() == public_key && !policy.revoked(),
          "installed generated policy");
  Require(policy.key_id() == signer.key_id(),
          "installed generated key identity");
  constexpr std::string_view message = "installed downstream payload";
  auto signature = signer.sign_payload(Bytes(message));
  const auto options = signing::trusted_ed25519_options(policy);
  Require(options.require_trusted_signature && options.signature_verifier,
          "installed trusted verification adapter");
  Require(options.signature_verifier(signature, Bytes(message)),
          "installed authorized verification");
  Require(!options.signature_verifier(signature, Bytes("altered payload")),
          "installed content tamper rejection");
  auto altered = signature;
  altered.signature_hex[0] = altered.signature_hex[0] == '0' ? '1' : '0';
  Require(!options.signature_verifier(altered, Bytes(message)),
          "installed signature tamper rejection");
  altered = signature;
  altered.key_id = "unauthorized.other-key";
  Require(!options.signature_verifier(altered, Bytes(message)),
          "installed unknown signer rejection");
  altered = signature;
  altered.algorithm = "ed25519ph";
  Require(!options.signature_verifier(altered, Bytes(message)),
          "installed algorithm rejection");

  const auto revoked_json =
      signing::serialize_ed25519_trust_policy(public_key, true);
  const auto revoked =
      signing::Ed25519TrustPolicy::parse(revoked_json, Digest(revoked_json));
  const auto reject = signing::make_ed25519_verifier(revoked);
  Require(!reject(signature, Bytes(message)), "installed revocation rejection");
  bool digest_rejected = false;
  try {
    static_cast<void>(signing::Ed25519TrustPolicy::parse(policy_json + " ",
                                                         Digest(policy_json)));
  } catch (const jyppx::liuding::InvalidArgument&) {
    digest_rejected = true;
  }
  Require(digest_rejected, "installed original policy bytes must match digest");
}
}  // namespace

int main() {
  try {
    Rfc8032Oracle();
    GeneratedPolicyAndTamper();
    std::cout << "Installed native Ed25519 consumer passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Installed signing consumer failed: " << error.what() << '\n';
    return 1;
  }
}
