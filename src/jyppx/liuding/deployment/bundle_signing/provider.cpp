// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "secure_file.hpp"

#include <jyppx/liuding/deployment/bundle_signing/provider.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>
#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/opensslv.h>
#include <openssl/pem.h>
#include <openssl/provider.h>
#include <string>
#include <unordered_set>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if OPENSSL_VERSION_NUMBER < 0x30000000L
#error "The optional bundle_signing component requires OpenSSL EVP 3.0 APIs."
#endif

namespace jyppx::liuding::bundle_signing {
namespace {

using Json = nlohmann::json;
using Key = std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)>;
using Context = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
using Bio = std::unique_ptr<BIO, decltype(&BIO_free)>;
using LibraryContext =
    std::unique_ptr<OSSL_LIB_CTX, decltype(&OSSL_LIB_CTX_free)>;
using Provider =
    std::unique_ptr<OSSL_PROVIDER, decltype(&OSSL_PROVIDER_unload)>;
using Signature = std::unique_ptr<EVP_SIGNATURE, decltype(&EVP_SIGNATURE_free)>;
using KeyContext = std::unique_ptr<EVP_PKEY_CTX, decltype(&EVP_PKEY_CTX_free)>;
constexpr std::size_t kPolicyBytes = 4096U;
constexpr std::size_t kPrivateKeyBytes = 65536U;
constexpr std::size_t kPassphraseBytes = 1024U;
constexpr std::size_t kPayloadBytes = (64U << 20U) + 128U;
constexpr std::string_view kFormat = "jyppx.liuding.bundle_trust_policy";
constexpr const char* kProperties = "provider=default";

/** @brief Do not let a sensitive backend error queue escape to later callers.
 */
class ErrorScope final {
 public:
  ErrorScope() { ERR_clear_error(); }
  ~ErrorScope() { ERR_clear_error(); }
  ErrorScope(const ErrorScope&) = delete;
  ErrorScope& operator=(const ErrorScope&) = delete;
};

[[noreturn]] void Invalid(std::string_view operation) {
  throw InvalidArgument(
      "Invalid bundle signing input",
      ErrorDetails{
          .component = "bundle_signing",
          .operation = std::string(operation),
          .expected = "the documented bounded Ed25519 contract",
          .actual = "input rejected",
          .hint = "Supply an authorized policy and supported PKCS#8 key."});
}

[[noreturn]] void MalformedPolicy() {
  throw SerializationError(
      "Malformed bundle trust policy",
      ErrorDetails{.component = "bundle_signing",
                   .operation = "policy.parse",
                   .expected = "strict single-key schema 1",
                   .actual = "policy rejected",
                   .hint = "Use the exact supported policy schema."});
}

[[noreturn]] void Backend(std::string_view operation) {
  // Never include ERR_error_string(), private key text, paths, or callback
  // text.
  throw BackendError(
      "Native Ed25519 operation failed",
      ErrorDetails{
          .component = "bundle_signing",
          .operation = std::string(operation),
          .expected = "an available authorized OpenSSL Ed25519 provider",
          .actual = "backend rejected the operation",
          .hint = "Check the locked crypto SDK and explicit key inputs."});
}

/** @brief Finishes this thread's context-local state before owner release.
 *
 * OpenSSL requires threads using a non-default context to stop their local
 * state before its final free. Declare this guard before local EVP objects so
 * their destructors run first. Every operation retains the backend separately.
 */
class ThreadContextGuard final {
 public:
  explicit ThreadContextGuard(OSSL_LIB_CTX* context) : context_(context) {}
  ~ThreadContextGuard() { OPENSSL_thread_stop_ex(context_); }
  ThreadContextGuard(const ThreadContextGuard&) = delete;
  ThreadContextGuard& operator=(const ThreadContextGuard&) = delete;

 private:
  OSSL_LIB_CTX* context_;
};

/** @brief Isolates all algorithm selection from the host's global context.
 *
 * No OSSL_LIB_CTX_load_config call is made: no host openssl.cnf, global
 * property or external provider is imported. The OpenSSL default provider is
 * built in.
 */
class CryptoBackend final {
 public:
  CryptoBackend()
      : context_(nullptr, OSSL_LIB_CTX_free),
        provider_(nullptr, OSSL_PROVIDER_unload) {
    // A major/minor-only ABI check would permit a different unreviewed patch.
    // This identity check does not replace provenance or dependency review.
    if (OpenSSL_version_num() != OPENSSL_VERSION_NUMBER)
      throw CompatibilityError(
          "OpenSSL runtime version differs from the compiled SDK",
          ErrorDetails{
              .component = "bundle_signing",
              .operation = "backend.version",
              .expected = "the exact OpenSSL version used at build time",
              .actual = "crypto runtime identity mismatch",
              .hint = "Load the crypto SDK declared by the dependency lock."});
    context_.reset(OSSL_LIB_CTX_new());
    if (!context_) Backend("provider.context");
    const ThreadContextGuard thread(context_.get());
    provider_.reset(OSSL_PROVIDER_load(context_.get(), "default"));
    if (!provider_) Backend("provider.default");
    Signature signature(
        EVP_SIGNATURE_fetch(context_.get(), "ED25519", kProperties),
        EVP_SIGNATURE_free);
    if (!signature ||
        EVP_SIGNATURE_get0_provider(signature.get()) != provider_.get())
      Backend("provider.ed25519");
  }
  ~CryptoBackend() { OPENSSL_thread_stop_ex(context_.get()); }
  CryptoBackend(const CryptoBackend&) = delete;
  CryptoBackend& operator=(const CryptoBackend&) = delete;
  OSSL_LIB_CTX* context() const noexcept { return context_.get(); }
  const OSSL_PROVIDER* provider() const noexcept { return provider_.get(); }

 private:
  // Reverse destruction order unloads the provider before freeing its context.
  LibraryContext context_;
  Provider provider_;
};

std::shared_ptr<const CryptoBackend> MakeBackend() {
  ErrorScope errors;
  return std::make_shared<const CryptoBackend>();
}

std::span<const std::byte> Bytes(std::string_view text) noexcept {
  return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
}

const unsigned char* Raw(std::span<const std::byte> bytes) noexcept {
  // Some crypto providers reject a null pointer even for an empty message.
  static constexpr unsigned char empty = 0U;
  return bytes.empty() ? &empty
                       : reinterpret_cast<const unsigned char*>(bytes.data());
}

bool Hex(std::string_view text, std::size_t bytes) noexcept {
  return text.size() == bytes * 2U &&
         std::all_of(text.begin(), text.end(), [](unsigned char value) {
           return (value >= '0' && value <= '9') ||
                  (value >= 'a' && value <= 'f');
         });
}

std::string EncodeHex(std::span<const std::byte> bytes) {
  constexpr std::string_view alphabet = "0123456789abcdef";
  std::string result(bytes.size() * 2U, '0');
  for (std::size_t index = 0U; index < bytes.size(); ++index) {
    const auto value = std::to_integer<unsigned int>(bytes[index]);
    result[index * 2U] = alphabet[value >> 4U];
    result[index * 2U + 1U] = alphabet[value & 15U];
  }
  return result;
}

template <std::size_t Size>
std::array<std::byte, Size> DecodeHex(std::string_view text) {
  if (!Hex(text, Size)) Invalid("hex");
  const auto digit = [](char value) {
    return value <= '9' ? value - '0' : value - 'a' + 10;
  };
  std::array<std::byte, Size> result{};
  for (std::size_t index = 0U; index < Size; ++index)
    result[index] = static_cast<std::byte>(digit(text[index * 2U]) * 16 +
                                           digit(text[index * 2U + 1U]));
  return result;
}

// RFC 8032 section 5.1.3 requires canonical y < 2^255-19. The five
// sign-independent y values below enumerate the curve's eight torsion points
// (also listed by libsodium 1.0.20 ge25519_has_small_order). These are public
// input constants, not a replacement curve/signature implementation. OpenSSL
// performs the remaining decoding and signature equation checks. This sieve
// does not claim full prime-subgroup membership validation.
bool AllowedPublicKey(const Ed25519PublicKey& public_key) {
  auto y = public_key;
  y.back() &= std::byte{0x7f};
  static const auto prime = DecodeHex<32U>(
      "edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f");
  if (!std::lexicographical_compare(y.rbegin(), y.rend(), prime.rbegin(),
                                    prime.rend()))
    return false;
  static const std::array<Ed25519PublicKey, 5U> torsion_y{
      DecodeHex<32U>(
          "0000000000000000000000000000000000000000000000000000000000000000"),
      DecodeHex<32U>(
          "0100000000000000000000000000000000000000000000000000000000000000"),
      DecodeHex<32U>(
          "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05"),
      DecodeHex<32U>(
          "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a"),
      DecodeHex<32U>(
          "ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f")};
  return std::find(torsion_y.begin(), torsion_y.end(), y) == torsion_y.end();
}

ArtifactDigest Digest(std::span<const std::byte> bytes) {
  return {.algorithm = ArtifactDigestAlgorithm::sha256,
          .hex = sha256_hex(bytes)};
}

void ExpectedDigest(const ArtifactDigest& digest) {
  if (digest.algorithm != ArtifactDigestAlgorithm::sha256 ||
      !Hex(digest.hex, 32U))
    Invalid("expected_digest");
}

bool White(unsigned char value) noexcept {
  return value == ' ' || value == '\t' || value == '\r' || value == '\n';
}

std::string_view Trim(std::string_view text) noexcept {
  while (!text.empty() && White(static_cast<unsigned char>(text.front())))
    text.remove_prefix(1U);
  while (!text.empty() && White(static_cast<unsigned char>(text.back())))
    text.remove_suffix(1U);
  return text;
}

bool EncryptedPem(std::span<const std::byte> bytes) {
  if (bytes.empty() || bytes.size() > kPrivateKeyBytes)
    Invalid("private_key.bytes");
  auto text = Trim({reinterpret_cast<const char*>(bytes.data()), bytes.size()});
  if (text.find('\0') != std::string_view::npos) Invalid("private_key.pem");
  constexpr std::string_view plain_begin = "-----BEGIN PRIVATE KEY-----";
  constexpr std::string_view plain_end = "-----END PRIVATE KEY-----";
  constexpr std::string_view encrypted_begin =
      "-----BEGIN ENCRYPTED PRIVATE KEY-----";
  constexpr std::string_view encrypted_end =
      "-----END ENCRYPTED PRIVATE KEY-----";
  const bool encrypted = text.starts_with(encrypted_begin);
  const auto begin = encrypted ? encrypted_begin : plain_begin;
  const auto end = encrypted ? encrypted_end : plain_end;
  if (!text.starts_with(begin) || !text.ends_with(end))
    Invalid("private_key.pem");
  text.remove_prefix(begin.size());
  text.remove_suffix(end.size());
  if (text.find("-----BEGIN") != std::string_view::npos ||
      text.find("-----END") != std::string_view::npos)
    Invalid("private_key.pem.objects");
  return encrypted;
}

struct PasswordState {
  PassphraseCallback callback;
  bool failed = false;
};

int Password(char* output, int maximum, int /*writing*/, void* user) noexcept {
  auto* state = static_cast<PasswordState*>(user);
  if (state == nullptr) return 0;
  if (output == nullptr || maximum <= 0 || !state->callback) {
    state->failed = true;
    return 0;
  }
  try {
    auto password = state->callback();
    struct Guard {
      std::vector<std::byte>& bytes;
      ~Guard() {
        if (!bytes.empty()) OPENSSL_cleanse(bytes.data(), bytes.size());
      }
    } guard{password};
    if (password.empty() || password.size() > kPassphraseBytes ||
        password.size() > static_cast<std::size_t>(maximum)) {
      state->failed = true;
      return 0;
    }
    std::memcpy(output, password.data(), password.size());
    return static_cast<int>(password.size());
  } catch (...) {
    // No C++ exception or caller-supplied message may cross OpenSSL's C ABI.
    state->failed = true;
    return 0;
  }
}

Ed25519PublicKey Public(EVP_PKEY* key) {
  Ed25519PublicKey result{};
  auto size = result.size();
  if (key == nullptr || EVP_PKEY_is_a(key, "ED25519") != 1 ||
      EVP_PKEY_get_raw_public_key(
          key, reinterpret_cast<unsigned char*>(result.data()), &size) != 1 ||
      size != result.size())
    Backend("public_key");
  return result;
}

bool Verify(const CryptoBackend& backend, const Ed25519PublicKey& public_key,
            std::span<const std::byte> payload,
            std::span<const std::byte> signature) {
  if (!AllowedPublicKey(public_key)) return false;
  ErrorScope errors;
  const ThreadContextGuard thread(backend.context());
  Key key(
      EVP_PKEY_new_raw_public_key_ex(backend.context(), "ED25519", kProperties,
                                     Raw(public_key), public_key.size()),
      EVP_PKEY_free);
  Context context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!key || EVP_PKEY_get0_provider(key.get()) != backend.provider() ||
      !context ||
      EVP_DigestVerifyInit_ex(context.get(), nullptr, nullptr,
                              backend.context(), kProperties, key.get(),
                              nullptr) != 1)
    Backend("verification.initialize");
  const auto result =
      EVP_DigestVerify(context.get(), Raw(signature), signature.size(),
                       Raw(payload), payload.size());
  if (result < 0) Backend("verification.verify");
  return result == 1;
}

Json ParsePolicy(std::string_view text) {
  if (text.empty() || text.size() > kPolicyBytes) MalformedPolicy();
  std::unordered_set<std::string> fields;
  std::size_t nodes = 0U;
  auto callback = [&](int depth, Json::parse_event_t event, Json& value) {
    if (++nodes > 32U || depth > 1) MalformedPolicy();
    if (event == Json::parse_event_t::array_start ||
        (event == Json::parse_event_t::object_start && depth != 0))
      MalformedPolicy();
    if (event == Json::parse_event_t::key) {
      if (!fields.insert(value.get<std::string>()).second) MalformedPolicy();
    }
    if ((event == Json::parse_event_t::key ||
         event == Json::parse_event_t::value) &&
        value.is_string() && value.get_ref<const std::string&>().size() > 256U)
      MalformedPolicy();
    return true;
  };
  try {
    auto result = Json::parse(text.begin(), text.end(), callback, true, false);
    if (!result.is_object() || result.size() != 6U) MalformedPolicy();
    for (const auto* field : {"format", "schema_version", "algorithm", "key_id",
                              "public_key_hex", "revoked"})
      if (!result.contains(field)) MalformedPolicy();
    if (!result.at("format").is_string() ||
        result.at("format").get_ref<const std::string&>() != kFormat ||
        !result.at("schema_version").is_number_unsigned() ||
        result.at("schema_version") != 1U ||
        !result.at("algorithm").is_string() ||
        result.at("algorithm") != "ed25519" ||
        !result.at("key_id").is_string() ||
        !result.at("public_key_hex").is_string() ||
        !result.at("revoked").is_boolean())
      MalformedPolicy();
    return result;
  } catch (const Json::exception&) {
    // JSON diagnostics may contain snippets of caller-supplied input.
    MalformedPolicy();
  }
}

std::filesystem::path AbsoluteBundlePath(const std::filesystem::path& value) {
  if (value.empty() ||
      value.native().find(std::filesystem::path::value_type{}) !=
          std::filesystem::path::string_type::npos)
    Invalid("bundle.path");
  std::error_code error;
  for (const auto& component : value)
    if (component == "..") Invalid("bundle.path");
  const auto absolute = std::filesystem::absolute(value, error);
  if (error) Invalid("bundle.path");
  return absolute.lexically_normal();
}

bool EqualPart(const std::filesystem::path& left,
               const std::filesystem::path& right) {
#if defined(_WIN32)
  const auto& a = left.native();
  const auto& b = right.native();
  return CompareStringOrdinal(a.data(), static_cast<int>(a.size()), b.data(),
                              static_cast<int>(b.size()), TRUE) == CSTR_EQUAL;
#else
  return left == right;
#endif
}

bool IsWithin(const std::filesystem::path& parent,
              const std::filesystem::path& child) {
  auto descendant = child.begin();
  for (auto ancestor = parent.begin(); ancestor != parent.end(); ++ancestor) {
    if (descendant == child.end() || !EqualPart(*ancestor, *descendant))
      return false;
    ++descendant;
  }
  return true;
}

void RejectAliasedDescendant(const std::filesystem::path& source,
                             const std::filesystem::path& destination) {
  // Native file-identity comparison also covers Windows short-name aliases.
  // The publisher independently enforces no-follow traversal when pinning its
  // parent. As with existing bundle APIs, this is not a future namespace lease.
  auto current = destination.parent_path();
  while (!current.empty()) {
    std::error_code error;
    const bool same = std::filesystem::equivalent(source, current, error);
    if (error) Invalid("publication.destination_parent");
    if (same) Invalid("publication.destination");
    const auto parent = current.parent_path();
    if (parent == current) break;
    current = parent;
  }
}

}  // namespace

struct Ed25519TrustPolicy::State {
  Ed25519PublicKey public_key;
  std::string key_id;
  ArtifactDigest digest;
  bool revoked;
};

Ed25519TrustPolicy::Ed25519TrustPolicy(std::shared_ptr<const State> state)
    : state_(std::move(state)) {}
Ed25519TrustPolicy::Ed25519TrustPolicy(Ed25519TrustPolicy&& other) noexcept
    : state_(other.state_) {}
Ed25519TrustPolicy& Ed25519TrustPolicy::operator=(
    Ed25519TrustPolicy&& other) noexcept {
  state_ = other.state_;
  return *this;
}

Ed25519TrustPolicy Ed25519TrustPolicy::parse(
    std::string_view bytes, const ArtifactDigest& expected_digest) {
  ExpectedDigest(expected_digest);
  if (bytes.empty() || bytes.size() > kPolicyBytes) MalformedPolicy();
  const auto digest = Digest(Bytes(bytes));
  if (!(digest == expected_digest)) Invalid("policy.digest");
  const auto policy = ParsePolicy(bytes);
  const auto public_hex = policy.at("public_key_hex").get<std::string>();
  if (!Hex(public_hex, 32U)) MalformedPolicy();
  const auto public_key = DecodeHex<32U>(public_hex);
  if (!AllowedPublicKey(public_key)) Invalid("policy.public_key");
  const auto key_id = policy.at("key_id").get<std::string>();
  if (key_id != ed25519_key_id(public_key)) Invalid("policy.key_id");
  return Ed25519TrustPolicy(std::make_shared<const State>(
      State{.public_key = public_key,
            .key_id = key_id,
            .digest = digest,
            .revoked = policy.at("revoked").get<bool>()}));
}

Ed25519TrustPolicy Ed25519TrustPolicy::load(
    const std::filesystem::path& path, const ArtifactDigest& expected_digest) {
  const auto input = detail::ReadBoundedFile(path, kPolicyBytes);
  const auto bytes = input.bytes();
  return parse({reinterpret_cast<const char*>(bytes.data()), bytes.size()},
               expected_digest);
}

const Ed25519PublicKey& Ed25519TrustPolicy::public_key() const noexcept {
  return state_->public_key;
}
const std::string& Ed25519TrustPolicy::key_id() const noexcept {
  return state_->key_id;
}
const ArtifactDigest& Ed25519TrustPolicy::digest() const noexcept {
  return state_->digest;
}
bool Ed25519TrustPolicy::revoked() const noexcept { return state_->revoked; }

std::string ed25519_key_id(const Ed25519PublicKey& public_key) {
  return "ed25519-sha256-" + sha256_hex(public_key);
}

std::string serialize_ed25519_trust_policy(const Ed25519PublicKey& public_key,
                                           bool revoked) {
  if (!AllowedPublicKey(public_key)) Invalid("policy.public_key");
  return Json{{"format", kFormat},
              {"schema_version", 1U},
              {"algorithm", "ed25519"},
              {"key_id", ed25519_key_id(public_key)},
              {"public_key_hex", EncodeHex(public_key)},
              {"revoked", revoked}}
      .dump();
}

struct Ed25519Signer::State {
  State(std::shared_ptr<const CryptoBackend> context, Key value)
      : backend(std::move(context)),
        key(std::move(value)),
        public_key(Public(key.get())) {
    if (EVP_PKEY_get0_provider(key.get()) != backend->provider())
      Backend("private_key.provider");
    KeyContext check(
        EVP_PKEY_CTX_new_from_pkey(backend->context(), key.get(), kProperties),
        EVP_PKEY_CTX_free);
    if (!check || EVP_PKEY_private_check(check.get()) != 1)
      Backend("private_key.check");
  }
  // The key is destroyed before its provider/context shared owner is released.
  std::shared_ptr<const CryptoBackend> backend;
  Key key;
  Ed25519PublicKey public_key;
};

Ed25519Signer::Ed25519Signer(std::unique_ptr<State> state)
    : state_(std::move(state)) {}
Ed25519Signer::~Ed25519Signer() = default;
Ed25519Signer::Ed25519Signer(Ed25519Signer&&) noexcept = default;
Ed25519Signer& Ed25519Signer::operator=(Ed25519Signer&&) noexcept = default;

Ed25519Signer Ed25519Signer::generate() {
  ErrorScope errors;
  auto backend = MakeBackend();
  const ThreadContextGuard thread(backend->context());
  Key key(EVP_PKEY_Q_keygen(backend->context(), kProperties, "ED25519"),
          EVP_PKEY_free);
  if (!key) Backend("key.generate");
  return Ed25519Signer(std::make_unique<State>(backend, std::move(key)));
}

Ed25519Signer Ed25519Signer::from_pkcs8_pem(std::span<const std::byte> pem,
                                            PassphraseCallback passphrase) {
  ErrorScope errors;
  const bool encrypted = EncryptedPem(pem);
  if (encrypted && !passphrase) Invalid("private_key.passphrase_callback");
  auto backend = MakeBackend();
  const ThreadContextGuard thread(backend->context());
  PasswordState password{.callback = std::move(passphrase), .failed = false};
  Bio input(BIO_new_mem_buf(pem.data(), static_cast<int>(pem.size())),
            BIO_free);
  if (!input) Backend("private_key.buffer");
  Key key(PEM_read_bio_PrivateKey_ex(input.get(), nullptr, Password, &password,
                                     backend->context(), kProperties),
          EVP_PKEY_free);
  if (password.failed || !key) Backend("private_key.parse");
  if (EVP_PKEY_is_a(key.get(), "ED25519") != 1)
    Invalid("private_key.algorithm");
  return Ed25519Signer(std::make_unique<State>(backend, std::move(key)));
}

Ed25519Signer Ed25519Signer::from_pkcs8_file(const std::filesystem::path& path,
                                             PassphraseCallback passphrase) {
  const auto input = detail::ReadBoundedFile(path, kPrivateKeyBytes);
  return from_pkcs8_pem(input.bytes(), std::move(passphrase));
}

Ed25519PublicKey Ed25519Signer::public_key() const {
  if (!state_) Invalid("signer.state");
  return state_->public_key;
}

std::string Ed25519Signer::key_id() const {
  return ed25519_key_id(public_key());
}

interop::BundleSignature Ed25519Signer::sign_payload(
    std::span<const std::byte> payload) const {
  if (!state_) Invalid("signer.state");
  if (payload.size() > kPayloadBytes) Invalid("payload.bytes");
  ErrorScope errors;
  const ThreadContextGuard thread(state_->backend->context());
  Context context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestSignInit_ex(context.get(), nullptr, nullptr,
                                        state_->backend->context(), kProperties,
                                        state_->key.get(), nullptr) != 1)
    Backend("signature.initialize");
  std::array<std::byte, 64U> signature{};
  auto size = signature.size();
  if (EVP_DigestSign(context.get(),
                     reinterpret_cast<unsigned char*>(signature.data()), &size,
                     Raw(payload), payload.size()) != 1 ||
      size != signature.size())
    Backend("signature.sign");
  return {.algorithm = "ed25519",
          .key_id = key_id(),
          .signature_hex = EncodeHex(signature)};
}

interop::BundleSignature Ed25519Signer::sign_manifest(
    const interop::ModelBundleManifest& manifest,
    interop::ModelBundleLimits limits) const {
  const auto payload = interop::model_bundle_signing_payload(manifest, limits);
  return sign_payload(Bytes(payload));
}

bool verify_ed25519_signature(const Ed25519PublicKey& public_key,
                              std::span<const std::byte> payload,
                              std::span<const std::byte> signature) {
  if (signature.size() != 64U || payload.size() > kPayloadBytes) return false;
  const auto backend = MakeBackend();
  return Verify(*backend, public_key, payload, signature);
}

interop::BundleSignatureVerifier make_ed25519_verifier(
    Ed25519TrustPolicy policy, std::size_t max_payload_bytes) {
  if (max_payload_bytes == 0U || max_payload_bytes > kPayloadBytes)
    Invalid("verification.payload_limit");
  auto backend = MakeBackend();
  return
      [policy = std::move(policy), max_payload_bytes,
       backend = std::move(backend)](const interop::BundleSignature& signature,
                                     std::span<const std::byte> payload) {
        if (policy.revoked() || signature.algorithm != "ed25519" ||
            signature.key_id != policy.key_id() ||
            payload.size() > max_payload_bytes ||
            !Hex(signature.signature_hex, 64U))
          return false;
        const auto decoded = DecodeHex<64U>(signature.signature_hex);
        return Verify(*backend, policy.public_key(), payload, decoded);
      };
}

interop::ModelBundleVerifyOptions trusted_ed25519_options(
    Ed25519TrustPolicy policy, interop::ModelBundleLimits limits) {
  if (limits.max_manifest_bytes == 0U ||
      limits.max_manifest_bytes > (64U << 20U))
    Invalid("verification.manifest_limit");
  return {.limits = limits,
          .require_trusted_signature = true,
          .signature_verifier = make_ed25519_verifier(
              std::move(policy), limits.max_manifest_bytes + 128U)};
}

SignedBundlePublication sign_model_bundle_directory(
    const std::filesystem::path& source,
    const ArtifactDigest& expected_manifest,
    const std::filesystem::path& destination, const Ed25519Signer& signer,
    Ed25519TrustPolicy policy, interop::ModelBundleLimits limits) {
  ExpectedDigest(expected_manifest);
  if (policy.revoked() || signer.public_key() != policy.public_key() ||
      signer.key_id() != policy.key_id())
    Invalid("publication.authorized_signer");
  const auto source_path = AbsoluteBundlePath(source);
  const auto destination_path = AbsoluteBundlePath(destination);
  if (IsWithin(source_path, destination_path))
    Invalid("publication.destination");
  auto source_options = trusted_ed25519_options(policy, limits);
  source_options.require_trusted_signature = false;
  auto snapshot = interop::read_model_bundle_snapshot(
      source_path, expected_manifest, std::span<const std::string_view>{},
      source_options, static_cast<std::uint64_t>(limits.max_manifest_bytes));
  RejectAliasedDescendant(source_path, destination_path);
  auto manifest = std::move(snapshot.verification.manifest);
  if (manifest.signatures.empty())
    manifest.signatures.push_back(signer.sign_manifest(manifest, limits));
  // The single-key source callback already rejects every foreign or invalid
  // declaration. A valid existing signature is kept; duplicate IDs never arise.
  std::vector<interop::BundlePublishSource> sources;
  sources.reserve(manifest.files.size());
  for (const auto& file : manifest.files)
    sources.push_back(
        {.file_id = file.id, .source_path = source_path / file.relative_path});
  auto verified =
      interop::publish_model_bundle(destination_path, manifest, sources,
                                    trusted_ed25519_options(policy, limits));
  return {.source_manifest_digest = snapshot.verification.manifest_digest,
          .trust_policy_digest = policy.digest(),
          .key_id = policy.key_id(),
          .published = std::move(verified)};
}

}  // namespace jyppx::liuding::bundle_signing
