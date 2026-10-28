// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TRAINING_EVALUATION_SESSION_STATE_HPP
#define JYPPX_LIUDING_TRAINING_EVALUATION_SESSION_STATE_HPP

#include <jyppx/liuding/foundation/artifact.hpp>
#include <jyppx/liuding/foundation/checkpoint.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::liuding::training::detail::evaluation_session {
using Bytes = std::vector<std::byte>;
constexpr std::uint64_t kTopologyBytes = 64ULL << 10U;
inline void append_integer(Bytes& bytes, std::uint64_t value) {
  if (bytes.size() > kTopologyBytes - 8)
    throw ResourceExhausted("Evaluation topology exceeds 64 KiB");
  for (unsigned i = 0; i < 8; ++i)
    bytes.push_back(static_cast<std::byte>((value >> (8U * i)) & 0xffU));
}
inline std::uint64_t read_integer(std::span<const std::byte> bytes,
                                  std::size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 8)
    throw SerializationError("Truncated evaluation progress");
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i)
    value |= static_cast<std::uint64_t>(bytes[offset + i]) << (8U * i);
  return value;
}
inline void append_text(Bytes& bytes, std::string_view value) {
  if (bytes.size() > kTopologyBytes - 8 ||
      value.size() > kTopologyBytes - 8 - bytes.size())
    throw ResourceExhausted("Evaluation topology exceeds 64 KiB");
  append_integer(bytes, value.size());
  for (const auto c : value) bytes.push_back(static_cast<std::byte>(c));
}
template <typename Identity>
void append_identity(Bytes& bytes, const Identity& identity) {
  for (const auto& value : {identity.evaluation_id, identity.build_fingerprint})
    if (value.empty() || value.size() > 1024 ||
        !std::all_of(value.begin(), value.end(),
                     [](unsigned char c) { return c >= 32 && c <= 126; }))
      throw InvalidArgument(
          "Evaluation identity requires bounded printable ASCII");
  validate_artifact_digest(identity.dataset_digest);
  validate_artifact_digest(identity.model_digest);
  validate_artifact_digest(identity.config_digest);
  validate_state_manifest({.participant_id = identity.participant_id,
                           .fields = {{.name = "topology", .max_bytes = 1}},
                           .degradation_reasons = {}});
  for (const auto& value :
       {identity.participant_id, identity.evaluation_id,
        identity.dataset_digest.hex, identity.model_digest.hex,
        identity.config_digest.hex, identity.build_fingerprint})
    append_text(bytes, value);
}
inline std::string prefix(std::size_t index) {
  return "slice." + std::to_string(index) + ".";
}
class SliceWriter final : public StateWriter {
 public:
  SliceWriter(StateWriter& writer, std::size_t index)
      : writer_(writer), prefix_(prefix(index)) {}
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    writer_.write_blob(prefix_ + std::string(name), data);
  }

 private:
  StateWriter& writer_;
  std::string prefix_;
};
class SliceReader final : public StateReader {
 public:
  SliceReader(const StateReader& reader, std::size_t index,
              StateManifest manifest)
      : reader_(reader),
        prefix_(prefix(index)),
        manifest_(std::move(manifest)) {}
  const StateManifest& manifest() const noexcept override { return manifest_; }
  bool contains(std::string_view name) const override {
    return reader_.contains(prefix_ + std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t limit) const override {
    return reader_.read_blob(prefix_ + std::string(name), limit);
  }

 private:
  const StateReader& reader_;
  std::string prefix_;
  StateManifest manifest_;
};
inline void validate_reader(const StateReader& reader,
                            const StateManifest& expected,
                            std::span<const std::byte> topology,
                            RestorePolicy policy) {
  const auto& stored = reader.manifest();
  validate_state_manifest(stored);
  if (stored.participant_id != expected.participant_id ||
      stored.schema_version != 1 ||
      stored.restore_level != RestoreLevel::exact ||
      !stored.degradation_reasons.empty() ||
      !restore_policy_accepts(policy, stored.restore_level) ||
      !state_fields_equal(stored.fields, expected.fields))
    throw CompatibilityError("Evaluation state contract differs");
  const auto actual = reader.read_blob("topology", topology.size());
  if (actual.size() != topology.size() ||
      !std::equal(actual.begin(), actual.end(), topology.begin()))
    throw CompatibilityError("Evaluation input/config identity differs");
}
class RunningGuard final {
 public:
  RunningGuard(bool& running, bool& boundary)
      : running_(running), boundary_(boundary) {
    if (running_) throw NotReady("Evaluation is already running");
    running_ = true;
  }
  ~RunningGuard() {
    running_ = false;
    boundary_ = false;
  }

 private:
  bool& running_;
  bool& boundary_;
};
template <typename Session, typename Hook>
void notify(const Session& session, const Hook& hook, bool& boundary) {
  if (!hook) return;
  boundary = true;
  try {
    hook(session);
  } catch (...) {
    boundary = false;
    throw;
  }
  boundary = false;
}
}  // namespace jyppx::liuding::training::detail::evaluation_session
#endif  // JYPPX_LIUDING_TRAINING_EVALUATION_SESSION_STATE_HPP
