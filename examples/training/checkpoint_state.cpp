// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

namespace liuding = jyppx::liuding;
namespace training = liuding::training;

/** @brief Owns one application counter with a portable eight-byte schema. */
class Counter final : public liuding::CheckpointParticipant {
 public:
  Counter(std::string id, std::uint64_t value, bool reject_saved = false)
      : id_(std::move(id)), value_(value), reject_saved_(reject_saved) {}

  std::string participant_id() const override { return id_; }

  liuding::StateManifest describe_state() const override {
    return {.participant_id = id_,
            .schema_version = 1,
            .restore_level = liuding::RestoreLevel::exact,
            .fields = {{.name = "counter",
                        .media_type = "application/octet-stream",
                        .max_bytes = 8}}};
  }

  void save_state(liuding::StateWriter& writer) const override {
    std::array<std::byte, 8> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      bytes[i] = static_cast<std::byte>((value_ >> (8U * i)) & 0xffU);
    }
    writer.write_blob("counter", bytes);
  }

  void load_state(const liuding::StateReader& reader,
                  liuding::RestorePolicy policy) override {
    ++load_calls_;
    const auto& stored = reader.manifest();
    const auto expected = describe_state();
    liuding::validate_state_manifest(stored);
    if (stored.participant_id != id_ || stored.schema_version != 1 ||
        stored.restore_level != liuding::RestoreLevel::exact ||
        !liuding::state_fields_equal(stored.fields, expected.fields) ||
        !liuding::restore_policy_accepts(policy, stored.restore_level)) {
      throw liuding::CompatibilityError("Counter schema or policy mismatch");
    }
    const auto bytes = reader.read_blob("counter", 8);
    if (bytes.size() != 8) {
      throw liuding::SerializationError("Counter requires exactly eight bytes");
    }
    std::uint64_t candidate = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      candidate |= std::to_integer<std::uint64_t>(bytes[i]) << (8U * i);
    }
    // Reject before changing live state; the rollback snapshot stays valid.
    if (reject_saved_ && candidate == 84) {
      throw liuding::CompatibilityError("Example counter rejects saved value");
    }
    value_ = candidate;
  }

  std::uint64_t value() const noexcept { return value_; }
  std::size_t load_calls() const noexcept { return load_calls_; }

 private:
  std::string id_;
  std::uint64_t value_;
  bool reject_saved_;
  // Diagnostic only; this count is not recoverable application state.
  std::size_t load_calls_ = 0;
};

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      throw std::invalid_argument(
          "usage: liuding-forge-example-checkpoint-state "
          "<save|restore|rollback|budget> <checkpoint-path>");
    }
    const std::string_view mode(argv[1]);
    if (mode != "save" && mode != "restore" && mode != "rollback" &&
        mode != "budget") {
      throw std::invalid_argument("Unknown checkpoint example mode");
    }
    const bool saving = mode == "save";
    auto first =
        std::make_shared<Counter>("application.first", saving ? 42 : 111);
    auto second = std::make_shared<Counter>(
        "application.second", saving ? 84 : 222, mode == "rollback");
    training::CheckpointParticipantRegistry registry;
    // Registration order differs from the canonical capture/restore order.
    registry.register_participants({second, first});
    training::CheckpointCoordinator coordinator(
        registry, {.max_participants = 2,
                   .max_rollback_bytes = mode == "budget" ? 15U : 16U});
    training::FileCheckpointStore file(
        std::filesystem::path(argv[2]),
        {.max_artifact_bytes = 65536, .max_field_bytes = 8});

    if (saving) {
      const auto descriptor = coordinator.save(file);
      require(descriptor.participants.size() == 2 &&
                  descriptor.participants.front().participant_id ==
                      first->participant_id(),
              "Capture order must follow participant identity");
    } else if (mode == "restore") {
      coordinator.restore(file, liuding::RestorePolicy::exact_only);
      require(first->value() == 42 && second->value() == 84,
              "A fresh execution domain must restore both saved counters");
    } else if (mode == "rollback") {
      bool rejected = false;
      try {
        coordinator.restore(file, liuding::RestorePolicy::exact_only);
      } catch (const liuding::CompatibilityError& error) {
        require(error.message() == "Example counter rejects saved value",
                "Expected the application precondition failure");
        rejected = true;
      }
      require(rejected && first->value() == 111 && second->value() == 222 &&
                  first->load_calls() == 2 && second->load_calls() == 2,
              "Failed restore must roll back both attempted participants");
    } else {
      bool rejected = false;
      try {
        coordinator.restore(file, liuding::RestorePolicy::exact_only);
      } catch (const liuding::ResourceExhausted&) {
        rejected = true;
      }
      require(rejected && first->value() == 111 && second->value() == 222 &&
                  first->load_calls() == 0 && second->load_calls() == 0,
              "Insufficient rollback budget must fail before participant load");
    }
    std::cout << "mode=" << mode << " first=" << first->value()
              << " second=" << second->value()
              << " first_loads=" << first->load_calls()
              << " second_loads=" << second->load_calls() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
