// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#pragma once
#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <span>

namespace jyppx::liuding::vision::tasks::detail {
// Search state is its scheduling journal; durable run/artifact ownership stays
// with the existing experiment store. File uploads stream into its shared CAS.
class RunLedger final {
  experiment::SQLiteRunStore store_;
  std::string id_;
  std::uint64_t created_ = 0;
  bool terminal_ = false;
  bool mark_interruption_ = true;
  static std::uint64_t Now() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
  }

 public:
  RunLedger(const std::filesystem::path& root, std::string id,
            std::string config, std::string data,
            std::map<std::string, std::string> tags = {},
            bool mark_interruption = true)
      : store_(root),
        id_(std::move(id)),
        mark_interruption_(mark_interruption) {
    if (auto existing = store_.find(id_)) {
      if (existing->lineage.config_digest != config ||
          existing->lineage.data_manifest_digest != data ||
          existing->tags != tags)
        throw CompatibilityError("Vision experiment ledger identity mismatch");
      created_ = existing->created_at_unix_ms;
      terminal_ = experiment::is_terminal_run_status(existing->status);
      return;
    }
    created_ = Now();
    experiment::RunMetadata metadata;
    metadata.run_id = id_;
    metadata.project_id = "vision";
    metadata.experiment_id = "training-suite";
    metadata.display_name = id_;
    metadata.status = experiment::RunStatus::running;
    metadata.created_at_unix_ms = created_;
    metadata.updated_at_unix_ms = created_;
    metadata.lineage.config_digest = std::move(config);
    metadata.lineage.data_manifest_digest = std::move(data);
    metadata.tags = std::move(tags);
    store_.start(metadata);
  }
  ~RunLedger() {
    if (!terminal_ && mark_interruption_) try {
        store_.transition(id_, experiment::RunStatus::interrupted,
                          (std::max)(created_, Now()));
      } catch (...) {
      }
  }
  void file(const std::filesystem::path& path, std::string role,
            bool checkpoint = false, std::stop_token stop = {}) {
    if (!std::filesystem::exists(path)) return;
    const auto size = std::filesystem::file_size(path);
    if (size > (32ULL << 30U))
      throw ResourceExhausted("Vision ledger file exceeds 32 GiB");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw IoError("Cannot open vision ledger artifact");
    Sha256 digest;
    std::array<std::byte, 65536> chunk;
    std::uint64_t count = 0;
    while (input) {
      if (stop.stop_requested())
        throw Cancelled("Vision ledger upload cancelled");
      input.read(reinterpret_cast<char*>(chunk.data()), chunk.size());
      auto n = static_cast<std::size_t>(input.gcount());
      digest.update(std::span(chunk).first(n));
      count += n;
    }
    if (!input.eof() || count != size)
      throw IoError("Vision ledger artifact changed while hashing");
    experiment::ArtifactDescriptor descriptor;
    descriptor.logical_name = role;
    descriptor.producer = "vision-suite/v1";
    descriptor.expected_size_bytes = size;
    store_.put_artifact_file(id_, role,
                             checkpoint
                                 ? experiment::RunArtifactKind::checkpoint
                                 : experiment::RunArtifactKind::artifact,
                             role, path, descriptor, created_,
                             {.hex = sha256_digest_hex(digest.finish())}, stop);
  }
  void finish(experiment::RunStatus status) {
    if (!terminal_) {
      store_.transition(id_, status, (std::max)(created_, Now()));
      terminal_ = true;
    }
  }
};
}  // namespace jyppx::liuding::vision::tasks::detail
