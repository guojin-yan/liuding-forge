#include <jyppx/interop/checkpoint_manifest.hpp>
#include <jyppx/interop/file_state_archive.hpp>

#include <array>
#include <chrono>
#include <filesystem>
#include <string>

namespace {
bool archive_consumer(const jyppx::StateManifest& manifest) {
  struct TempRoot {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("jyppx-interop-consumer-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    ~TempRoot() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } root;
  const std::array<std::byte, 8> content{};
  {
    jyppx::interop::FileStateArchive archive(root.path / "last.bin");
    archive.begin({manifest});
    archive.writer_for("consumer.state").write_blob("value", content);
    archive.commit();
  }
  jyppx::interop::FileStateArchive reopened(root.path / "last.bin");
  return reopened.reader_for("consumer.state").read_blob("value", 8U).size() ==
         8U;
}
}  // namespace

int main() {
  jyppx::interop::CheckpointArtifactManifest manifest{
      .schema_version =
          jyppx::interop::CheckpointArtifactManifest::format_version,
      .build = {.jyppx_version = "0.3.0",
                .git_commit = "consumer",
                .compiler = "consumer",
                .cxx_standard = "202002",
                .os_arch = "consumer",
                .build_type = "Release",
                .libtorch_version = "2.5.1",
                .libtorch_build = "consumer",
                .accelerator_runtime = "cpu",
                .dependency_lock_digest = "unlocked"},
      .participants =
          {{.state = {.participant_id = "consumer.state",
                      .schema_version = 1U,
                      .restore_level = jyppx::RestoreLevel::exact,
                      .fields = {{.name = "value",
                                  .media_type = "application/octet-stream",
                                  .max_bytes = 8U,
                                  .required = true}}},
            .blobs = {{.field_name = "value",
                       .relative_path = "blobs/value.bin",
                       .size_bytes = 8U,
                       .digest =
                           {.algorithm = jyppx::ArtifactDigestAlgorithm::sha256,
                            .hex = "0123456789abcdef0123456789abcdef"
                                   "0123456789abcdef0123456789abcdef"}}}}},
  };
  const auto encoded =
      jyppx::interop::checkpoint_artifact_manifest_json(manifest);
  const auto decoded =
      jyppx::interop::parse_checkpoint_artifact_manifest_json(encoded);
  const auto capability = jyppx::interop::make_checkpoint_manifest_capability();
  return decoded.participants.size() == 1U &&
                 decoded.participants.front().state.participant_id ==
                     "consumer.state" &&
                 capability.feature_id == "interop.checkpoint_manifest" &&
                 encoded == jyppx::interop::checkpoint_artifact_manifest_json(
                                decoded) &&
                 archive_consumer(decoded.participants.front().state)
             ? 0
             : 1;
}
