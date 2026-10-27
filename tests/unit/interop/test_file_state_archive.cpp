// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/file_state_archive.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <latch>
#include <limits>
#include <string_view>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using jyppx::liuding::interop::FileStateArchive;

jyppx::liuding::StateManifest state() {
  return {
      .participant_id = "test.state",
      .schema_version = 7U,
      .restore_level = jyppx::liuding::RestoreLevel::bounded,
      .fields = {{.name = "payload", .max_bytes = 32768U},
                 {.name = "optional", .max_bytes = 4U, .required = false}},
      .degradation_reasons = {"numeric tolerance fixed by the participant"}};
}

std::vector<std::byte> read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  const auto size = input.tellg();
  if (size < 0) throw jyppx::liuding::IoError("Cannot read test artifact");
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  input.seekg(0);
  input.read(reinterpret_cast<char*>(bytes.data()), size);
  return bytes;
}

void write_file(const std::filesystem::path& path,
                std::span<const std::byte> bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

class FileStateArchiveTest : public ::testing::Test {
 protected:
  void SetUp() override {
    directory_ =
        std::filesystem::temp_directory_path() /
        (std::string("jyppx-archive-") +
         ::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::filesystem::create_directories(directory_);
    path_ = directory_ / "state.bin";
  }
  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }
  void save(std::span<const std::byte> bytes) {
    FileStateArchive archive(path_);
    archive.begin({state()});
    archive.writer_for("test.state").write_blob("payload", bytes);
    archive.commit();
  }
  std::filesystem::path directory_;
  std::filesystem::path path_;
};

TEST_F(FileStateArchiveTest, ReopensEmbeddedManifestAndValidatesAllDigests) {
  const std::array<std::byte, 3> content{std::byte{'a'}, std::byte{'b'},
                                         std::byte{'c'}};
  save(content);
  FileStateArchive archive(path_);
  ASSERT_EQ(archive.manifest().participants.size(), 1U);
  const auto& entry = archive.manifest().participants.front();
  EXPECT_EQ(entry.state.schema_version, 7U);
  EXPECT_EQ(entry.state.restore_level, jyppx::liuding::RestoreLevel::bounded);
  EXPECT_EQ(entry.state.degradation_reasons, state().degradation_reasons);
  ASSERT_EQ(entry.blobs.size(), 1U);
  EXPECT_EQ(entry.blobs[0].digest.hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(entry.blobs[0].size_bytes, content.size());
  EXPECT_FALSE(archive.reader_for("test.state").contains("optional"));
  EXPECT_THROW(archive.reader_for("test.state").read_blob("payload", 2U),
               jyppx::liuding::ResourceExhausted);
  const auto encoded = read_file(path_);
  const auto digest = jyppx::liuding::sha256(
      std::span<const std::byte>(encoded).first(encoded.size() - 32U));
  EXPECT_TRUE(std::equal(digest.begin(), digest.end(), encoded.end() - 32U));
}

TEST_F(FileStateArchiveTest, CanonicalWireBytesSpanParticipantsAndEmptyFields) {
  auto first = state();
  first.participant_id = "a.state";
  first.fields = {{.name = "empty", .max_bytes = 1U, .required = true},
                  {.name = "payload", .max_bytes = 32769U, .required = true},
                  {.name = "unused", .max_bytes = 1U, .required = false}};
  auto last = state();
  last.participant_id = "z.state";
  last.fields.front().max_bytes = 3U;
  std::vector<std::byte> input(32769U, std::byte{7});
  const std::array tail{std::byte{0}, std::byte{255}, std::byte{3}};
  FileStateArchive archive(path_);
  archive.begin({first, last});
  archive.writer_for("z.state").write_blob("payload", tail);
  archive.writer_for("a.state").write_blob("payload", input);
  archive.writer_for("a.state").write_blob("empty", {});
  // Writers must consume the caller's bytes before returning, even though
  // publication later streams the captured fields.
  std::fill(input.begin(), input.end(), std::byte{99});
  archive.commit();

  const auto json = jyppx::liuding::interop::checkpoint_artifact_manifest_json(
      archive.manifest());
  std::vector<std::byte> expected;
  for (const char value : std::string_view("JYPPXCK1"))
    expected.push_back(static_cast<std::byte>(value));
  const auto integer = [&](std::uint64_t value, unsigned size) {
    for (unsigned index = 0; index < size; ++index)
      expected.push_back(static_cast<std::byte>(value >> (8U * index)));
  };
  integer(1U, 4U);
  integer(json.size(), 8U);
  for (const unsigned char value : json)
    expected.push_back(static_cast<std::byte>(value));
  expected.insert(expected.end(), 32769U, std::byte{7});
  expected.insert(expected.end(), tail.begin(), tail.end());
  const auto digest = jyppx::liuding::sha256(expected);
  expected.insert(expected.end(), digest.begin(), digest.end());
  EXPECT_EQ(read_file(path_), expected);
  FileStateArchive reopened(path_);
  EXPECT_TRUE(reopened.reader_for("a.state").contains("empty"));
  EXPECT_TRUE(reopened.reader_for("a.state").read_blob("empty", 0U).empty());
  EXPECT_FALSE(reopened.reader_for("a.state").contains("unused"));
  EXPECT_EQ(reopened.reader_for("a.state").read_blob("payload", 32769U)[0],
            std::byte{7});
}

TEST_F(FileStateArchiveTest, ExactArtifactLimitIncludesFramingAndTrailer) {
  auto descriptor = state();
  descriptor.fields.front().max_bytes = 64U;
  const std::array<std::byte, 64> payload{};
  {
    FileStateArchive reference(path_);
    reference.begin({descriptor});
    reference.writer_for("test.state").write_blob("payload", payload);
    reference.commit();
  }
  const auto expected = read_file(path_);
  for (const auto shortage : {0U, 1U}) {
    SCOPED_TRACE(shortage);
    const auto destination =
        directory_ / ("bounded-" + std::to_string(shortage));
    const std::array prior{std::byte{42}};
    {
      FileStateArchive previous(destination);
      previous.begin({descriptor});
      previous.writer_for("test.state").write_blob("payload", prior);
      previous.commit();
    }
    const auto previous = read_file(destination);
    jyppx::liuding::interop::FileStateArchiveOptions options;
    options.max_artifact_bytes = expected.size() - shortage;
    options.max_field_bytes = 64U;
    FileStateArchive archive(destination, options);
    archive.begin({descriptor});
    archive.writer_for("test.state").write_blob("payload", payload);
    if (shortage == 0U) {
      EXPECT_NO_THROW(archive.commit());
      EXPECT_EQ(read_file(destination), expected);
    } else {
      EXPECT_THROW(archive.commit(), jyppx::liuding::ResourceExhausted);
      EXPECT_EQ(read_file(destination), previous);
      EXPECT_EQ(archive.reader_for("test.state").read_blob("payload", 1U)[0],
                prior[0]);
    }
    EXPECT_FALSE(std::filesystem::exists(destination.string() + ".tmp"));
  }
}

TEST_F(FileStateArchiveTest, LoadedBlobsSurviveUntilSuccessfulReplacement) {
  auto last = state();
  last.participant_id = "z.state";
  const std::vector<std::byte> original(32768U, std::byte{42});
  const std::array optional{std::byte{0}, std::byte{255}, std::byte{7}};
  const std::array replacement{std::byte{19}, std::byte{20}};
  {
    FileStateArchive destination(path_);
    destination.begin({state(), last});
    destination.writer_for("test.state").write_blob("payload", original);
    destination.writer_for("test.state").write_blob("optional", optional);
    destination.writer_for("z.state").write_blob("payload", {});
    destination.commit();
  }
  FileStateArchive loaded(path_);
  const auto& reader = loaded.reader_for("test.state");
  const auto payload = reader.read_blob("payload", original.size());
  const auto extra = reader.read_blob("optional", optional.size());
  const auto verify_original = [&] {
    EXPECT_EQ(&loaded.reader_for("test.state"), &reader);
    EXPECT_EQ(reader.read_blob("payload", original.size()).data(),
              payload.data());
    EXPECT_TRUE(std::equal(payload.begin(), payload.end(), original.begin(),
                           original.end()));
    EXPECT_TRUE(std::equal(extra.begin(), extra.end(), optional.begin(),
                           optional.end()));
    EXPECT_TRUE(loaded.reader_for("z.state").contains("payload"));
    EXPECT_TRUE(loaded.reader_for("z.state").read_blob("payload", 0U).empty());
  };
  verify_original();
  loaded.begin({state()});
  loaded.writer_for("test.state").write_blob("payload", replacement);
  verify_original();
  loaded.rollback();
  verify_original();
  loaded.begin({state()});
  EXPECT_THROW(loaded.commit(), jyppx::liuding::SerializationError);
  verify_original();
  // An independent atomic publication, then an in-place file mutation, must
  // not alter any bytes already exposed by this immutable snapshot.
  save(replacement);
  verify_original();
  write_file(path_, replacement);
  verify_original();
  loaded.begin({state()});
  loaded.writer_for("test.state").write_blob("payload", replacement);
  loaded.commit();
  // The old borrowed reader/spans are invalid now; acquire the new generation.
  const auto current =
      loaded.reader_for("test.state").read_blob("payload", replacement.size());
  EXPECT_TRUE(std::equal(current.begin(), current.end(), replacement.begin(),
                         replacement.end()));
  EXPECT_THROW(loaded.reader_for("z.state"),
               jyppx::liuding::SerializationError);
  const FileStateArchive reopened(path_);
  const auto published = reopened.reader_for("test.state")
                             .read_blob("payload", replacement.size());
  EXPECT_TRUE(std::equal(published.begin(), published.end(), current.begin(),
                         current.end()));
}

TEST_F(FileStateArchiveTest, ReaderCannotCleanAnActiveWritersTemporaryFile) {
  const std::array<std::byte, 1> content{std::byte{42}};
  save(content);
  FileStateArchive writer(path_);
  writer.begin({state()});
  write_file(path_.string() + ".tmp", content);
  FileStateArchive reader(path_);
  EXPECT_TRUE(std::filesystem::exists(path_.string() + ".tmp"));
  EXPECT_EQ(reader.reader_for("test.state").read_blob("payload", 1U)[0],
            content[0]);
  FileStateArchive competitor(path_);
  EXPECT_THROW(competitor.begin({state()}), jyppx::liuding::NotReady);
  writer.rollback();
  FileStateArchive recovered(path_);
  EXPECT_FALSE(std::filesystem::exists(path_.string() + ".tmp"));
  EXPECT_NO_THROW(competitor.begin({state()}));
  competitor.rollback();
}

TEST_F(FileStateArchiveTest, CreatesAndReopensNestedDurableDestination) {
  path_ = directory_ / "runs" / "session" / "checkpoints" / "last.bin";
  const std::array<std::byte, 1> content{std::byte{42}};
  save(content);
  FileStateArchive reopened(path_);
  EXPECT_EQ(reopened.reader_for("test.state").read_blob("payload", 1U)[0],
            content[0]);
}

TEST_F(FileStateArchiveTest,
       ReadOnlySnapshotPreservesRecoveryFilesAndRejectsWrites) {
  const std::array<std::byte, 1> content{std::byte{42}};
  save(content);
  const auto temporary = std::filesystem::path(path_.string() + ".tmp");
  const auto lock = std::filesystem::path(path_.string() + ".lock");
  std::filesystem::remove(lock);
  write_file(temporary, content);
  jyppx::liuding::interop::FileStateArchiveOptions options;
  options.read_only = true;
  FileStateArchive reader(path_, options);
  EXPECT_EQ(reader.reader_for("test.state").read_blob("payload", 1U)[0],
            content[0]);
  EXPECT_THROW(reader.begin({state()}), jyppx::liuding::InvalidArgument);
  EXPECT_EQ(read_file(temporary), std::vector(content.begin(), content.end()));
  EXPECT_FALSE(std::filesystem::exists(lock));
}

TEST_F(FileStateArchiveTest, ReadOnlyMissingArchiveDoesNotCreateItsParent) {
  const auto missing = directory_ / "missing" / "state.bin";
  jyppx::liuding::interop::FileStateArchiveOptions options;
  options.read_only = true;
  EXPECT_THROW(FileStateArchive(missing, options), jyppx::liuding::IoError);
  EXPECT_FALSE(std::filesystem::exists(missing.parent_path()));
}

TEST_F(FileStateArchiveTest, ConcurrentReplacementReadsOneCompleteGeneration) {
  const std::array<std::byte, 1> original{std::byte{42}};
  save(original);
  FileStateArchive snapshot(path_);
  std::latch ready(1);
  std::exception_ptr writer_failure;
  std::jthread writer([&] {
    ready.wait();
    try {
      FileStateArchive archive(path_);
      for (unsigned iteration = 0; iteration < 32U; ++iteration) {
        const std::vector<std::byte> content(
            iteration % 2U == 0U ? 32768U : 1U,
            iteration % 2U == 0U ? std::byte{7} : std::byte{42});
        archive.begin({state()});
        archive.writer_for("test.state").write_blob("payload", content);
        archive.commit();
      }
    } catch (...) {
      writer_failure = std::current_exception();
    }
  });
  ready.count_down();
  for (unsigned iteration = 0; iteration < 128U; ++iteration) {
    EXPECT_NO_THROW({
      FileStateArchive reader(path_);
      const auto content =
          reader.reader_for("test.state").read_blob("payload", 32768U);
      ASSERT_TRUE(content.size() == 1U || content.size() == 32768U);
      const auto value = content.size() == 1U ? std::byte{42} : std::byte{7};
      EXPECT_TRUE(std::all_of(content.begin(), content.end(),
                              [value](auto byte) { return byte == value; }));
    });
  }
  writer.join();
  if (writer_failure) {
    EXPECT_NO_THROW(std::rethrow_exception(writer_failure));
  }
  EXPECT_EQ(snapshot.reader_for("test.state").read_blob("payload", 1U)[0],
            original[0]);
}

#if defined(_WIN32)
TEST_F(FileStateArchiveTest,
       RetriesTransientWindowsPublicationSharingConflict) {
  const std::array<std::byte, 1> original{std::byte{42}};
  const std::array<std::byte, 1> replacement{std::byte{7}};
  save(original);
  const auto handle = CreateFileW(
      path_.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  ASSERT_NE(handle, INVALID_HANDLE_VALUE);
  std::jthread reader([handle] {
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    CloseHandle(handle);
  });
  EXPECT_NO_THROW(save(replacement));
  reader.join();
  FileStateArchive reopened(path_);
  EXPECT_EQ(reopened.reader_for("test.state").read_blob("payload", 1U)[0],
            replacement[0]);
}

TEST_F(FileStateArchiveTest,
       PersistentWindowsSharingConflictPreservesSnapshot) {
  const std::array<std::byte, 1> original{std::byte{42}};
  const std::array<std::byte, 1> replacement{std::byte{7}};
  save(original);
  const auto handle = CreateFileW(
      path_.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  ASSERT_NE(handle, INVALID_HANDLE_VALUE);
  EXPECT_THROW(save(replacement), jyppx::liuding::IoError);
  CloseHandle(handle);
  FileStateArchive reopened(path_);
  EXPECT_EQ(reopened.reader_for("test.state").read_blob("payload", 1U)[0],
            original[0]);
  EXPECT_FALSE(std::filesystem::exists(path_.string() + ".tmp"));
}
#endif

#if GTEST_HAS_DEATH_TEST
TEST_F(FileStateArchiveTest, ProcessExitDuringStagingRecoversPriorSnapshot) {
  const std::array<std::byte, 1> content{std::byte{42}};
  save(content);
  const auto previous = read_file(path_);
  ASSERT_EXIT(
      ([&] {
        FileStateArchive writer(path_);
        writer.begin({state()});
        write_file(path_.string() + ".tmp", content);
        std::_Exit(17);  // Kernel releases the lock; C++ cleanup never runs.
      }()),
      ::testing::ExitedWithCode(17), "");
  ASSERT_TRUE(std::filesystem::exists(path_.string() + ".tmp"));
  FileStateArchive recovered(path_);
  EXPECT_EQ(read_file(path_), previous);
  EXPECT_FALSE(std::filesystem::exists(path_.string() + ".tmp"));
  EXPECT_EQ(recovered.reader_for("test.state").read_blob("payload", 1U)[0],
            content[0]);
}

TEST_F(FileStateArchiveTest, ProcessExitAfterPublicationReopensNewSnapshot) {
  const std::array<std::byte, 1> content{std::byte{42}};
  save(content);
  ASSERT_EXIT(
      ([&] {
        FileStateArchive writer(path_);
        writer.begin({state()});
        const std::array<std::byte, 2> replacement{std::byte{7}, std::byte{7}};
        writer.writer_for("test.state").write_blob("payload", replacement);
        writer.commit();
        std::_Exit(23);
      }()),
      ::testing::ExitedWithCode(23), "");
  FileStateArchive recovered(path_);
  const auto bytes =
      recovered.reader_for("test.state").read_blob("payload", 2U);
  ASSERT_EQ(bytes.size(), 2U);
  EXPECT_EQ(bytes[0], std::byte{7});
  EXPECT_EQ(bytes[1], std::byte{7});
}
#endif

TEST_F(FileStateArchiveTest,
       InsufficientStorageReservePreservesOldCompleteFile) {
  const std::array<std::byte, 1> content{std::byte{42}};
  save(content);
  const auto previous = read_file(path_);
  jyppx::liuding::interop::FileStateArchiveOptions options;
  options.min_free_storage_bytes = (std::numeric_limits<std::uint64_t>::max)();
  FileStateArchive writer(path_, options);
  writer.begin({state()});
  writer.writer_for("test.state").write_blob("payload", content);
  EXPECT_THROW(writer.commit(), jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(read_file(path_), previous);
  EXPECT_FALSE(std::filesystem::exists(path_.string() + ".tmp"));
  EXPECT_NO_THROW(writer.begin({state()}));
  writer.rollback();
}

TEST_F(FileStateArchiveTest, TruncationCorruptionAndForgedFramingAreRejected) {
  const std::array<std::byte, 3> content{std::byte{'a'}, std::byte{'b'},
                                         std::byte{'c'}};
  save(content);
  const auto original = read_file(path_);
  for (const auto size : {0U, 8U, 20U, 31U, 32U, 100U}) {
    write_file(path_, std::span<const std::byte>(original).first(size));
    EXPECT_THROW({ FileStateArchive invalid(path_); },
                 jyppx::liuding::SerializationError);
  }
  for (const auto offset :
       {0U, 8U, 12U, 20U, static_cast<unsigned>(original.size() - 33U)}) {
    auto corrupt = original;
    corrupt[offset] ^= std::byte{1};
    write_file(path_, corrupt);
    EXPECT_THROW({ FileStateArchive invalid(path_); },
                 jyppx::liuding::SerializationError);
  }
  auto forged = original;
  std::fill(forged.begin() + 12U, forged.begin() + 20U, std::byte{0xff});
  const auto digest = jyppx::liuding::sha256(
      std::span<const std::byte>(forged).first(forged.size() - 32U));
  std::copy(digest.begin(), digest.end(), forged.end() - 32U);
  write_file(path_, forged);
  EXPECT_THROW({ FileStateArchive invalid(path_); },
               jyppx::liuding::ResourceExhausted);
}

TEST_F(FileStateArchiveTest, ValidatesLimitsFieldsAndTransactionLifecycle) {
  jyppx::liuding::interop::FileStateArchiveOptions invalid_options;
  invalid_options.max_field_bytes = 0;
  EXPECT_THROW({ FileStateArchive invalid(path_, invalid_options); },
               jyppx::liuding::InvalidArgument);
  FileStateArchive archive(path_);
  EXPECT_THROW(archive.commit(), jyppx::liuding::NotReady);
  EXPECT_THROW(archive.begin({}), jyppx::liuding::InvalidArgument);
  archive.begin({state()});
  EXPECT_THROW(archive.begin({state()}), jyppx::liuding::NotReady);
  EXPECT_THROW(archive.writer_for("absent"), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(archive.writer_for("test.state").write_blob("undeclared", {}),
               jyppx::liuding::SerializationError);
  EXPECT_THROW(archive.commit(), jyppx::liuding::SerializationError);
  EXPECT_FALSE(std::filesystem::exists(path_));
  archive.begin({state()});
  archive.writer_for("test.state").write_blob("payload", {});
  EXPECT_THROW(archive.writer_for("test.state").write_blob("payload", {}),
               jyppx::liuding::SerializationError);
  archive.commit();
  FileStateArchive reopened(path_);
  EXPECT_TRUE(
      reopened.reader_for("test.state").read_blob("payload", 0U).empty());
}

#if !defined(_WIN32)
TEST_F(FileStateArchiveTest,
       ActualShortWritePreservesPublishedFileAndRemovesPartialTemporary) {
  const std::array<std::byte, 1> content{std::byte{42}};
  save(content);
  const auto previous = read_file(path_);
  const auto child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    // Limit the child process's real file writes without changing the host
    // filesystem or allocating a disposable disk. EFBIG is reported by the OS.
    ::signal(SIGXFSZ, SIG_IGN);
    const rlimit limit{4096, 4096};
    if (::setrlimit(RLIMIT_FSIZE, &limit) != 0) ::_exit(2);
    try {
      FileStateArchive archive(path_);
      archive.begin({state()});
      const std::vector<std::byte> payload(16384U, std::byte{7});
      archive.writer_for("test.state").write_blob("payload", payload);
      archive.commit();
      ::_exit(3);
    } catch (const jyppx::liuding::ResourceExhausted&) {
      ::_exit(0);
    } catch (...) {
      ::_exit(4);
    }
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  EXPECT_EQ(read_file(path_), previous);
  EXPECT_FALSE(std::filesystem::exists(path_.string() + ".tmp"));
}
#endif

}  // namespace
