// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/manifests.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <vector>

namespace {
using namespace jyppx::liuding::data::image;
namespace fs = std::filesystem;

ManifestReadOptions Classes() {
  ManifestReadOptions options;
  options.categories = {{3, "cat"}, {7, "dog"}, {9, "cat,dog"}};
  return options;
}

class TempDirectory final {
 public:
  TempDirectory() {
    static std::atomic<unsigned> counter{0U};
    const auto base = fs::temp_directory_path();
    for (unsigned attempt = 0; attempt != 64U; ++attempt) {
      const auto candidate =
          base /
          ("jyppx-image-manifest-" +
           std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count()) +
           "-" + std::to_string(++counter) + "-" +
           std::to_string(std::random_device{}()));
      std::error_code error;
      if (fs::create_directory(candidate, error)) {
        path_ = candidate;
        return;
      }
    }
    throw std::runtime_error("could not create manifest fixture directory");
  }
  ~TempDirectory() {
    std::error_code error;
    fs::remove_all(path_, error);
  }
  TempDirectory(const TempDirectory&) = delete;
  TempDirectory& operator=(const TempDirectory&) = delete;
  const fs::path& path() const noexcept { return path_; }

 private:
  fs::path path_;
};

void Write(const fs::path& path, std::string_view text) {
  fs::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary);
  ASSERT_TRUE(stream.good());
  stream << text;
  ASSERT_TRUE(stream.good());
}

TEST(ImageManifestTest, CsvSupportsQuotedCommaClassAndStableLabels) {
  auto options = Classes();
  const auto index = parse_manifest(
      "\xef\xbb\xbf"
      "path,labels\r\n"
      "a.jpg,cat\r\n"
      "b.jpg,\"cat,dog\"\r\n",
      ManifestFormat::csv, ManifestTask::classification, options);
  ASSERT_EQ(index.records().size(), 2U);
  EXPECT_EQ(index.records()[0].labels, (std::vector<std::size_t>{0U}));
  EXPECT_EQ(index.records()[1].labels, (std::vector<std::size_t>{2U}));
  EXPECT_EQ(index.digest_scope(), ManifestDigestScope::document_bytes);
  EXPECT_EQ(index.source_sha256().size(), 64U);
}

TEST(ImageManifestTest, TsvSupportsMultiLabelAndRejectsDuplicateClass) {
  auto options = Classes();
  options.multi_label = true;
  const auto index =
      parse_manifest("path\tlabels\nimage.png\tcat|dog\n", ManifestFormat::tsv,
                     ManifestTask::classification, options);
  ASSERT_EQ(index.records().size(), 1U);
  EXPECT_EQ(index.records()[0].labels, (std::vector<std::size_t>{0U, 1U}));
  EXPECT_THROW(
      parse_manifest("path\tlabels\nimage.png\tcat|cat\n", ManifestFormat::tsv,
                     ManifestTask::classification, options),
      jyppx::liuding::SerializationError);
}

TEST(ImageManifestTest, CsvDoubledQuotesAndJsonlPipeClassHaveDistinctRules) {
  auto options = Classes();
  options.categories[0].name = "a\"b";
  const auto csv =
      parse_manifest("path,labels\na.jpg,\"a\"\"b\"\n", ManifestFormat::csv,
                     ManifestTask::classification, options);
  EXPECT_EQ(csv.records()[0].labels, (std::vector<std::size_t>{0U}));
  options.categories[0].name = "cat|dog";
  EXPECT_THROW(
      parse_manifest("path,labels\na.jpg,cat|dog\n", ManifestFormat::csv,
                     ManifestTask::classification, options),
      jyppx::liuding::InvalidArgument);
  const auto jsonl = parse_manifest(
      "{\"path\":\"a.jpg\",\"labels\":[\"cat|dog\"]}", ManifestFormat::jsonl,
      ManifestTask::classification, options);
  EXPECT_EQ(jsonl.records()[0].labels, (std::vector<std::size_t>{0U}));
}

TEST(ImageManifestTest, JsonlRequiresExactPathLabelsOrInputTargetSchema) {
  auto options = Classes();
  options.multi_label = true;
  const auto index = parse_manifest(
      "{\"path\":\"a.jpg\",\"labels\":[\"cat\",\"dog\"]}\n",
      ManifestFormat::jsonl, ManifestTask::classification, options);
  EXPECT_EQ(index.records()[0].path, "a.jpg");
  EXPECT_EQ(index.records()[0].labels, (std::vector<std::size_t>{0U, 1U}));
  EXPECT_THROW(parse_manifest("{\"path\":\"a.jpg\",\"labels\":[]}",
                              ManifestFormat::jsonl,
                              ManifestTask::classification, Classes()),
               jyppx::liuding::SerializationError);
  EXPECT_THROW(parse_manifest("{\"path\":\"a.jpg\",\"path\":\"b.jpg\","
                              "\"labels\":[\"cat\"]}",
                              ManifestFormat::jsonl,
                              ManifestTask::classification, Classes()),
               jyppx::liuding::SerializationError);
}

TEST(ImageManifestTest, JsonlEscapedDuplicateKeysAndUnknownFieldsAreRejected) {
  EXPECT_THROW(
      parse_manifest(R"({"path":"a.jpg","p\u0061th":"b.jpg","labels":["cat"]})",
                     ManifestFormat::jsonl, ManifestTask::classification,
                     Classes()),
      jyppx::liuding::SerializationError);
  EXPECT_THROW(parse_manifest(R"({"path":"a.jpg","labels":["cat"],"weight":1})",
                              ManifestFormat::jsonl,
                              ManifestTask::classification, Classes()),
               jyppx::liuding::SerializationError);
}

TEST(ImageManifestTest, PairedManifestKeepsTargetPathWithoutOpeningIt) {
  auto options = ManifestReadOptions{};
  const auto index =
      parse_manifest("input,target\nimages/a.png,masks/a.tif\n",
                     ManifestFormat::csv, ManifestTask::paired, options);
  ASSERT_EQ(index.records().size(), 1U);
  EXPECT_EQ(index.records()[0].path, "images/a.png");
  ASSERT_TRUE(index.records()[0].target_path);
  EXPECT_EQ(*index.records()[0].target_path, "masks/a.tif");
}

TEST(ImageManifestTest, RejectsTraversalUnknownClassAndStrictLineContract) {
  const auto options = Classes();
  EXPECT_THROW(
      parse_manifest("path,labels\n../secret.jpg,cat\n", ManifestFormat::csv,
                     ManifestTask::classification, options),
      jyppx::liuding::SerializationError);
  EXPECT_THROW(
      parse_manifest("path,labels\nimage.jpg,unknown\n", ManifestFormat::csv,
                     ManifestTask::classification, options),
      jyppx::liuding::SerializationError);
  EXPECT_THROW(
      parse_manifest("path,labels\nimage.jpg,cat\n\n", ManifestFormat::csv,
                     ManifestTask::classification, options),
      jyppx::liuding::SerializationError);
}

TEST(ImageManifestTest, DuplicateAndPortablePathCollisionsAreRejected) {
  for (const auto* text :
       {"path,labels\na.jpg,cat\na.jpg,dog\n",
        "path,labels\nA.jpg,cat\na.jpg,dog\n", "path,labels\nC:/a.jpg,cat\n",
        "path,labels\nCON.jpg,cat\n", "path,labels\nimages//a.jpg,cat\n",
        "path,labels\nimages/a.jpg.,cat\n"})
    EXPECT_THROW(parse_manifest(text, ManifestFormat::csv,
                                ManifestTask::classification, Classes()),
                 jyppx::liuding::SerializationError);
}

TEST(ImageManifestTest, LimitsBoundRecordsLabelsTokensAndOwnedIndex) {
  auto options = Classes();
  options.limits.max_images = 1;
  EXPECT_THROW(
      parse_manifest("path,labels\na.jpg,cat\nb.jpg,dog\n", ManifestFormat::csv,
                     ManifestTask::classification, options),
      jyppx::liuding::ResourceExhausted);
  options = Classes();
  options.max_record_bytes = 8;
  EXPECT_THROW(parse_manifest("path,labels\na.jpg,cat\n", ManifestFormat::csv,
                              ManifestTask::classification, options),
               jyppx::liuding::ResourceExhausted);
  options = Classes();
  options.max_index_bytes = 1;
  EXPECT_THROW(parse_manifest("path,labels\na.jpg,cat\n", ManifestFormat::csv,
                              ManifestTask::classification, options),
               jyppx::liuding::ResourceExhausted);
  options = Classes();
  options.multi_label = true;
  options.max_labels_per_record = 1;
  EXPECT_THROW(
      parse_manifest("path,labels\na.jpg,cat|dog\n", ManifestFormat::csv,
                     ManifestTask::classification, options),
      jyppx::liuding::ResourceExhausted);
  options = Classes();
  options.limits.max_nodes = 2;
  EXPECT_THROW(parse_manifest("{\"path\":\"a.jpg\",\"labels\":[\"cat\"]}",
                              ManifestFormat::jsonl,
                              ManifestTask::classification, options),
               jyppx::liuding::ResourceExhausted);
}

TEST(ImageManifestTest, ActualFileReaderHashesDocumentAndOwnsClassMap) {
  TempDirectory fixture;
  Write(fixture.path() / "manifest.csv", "path,labels\na.jpg,cat\n");
  auto options = Classes();
  const auto index =
      read_manifest_file(fixture.path() / "manifest.csv", ManifestFormat::csv,
                         ManifestTask::classification, options);
  options.categories[0].name = "changed externally";
  EXPECT_EQ(index.categories()[0].name, "cat");
  const auto parsed =
      parse_manifest("path,labels\na.jpg,cat\n", ManifestFormat::csv,
                     ManifestTask::classification, Classes());
  EXPECT_EQ(index.source_sha256(), parsed.source_sha256());
  EXPECT_THROW(
      read_manifest_file(fixture.path() / "absent.csv", ManifestFormat::csv,
                         ManifestTask::classification, Classes()),
      jyppx::liuding::IoError);
}

TEST(ImageManifestTest, ImageFolderSortsClassesAndFilesAndHashesInventory) {
  TempDirectory fixture;
  Write(fixture.path() / "dog" / "z.JPG", "not decoded");
  Write(fixture.path() / "dog" / "a.png", "not decoded");
  Write(fixture.path() / "cat" / "b.jpeg", "not decoded");
  Write(fixture.path() / ".git" / "ignored.jpg", "ignored");
  const auto index = index_image_folder(fixture.path());
  ASSERT_EQ(index.categories().size(), 2U);
  EXPECT_EQ(index.categories()[0].name, "cat");
  EXPECT_EQ(index.categories()[1].name, "dog");
  ASSERT_EQ(index.records().size(), 3U);
  EXPECT_EQ(index.records()[0].path, "cat/b.jpeg");
  EXPECT_EQ(index.records()[1].path, "dog/a.png");
  EXPECT_EQ(index.records()[2].path, "dog/z.JPG");
  EXPECT_EQ(index.digest_scope(),
            ManifestDigestScope::folder_paths_and_classes);
  EXPECT_EQ(index.source_sha256().size(), 64U);
  const auto original_digest = index.source_sha256();
  Write(fixture.path() / "dog" / "a.png",
        "a changed file is outside digest scope");
  EXPECT_EQ(index_image_folder(fixture.path()).source_sha256(),
            original_digest);
}

TEST(ImageManifestTest,
     ImageFolderNestedPathsVisibilityAndSelectionAreExplicit) {
  TempDirectory fixture;
  Write(fixture.path() / "cat" / "a.png", "image");
  Write(fixture.path() / "cat" / "nested" / "b.jpg", "image");
  Write(fixture.path() / "cat" / ".hidden.jpg", "hidden");
  Write(fixture.path() / "cat" / "ignore.txt", "metadata");
  const auto recursive = index_image_folder(fixture.path());
  EXPECT_EQ(recursive.records().size(), 2U);
  ImageFolderOptions options;
  options.recursive = false;
  EXPECT_EQ(index_image_folder(fixture.path(), options).records().size(), 1U);
  options.recursive = true;
  options.include_hidden = true;
  EXPECT_EQ(index_image_folder(fixture.path(), options).records().size(), 3U);
  options.include_hidden = false;
  options.max_directory_depth = 1;
  EXPECT_THROW(index_image_folder(fixture.path(), options),
               jyppx::liuding::ResourceExhausted);
}

TEST(ImageManifestTest, ImageFolderRejectsSymbolicLinksWhenAvailable) {
  TempDirectory fixture;
  Write(fixture.path() / "cat" / "a.jpg", "image");
  std::error_code error;
  fs::create_symlink(fixture.path() / "cat" / "a.jpg",
                     fixture.path() / "cat" / "linked.jpg", error);
  if (error)
    GTEST_SKIP() << "Creating filesystem symbolic links is unavailable";
  EXPECT_THROW(index_image_folder(fixture.path()), jyppx::liuding::IoError);
}

TEST(ImageManifestTest, ImageFolderRejectsRootImagesAndEmptyClasses) {
  TempDirectory fixture;
  Write(fixture.path() / "root.png", "image");
  fs::create_directory(fixture.path() / "cat");
  EXPECT_THROW(index_image_folder(fixture.path()),
               jyppx::liuding::SerializationError);
  std::error_code error;
  fs::remove(fixture.path() / "root.png", error);
  EXPECT_THROW(index_image_folder(fixture.path()),
               jyppx::liuding::SerializationError);
  ImageFolderOptions options;
  options.allow_empty_classes = true;
  EXPECT_THROW(index_image_folder(fixture.path(), options),
               jyppx::liuding::SerializationError);
}

TEST(ImageManifestTest, PairedJsonlPreservesPathsAndRejectsAmbiguousSchemas) {
  const auto index = parse_manifest(
      "{\"input\":\"photos/a.png\",\"target\":\"labels/a.png\"}\r\n"
      "{\"target\":\"labels/b.png\",\"input\":\"photos/b.png\"}\n",
      ManifestFormat::jsonl, ManifestTask::paired);
  ASSERT_EQ(index.records().size(), 2U);
  EXPECT_EQ(index.task(), ManifestTask::paired);
  EXPECT_TRUE(index.categories().empty());
  EXPECT_EQ(index.records()[0].path, "photos/a.png");
  EXPECT_EQ(index.records()[1].target_path, "labels/b.png");
  EXPECT_TRUE(index.records()[0].labels.empty());
  for (const auto* text :
       {"{\"input\":\"a.png\",\"other\":\"b.png\"}",
        "{\"input\":\"a.png\",\"target\":1}",
        "{\"input\":\"a.png\",\"target\":\"b.png\",\"extra\":true}",
        "{\"input\":\"a.png\",\"input\":\"c.png\"}", "[]", "\n",
        "{\"input\":\"a.png\",\"target\":\"b.png\"}\n\n"})
    EXPECT_THROW(
        parse_manifest(text, ManifestFormat::jsonl, ManifestTask::paired),
        jyppx::liuding::SerializationError)
        << text;
  EXPECT_THROW(parse_manifest(std::string("\xff", 1), ManifestFormat::jsonl,
                              ManifestTask::paired),
               jyppx::liuding::SerializationError);
  auto options = ManifestReadOptions{};
  options.limits.max_depth = 1;
  EXPECT_THROW(
      parse_manifest("{\"input\":[[\"a.png\"]],\"target\":\"b.png\"}",
                     ManifestFormat::jsonl, ManifestTask::paired, options),
      jyppx::liuding::ResourceExhausted);
  options = {};
  options.limits.max_string_bytes = 6;
  EXPECT_THROW(
      parse_manifest("{\"input\":\"long-name.png\",\"target\":\"b.png\"}",
                     ManifestFormat::jsonl, ManifestTask::paired, options),
      jyppx::liuding::ResourceExhausted);
  options = {};
  options.limits.max_input_bytes = 8;
  EXPECT_THROW(
      parse_manifest("{\"input\":\"a.png\",\"target\":\"b.png\"}",
                     ManifestFormat::jsonl, ManifestTask::paired, options),
      jyppx::liuding::ResourceExhausted);
}

TEST(ImageManifestTest,
     ImageFolderLimitsAndExtensionsRejectWithoutPartialIndex) {
  TempDirectory fixture;
  Write(fixture.path() / "cat" / "a.png", "image");
  Write(fixture.path() / "cat" / "b.jpg", "image");
  Write(fixture.path() / "dog" / "c.png", "image");
  for (const auto& extensions :
       std::vector<std::vector<std::string>>{{},
                                             {"png"},
                                             {"."},
                                             {".p?g"},
                                             {".png", ".PNG"},
                                             {".abcdefghijklmnop"}}) {
    ImageFolderOptions options;
    options.extensions = extensions;
    EXPECT_THROW(index_image_folder(fixture.path(), options),
                 jyppx::liuding::InvalidArgument);
  }
  for (int field = 0; field < 5; ++field) {
    ImageFolderOptions options;
    if (field == 0) options.max_directory_entries = 1;
    if (field == 1) options.limits.max_categories = 1;
    if (field == 2) options.limits.max_images = 1;
    if (field == 3) options.max_index_bytes = 1;
    if (field == 4) options.limits.max_string_bytes = 1;
    EXPECT_THROW(index_image_folder(fixture.path(), options),
                 jyppx::liuding::ResourceExhausted)
        << field;
  }
  ImageFolderOptions options;
  options.extensions = {".PNG"};
  EXPECT_EQ(index_image_folder(fixture.path(), options).records().size(), 2U);
  EXPECT_THROW(index_image_folder({}), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(index_image_folder(fixture.path() / ".." / "unused"),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(index_image_folder(fixture.path() / "missing"),
               jyppx::liuding::IoError);
  EXPECT_EQ(index_image_folder(fixture.path()).records().size(), 3U);
}

TEST(ImageManifestTest, InvalidOptionsAndClassificationSchemaFailBeforeIndex) {
  using jyppx::liuding::InvalidArgument;
  using jyppx::liuding::SerializationError;
  for (int field = 0; field < 7; ++field) {
    auto options = Classes();
    if (field == 0) options.limits.max_input_bytes = 0;
    if (field == 1) options.limits.max_depth = 129;
    if (field == 2) options.max_record_bytes = 0;
    if (field == 3) options.max_labels_per_record = 0;
    if (field == 4) options.categories.clear();
    if (field == 5) options.max_index_bytes = 0;
    if (field == 6) options.limits.max_categories = 0;
    EXPECT_THROW(parse_manifest("path,labels\na.png,cat\n", ManifestFormat::csv,
                                ManifestTask::classification, options),
                 InvalidArgument)
        << field;
  }
  EXPECT_THROW(parse_manifest("path,labels\na.png,cat\n",
                              static_cast<ManifestFormat>(99),
                              ManifestTask::classification, Classes()),
               InvalidArgument);
  EXPECT_THROW(parse_manifest("path,labels\na.png,cat\n", ManifestFormat::csv,
                              static_cast<ManifestTask>(99), Classes()),
               InvalidArgument);
  EXPECT_THROW(
      parse_manifest("input,target\na.png,b.png\n", ManifestFormat::csv,
                     ManifestTask::paired, Classes()),
      InvalidArgument);
  auto options = ManifestReadOptions{};
  options.multi_label = true;
  EXPECT_THROW(
      parse_manifest("input,target\na.png,b.png\n", ManifestFormat::csv,
                     ManifestTask::paired, options),
      InvalidArgument);
  for (const auto* json : {"{\"path\":\"a.png\",\"wrong\":[\"cat\"]}",
                           "{\"path\":\"a.png\",\"labels\":\"cat\"}",
                           "{\"path\":\"a.png\",\"labels\":[\"cat\"]",
                           "{\"path\":\"a\\u0001.png\",\"labels\":[\"cat\"]}"})
    EXPECT_THROW(parse_manifest(json, ManifestFormat::jsonl,
                                ManifestTask::classification, Classes()),
                 SerializationError)
        << json;
  EXPECT_THROW(
      parse_manifest("path,labels\na.png,cat\rbroken", ManifestFormat::csv,
                     ManifestTask::classification, Classes()),
      SerializationError);
  EXPECT_THROW(parse_manifest("input,target\na.png,\n", ManifestFormat::csv,
                              ManifestTask::paired),
               SerializationError);
}

TEST(ImageManifestTest,
     RichCategoryIndexIsOwnedAndFolderCaseCollisionsAreExplicit) {
  auto options = Classes();
  options.categories[0].keypoint_names = {"left", "right"};
  options.categories[0].skeleton = {{0, 1}};
  const auto index =
      parse_manifest("path,labels\na.png,cat\n", ManifestFormat::csv,
                     ManifestTask::classification, options);
  options.categories[0].skeleton.clear();
  EXPECT_EQ(index.categories()[0].keypoint_names,
            (std::vector<std::string>{"left", "right"}));
  EXPECT_EQ(index.categories()[0].skeleton,
            (std::vector<std::array<std::size_t, 2>>{{0, 1}}));
  TempDirectory fixture;
  Write(fixture.path() / "cat" / "a.png", "first");
  Write(fixture.path() / "cat" / "A.PNG", "second");
  if (fs::equivalent(fixture.path() / "cat" / "a.png",
                     fixture.path() / "cat" / "A.PNG"))
    EXPECT_EQ(index_image_folder(fixture.path()).records().size(), 1U);
  else
    EXPECT_THROW(index_image_folder(fixture.path()),
                 jyppx::liuding::SerializationError);
}

TEST(ImageManifestTest, CancellationIsObservedBeforePublishing) {
  std::stop_source source;
  source.request_stop();
  EXPECT_THROW(parse_manifest("path,labels\na.jpg,cat\n", ManifestFormat::csv,
                              ManifestTask::classification, Classes(),
                              source.get_token()),
               jyppx::liuding::Cancelled);
  TempDirectory fixture;
  Write(fixture.path() / "cat" / "a.jpg", "image");
  EXPECT_THROW(index_image_folder(fixture.path(), {}, source.get_token()),
               jyppx::liuding::Cancelled);
}

}  // namespace
