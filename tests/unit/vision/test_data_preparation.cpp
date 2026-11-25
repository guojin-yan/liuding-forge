// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/data/preparation.hpp>
#include <jyppx/liuding/vision/tasks/workflow/recipe.hpp>

#include <torch/cuda.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <map>

namespace jyppx::liuding::vision::tests {
namespace {
namespace im = data::image;
namespace t = tasks;

class PipelineState final : public StateReader, public StateWriter {
 public:
  explicit PipelineState(StateManifest manifest)
      : manifest_(std::move(manifest)) {}
  const StateManifest& manifest() const noexcept override { return manifest_; }
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t maximum) const override {
    const auto& bytes = blobs_.at(std::string(name));
    if (bytes.size() > maximum) throw std::runtime_error("state byte limit");
    return bytes;
  }
  void write_blob(std::string_view name,
                  std::span<const std::byte> bytes) override {
    blobs_[std::string(name)] = {bytes.begin(), bytes.end()};
  }

 private:
  StateManifest manifest_;
  std::map<std::string, std::vector<std::byte>> blobs_;
};

class VisionDataPreparation : public ::testing::Test {
 protected:
  std::filesystem::path root;
  std::shared_ptr<im::ImageDataset> dataset;
  void SetUp() override {
    torch::set_num_threads(1);
    root = std::filesystem::temp_directory_path() /
           ("liuding-numeric-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(root);
    im::AnnotationIndex index;
    index.categories = {{.source_id = 1, .name = "object"}};
    for (int i = 0; i < 4; ++i) {
      const auto file = std::to_string(i) + ".png";
      std::ofstream(root / file) << "deterministic-reader-fixture";
      index.images.push_back(
          {.source_image_id = i + 1,
           .image_path = file,
           .width = 32,
           .height = 32,
           .objects = {{.source_id = i + 1,
                        .category = 0,
                        .box = im::ImageBox{1, 2, 12, 20}}}});
    }
    im::ImageDatasetOptions decode;
    decode.image_reader = [](const std::filesystem::path& path,
                             const im::ImageDecodeOptions&, std::stop_token) {
      im::DecodedImage out;
      out.tensor = Pixels(std::stoi(path.stem().string()));
      out.width = out.height = 32;
      out.channels = 3;
      return out;
    };
    dataset =
        std::make_shared<im::ImageDataset>(root, std::move(index), decode);
  }
  void TearDown() override {
    dataset.reset();
    std::error_code error;
    std::filesystem::remove_all(root, error);
  }
  static Tensor Pixels(int index) {
    return (torch::arange(3 * 32 * 32, torch::kInt64) + index)
        .remainder(256)
        .to(torch::kUInt8)
        .view({3, 32, 32});
  }
  t::YoloDataOptions Options(bool deferred) {
    t::YoloDataOptions o;
    o.num_classes = 1;
    o.letterbox.canvas = {32, 32};
    o.numeric_on_consumer = deferred;
    o.normalization_mean = {.1, .2, .3};
    o.normalization_std = {.5, .25, .125};
    return o;
  }
  data::DataPipelineOptions Pipeline(std::size_t workers, Device device) {
    data::DataPipelineOptions p;
    p.pipeline_id = "vision.numeric";
    p.dataset_version = "fixture/v1";
    p.batch_size = 2;
    p.num_workers = workers;
    p.memory.target_device = device;
    p.memory.pin_memory = device.is_cuda();
    p.memory.non_blocking = device.is_cuda();
    return p;
  }
  void CheckDelivery(Device device) {
    for (const auto workers : {0U, 2U}) {
      auto p = Pipeline(workers, device);
      auto original = t::make_yolo_data_pipeline(dataset, p, Options(false));
      auto deferred = t::make_yolo_data_pipeline(dataset, p, Options(true));
      original->set_epoch(3);
      deferred->set_epoch(3);
      auto a = original->next(), b = deferred->next();
      ASSERT_TRUE(a);
      ASSERT_TRUE(b);
      ASSERT_TRUE(b->transformed_memory);
      const auto old = im::image_batch_from_value(a->value);
      const auto batch = im::image_batch_from_value(b->value);
      EXPECT_EQ(batch.images.device().type(), device.type());
      EXPECT_EQ(batch.pixels.dtype, torch::kFloat32);
      EXPECT_TRUE(torch::allclose(batch.images, old.images, 1e-6, 1e-7));
      auto expected =
          torch::stack({Pixels(0), Pixels(1)}).to(device).to(torch::kFloat32) /
          255.;
      const auto mean =
          torch::tensor({.1F, .2F, .3F}, expected.options()).view({1, 3, 1, 1});
      const auto std = torch::tensor({.5F, .25F, .125F}, expected.options())
                           .view({1, 3, 1, 1});
      EXPECT_TRUE(
          torch::allclose(batch.images, (expected - mean) / std, 1e-6, 1e-7));
      EXPECT_EQ(a->memory.tensor_bytes - b->memory.tensor_bytes,
                2U * 3U * 32U * 32U * 3U);
      EXPECT_EQ(b->transformed_memory->tensor_bytes, a->memory.tensor_bytes);
      EXPECT_TRUE(torch::equal(t::build_yolo_targets(batch, Options(true)),
                               t::build_yolo_targets(old, Options(false))));
      ASSERT_EQ(batch.items.size(), 2U);
      EXPECT_EQ(batch.items[1].key.id, "1.png");
      EXPECT_EQ(batch.items[1].annotation->objects[0].box->x_max, 12);
      PipelineState saved(deferred->describe_state());
      deferred->save_state(saved);
      auto resumed = t::make_yolo_data_pipeline(dataset, p, Options(true));
      resumed->load_state(saved, RestorePolicy::exact_only);
      const auto next = deferred->next(), replay = resumed->next();
      ASSERT_TRUE(next);
      ASSERT_TRUE(replay);
      EXPECT_EQ(next->sample_ids, replay->sample_ids);
      EXPECT_TRUE(
          torch::equal(im::image_batch_from_value(next->value).images,
                       im::image_batch_from_value(replay->value).images));
      EXPECT_THROW(original->load_state(saved, RestorePolicy::exact_only),
                   CompatibilityError);
    }
  }
};

TEST_F(VisionDataPreparation, DeferredNumericCpuPreservesTargetsAndResume) {
  CheckDelivery(Device(torch::kCPU));
}
TEST_F(VisionDataPreparation, DeferredNumericCudaPreservesTargetsAndResume) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  CheckDelivery(Device(torch::kCUDA, 0));
}
TEST_F(VisionDataPreparation,
       DeferredNumericRejectsConflictingHookAndOutputBudget) {
  auto p = Pipeline(0, Device(torch::kCPU));
  p.consumer_transform = data::BatchConsumerTransform{
      .id = "caller",
      .apply = [](const c10::IValue& v, const data::BatchConsumerContext&,
                  std::stop_token) { return v; }};
  EXPECT_THROW(t::make_yolo_data_pipeline(dataset, p, Options(true)),
               InvalidArgument);
  p.consumer_transform.reset();
  p.memory.max_bytes = 10000;
  auto pipeline = t::make_yolo_data_pipeline(dataset, p, Options(true));
  pipeline->set_epoch(0);
  EXPECT_THROW(pipeline->next(), ResourceExhausted);
  EXPECT_EQ(pipeline->state().next_cursor, 0U);
}
TEST_F(VisionDataPreparation, StandalonePreparationRemainsNormalized) {
  const std::array samples{dataset->get(0)};
  const std::array keys{im::ImageBatchKey{.id = "0.png"}};
  const auto original = t::prepare_yolo_batch(samples, keys, Options(false));
  const auto deferred = t::prepare_yolo_batch(samples, keys, Options(true));
  EXPECT_EQ(deferred.images.scalar_type(), torch::kFloat32);
  EXPECT_TRUE(torch::equal(original.images, deferred.images));
  std::stop_source stopped;
  stopped.request_stop();
  EXPECT_THROW(
      t::prepare_yolo_batch(samples, keys, Options(true), stopped.get_token()),
      Cancelled);
}

TEST(VisionDataPreparationRecipe,
     NumericStageAndTransferRoundTripBindIdentity) {
  t::RecipeSpec r;
  r.training.source = std::filesystem::absolute("annotations.json");
  const auto original = t::recipe_json(r);
  const auto identity = t::recipe_identity(r);
  const auto data_id = t::yolo_data_identity(r.data);
  r.data.numeric_on_consumer = true;
  r.non_blocking_transfer = true;
  auto restored =
      t::parse_recipe(t::recipe_json(r), std::filesystem::current_path());
  EXPECT_TRUE(restored.data.numeric_on_consumer);
  EXPECT_TRUE(restored.non_blocking_transfer);
  EXPECT_NE(identity, t::recipe_identity(r));
  EXPECT_NE(data_id, t::yolo_data_identity(r.data));
  EXPECT_EQ(t::recipe_identity(r), t::recipe_identity(restored));
  r.data.numeric_on_consumer = false;
  r.non_blocking_transfer = false;
  EXPECT_EQ(t::recipe_json(r), original);
  for (const auto* invalid : {"\"other\"", "true", "0", "null"}) {
    EXPECT_THROW(
        t::parse_recipe(
            std::string(
                R"({"model":{"architecture":"yolo26n"},"dataset":{"format":"coco","source":"a.json"},"preprocessing":{"numeric_stage":)") +
                invalid + "}}",
            std::filesystem::current_path()),
        InvalidArgument);
  }
  for (const auto* invalid : {"1", "\"true\"", "null"}) {
    EXPECT_THROW(
        t::parse_recipe(
            std::string(
                R"({"model":{"architecture":"yolo26n"},"dataset":{"format":"coco","source":"a.json"},"non_blocking_transfer":)") +
                invalid + "}",
            std::filesystem::current_path()),
        InvalidArgument);
  }
}
}  // namespace
}  // namespace jyppx::liuding::vision::tests
