// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../../../src/jyppx/liuding/vision/tasks/data/yolo_original_detail.hpp"

#include <jyppx/liuding/data/samplers/random_sampler.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/vision/tasks/workflow/recipe.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <nlohmann/json.hpp>
#include <thread>

namespace jyppx::liuding::vision::tests {
namespace {
namespace t = tasks;
namespace im = data::image;
using J = nlohmann::json;
class Saved final : public StateReader, public StateWriter {
 public:
  explicit Saved(StateManifest manifest) : manifest_(std::move(manifest)) {}
  const StateManifest& manifest() const noexcept override { return manifest_; }
  bool contains(std::string_view name) const override {
    return fields.contains(std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t maximum) const override {
    const auto& out = fields.at(std::string(name));
    if (out.size() > maximum) throw std::runtime_error("test state bound");
    return out;
  }
  void write_blob(std::string_view name,
                  std::span<const std::byte> value) override {
    fields[std::string(name)] = {value.begin(), value.end()};
  }
  std::map<std::string, std::vector<std::byte>> fields;

 private:
  StateManifest manifest_;
};
t::YoloDataOptions Options() {
  t::YoloDataOptions o;
  o.num_classes = 1;
  o.letterbox.canvas = {64, 64};
  o.yolo_original.enabled = true;
  o.yolo_original.decoded_cache_bytes = 0;
  o.augmentation.dataset_id = "original-yolo-fixture/v1";
  o.augmentation.seed = 191;
  auto& a = o.augmentation;
  a.multi_source.mosaic_probability = .90863;
  a.multi_source.mixup_probability = .5;
  a.multi_source.total_epochs = 4;
  a.multi_source.close_epochs = 1;
  a.multi_source.copypaste_probability = .07504;
  a.projective.degrees = 1.11032;
  a.projective.translate_fraction = .07105;
  a.projective.scale_gain = .56232;
  a.projective.shear_degrees = 1.46386;
  a.projective.perspective = .00011;
  a.photo.hue_shift = .01373;
  a.photo.saturation_gain = .64481;
  a.photo.value_gain = .56565;
  a.horizontal_flip_probability = .60571;
  a.vertical_flip_probability = .05854;
  a.bgr_probability = .10567;
  return o;
}
data::DataPipelineOptions Pipeline(std::size_t workers = 2) {
  data::DataPipelineOptions p;
  p.pipeline_id = "original-yolo.fixture";
  p.dataset_version = "original-yolo-fixture/v1";
  p.batch_size = 2;
  p.num_workers = workers;
  p.prefetch_batches = 4;
  if (workers) p.batch_timeout = std::chrono::seconds(30);
  return p;
}
struct Fixture {
  std::filesystem::path root;
  std::shared_ptr<im::ImageDataset> dataset;
  std::shared_ptr<std::atomic<int>> reads =
      std::make_shared<std::atomic<int>>(0);
  Fixture() {
    torch::set_num_threads(1);
    root = std::filesystem::temp_directory_path() /
           ("liuding-yolo-original-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(root);
    im::AnnotationIndex index;
    index.categories = {{.source_id = 1, .name = "object"}};
    for (int i = 0; i < 9; ++i) {
      auto path = std::to_string(i) + ".png";
      std::ofstream(root / path) << "deterministic-decoder-fixture";
      index.images.push_back(
          {.source_image_id = i + 1,
           .image_path = path,
           .width = 32,
           .height = 24,
           .objects = {{.source_id = 1,
                        .category = 0,
                        .box = im::ImageBox{2, 3, 28, 20}}}});
    }
    im::ImageDatasetOptions decode;
    decode.image_reader = [reads = reads](const auto& path, const auto&,
                                          std::stop_token stop) {
      if (stop.stop_requested()) throw Cancelled("fixture cancelled");
      const auto i = std::stoi(path.stem().string());
      ++*reads;
      if (i % 2 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      im::DecodedImage image;
      image.tensor = torch::empty({3, 24, 32}, torch::kUInt8);
      auto* p = image.tensor.mutable_data_ptr<std::uint8_t>();
      for (int c = 0; c < 3; ++c)
        for (int y = 0; y < 24; ++y)
          for (int x = 0; x < 32; ++x)
            p[(c * 24 + y) * 32 + x] =
                std::uint8_t((i * 47 + c * 63 + y * 31 + x * 17) % 256);
      image.width = 32;
      image.height = 24;
      image.channels = 3;
      return image;
    };
    dataset = std::make_shared<im::ImageDataset>(root, std::move(index),
                                                 std::move(decode));
  }
  ~Fixture() {
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
  }
  auto Make(t::YoloDataOptions o = Options(),
            data::DataPipelineOptions p = Pipeline()) {
    return t::make_yolo_data_pipeline(
        dataset, p, o,
        std::make_shared<data::samplers::RandomSampler>(dataset->size(), 42));
  }
};
void Equal(const data::BatchEnvelope& a, const data::BatchEnvelope& b,
           const t::YoloDataOptions& o) {
  const auto aa = im::image_batch_from_value(a.value),
             bb = im::image_batch_from_value(b.value);
  EXPECT_EQ(a.sample_ids, b.sample_ids);
  EXPECT_TRUE(torch::equal(aa.images, bb.images));
  EXPECT_TRUE(
      torch::equal(t::build_yolo_targets(aa, o), t::build_yolo_targets(bb, o)));
  ASSERT_EQ(aa.items.size(), bb.items.size());
  for (std::size_t i = 0; i < aa.items.size(); ++i) {
    const auto& x = aa.items[i].annotation->objects;
    const auto& y = bb.items[i].annotation->objects;
    ASSERT_EQ(x.size(), y.size());
    for (std::size_t j = 0; j < x.size(); ++j) {
      ASSERT_EQ(x[j].augmentation_sources.size(),
                y[j].augmentation_sources.size());
      for (std::size_t k = 0; k < x[j].augmentation_sources.size(); ++k) {
        EXPECT_EQ(x[j].augmentation_sources[k].image_path,
                  y[j].augmentation_sources[k].image_path);
        EXPECT_EQ(x[j].augmentation_sources[k].matrix,
                  y[j].augmentation_sources[k].matrix);
        EXPECT_DOUBLE_EQ(x[j].augmentation_sources[k].weight,
                         y[j].augmentation_sources[k].weight);
      }
    }
  }
}
TEST(YoloOriginal,
     BetaDistributionAndBoundedHistoryRetainOfficialSamplingSemantics) {
  auto o = Options();
  o.augmentation.multi_source.mosaic_probability = 1;
  o.augmentation.multi_source.mixup_probability = 0;
  t::detail::OriginalYoloBuffer buffer;
  for (std::size_t i = 0; i < 8; ++i) {
    auto p = t::detail::plan_original_yolo_sample(i, 20, 4, buffer, o, 0, i);
    ASSERT_TRUE(p.primary.geometry.mosaic);
    EXPECT_EQ(p.primary.indices[0], i);
    EXPECT_LE(buffer.size(), 3U);
    for (auto partner : p.primary.indices)
      EXPECT_NE(std::find(buffer.begin(), buffer.end(), partner), buffer.end());
  }
  const auto before = buffer;
  static_cast<void>(t::detail::plan_original_yolo_sample(buffer.back(), 20, 4,
                                                         buffer, o, 0, 9));
  EXPECT_EQ(buffer, before);
  double sum = 0, square = 0, minimum = 1, maximum = 0;
  for (std::uint64_t i = 0; i < 10000; ++i) {
    im::SampleRandom random({.dataset_id = "beta",
                             .sample_id = std::to_string(i),
                             .transform_id = "beta32",
                             .seed = 71});
    double x = t::detail::original_yolo_beta32(random, 0);
    sum += x;
    square += x * x;
    minimum = std::min(minimum, x);
    maximum = std::max(maximum, x);
  }
  EXPECT_NEAR(sum / 10000, .5, .002);
  EXPECT_NEAR(square / 10000 - std::pow(sum / 10000, 2), 1. / 260, .0003);
  EXPECT_LT(minimum, .35);
  EXPECT_GT(maximum, .65);
  EXPECT_THROW(
      t::detail::original_yolo_beta32(
          im::SampleRandom(
              {.dataset_id = "x", .sample_id = "y", .transform_id = "z"}),
          UINT64_MAX),
      InvalidArgument);
}
TEST(YoloOriginal, ParallelPrefetchAndExactRestoreKeepEpochHistoryAndPixels) {
  if (!t::yolo_opencv_available()) {
    EXPECT_THROW(t::yolo_data_identity(Options()), NotImplemented);
    return;
  }
  Fixture fixture;
  auto o = Options();
  auto p = Pipeline();
  auto a = fixture.Make(o, p), b = fixture.Make(o, p);
  a->set_epoch(0);
  b->set_epoch(0);
  for (int i = 0; i < 2; ++i) {
    auto x = a->next(), y = b->next();
    ASSERT_TRUE(x);
    ASSERT_TRUE(y);
    Equal(*x, *y, o);
  }
  Saved saved(a->describe_state());
  a->save_state(saved);
  auto resumed = fixture.Make(o, p);
  resumed->load_state(saved, RestorePolicy::exact_only);
  while (auto x = a->next()) {
    auto y = b->next(), z = resumed->next();
    ASSERT_TRUE(y);
    ASSERT_TRUE(z);
    Equal(*x, *y, o);
    Equal(*x, *z, o);
  }
  EXPECT_FALSE(b->next());
  EXPECT_FALSE(resumed->next());
  a->set_epoch(1);
  resumed->set_epoch(1);
  while (auto x = a->next()) {
    auto y = resumed->next();
    ASSERT_TRUE(y);
    Equal(*x, *y, o);
  }
  EXPECT_FALSE(resumed->next());
  EXPECT_THROW(a->set_epoch(3), InvalidArgument);
}
TEST(YoloOriginal, DeferredNormalizationAndDecodeMemoizationDoNotChangeValues) {
  if (!t::yolo_opencv_available()) {
    EXPECT_THROW(t::yolo_data_identity(Options()), NotImplemented);
    return;
  }
  Fixture fixture;
  auto plain = Options(), deferred = plain;
  deferred.numeric_on_consumer = true;
  deferred.yolo_original.decoded_cache_bytes = 1 << 20;
  auto a = fixture.Make(plain), b = fixture.Make(deferred);
  a->set_epoch(0);
  b->set_epoch(0);
  while (auto x = a->next()) {
    auto y = b->next();
    ASSERT_TRUE(y);
    Equal(*x, *y, plain);
  }
  EXPECT_FALSE(b->next());
  auto eval = t::yolo_evaluation_options(deferred);
  ASSERT_FALSE(eval.yolo_original.training);
  const auto sample = fixture.dataset->get(0);
  const std::array samples{sample};
  const std::array keys{im::ImageBatchKey{.id = "evaluation"}};
  EXPECT_THROW(t::prepare_yolo_batch(samples, keys, plain), InvalidArgument);
  EXPECT_NO_THROW(t::prepare_yolo_batch(samples, keys, eval));
  const auto before = fixture.reads->load();
  auto p = Pipeline(0);
  p.batch_size = 9;
  auto cached = fixture.Make(eval, p);
  cached->set_epoch(0);
  ASSERT_TRUE(cached->next());
  const auto after = fixture.reads->load();
  cached->set_epoch(1);
  ASSERT_TRUE(cached->next());
  EXPECT_EQ(fixture.reads->load(), after);
  EXPECT_GT(after, before);
}
TEST(YoloOriginal, ClosureKeepsPhotometryAndInvalidPoliciesOrRestoreReject) {
  auto o = Options();
  o.augmentation.multi_source.mosaic_probability = 1;
  o.augmentation.multi_source.mixup_probability = 1;
  t::detail::OriginalYoloBuffer buffer;
  const auto p = t::detail::plan_original_yolo_sample(0, 9, 8, buffer, o, 3, 0);
  EXPECT_FALSE(p.primary.geometry.mosaic);
  EXPECT_FALSE(p.mixup);
  EXPECT_TRUE(p.color.hsv);
  if (!t::yolo_opencv_available()) {
    EXPECT_THROW(t::yolo_data_identity(o), NotImplemented);
    return;
  }
  Fixture fixture;
  o = Options();
  o.yolo_original.max_plan_bytes = 1024;
  EXPECT_THROW(fixture.Make(o), ResourceExhausted);
  o = Options();
  o.augmentation.multi_source.cutmix_probability = .1;
  EXPECT_THROW(fixture.Make(o), InvalidArgument);
  o = Options();
  o.rtdetr.enabled = true;
  EXPECT_THROW(fixture.Make(o), InvalidArgument);
  auto a = fixture.Make();
  a->set_epoch(0);
  ASSERT_TRUE(a->next());
  Saved saved(a->describe_state());
  a->save_state(saved);
  o = Options();
  o.yolo_original.decoded_cache_bytes = 1000;
  auto different = fixture.Make(o);
  EXPECT_THROW(different->load_state(saved, RestorePolicy::exact_only),
               CompatibilityError);
  auto replay = fixture.Make();
  Saved before(replay->describe_state());
  replay->save_state(before);
  auto damaged = saved;
  damaged.fields["topology"][0] ^= std::byte{1};
  EXPECT_THROW(replay->load_state(damaged, RestorePolicy::exact_only), Error);
  Saved after(replay->describe_state());
  replay->save_state(after);
  EXPECT_EQ(before.fields, after.fields);
}
TEST(YoloOriginal, RecipeRoundTripBindsOnlyExplicitProfile) {
  auto recipe = t::parse_recipe(
      R"({"model":{"architecture":"yolo26n","task":"detect","num_classes":1},"dataset":{"format":"coco","source":"annotations.json","image_root":"images"},"image_size":64,"epochs":4,"augmentation":{"close_epochs":1}})",
      std::filesystem::current_path());
  auto original = t::recipe_json(recipe);
  EXPECT_EQ(original.find("yolo_original_data"), std::string::npos);
  const auto j = J::parse(original);
  auto explicit_recipe = j;
  explicit_recipe["yolo_original_data"] = J::object();
  if (!t::yolo_opencv_available()) {
    EXPECT_THROW(t::parse_recipe(explicit_recipe.dump(),
                                 std::filesystem::current_path()),
                 NotImplemented);
    return;
  }
  auto legacy_copy = j;
  legacy_copy["augmentation"]["copypaste"] = .07504;
  EXPECT_THROW(
      t::parse_recipe(legacy_copy.dump(), std::filesystem::current_path()),
      InvalidArgument);
  explicit_recipe["augmentation"]["copypaste"] = .07504;
  auto configured =
      t::parse_recipe(explicit_recipe.dump(), std::filesystem::current_path());
  EXPECT_NE(t::recipe_identity(configured), t::recipe_identity(recipe));
  EXPECT_TRUE(configured.data.yolo_original.enabled);
  EXPECT_EQ(t::recipe_json(t::parse_recipe(t::recipe_json(configured),
                                           std::filesystem::current_path())),
            t::recipe_json(configured));
  explicit_recipe["yolo_original_data"]["unknown"] = true;
  EXPECT_THROW(
      t::parse_recipe(explicit_recipe.dump(), std::filesystem::current_path()),
      InvalidArgument);
}
}  // namespace
}  // namespace jyppx::liuding::vision::tests
