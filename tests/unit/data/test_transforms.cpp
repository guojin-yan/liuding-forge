// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

/**
 * @file test_transforms.cpp
 * @brief Tests tensor transforms and their current pipeline integration.
 */

#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/dataset.hpp>
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>
#include <jyppx/liuding/data/tensor_dataset.hpp>
#include <jyppx/liuding/data/transforms/compose.hpp>
#include <jyppx/liuding/data/transforms/transform.hpp>
#include <jyppx/liuding/data/transforms/vision.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <span>
#include <stop_token>
#include <vector>

namespace {

using namespace jyppx::liuding;
using namespace jyppx::liuding::data::transforms;

// Records the order it is applied in; used to verify Compose chaining.
class AppendTransform : public TensorTransform {
 public:
  explicit AppendTransform(std::vector<std::string>& log, std::string tag)
      : log_(log), tag_(std::move(tag)) {}

  Tensor operator()(const Tensor& input) const override {
    log_.push_back(tag_);
    return input + 1.0;
  }
  std::string name() const override { return tag_; }

 private:
  std::vector<std::string>& log_;
  std::string tag_;
};

class TransformsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Deterministic 2x3 "image", channel-major {C=2, H=2, W=3}.
    image_ = torch::arange(12, torch::TensorOptions().dtype(torch::kFloat32))
                 .view({2, 2, 3});
  }

  Tensor image_;
};

TEST_F(TransformsTest, ComposeAppliesTransformsInOrder) {
  std::vector<std::string> log;
  Compose compose({
      std::make_shared<AppendTransform>(log, "first"),
      std::make_shared<AppendTransform>(log, "second"),
      std::make_shared<AppendTransform>(log, "third"),
  });

  const auto result = compose(torch::zeros({1}));
  EXPECT_EQ(log, (std::vector<std::string>{"first", "second", "third"}));
  EXPECT_TRUE(torch::equal(result, torch::full({1}, 3.0f)));
  EXPECT_EQ(compose.name(), "Compose(first, second, third)");
  EXPECT_EQ(compose.size(), 3);
}

TEST_F(TransformsTest, ComposeRejectsNullTransforms) {
  try {
    (void)Compose({nullptr});
    FAIL() << "a null transform must be rejected";
  } catch (const jyppx::liuding::InvalidArgument& error) {
    EXPECT_EQ(error.code(), jyppx::liuding::ErrorCode::invalid_argument);
    EXPECT_EQ(error.details().component, "data.transforms.compose");
    EXPECT_EQ(error.details().operation, "constructor.transforms");
  }
  EXPECT_THROW(RandomChoice({}), jyppx::liuding::InvalidArgument);
}

TEST_F(TransformsTest, NormalizeSubtractsMeanAndDividesStd) {
  const Normalize normalize({1.0, 3.0}, {2.0, 4.0});
  const auto result = normalize(image_);

  // Channel 0: (x - 1) / 2; channel 1: (x - 3) / 4.
  EXPECT_NEAR(result[0][0][0].item<double>(), (0.0 - 1.0) / 2.0, 1e-6);
  EXPECT_NEAR(result[1][1][2].item<double>(), (11.0 - 3.0) / 4.0, 1e-6);
  EXPECT_EQ(result.scalar_type(), at::kFloat);
}

TEST_F(TransformsTest, NormalizePromotesIntegralInputs) {
  auto integral = image_.to(at::kByte);
  const Normalize normalize({0.0, 0.0}, {1.0, 1.0});
  const auto result = normalize(integral);
  EXPECT_EQ(result.scalar_type(), at::kFloat);
}

TEST_F(TransformsTest, NormalizeValidatesArguments) {
  EXPECT_THROW(Normalize({}, {1.0}), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(Normalize({1.0}, {1.0, 2.0}), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(Normalize({1.0}, {0.0}), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(Normalize({1.0, 2.0}, {1.0, 2.0})(torch::ones({1, 2, 2})),
               jyppx::liuding::InvalidArgument);  // channel mismatch
}

TEST_F(TransformsTest, ResizeProducesRequestedShape) {
  const Resize resize({4, 6});
  const auto result = resize(image_);
  EXPECT_EQ(result.sizes(), (std::vector<int64_t>{2, 4, 6}));

  const Resize square({3});
  EXPECT_EQ(square(image_).sizes(), (std::vector<int64_t>{2, 3, 3}));
  EXPECT_EQ(resize.name(), "Resize(4, 6)");
}

TEST_F(TransformsTest, ResizeNearestPreservesExactValues) {
  // Upscaling a constant image keeps the constant under nearest sampling.
  const auto constant = torch::full({1, 2, 2}, 5.0f);
  const Resize resize({4, 4}, "nearest");
  const auto result = resize(constant);
  EXPECT_TRUE(torch::equal(result, torch::full({1, 4, 4}, 5.0f)));
}

TEST_F(TransformsTest, CenterCropTakesTheMiddle) {
  const CenterCrop crop({1, 1});
  const auto result = crop(image_);

  // {2, 2, 3}: top = (2-1)/2 = 0, left = (3-1)/2 = 1 -> element [c][0][1].
  EXPECT_EQ(result.sizes(), (std::vector<int64_t>{2, 1, 1}));
  EXPECT_NEAR(result[0][0][0].item<double>(), 1.0, 1e-6);  // image_[0][0][1]
  EXPECT_NEAR(result[1][0][0].item<double>(), 7.0, 1e-6);  // image_[1][0][1]
  EXPECT_EQ(crop.name(), "CenterCrop(1, 1)");
}

TEST_F(TransformsTest, RandomCropIsDeterministicUnderSeed) {
  torch::manual_seed(42);
  const RandomCrop crop({1, 2});
  const auto first = crop(image_);
  torch::manual_seed(42);
  const auto second = crop(image_);
  EXPECT_TRUE(torch::equal(first, second));

  // The crop result is a sub-block of the input (all values come from it).
  EXPECT_EQ(first.sizes(), (std::vector<int64_t>{2, 1, 2}));
  EXPECT_TRUE(torch::isin(first, image_).all().item<bool>());
}

TEST_F(TransformsTest, RandomCropRejectsOversizedCrops) {
  const RandomCrop crop({4, 4});
  EXPECT_THROW(crop(image_), jyppx::liuding::InvalidArgument);
  const CenterCrop center({4, 4});
  EXPECT_THROW(center(image_), jyppx::liuding::InvalidArgument);
}

TEST_F(TransformsTest, FlipsWithProbabilityOneAndZero) {
  const RandomHorizontalFlip always(1.0);
  EXPECT_TRUE(torch::equal(always(image_), torch::flip(image_, {2})));

  const RandomHorizontalFlip never(0.0);
  EXPECT_TRUE(torch::equal(never(image_), image_));

  const RandomVerticalFlip always_v(1.0);
  EXPECT_TRUE(torch::equal(always_v(image_), torch::flip(image_, {1})));

  EXPECT_THROW(RandomHorizontalFlip(1.5), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(RandomVerticalFlip(-0.1), jyppx::liuding::InvalidArgument);
}

TEST_F(TransformsTest, RejectsNonFiniteRangesAndMalformedImageShapes) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();

  try {
    (void)Pad({1, 2, 3});
    FAIL() << "three-value padding must be rejected";
  } catch (const jyppx::liuding::InvalidArgument& error) {
    EXPECT_EQ(error.code(), jyppx::liuding::ErrorCode::invalid_argument);
    EXPECT_EQ(error.details().component, "data.transforms.vision");
    EXPECT_EQ(error.details().operation, "pad.padding");
  }
  EXPECT_THROW((void)RandomHorizontalFlip{nan}, jyppx::liuding::InvalidArgument);
  EXPECT_THROW((void)RandomRotation{nan}, jyppx::liuding::InvalidArgument);
  EXPECT_THROW((void)RandomAffine{nan}, jyppx::liuding::InvalidArgument);
  EXPECT_THROW((void)ColorJitter{nan}, jyppx::liuding::InvalidArgument);
  EXPECT_THROW(Normalize({nan}, {1.0}), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(Normalize({0.0}, {infinity}), jyppx::liuding::InvalidArgument);

  const auto matrix = torch::ones({2, 2});
  EXPECT_THROW(RandomRotation(10.0)(matrix), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(RandomAffine(10.0)(matrix), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(Resize({4})(Tensor{}), jyppx::liuding::InvalidArgument);
}

TEST_F(TransformsTest, ExecutesTensorNativePipelineOnCudaWhenAvailable) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }

  Compose pipeline({
      std::make_shared<Resize>(std::vector<int64_t>{10, 10}),
      std::make_shared<CenterCrop>(std::vector<int64_t>{8, 8}),
      std::make_shared<RandomHorizontalFlip>(1.0),
      std::make_shared<Normalize>(std::vector<double>{0.5, 0.5, 0.5},
                                  std::vector<double>{0.5, 0.5, 0.5}),
      std::make_shared<RandomRotation>(0.0),
      std::make_shared<Grayscale>(3),
  });
  const auto input = torch::rand(
      {3, 8, 8},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  const auto result = pipeline(input);

  EXPECT_TRUE(result.is_cuda());
  EXPECT_EQ(result.sizes(), (std::vector<int64_t>{3, 8, 8}));
  EXPECT_TRUE(torch::isfinite(result).all().item<bool>());
}

TEST_F(TransformsTest, ColorJitterExecutesEveryEnabledAdjustment) {
  const auto rgb =
      torch::linspace(0.0, 1.0, 48,
                      torch::TensorOptions().dtype(torch::kFloat32))
          .view({3, 4, 4});
  const ColorJitter jitter(/*brightness=*/0.2, /*contrast=*/0.2,
                           /*saturation=*/0.2, /*hue=*/0.1);
  torch::manual_seed(2026);

  const auto result = jitter(rgb);

  EXPECT_EQ(result.sizes(), rgb.sizes());
  EXPECT_EQ(result.scalar_type(), torch::kFloat32);
  EXPECT_TRUE(torch::isfinite(result).all().item<bool>());
  EXPECT_FALSE(torch::equal(result, rgb));
  EXPECT_EQ(jitter.name(), "ColorJitter");
}

TEST_F(TransformsTest, RandomChoiceSelectsConfiguredTransforms) {
  // Two transforms produce distinct constants; repeated draws over many
  // calls must only ever yield those two outputs.
  struct Constant : TensorTransform {
    explicit Constant(double v) : value(v) {}
    Tensor operator()(const Tensor&) const override {
      return torch::full({}, value);
    }
    std::string name() const override { return "Constant"; }
    double value;
  };

  const RandomChoice choice({
      std::make_shared<Constant>(1.0),
      std::make_shared<Constant>(2.0),
  });

  torch::manual_seed(0);
  double first = 0.0;
  double second = -1.0;
  bool saw_both = false;
  for (int i = 0; i < 64; ++i) {
    const double value = choice(torch::zeros({})).item<double>();
    EXPECT_TRUE(value == 1.0 || value == 2.0);
    if (i == 0) first = value;
    if (i == 1) second = value;
    if (first != second) saw_both = true;
  }
  EXPECT_TRUE(saw_both);  // both branches are reachable
  EXPECT_EQ(choice.name(), "RandomChoice");
}

TEST_F(TransformsTest, PublishesStableNamesForCurrentVisionTransforms) {
  EXPECT_EQ(RandomCrop({1, 2}).name(), "RandomCrop(1, 2)");
  EXPECT_EQ(RandomVerticalFlip().name(), "RandomVerticalFlip");
  EXPECT_EQ(Pad({1}).name(), "Pad");
  EXPECT_EQ(RandomRotation(10.0).name(), "RandomRotation");
  EXPECT_EQ(RandomAffine(10.0).name(), "RandomAffine");
  EXPECT_EQ(Grayscale().name(), "Grayscale");
  EXPECT_EQ(RandomGrayscale().name(), "RandomGrayscale");
}

TEST_F(TransformsTest, TransformPipelineWithCompose) {
  // The documented pipeline: resize -> center crop -> flip -> normalize.
  Compose pipeline({
      std::make_shared<Resize>(std::vector<int64_t>{8, 8}),
      std::make_shared<CenterCrop>(std::vector<int64_t>{4, 4}),
      std::make_shared<RandomHorizontalFlip>(
          0.0),  // deterministic for the test
      std::make_shared<Normalize>(std::vector<double>{0.5, 0.5},
                                  std::vector<double>{0.5, 0.5}),
  });

  const auto result = pipeline(image_);
  EXPECT_EQ(result.sizes(), (std::vector<int64_t>{2, 4, 4}));
  EXPECT_NEAR(result.mean().item<double>(),
              (image_.mean().item<double>() - 0.5) / 0.5, 0.1);
  EXPECT_EQ(pipeline.name(),
            "Compose(Resize(8, 8), CenterCrop(4, 4), "
            "RandomHorizontalFlip, Normalize)");
}

TEST_F(TransformsTest, TransformsWorkInsideDataPipeline) {
  // A dataset whose get() applies the transform: the standard integration.
  class TransformedDataset : public jyppx::liuding::data::Dataset<jyppx::liuding::data::Example> {
   public:
    TransformedDataset(Tensor data, Tensor targets, Normalize normalize)
        : data_(std::move(data)),
          targets_(std::move(targets)),
          normalize_(std::move(normalize)) {}

    jyppx::liuding::data::Example get(size_t index) const override {
      return {normalize_(data_[index].view({1, 2, 3})).view({6}),
              targets_[index]};
    }
    size_t size() const override { return data_.size(0); }

   private:
    Tensor data_;
    Tensor targets_;
    Normalize normalize_;
  };

  auto dataset = std::make_shared<TransformedDataset>(image_, torch::zeros({2}),
                                                      Normalize({0.0}, {1.0}));
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(
          dataset->size()),
      {.pipeline_id = "data.transforms.pipeline",
       .dataset_version = "test.transforms.dataset/v1",
       .dataset_size = dataset->size(),
       .schema = jyppx::liuding::data::BatchSchema::any("test.transforms.batch", 1),
       .batch_factory_id = "test.transforms.collate/v1",
       .batch_size = 2},
      [dataset](std::span<const std::size_t> indices,
                const jyppx::liuding::data::DataWorkerContext&, std::stop_token) {
        std::vector<Tensor> inputs;
        std::vector<Tensor> targets;
        inputs.reserve(indices.size());
        targets.reserve(indices.size());
        for (const auto index : indices) {
          const auto sample = dataset->get(index);
          inputs.push_back(sample.data);
          targets.push_back(sample.target);
        }
        return c10::ivalue::Tuple::create(
            {torch::stack(inputs), torch::stack(targets)});
      });
  pipeline.set_epoch(0);

  int batches = 0;
  while (const auto batch = pipeline.next()) {
    const auto inputs = batch->value.toTuple()->elements().at(0).toTensor();
    EXPECT_EQ(inputs.size(1), 6);  // Normalized rows keep their shape.
    ++batches;
  }
  EXPECT_EQ(batches, 1);
}

}  // namespace
