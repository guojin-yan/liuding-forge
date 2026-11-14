// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/nn/recompute.hpp>
#include <jyppx/liuding/vision/ops/assignment.hpp>
#include <jyppx/liuding/vision/ops/boxes.hpp>
#include <jyppx/liuding/vision/ops/deformable.hpp>
#include <jyppx/liuding/vision/ops/roi.hpp>
#include <jyppx/liuding/vision/ops/rotated.hpp>
#include <jyppx/liuding/vision/tasks/advanced_metrics.hpp>
#include <jyppx/liuding/vision/tasks/data/augmentation.hpp>
#include <jyppx/liuding/vision/tasks/data/composition.hpp>
#include <jyppx/liuding/vision/tasks/tracking/metrics.hpp>
#include <jyppx/liuding/vision/tasks/tracking/tracker.hpp>
#include <jyppx/liuding/vision/tasks/training/advanced.hpp>
#include <jyppx/liuding/vision/tasks/training/strategy.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <set>
#include <stop_token>

namespace jyppx::liuding::vision::tests {
namespace {
namespace ops = vision_ops;
namespace t = tasks;
namespace tr = tasks::tracking;
using I = std::int64_t;

class VisionContracts : public ::testing::Test {
 protected:
  void SetUp() override { torch::set_num_threads(1); }
};

TEST_F(VisionContracts,
       DisabledCompositionPreservesDenseTargetsAndChecksLimits) {
  namespace im = data::image;
  im::VisionSample sample;
  sample.image = torch::full({3, 4, 4}, 17, torch::kUInt8);
  sample.width = sample.height = 4;
  sample.mask_target.emplace();
  sample.mask_target->ids = torch::arange(16, torch::kInt64).view({4, 4});
  sample.mask_target->width = sample.mask_target->height = 4;
  const im::SampleRandom random({.dataset_id = "dense/v1",
                                 .sample_id = "sample",
                                 .transform_id = "composition/v1"});
  t::MultiSourceOptions options;
  const auto apply = [&](std::stop_token stop = {}) {
    return t::augment_multisource(sample, 0, {}, options, {}, random, 0, stop);
  };
  for (bool paired : {false, true}) {
    if (paired) sample.paired_target = torch::ones({3, 4, 4}, torch::kUInt8);
    for (bool closed : {false, true}) {
      options.mosaic_probability = closed ? 1 : 0;
      options.close_epochs = closed ? options.total_epochs : 0;
      const auto result = apply();
      EXPECT_EQ(result.image.data_ptr(), sample.image.data_ptr());
      ASSERT_TRUE(result.mask_target);
      EXPECT_TRUE(
          torch::equal(result.mask_target->ids, sample.mask_target->ids));
      if (paired) {
        ASSERT_TRUE(result.paired_target);
        EXPECT_TRUE(torch::equal(*result.paired_target, *sample.paired_target));
      }
    }
  }
  options.close_epochs = 0;
  EXPECT_THROW(apply(),
               InvalidArgument);  // Active mixing rejects dense targets.
  options.mosaic_probability = 0;
  options.max_pixels = 15;
  EXPECT_THROW(apply(), ResourceExhausted);
  options.max_pixels = 16;
  options.total_epochs = 0;
  EXPECT_THROW(apply(), InvalidArgument);
  options.total_epochs = 1;
  std::stop_source stopped;
  stopped.request_stop();
  EXPECT_THROW(apply(stopped.get_token()), Cancelled);
}

// Enumerate every injective row/column selection. This intentionally shares
// neither the Hungarian algorithm nor its padding with the implementation.
double BruteAssignment(const Tensor& costs, std::optional<double> limit) {
  const auto rows = costs.size(0), cols = costs.size(1);
  double best = std::numeric_limits<double>::infinity();
  std::vector<bool> used(cols);
  std::function<void(I, I, double)> visit = [&](I row, I matched, double sum) {
    if (row == rows) {
      if (limit)
        sum += (rows + cols - 2 * matched) * *limit / 2;
      else if (matched != (std::min)(rows, cols))
        return;
      best = (std::min)(best, sum);
      return;
    }
    visit(row + 1, matched, sum);
    for (I col = 0; col < cols; ++col) {
      const auto cost = costs[row][col].item<double>();
      if (used[col] || (limit && cost > *limit)) continue;
      used[col] = true;
      visit(row + 1, matched + 1, sum + cost);
      used[col] = false;
    }
  };
  visit(0, 0, 0);
  return best;
}

TEST_F(VisionContracts, AssignmentMatchesExhaustiveRectangularOptima) {
  for (I rows = 1; rows <= 4; ++rows) {
    for (I cols = 1; cols <= 4; ++cols) {
      for (I seed = 0; seed < 4; ++seed) {
        std::vector<double> values;
        for (I i = 0; i < rows * cols; ++i)
          values.push_back(((i * 13 + seed * 7 + i * i) % 19 - 7) / 8.);
        const auto costs =
            torch::tensor(values, torch::kFloat64).view({rows, cols});
        const auto original = costs.clone();
        for (const auto limit :
             {std::optional<double>{}, std::optional<double>{.5}}) {
          const auto matched =
              ops::linear_assignment(costs, {.cost_limit = limit});
          std::set<I> seen_rows, seen_cols;
          double actual = 0;
          for (I i = 0; i < matched.rows.numel(); ++i) {
            const auto row = matched.rows[i].item<I>();
            const auto col = matched.columns[i].item<I>();
            ASSERT_TRUE(seen_rows.insert(row).second);
            ASSERT_TRUE(seen_cols.insert(col).second);
            const auto cost = costs[row][col].item<double>();
            if (limit) EXPECT_LE(cost, *limit);
            actual += cost;
          }
          if (limit)
            actual += (rows + cols - 2 * matched.rows.numel()) * *limit / 2;
          EXPECT_NEAR(actual, BruteAssignment(costs, limit), 1e-12)
              << "rows=" << rows << " cols=" << cols << " seed=" << seed;
        }
        EXPECT_TRUE(torch::equal(costs, original));
      }
    }
  }
}

TEST_F(VisionContracts, AssignmentEnforcesCancellationAndWorkspaceBudgets) {
  auto costs = torch::zeros({2, 3}, torch::kFloat64);
  EXPECT_THROW(ops::linear_assignment(costs, {.max_pairs = 5}),
               ResourceExhausted);
  EXPECT_THROW(ops::linear_assignment(costs, {.max_work = 1}),
               ResourceExhausted);
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(ops::linear_assignment(costs, {}, stop.get_token()), Cancelled);
  EXPECT_EQ(ops::linear_assignment(torch::empty({0, 3})).rows.numel(), 0);
  costs[0][0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(ops::linear_assignment(costs), InvalidArgument);
}

TEST_F(VisionContracts, NmsUsesContinuousAreaAndStableTies) {
  const auto boxes = torch::tensor({0., 0., 2., 2., 0., 0., 2., 2., 1., 1., 3.,
                                    3., 5., 5., 5., 5.})
                         .view({4, 4});
  const auto keep = ops::nms(boxes, torch::ones({4}), {.iou_threshold = .5});
  EXPECT_TRUE(torch::equal(keep, torch::tensor({0, 2, 3}, torch::kInt64)));
  EXPECT_NEAR(ops::pairwise_box_iou(boxes, boxes)[0][2].item<double>(), 1. / 7,
              1e-7);
  EXPECT_THROW(ops::pairwise_box_iou(boxes, boxes, {.max_pairs = 15}),
               ResourceExhausted);
  auto centers =
      ops::convert_boxes(boxes, ops::BoxFormat::xyxy, ops::BoxFormat::cxcywh);
  EXPECT_TRUE(torch::equal(
      ops::convert_boxes(centers, ops::BoxFormat::cxcywh, ops::BoxFormat::xyxy),
      boxes));
}

TEST_F(VisionContracts, RotatedIntersectionMatchesAnalyticCross) {
  const auto a =
      torch::tensor({0., 0., 4., 2., 0.}, torch::kFloat64).view({1, 5});
  const auto b =
      torch::tensor({0., 0., 4., 2., std::acos(-1.) / 2}, torch::kFloat64)
          .view({1, 5});
  EXPECT_NEAR(ops::pairwise_rotated_iou(a, b).item<double>(), 1. / 3, 1e-12);
  EXPECT_NEAR(ops::pairwise_rotated_iou(a, a).item<double>(), 1., 1e-12);
}

TEST_F(VisionContracts, DeformableSamplingHasAnalyticValueAndGradients) {
  auto value = torch::tensor({1., 2., 3., 4.}, torch::kFloat64)
                   .view({1, 4, 1, 1})
                   .requires_grad_(true);
  auto locations =
      torch::full({1, 1, 1, 1, 1, 2}, .5, torch::kFloat64).requires_grad_(true);
  auto weights =
      torch::ones({1, 1, 1, 1, 1}, torch::kFloat64).requires_grad_(true);
  const std::array<std::array<I, 2>, 1> shapes{{{2, 2}}};
  auto result =
      ops::multiscale_deformable_attention(value, shapes, locations, weights);
  EXPECT_DOUBLE_EQ(result.item<double>(), 2.5);
  result.sum().backward();
  EXPECT_TRUE(
      torch::allclose(value.grad(), torch::full_like(value, .25), 0, 1e-12));
  EXPECT_TRUE(torch::allclose(locations.grad().reshape({2}),
                              torch::tensor({2., 4.}, torch::kFloat64), 0,
                              1e-12));
  EXPECT_DOUBLE_EQ(weights.grad().item<double>(), 2.5);
  EXPECT_THROW(ops::multiscale_deformable_attention(value, shapes, locations,
                                                    weights, 1),
               ResourceExhausted);
}

TEST_F(VisionContracts, RoiOperatorsMatchPlaneAndDiscreteMaxima) {
  auto feature = torch::arange(16, torch::kFloat64)
                     .view({1, 1, 4, 4})
                     .requires_grad_(true);
  const auto rois =
      torch::tensor({0., 0., 0., 3., 3.}, torch::kFloat64).view({1, 5});
  auto aligned = ops::roi_align(feature, rois, {.output_size = {1, 1}});
  EXPECT_NEAR(aligned.item<double>(), 7.5, 1e-12);
  aligned.sum().backward();
  EXPECT_NEAR(feature.grad().sum().item<double>(), 1., 1e-12);
  const auto pooled = ops::roi_pool(feature.detach(), rois, {2, 2});
  EXPECT_TRUE(torch::equal(pooled.reshape({4}),
                           torch::tensor({5., 7., 13., 15.}, torch::kFloat64)));
}

TEST_F(VisionContracts, BoxCoderRoundTripsNonUnitWeights) {
  const auto anchors =
      torch::tensor({1., 2., 6., 10., 0., 1., 2., 3.}).view({2, 4});
  const auto targets =
      torch::tensor({2., 4., 10., 13., 1., 0., 5., 7.}).view({2, 4});
  const std::array<double, 4> weights{10, 10, 5, 5};
  const auto encoded = ops::encode_box_deltas(anchors, targets, weights);
  EXPECT_TRUE(torch::allclose(ops::decode_box_deltas(anchors, encoded, weights),
                              targets, 1e-6, 1e-6));
}

TEST_F(VisionContracts, MultilabelUnknownsHaveNoLossOrGradient) {
  auto logits = torch::tensor({0., 0., 100.}).view({1, 3}).requires_grad_(true);
  const auto labels = torch::tensor({1., 0., 1.}).view({1, 3});
  const auto known =
      torch::tensor({true, true, false}, torch::kBool).view({1, 3});
  auto loss = t::multilabel_loss(logits, labels, known);
  EXPECT_NEAR(loss.item<double>(), std::log(2.), 1e-7);
  loss.backward();
  EXPECT_TRUE(
      torch::equal(logits.grad().reshape({3}), torch::tensor({-.25, .25, 0.})));
  EXPECT_THROW(t::multilabel_loss(logits, labels, torch::zeros_like(known)),
               InvalidArgument);
}

TEST_F(VisionContracts, AsymmetricZeroFocusingReducesToBce) {
  const auto logits = torch::tensor({-2., 1., .5, 3.}).view({2, 2});
  const auto labels = torch::tensor({1., 0., 0., 1.}).view({2, 2});
  const auto known = torch::ones({2, 2}, torch::kBool);
  EXPECT_NEAR(t::multilabel_loss(logits, labels, known).item<double>(),
              t::multilabel_loss(
                  logits, labels, known,
                  {.asymmetric = true, .gamma_negative = 0, .negative_clip = 0})
                  .item<double>(),
              1e-6);
}

TEST_F(VisionContracts, DistillationDetachesTeacherAndUsesTemperatureSquared) {
  auto student = torch::zeros({1, 2}).requires_grad_(true);
  auto teacher =
      torch::tensor({0., std::log(3.) * 2}).view({1, 2}).requires_grad_(true);
  auto loss = t::logits_distillation_loss(student, teacher, 2.);
  const auto expected = 4 * (.25 * std::log(.5) + .75 * std::log(1.5));
  EXPECT_NEAR(loss.item<double>(), expected, 1e-6);
  loss.backward();
  EXPECT_TRUE(torch::allclose(
      student.grad(), torch::tensor({.5, -.5}).view({1, 2}), 1e-6, 1e-6));
  EXPECT_FALSE(teacher.grad().defined());
}

TEST_F(VisionContracts, SemanticIgnorePixelsHaveZeroGradient) {
  auto logits = torch::zeros({1, 2, 1, 2}).requires_grad_(true);
  const auto labels = torch::tensor({0, 255}, torch::kInt64).view({1, 1, 2});
  auto loss = t::semantic_segmentation_loss(logits, labels);
  EXPECT_NEAR(loss.item<double>(), std::log(2.), 1e-7);
  loss.backward();
  EXPECT_DOUBLE_EQ(logits.grad().select(3, 1).abs().sum().item<double>(), 0);
  EXPECT_THROW(
      t::semantic_segmentation_loss(logits, torch::full_like(labels, 255)),
      InvalidArgument);
}

TEST_F(VisionContracts, DepthMasksExcludeNanBeforeArithmetic) {
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  auto prediction =
      torch::tensor({2.f, nan}).view({1, 1, 1, 2}).requires_grad_(true);
  const auto target = torch::tensor({1.f, nan}).view({1, 1, 1, 2});
  const auto valid =
      torch::tensor({true, false}, torch::kBool).view({1, 1, 1, 2});
  auto loss = t::depth_loss(prediction, target, valid);
  EXPECT_NEAR(loss.item<double>(), std::log(2.), 1e-7);
  loss.backward();
  EXPECT_TRUE(torch::isfinite(prediction.grad()).all().item<bool>());
  EXPECT_DOUBLE_EQ(prediction.grad()[0][0][0][0].item<double>(), .5);
  EXPECT_DOUBLE_EQ(prediction.grad()[0][0][0][1].item<double>(), 0);
}

TEST_F(VisionContracts, RelativeDepthRemovesScaleAndShift) {
  const auto prediction = torch::tensor({1., 2., 3.}).view({1, 1, 1, 3});
  const auto target = torch::tensor({5., 7., 9.}).view({1, 1, 1, 3});
  EXPECT_NEAR(
      t::depth_loss(prediction, target, torch::ones_like(target, torch::kBool),
                    {.scale_shift_invariant = true})
          .item<double>(),
      0., 1e-6);
}

TEST_F(VisionContracts, SimmimMaskPreservesBlockCountAndReplay) {
  const auto mask = t::simmim_mask({64, 64}, 4, 32, .5, 17);
  EXPECT_EQ(mask.sum().item<I>(), 128);
  EXPECT_TRUE(torch::equal(mask, t::simmim_mask({64, 64}, 4, 32, .5, 17)));
  for (I y = 0; y < 16; y += 8)
    for (I x = 0; x < 16; x += 8) {
      const auto block = mask.slice(0, y, y + 8).slice(1, x, x + 8);
      EXPECT_TRUE(block.eq(block[0][0]).all().item<bool>());
    }
  EXPECT_THROW(t::simmim_mask({63, 64}, 4, 32, .5, 17), InvalidArgument);
}

TEST_F(VisionContracts, SimmimReconstructionOnlySupervisesMaskedPixels) {
  auto reconstructed = torch::tensor({1., 100., 3., 100.})
                           .view({1, 1, 2, 2})
                           .requires_grad_(true);
  const auto target = torch::zeros_like(reconstructed);
  const auto mask =
      torch::tensor({true, false, true, false}, torch::kBool).view({1, 2, 2});
  auto loss = t::simmim_reconstruction_loss(reconstructed, target, mask, 1);
  EXPECT_DOUBLE_EQ(loss.item<double>(), 2.);
  loss.backward();
  EXPECT_TRUE(torch::equal(reconstructed.grad().reshape({4}),
                           torch::tensor({.5, 0., .5, 0.})));
}

TEST_F(VisionContracts, MultilabelApGroupsTiesAndIgnoresUnknowns) {
  const auto scores = torch::tensor({.5, .9, .5, .8, .1, .99}).view({3, 2});
  const auto labels = torch::tensor({1., 0., 0., 1., 1., 1.}).view({3, 2});
  const auto known =
      torch::tensor({true, true, true, true, true, false}, torch::kBool)
          .view({3, 2});
  const auto result = t::multilabel_metrics(scores, labels, known);
  // Class 0 AP = .5/2 + (2/3)/2 = 7/12; class 1 AP = 1/2.
  EXPECT_NEAR(result.at("map"), 13. / 24, 1e-12);
  EXPECT_NEAR(result.at("micro_f1"), 4. / 7, 1e-12);
  EXPECT_NEAR(result.at("macro_f1"), 7. / 12, 1e-12);
  EXPECT_THROW(t::calibrate_multilabel_thresholds(scores, labels, known, ""),
               InvalidArgument);
}

TEST_F(VisionContracts, RetrievalExcludesSelfAndSameCameraPositives) {
  const auto features =
      torch::tensor({1., 0., 1., 0., 0., 1., 0., 1.}).view({4, 2});
  const auto ids = torch::tensor({0, 0, 1, 1}, torch::kInt64);
  const auto result =
      t::reid_retrieval_metrics(features, ids, {"a", "b", "a", "b"});
  EXPECT_DOUBLE_EQ(result.at("map"), 1.);
  EXPECT_DOUBLE_EQ(result.at("rank1"), 1.);
  EXPECT_DOUBLE_EQ(result.at("queries"), 4.);
  EXPECT_THROW(t::reid_retrieval_metrics(features, ids, {"a", "a", "a", "a"}),
               InvalidArgument);
}

t::ImageResult Detection(double score = .9) {
  t::ImageResult result;
  result.boxes = torch::tensor({0., 0., 20., 20.}).view({1, 4});
  result.scores = torch::tensor({score}, torch::kFloat32);
  result.classes = torch::zeros({1}, torch::kInt64);
  return result;
}

TEST_F(VisionContracts, ByteTrackRetainsIdentityThroughLowConfidenceAndResume) {
  tr::Tracker tracker;
  tracker.reset("sequence");
  const auto first = tracker.step(Detection(), 0);
  ASSERT_EQ(first.size(), 1);
  const auto low = tracker.step(Detection(.2), 1);
  ASSERT_EQ(low.size(), 1);
  EXPECT_EQ(low.front().track_id, first.front().track_id);
  const auto snapshot = tracker.snapshot();
  tr::Tracker restored;
  restored.restore(snapshot);
  const auto a = tracker.step(Detection(), 2),
             b = restored.step(Detection(), 2);
  ASSERT_EQ(a.size(), 1);
  ASSERT_EQ(b.size(), 1);
  EXPECT_EQ(a.front().track_id, b.front().track_id);
  EXPECT_EQ(tracker.snapshot(), restored.snapshot());
}

TEST_F(VisionContracts, TrackerFailedFrameAndRestoreAreTransactional) {
  tr::Tracker tracker;
  tracker.reset("sequence");
  tracker.step(Detection(), 0);
  const auto before = tracker.snapshot();
  EXPECT_THROW(tracker.step(Detection(), 0), InvalidArgument);
  EXPECT_THROW(tracker.step(Detection(1.1), 1), InvalidArgument);
  EXPECT_EQ(tracker.snapshot(), before);
  tr::Tracker other({.track_buffer = 1});
  other.reset("other");
  const auto other_before = other.snapshot();
  EXPECT_THROW(other.restore(before), CompatibilityError);
  EXPECT_EQ(other.snapshot(), other_before);
}

TEST_F(VisionContracts, TrackingMetricsMeasureIdentitySwitch) {
  tr::TrackResult gt{.track_id = 1, .box = {0, 0, 10, 10}, .confidence = 1};
  auto p1 = gt, p2 = gt;
  p1.track_id = 10;
  p2.track_id = 20;
  const std::array<tr::TrackingFrame, 2> frames{
      {{0, {gt}, {p1}}, {1, {gt}, {p2}}}};
  const auto metrics = tr::evaluate_tracking(frames);
  EXPECT_DOUBLE_EQ(metrics.at("mota"), .5);
  EXPECT_DOUBLE_EQ(metrics.at("idf1"), .5);
  EXPECT_DOUBLE_EQ(metrics.at("id_switches"), 1.);
  EXPECT_NEAR(metrics.at("hota"), std::sqrt(.5), 1e-12);
}

TEST_F(VisionContracts, RecomputePreservesDropoutGradientsAndCallerRng) {
  torch::manual_seed(29);
  auto region = std::make_shared<torch::nn::SequentialImpl>();
  region->push_back(torch::nn::Linear(4, 5));
  region->push_back(torch::nn::Dropout(.4));
  region->push_back(torch::nn::Linear(5, 2));
  const auto x = torch::randn({3, 4});
  torch::manual_seed(73);
  auto expected = region->forward(x);
  expected.square().mean().backward();
  std::vector<Tensor> gradients;
  for (auto& parameter : region->parameters()) {
    gradients.push_back(parameter.grad().clone());
    parameter.mutable_grad() = Tensor{};
  }
  const auto next = torch::randn({4});
  torch::manual_seed(73);
  auto actual =
      nn::recompute(x, region, [](torch::nn::Module& m, const Tensor& input) {
        return dynamic_cast<torch::nn::SequentialImpl&>(m).forward(input);
      });
  actual.square().mean().backward();
  EXPECT_TRUE(torch::equal(expected, actual));
  EXPECT_TRUE(torch::equal(next, torch::randn({4})));
  std::size_t index = 0;
  for (const auto& parameter : region->parameters())
    EXPECT_TRUE(
        torch::allclose(parameter.grad(), gradients[index++], 1e-5, 1e-6));
}

TEST_F(VisionContracts, WarmupCosineUsesSuccessfulUpdateBoundaries) {
  const t::WarmupCosineOptions options{.total_updates = 10,
                                       .warmup_updates = 2,
                                       .warmup_start_factor = .1,
                                       .final_factor = .01};
  EXPECT_DOUBLE_EQ(t::warmup_cosine_factor(options, 0), .1);
  EXPECT_DOUBLE_EQ(t::warmup_cosine_factor(options, 1), .55);
  EXPECT_DOUBLE_EQ(t::warmup_cosine_factor(options, 2), 1.);
  EXPECT_NEAR(t::warmup_cosine_factor(options, 6), .505, 1e-12);
  EXPECT_DOUBLE_EQ(t::warmup_cosine_factor(options, 10), .01);
  EXPECT_DOUBLE_EQ(t::warmup_cosine_factor(options, 100), .01);
}

data::image::VisionSample ColorBars() {
  data::image::VisionSample sample;
  sample.width = 3;
  sample.height = 1;
  sample.image_path = "color-bars";
  sample.image = torch::tensor({255, 0, 0, 0, 255, 0, 0, 0, 255}, torch::kUInt8)
                     .view({3, 1, 3});
  return sample;
}

data::image::SampleRandom PhotoRandom(const t::PhotometricOptions& options,
                                      std::uint64_t epoch = 0) {
  return data::image::SampleRandom(
      {.dataset_id = "fixture",
       .sample_id = "color-bars",
       .transform_id = t::photometric_identity(options),
       .seed = 31,
       .epoch = epoch});
}

TEST_F(VisionContracts, PhotometricGrayscaleMatchesBt601AndPreservesSource) {
  const auto sample = ColorBars();
  const auto before = sample.image.clone();
  const t::PhotometricOptions options{.grayscale_probability = 1};
  const auto result =
      t::augment_photometric(sample, options, PhotoRandom(options));
  const auto expected = torch::tensor({76, 150, 29}, torch::kUInt8)
                            .view({1, 1, 3})
                            .expand({3, 1, 3});
  EXPECT_TRUE(torch::equal(result.image, expected));
  EXPECT_TRUE(torch::equal(sample.image, before));
}

TEST_F(VisionContracts, PhotometricReplayDoesNotConsumeGlobalTorchRng) {
  const auto sample = ColorBars();
  const t::PhotometricOptions options{
      .hue_shift = .2, .saturation_gain = .4, .value_gain = .4};
  torch::manual_seed(37);
  const auto expected_rng = torch::randn({8});
  torch::manual_seed(37);
  const auto a = t::augment_photometric(sample, options, PhotoRandom(options));
  const auto b = t::augment_photometric(sample, options, PhotoRandom(options));
  EXPECT_TRUE(torch::equal(a.image, b.image));
  EXPECT_TRUE(torch::equal(expected_rng, torch::randn({8})));
  EXPECT_NE(PhotoRandom(options).identity(),
            PhotoRandom(options, 1).identity());
}

TEST_F(VisionContracts, PhotometricEnforcesCancellationAndWorkLimits) {
  const auto sample = ColorBars();
  t::PhotometricOptions options{.blur_probability = 1};
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(t::augment_photometric(sample, options, PhotoRandom(options),
                                      stop.get_token()),
               Cancelled);
  options.max_filter_operations = 1;
  EXPECT_THROW(t::augment_photometric(sample, options, PhotoRandom(options)),
               ResourceExhausted);
}

TEST_F(VisionContracts, CatalogResolvesEveryEntryWithoutImplicitAliases) {
  std::set<std::string> names;
  for (const auto& entry : t::vision_catalog()) {
    const auto identity = entry.architecture + "/" +
                          std::string(t::task_name(entry.task)) + "/" +
                          entry.variant;
    ASSERT_TRUE(names.insert(identity).second) << identity;
    t::ModelSpec spec;
    spec.architecture = entry.architecture;
    spec.task = entry.task;
    spec.variant = entry.variant;
    EXPECT_EQ(t::describe_binding(spec).revision, entry.revision);
    EXPECT_FALSE(entry.revision.empty());
    EXPECT_EQ(t::parse_task(t::task_name(entry.task)), entry.task);
    EXPECT_FALSE(entry.architecture.starts_with("yolov6"));
    EXPECT_FALSE(entry.architecture.starts_with("yolov7"));
    EXPECT_FALSE(entry.architecture.starts_with("yolov9"));
    EXPECT_FALSE(entry.architecture.starts_with("yolov10"));
  }
  EXPECT_GE(names.size(), 200);
}

TEST_F(VisionContracts, SelectionMetricsFollowTaskAndDepthAlignment) {
  t::RecipeSpec recipe;
  recipe.schema_version = 2;
  recipe.extension.training_manifest = "__fixture__";
  recipe.extension.validation_manifest = "__validation__";
  recipe.full_validation_metrics = true;
  const std::vector<std::pair<t::TaskKind, std::vector<std::string>>> cases{
      {t::TaskKind::classify, {"validation.top1", "validation.top5"}},
      {t::TaskKind::classify_multilabel,
       {"validation.map", "validation.micro_f1", "validation.macro_f1"}},
      {t::TaskKind::detect, {"validation.map", "validation.ap50"}},
      {t::TaskKind::semantic_segment,
       {"validation.miou", "validation.pixel_accuracy"}},
      {t::TaskKind::depth,
       {"validation.rmse", "validation.mae", "validation.absrel",
        "validation.delta1"}},
      {t::TaskKind::reid, {"validation.rank1", "validation.map"}},
      {t::TaskKind::pretrain_simmim, {"validation.loss"}}};
  for (const auto& [task, metrics] : cases) {
    recipe.model.task = task;
    for (const auto& metric : metrics) {
      recipe.selection_metric = metric;
      EXPECT_NO_THROW(t::check_recipe(recipe)) << metric;
    }
    recipe.selection_metric = "validation.unknown";
    EXPECT_THROW(t::check_recipe(recipe), InvalidArgument);
  }
  recipe.model.task = t::TaskKind::semantic_segment;
  recipe.selection_metric = "validation.top1";
  EXPECT_THROW(t::check_recipe(recipe), InvalidArgument);
  recipe.model.task = t::TaskKind::depth;
  recipe.extension.relative_inverse_depth = true;
  recipe.selection_metric = "validation.absrel";
  EXPECT_THROW(t::check_recipe(recipe), InvalidArgument);
  recipe.selection_metric = "validation.rmse";
  EXPECT_NO_THROW(t::check_recipe(recipe));
  recipe.full_validation_metrics = false;
  EXPECT_THROW(t::check_recipe(recipe), InvalidArgument);
}

TEST_F(VisionContracts, ExecutionPolicyRoundTripsAndChangesResumeIdentity) {
  t::RecipeSpec recipe;
  recipe.schema_version = 2;
  recipe.extension.training_manifest =
      std::filesystem::absolute("fixture.json");
  ASSERT_TRUE(recipe.deterministic);
  EXPECT_DOUBLE_EQ(recipe.initial_loss_scale, 65536.);
  const auto identity = t::recipe_identity(recipe);
  recipe.deterministic = false;
  EXPECT_NE(t::recipe_identity(recipe), identity);
  const auto non_deterministic = t::recipe_identity(recipe);
  recipe.initial_loss_scale = 1.;
  EXPECT_NE(t::recipe_identity(recipe), non_deterministic);
  const auto restored =
      t::parse_recipe(t::recipe_json(recipe), std::filesystem::current_path());
  EXPECT_FALSE(restored.deterministic);
  EXPECT_DOUBLE_EQ(restored.initial_loss_scale, 1.);
  for (double invalid : {0., -1., std::numeric_limits<double>::infinity()}) {
    recipe.initial_loss_scale = invalid;
    EXPECT_THROW(t::check_recipe(recipe), InvalidArgument);
  }
}

}  // namespace
}  // namespace jyppx::liuding::vision::tests
