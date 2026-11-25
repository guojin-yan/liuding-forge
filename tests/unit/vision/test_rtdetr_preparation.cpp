// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../../../src/jyppx/liuding/vision/tasks/data/rtdetr_preparation_detail.hpp"
#include "rtdetr_data_reference.hpp"

#include <jyppx/liuding/vision/tasks/rtdetr.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>
#include <jyppx/liuding/vision/tasks/workflow/recipe.hpp>

#include <gtest/gtest.h>

namespace jyppx::liuding::vision::tests {
namespace {
namespace t = tasks;
namespace im = data::image;
namespace ref = rtdetr_data_reference;
template <std::size_t N>
Tensor Bytes(const std::array<std::uint8_t, N>& a, int h, int w) {
  return torch::tensor(std::vector<std::uint8_t>(a.begin(), a.end()),
                       torch::kUInt8)
      .view({h, w, 3})
      .permute({2, 0, 1})
      .contiguous();
}
im::VisionSample Sample() {
  im::VisionSample s;
  s.image = Bytes(ref::pixels, 8, 12);
  s.width = 12;
  s.height = 8;
  s.image_path = "reference";
  s.geometry.emplace(im::ImageExtent{12, 8});
  s.categories = std::make_shared<const std::vector<im::VisionCategory>>(
      std::vector<im::VisionCategory>{{.source_id = 1, .name = "object"}});
  auto a = std::make_shared<im::VisionAnnotationRecord>();
  a->width = 12;
  a->height = 8;
  a->image_path = s.image_path;
  a->objects = {
      {.source_id = 1, .category = 0, .box = im::ImageBox{1, 1, 7, 6}},
      {.source_id = 2, .category = 0, .box = im::ImageBox{8, 2, 11, 7}}};
  s.annotation = a;
  return s;
}
im::SampleRandom Random(const t::RtDetrDataOptions& o, std::uint64_t seed = 0) {
  return im::SampleRandom({.dataset_id = "fixture",
                           .sample_id = "reference",
                           .transform_id = t::rtdetr_data_identity(o),
                           .seed = seed});
}
TEST(RtDetrData, FixedPhotometricOperationsMatchIndependentPilBytes) {
  torch::set_num_threads(1);
  const auto s = Sample();
  const auto before = s.image.clone();
  for (int i = 0; i < 4; ++i) {
    SCOPED_TRACE(i);
    t::detail::RtDetrSamplePlan p;
    Tensor expected;
    if (i == 0) {
      p.brightness = 1.1;
      expected = Bytes(ref::brightness, 8, 12);
    }
    if (i == 1) {
      p.contrast = .65;
      expected = Bytes(ref::contrast, 8, 12);
    }
    if (i == 2) {
      p.saturation = 1.25;
      expected = Bytes(ref::saturation, 8, 12);
    }
    if (i == 3) {
      p.hue = -.035;
      expected = Bytes(ref::hue, 8, 12);
    }
    const auto out = t::detail::apply_rtdetr_plan(s, {12, 8}, p, {});
    EXPECT_TRUE(torch::equal(out.image, expected))
        << "maximum error "
        << (out.image.to(torch::kInt32) - expected.to(torch::kInt32))
               .abs()
               .max()
               .item<int>();
  }
  EXPECT_TRUE(torch::equal(s.image, before));
}
TEST(RtDetrData, FixedGeometryMatchesPilStretchPixelsAndOriginalBoxMapping) {
  const auto s = Sample();
  t::detail::RtDetrSamplePlan p;
  p.padding = {3, 2, 5, 4};
  p.crop = im::ImageGeometryStep{.kind = im::ImageGeometryKind::crop,
                                 .extent = {17, 10},
                                 .left = 2,
                                 .top = 1};
  p.horizontal_flip = true;
  const auto out = t::detail::apply_rtdetr_plan(s, {32, 32}, p, {});
  EXPECT_TRUE(torch::equal(out.image, Bytes(ref::geometry_pixels, 32, 32)));
  ASSERT_TRUE(out.annotation);
  ASSERT_EQ(out.annotation->objects.size(), 2U);
  for (std::size_t i = 0; i < 2; ++i) {
    const auto b = *out.annotation->objects[i].box;
    const std::array actual{b.x_min, b.y_min, b.x_max, b.y_max};
    for (int j = 0; j < 4; ++j)
      EXPECT_NEAR(actual[j], ref::geometry_boxes[i * 4 + j], 2e-6);
    const auto inverse = out.geometry->inverse_box(b);
    ASSERT_TRUE(inverse);
    EXPECT_NEAR(inverse->x_min, s.annotation->objects[i].box->x_min, 1e-10);
  }
}
TEST(RtDetrData, WideColorPhotometryMatchesIndependentPilBytes) {
  auto s = Sample();
  s.image = Bytes(ref::wide_pixels, 16, 16);
  s.width = s.height = 16;
  s.geometry.emplace(im::ImageExtent{16, 16});
  s.annotation.reset();
  for (int i = 0; i < 4; ++i) {
    SCOPED_TRACE(i);
    t::detail::RtDetrSamplePlan p;
    Tensor expected;
    if (i == 0) {
      p.brightness = 1.1;
      expected = Bytes(ref::wide_brightness, 16, 16);
    }
    if (i == 1) {
      p.contrast = .65;
      expected = Bytes(ref::wide_contrast, 16, 16);
    }
    if (i == 2) {
      p.saturation = 1.25;
      expected = Bytes(ref::wide_saturation, 16, 16);
    }
    if (i == 3) {
      p.hue = -.035;
      expected = Bytes(ref::wide_hue, 16, 16);
    }
    const auto out = t::detail::apply_rtdetr_plan(s, {16, 16}, p, {});
    EXPECT_TRUE(torch::equal(out.image, expected))
        << "maximum error "
        << (out.image.to(torch::kInt32) - expected.to(torch::kInt32))
               .abs()
               .max()
               .item<int>();
  }
}
TEST(RtDetrData, IouCropDropsOutsideCentersAndSanitizesSmallBoxes) {
  auto s = Sample();
  t::detail::RtDetrSamplePlan p;
  p.crop = im::ImageGeometryStep{.kind = im::ImageGeometryKind::crop,
                                 .extent = {7, 8}};
  const auto out = t::detail::apply_rtdetr_plan(s, {32, 32}, p, {});
  ASSERT_EQ(out.annotation->objects.size(), 1U);
  EXPECT_EQ(out.annotation->objects[0].source_id, 1);
  auto a = std::make_shared<im::VisionAnnotationRecord>(*s.annotation);
  a->objects[0].box = im::ImageBox{1, 1, 1.5, 6};
  a->objects[1].crowd = true;
  s.annotation = a;
  const auto empty = t::detail::apply_rtdetr_plan(s, {32, 32}, {}, {});
  EXPECT_TRUE(empty.annotation->objects.empty());
}
TEST(RtDetrData, RandomReplayIsWorkerIndependentAndEvaluationHasOnlyStretch) {
  auto s = Sample();
  t::RtDetrDataOptions o;
  o.enabled = true;
  torch::manual_seed(17);
  const auto rng = torch::rand({4});
  torch::manual_seed(17);
  for (std::uint64_t seed = 0; seed < 8; ++seed) {
    const auto a = t::prepare_rtdetr_sample(s, {32, 32}, o, Random(o, seed));
    const auto b = t::prepare_rtdetr_sample(s, {32, 32}, o, Random(o, seed));
    EXPECT_TRUE(torch::equal(a.image, b.image));
    EXPECT_EQ(a.annotation->objects.size(), b.annotation->objects.size());
  }
  EXPECT_TRUE(torch::equal(rng, torch::rand({4})));
  o.training = false;
  EXPECT_TRUE(torch::equal(
      t::prepare_rtdetr_sample(s, {32, 32}, o, Random(o)).image,
      t::prepare_rtdetr_sample(s, {32, 32}, o, Random(o, 34)).image));
  t::YoloDataOptions data;
  data.rtdetr = o;
  data.rtdetr.training = true;
  EXPECT_FALSE(t::yolo_evaluation_options(data).rtdetr.training);
}
TEST(RtDetrData, RejectsRichTargetsCancellationAndAllocationBounds) {
  auto s = Sample();
  t::RtDetrDataOptions o;
  o.enabled = true;
  o.training = false;
  auto a = std::make_shared<im::VisionAnnotationRecord>(*s.annotation);
  a->objects[0].keypoints.push_back({{1, 1}, 2});
  s.annotation = a;
  EXPECT_THROW(t::prepare_rtdetr_sample(s, {32, 32}, o, Random(o)),
               InvalidArgument);
  s = Sample();
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(
      t::prepare_rtdetr_sample(s, {32, 32}, o, Random(o), {}, stop.get_token()),
      Cancelled);
  im::JointTransformOptions g;
  g.limits.max_working_bytes = 1;
  EXPECT_THROW(t::prepare_rtdetr_sample(s, {32, 32}, o, Random(o), g),
               ResourceExhausted);
  auto budget = std::make_shared<im::ImageResourceBudget>(
      im::ImageResourceBudgetLimits{.max_decoded_bytes = 1});
  g = {};
  g.resource_budget = budget;
  EXPECT_THROW(t::prepare_rtdetr_sample(s, {32, 32}, o, Random(o), g),
               ResourceExhausted);
  EXPECT_EQ(budget->usage(im::ImageResourceKind::pixels).current, 0U);
}
TEST(RtDetrData, RecipeBindsExplicitProfileAndRejectsMixedAugmentation) {
  t::RecipeSpec r;
  r.model.architecture = "rtdetr_resnet18_vd_coco72";
  r.training.source = std::filesystem::absolute("ann.json");
  const auto old = t::recipe_json(r), id = t::recipe_identity(r);
  r.data.rtdetr.enabled = true;
  r.rtdetr_training_sizes = {480, 512, 544, 576, 608, 640, 640,
                             640, 672, 704, 736, 768, 800};
  const auto out =
      t::parse_recipe(t::recipe_json(r), std::filesystem::current_path());
  EXPECT_EQ(out.rtdetr_training_sizes, r.rtdetr_training_sizes);
  EXPECT_TRUE(out.data.rtdetr.enabled);
  EXPECT_EQ(t::recipe_identity(r), t::recipe_identity(out));
  EXPECT_NE(id, t::recipe_identity(r));
  r.data.augmentation.horizontal_flip_probability = .5;
  EXPECT_THROW(t::check_recipe(r), InvalidArgument);
  r.data.augmentation.horizontal_flip_probability = 0;
  r.model.architecture = "yolo26n";
  EXPECT_THROW(t::check_recipe(r), InvalidArgument);
  r.model.architecture = "rtdetr_resnet18_vd_coco72";
  r.rtdetr_training_sizes = {33};
  EXPECT_THROW(t::check_recipe(r), InvalidArgument);
  r.rtdetr_training_sizes.clear();
  r.data.rtdetr.enabled = false;
  EXPECT_EQ(t::recipe_json(r), old);
}
TEST(RtDetrData, ModelMultiscaleUsesNearestAndPreservesNormalizedBoxTargets) {
  torch::set_num_threads(1);
  t::RecipeSpec r;
  r.model.architecture = "rtdetr_resnet18_vd_coco72";
  r.model.num_classes = 1;
  r.training.source = "fixture";
  auto binding = t::bind_task(r);
  binding.model->state().module().eval();
  Tensor received_image, received_targets;
  auto wrapped = nn::adapt_module(
      binding.model->state().shared_module(),
      [model = binding.model, &received_image, &received_targets](
          torch::nn::Module&, const nn::Invocation& in) {
        received_image = in.positional[0].toTensor();
        received_targets = in.positional[1].toTensor();
        return model->invoke(in).value;
      },
      binding.model->input_schema(), binding.model->output_schema());
  const auto image = torch::arange(2 * 3 * 256 * 256, torch::kFloat32)
                         .remainder(251)
                         .view({2, 3, 256, 256}) /
                     251;
  const auto target = torch::tensor({0.f, 0.f, 8.f, 16.f, 32.f, 40.f, 1.f, 0.f,
                                     16.f, 8.f, 40.f, 32.f})
                          .view({2, 6});
  const nn::Invocation invocation{{image, target}, {}};
  const std::optional<c10::IValue> targets{target};
  data::BatchEnvelope batch;
  training::StepContext context;
  const auto step = t::make_rtdetr_step(1, true, {384}, 5);
  auto out = step({wrapped, batch, invocation, targets, context});
  const auto indices =
      torch::arange(384, torch::kInt64).mul(256).div(384, "floor");
  const auto expected = image.index_select(2, indices).index_select(3, indices);
  EXPECT_TRUE(torch::equal(received_image, expected));
  EXPECT_TRUE(
      torch::equal(received_targets.slice(1, 0, 2), target.slice(1, 0, 2)));
  EXPECT_TRUE(torch::equal(received_targets.slice(1, 2, 6),
                           target.slice(1, 2, 6) * 1.5));
  EXPECT_TRUE(torch::isfinite(out.loss).item<bool>());
  out.loss.backward();
  bool gradient = false;
  for (const auto& parameter : binding.model->state().parameters())
    if (parameter.grad().defined()) {
      EXPECT_TRUE(torch::isfinite(parameter.grad()).all().item<bool>());
      gradient = true;
    }
  EXPECT_TRUE(gradient);
  context.phase = training::StepPhase::validation;
  static_cast<void>(step({wrapped, batch, invocation, targets, context}));
  EXPECT_TRUE(torch::equal(received_image, image));
  EXPECT_TRUE(torch::equal(received_targets, target));
  EXPECT_TRUE(torch::equal(invocation.positional[0].toTensor(), image));
  EXPECT_THROW(t::make_rtdetr_step(1, true, {33}), InvalidArgument);
}
}  // namespace
}  // namespace jyppx::liuding::vision::tests
