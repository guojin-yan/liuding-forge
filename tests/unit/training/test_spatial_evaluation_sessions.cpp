// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/checkpoint_coordinator.hpp>
#include <jyppx/liuding/training/detection_evaluation.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>
#include <jyppx/liuding/training/segmentation_evaluation.hpp>
#if defined(JYPPX_LIUDING_EVALUATION_REPORT_TEST)
#include <jyppx/liuding/experiment/evaluation.hpp>
#include <jyppx/liuding/experiment/run_store.hpp>
#endif

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::training;
struct SegmentationIoU {
  static constexpr bool segmentation = true;
  using Session = SegmentationEvaluationSession;
  using Identity = SegmentationEvaluationIdentity;
  using Options = SegmentationEvaluationOptions;
  using Batch = SegmentationEvaluationBatch;
  static Options options() {
    Options o;
    o.metric.num_classes = 3;
    o.metric.ignore_index = -1;
    o.sources = {"a", "b", "empty"};
    o.max_batches = 3;
    o.max_images = 6;
    o.max_pixels = 24;
    o.max_counts_bytes = 648;
    o.limits.max_batch_pixels = 8;
    return o;
  }
  static nn::ModelAdapter model(bool cuda, int& calls) {
    auto layer = std::make_shared<torch::nn::Conv2dImpl>(
        torch::nn::Conv2dOptions(1, 3, 1));
    layer->register_module("dropout", torch::nn::Dropout(.9));
    {
      torch::NoGradGuard guard;
      layer->weight.copy_(
          torch::tensor({-1.f, 1.f, 0.f}).reshape({3, 1, 1, 1}));
      layer->bias.copy_(torch::tensor({.5f, -.5f, -2.f}));
    }
    if (cuda) layer->to(torch::kCUDA);
    return nn::adapt_module(
        layer,
        [&calls](torch::nn::Conv2dImpl& m,
                 const nn::Invocation& in) -> c10::IValue {
          ++calls;
          if (m.is_training() || m.children().front()->is_training() ||
              torch::GradMode::is_enabled())
            throw std::runtime_error("inference guards missing");
          return m.forward(in.positional.front().toTensor());
        },
        {.positional = {ValueSchema::tensor()}}, ValueSchema::tensor());
  }
  static auto source(bool cuda = false) {
    return [cuda](std::uint64_t i, std::stop_token) -> std::optional<Batch> {
      if (i == 3) return std::nullopt;
      const auto device = cuda ? torch::kCUDA : torch::kCPU;
      auto x = torch::tensor({0.f, 1.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f})
                   .reshape({2, 1, 2, 2})
                   .to(device);
      auto y = torch::tensor({0, 1, 0, -1, 1, 0, 1, 0}, torch::kInt64)
                   .reshape({2, 2, 2})
                   .to(device);
      if (i == 2) y.fill_(-1);
      return Batch{.invocation = {.positional = {x}},
                   .targets = y,
                   .source_indices = {0, 1}};
    };
  }
  static void corrupt_batch(Batch& batch) { batch.targets[1][1][1] = 99; }
  static void corrupt_counts(
      std::map<std::string, std::vector<std::byte>>& fields) {
    fields.at("slice.3.counts").pop_back();
  }
  static void shrink(Options& o) { --o.max_counts_bytes; }
};
struct SegmentationDice : SegmentationIoU {
  static Options options() {
    auto o = SegmentationIoU::options();
    o.metric.score = metrics::SegmentationScore::dice;
    return o;
  }
};
struct DetectionAP {
  static constexpr bool segmentation = false;
  using Session = DetectionEvaluationSession;
  using Identity = DetectionEvaluationIdentity;
  using Options = DetectionEvaluationOptions;
  using Batch = DetectionEvaluationBatch;
  static Options options() {
    Options o;
    o.metric.dataset_id = "fixed.detection.dataset/v1";
    o.metric.class_vocabulary_id = "fixed.classes/v1";
    o.metric.iou_thresholds = {.5, .75};
    o.metric.sources = {"a", "b", "empty"};
    o.max_batches = 3;
    o.expected_images = 6;
    o.limits.max_images = 6;
    return o;
  }
  static nn::ModelAdapter model(bool cuda, int& calls) {
    auto layer = std::make_shared<torch::nn::LinearImpl>(5, 5);
    layer->register_module("dropout", torch::nn::Dropout(.9));
    {
      torch::NoGradGuard guard;
      layer->weight.copy_(torch::eye(5));
      layer->bias.zero_();
    }
    if (cuda) layer->to(torch::kCUDA);
    return nn::adapt_module(
        layer,
        [&calls](torch::nn::LinearImpl& m,
                 const nn::Invocation& in) -> c10::IValue {
          ++calls;
          if (m.is_training() || m.children().front()->is_training() ||
              torch::GradMode::is_enabled())
            throw std::runtime_error("inference guards missing");
          const auto y = m.forward(in.positional.front().toTensor());
          c10::impl::GenericList output(c10::AnyType::get());
          for (std::int64_t i = 0; i < y.size(0); ++i)
            output.push_back(c10::ivalue::Tuple::create(
                {y[i].narrow(1, 0, 4), y[i].select(1, 4),
                 torch::zeros({y.size(1)}, y.options().dtype(torch::kInt32))}));
          return output;
        },
        {.positional = {ValueSchema::tensor()}},
        ValueSchema::list(
            ValueSchema::tuple({ValueSchema::tensor(), ValueSchema::tensor(),
                                ValueSchema::tensor()})));
  }
  static auto source(bool cuda = false) {
    return [cuda](std::uint64_t index,
                  std::stop_token) -> std::optional<Batch> {
      if (index == 3) return std::nullopt;
      const auto x = torch::tensor({20.f, 20.f, 30.f, 30.f, .9f,  0.f,  0.f,
                                    10.f, 10.f, .1f,  0.f,  0.f,  10.f, 10.f,
                                    .8f,  20.f, 20.f, 30.f, 30.f, .05f})
                         .reshape({2, 2, 5})
                         .to(cuda ? torch::kCUDA : torch::kCPU);
      Batch batch{.invocation = {.positional = {x}}, .images = {}};
      for (std::size_t i = 0; i < 2; ++i)
        batch.images.push_back({.image_id = index * 2 + i,
                                .source_index = i,
                                .targets = {{.box = {0, 0, 10, 10},
                                             .class_index = 0,
                                             .crowd = index == 2,
                                             .ignore = false,
                                             .area = std::nullopt}},
                                .predictions = {}});
      return batch;
    };
  }
  static void corrupt_batch(Batch& batch) {
    batch.images.back().image_id = batch.images.front().image_id;
  }
  static void corrupt_counts(
      std::map<std::string, std::vector<std::byte>>& fields) {
    fields.at("slice.0.matches").push_back(std::byte{});
  }
  static void shrink(Options& o) { o.max_conversion_bytes = 1; }
};
template <typename T>
typename T::Identity binding() {
  typename T::Identity id;
  id.evaluation_id = "spatial.fixture";
  id.dataset_digest.hex = std::string(64, 'a');
  id.model_digest.hex = std::string(64, 'b');
  id.config_digest.hex = std::string(64, 'c');
  id.build_fingerprint = "fixed-spatial-fixture/v1";
  return id;
}
template <typename T>
auto session(typename T::Options options = T::options(),
             typename T::Identity identity = binding<T>(),
             std::optional<std::uint64_t> expected = 6) {
  if constexpr (T::segmentation)
    return std::make_shared<typename T::Session>(
        std::move(identity), std::move(options), expected,
        expected ? std::optional<std::uint64_t>(*expected * 4) : std::nullopt);
  else {
    options.expected_images = expected;
    return std::make_shared<typename T::Session>(std::move(identity),
                                                 std::move(options));
  }
}
template <typename T>
auto baseline(bool cuda = false) {
  int calls = 0;
  auto model = T::model(cuda, calls);
  auto source = T::source(cuda);
  std::uint64_t i = 0;
  if constexpr (T::segmentation)
    return evaluate_semantic_segmentation(
        model, [&](auto stop) { return source(i++, stop); }, T::options());
  else
    return evaluate_detection(
        model, [&](auto stop) { return source(i++, stop); }, T::options());
}
template <typename R>
void same_result(const R& a, const R& b) {
  EXPECT_EQ(a.batches, b.batches);
  ASSERT_EQ(a.by_source.size(), b.by_source.size());
  if constexpr (std::is_same_v<R, SegmentationEvaluationResult>) {
    EXPECT_EQ(a.images, b.images);
    EXPECT_EQ(a.pixels_seen, b.pixels_seen);
    const auto compare = [](const auto& x, const auto& y) {
      EXPECT_EQ(x.pixels, y.pixels);
      EXPECT_EQ(x.score, y.score);
      EXPECT_EQ(x.per_class, y.per_class);
      ASSERT_EQ(x.counts.size(), y.counts.size());
      for (std::size_t i = 0; i < x.counts.size(); ++i) {
        EXPECT_EQ(x.counts[i].intersection, y.counts[i].intersection);
        EXPECT_EQ(x.counts[i].target, y.counts[i].target);
        EXPECT_EQ(x.counts[i].predicted, y.counts[i].predicted);
      }
    };
    compare(a.overall, b.overall);
    for (std::size_t i = 0; i < a.by_source.size(); ++i)
      compare(a.by_source[i], b.by_source[i]);
  } else {
    const auto compare = [](const auto& x, const auto& y) {
      EXPECT_EQ(x.images, y.images);
      EXPECT_EQ(x.mean_average_precision, y.mean_average_precision);
      ASSERT_EQ(x.per_class.size(), y.per_class.size());
      for (std::size_t i = 0; i < x.per_class.size(); ++i) {
        EXPECT_EQ(x.per_class[i].ground_truth, y.per_class[i].ground_truth);
        EXPECT_EQ(x.per_class[i].predictions, y.per_class[i].predictions);
        EXPECT_EQ(x.per_class[i].average_precision,
                  y.per_class[i].average_precision);
        EXPECT_EQ(x.per_class[i].ap_by_iou, y.per_class[i].ap_by_iou);
        EXPECT_EQ(x.per_class[i].recall_by_iou, y.per_class[i].recall_by_iou);
      }
    };
    compare(a.overall, b.overall);
    for (std::size_t i = 0; i < a.by_source.size(); ++i)
      compare(a.by_source[i], b.by_source[i]);
  }
  ASSERT_EQ(a.observations.size(), b.observations.size());
  for (std::size_t i = 0; i < a.observations.size(); ++i) {
    EXPECT_EQ(a.observations[i].definition.parameters,
              b.observations[i].definition.parameters);
    EXPECT_EQ(a.observations[i].observation.value,
              b.observations[i].observation.value);
    EXPECT_EQ(a.observations[i].observation.status,
              b.observations[i].observation.status);
    EXPECT_EQ(a.observations[i].context, b.observations[i].context);
    EXPECT_EQ(a.observations[i].sample_count, b.observations[i].sample_count);
  }
}
class Snapshot final : public StateWriter, public StateReader {
 public:
  explicit Snapshot(const CheckpointParticipant& p)
      : descriptor(p.describe_state()) {}
  void write_blob(std::string_view name,
                  std::span<const std::byte> b) override {
    fields[std::string(name)] = {b.begin(), b.end()};
  }
  const StateManifest& manifest() const noexcept override { return descriptor; }
  bool contains(std::string_view name) const override {
    return fields.contains(std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t limit) const override {
    if (!contains(name)) throw SerializationError("fixture missing field");
    const auto& b = fields.at(std::string(name));
    if (b.size() > limit) throw ResourceExhausted("fixture read bound");
    return b;
  }
  StateManifest descriptor;
  std::map<std::string, std::vector<std::byte>> fields;
};
Snapshot snapshot(const CheckpointParticipant& p) {
  Snapshot s(p);
  p.save_state(s);
  return s;
}
void set_integer(std::vector<std::byte>& b, std::size_t offset,
                 std::uint64_t value) {
  for (unsigned i = 0; i < 8; ++i)
    b.at(offset + i) = static_cast<std::byte>((value >> (8U * i)) & 0xffU);
}
using SpatialTypes =
    ::testing::Types<SegmentationIoU, SegmentationDice, DetectionAP>;
template <typename T>
class SpatialEvaluationSession : public ::testing::Test {};
TYPED_TEST_SUITE(SpatialEvaluationSession, SpatialTypes);
template <typename T>
void rebuild(bool cuda) {
  std::optional<Snapshot> saved;
  {
    int calls = 0;
    auto model = T::model(cuda, calls);
    auto s = session<T>();
    EXPECT_THROW(s->result(), NotReady);
    EXPECT_THROW(s->run(model, T::source(cuda),
                        [&](const auto& state) {
                          saved = snapshot(state);
                          throw std::runtime_error("interrupt");
                        }),
                 std::runtime_error);
    EXPECT_EQ(s->progress().batches, 1);
    EXPECT_EQ(calls, 1);
    EXPECT_TRUE(model.state().is_training());
  }
  int calls = 0;
  auto model = T::model(cuda, calls);
  auto s = session<T>();
  s->load_state(*saved, RestorePolicy::exact_only);
  auto source = T::source(cuda);
  std::vector<std::uint64_t> requested;
  auto result = s->run(model, [&](auto i, auto stop) {
    requested.push_back(i);
    return source(i, stop);
  });
  EXPECT_EQ(requested, (std::vector<std::uint64_t>{1, 2, 3}));
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(s->progress().images, 6);
  EXPECT_TRUE(s->progress().complete);
  EXPECT_TRUE(model.state().is_training());
  same_result(result, baseline<T>(cuda));
  same_result(s->result(), result);
}
TYPED_TEST(SpatialEvaluationSession, RebuildsCommittedStatisticsCPU) {
  rebuild<TypeParam>(false);
}
TYPED_TEST(SpatialEvaluationSession, RebuildsCommittedStatisticsCUDA) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA unavailable";
  rebuild<TypeParam>(true);
}
TYPED_TEST(SpatialEvaluationSession,
           LateBatchFailureAndCancellationPreserveAllState) {
  int calls = 0;
  auto model = TypeParam::model(false, calls);
  auto s = session<TypeParam>();
  const auto before = snapshot(*s);
  auto source = TypeParam::source();
  EXPECT_THROW(s->run(model,
                      [&](auto i, auto stop) {
                        auto b = source(i, stop);
                        TypeParam::corrupt_batch(*b);
                        return b;
                      }),
               Error);
  EXPECT_EQ(snapshot(*s).fields, before.fields);
  std::stop_source stop;
  EXPECT_THROW(s->run(
                   model,
                   [&](auto i, auto token) {
                     auto b = source(i, token);
                     stop.request_stop();
                     return b;
                   },
                   {}, stop.get_token()),
               Cancelled);
  EXPECT_EQ(snapshot(*s).fields, before.fields);
  EXPECT_THROW(
      s->run(model, source, {}, {},
             std::chrono::steady_clock::now() - std::chrono::seconds(1)),
      DeadlineExceeded);
  same_result(s->run(model, source), baseline<TypeParam>());
}
TYPED_TEST(SpatialEvaluationSession,
           HookFailureKeepsCommitAndReentryIsRejected) {
  int calls = 0;
  auto model = TypeParam::model(false, calls);
  auto s = session<TypeParam>();
  auto source = TypeParam::source();
  const auto before = snapshot(*s);
  const auto checked_source = [&](auto i, auto stop) {
    Snapshot unsafe(*s);
    EXPECT_THROW(s->save_state(unsafe), NotReady);
    return source(i, stop);
  };
  const auto failing_hook = [&](const auto& state) {
    EXPECT_NO_THROW(snapshot(state));
    EXPECT_THROW(s->run(model, source), NotReady);
    EXPECT_THROW(s->load_state(before, RestorePolicy::exact_only), NotReady);
    throw std::runtime_error("hook failed");
  };
  EXPECT_THROW(s->run(model, checked_source, failing_hook), std::runtime_error);
  EXPECT_EQ(s->progress().batches, 1);
  same_result(s->run(model, source), baseline<TypeParam>());
}
TYPED_TEST(SpatialEvaluationSession, CompleteStateRetriesOnlyFinalHook) {
  int calls = 0;
  auto model = TypeParam::model(false, calls);
  auto s = session<TypeParam>();
  const auto expected = s->run(model, TypeParam::source());
  const auto complete = snapshot(*s);
  auto restored = session<TypeParam>();
  restored->load_state(complete, RestorePolicy::exact_only);
  calls = 0;
  int hooks = 0;
  const auto source = [](auto,
                         auto) -> std::optional<typename TypeParam::Batch> {
    throw std::runtime_error("must not read");
  };
  EXPECT_THROW(restored->run(model, source,
                             [&](const auto&) {
                               ++hooks;
                               throw std::runtime_error("publish");
                             }),
               std::runtime_error);
  same_result(restored->run(model, source, [&](const auto&) { ++hooks; }),
              expected);
  EXPECT_EQ(hooks, 2);
  EXPECT_EQ(calls, 0);
}
TYPED_TEST(SpatialEvaluationSession,
           RejectsIdentityPolicyAndResourceContractChanges) {
  auto s = session<TypeParam>();
  const auto saved = snapshot(*s);
  for (int variant = 0; variant < 9; ++variant) {
    auto id = binding<TypeParam>();
    auto o = TypeParam::options();
    if (variant == 0) id.evaluation_id += ".other";
    if (variant == 1) id.dataset_digest.hex[0] = 'd';
    if (variant == 2) id.model_digest.hex[0] = 'd';
    if (variant == 3) id.config_digest.hex[0] = 'd';
    if (variant == 4) id.build_fingerprint += ".other";
    if (variant == 5) id.participant_id += ".other";
    if (variant == 6) ++o.max_batches;
    if (variant == 7) ++o.metric.num_classes;
    if (variant == 8) ++o.limits.max_state_bytes;
    // Extra class needs a larger logical count budget to construct
    // legitimately.
    if constexpr (TypeParam::segmentation)
      if (variant == 7) o.max_counts_bytes += 216;
    auto other = session<TypeParam>(o, id);
    const auto before = snapshot(*other);
    EXPECT_THROW(other->load_state(saved, RestorePolicy::exact_only),
                 CompatibilityError)
        << variant;
    EXPECT_EQ(snapshot(*other).fields, before.fields);
  }
}
TYPED_TEST(SpatialEvaluationSession,
           LateCorruptionAndForgedCursorLoadAtomically) {
  int calls = 0;
  auto model = TypeParam::model(false, calls);
  auto s = session<TypeParam>();
  s->run(model, TypeParam::source());
  const auto saved = snapshot(*s);
  auto dest = session<TypeParam>();
  const auto before = snapshot(*dest);
  auto bad = saved;
  TypeParam::corrupt_counts(bad.fields);
  EXPECT_THROW(dest->load_state(bad, RestorePolicy::exact_only), Error);
  EXPECT_EQ(snapshot(*dest).fields, before.fields);
  bad = saved;
  set_integer(bad.fields.at("progress"), 16, 5);
  EXPECT_THROW(dest->load_state(bad, RestorePolicy::exact_only),
               SerializationError);
  EXPECT_EQ(snapshot(*dest).fields, before.fields);
  bad = saved;
  bad.descriptor.schema_version = 2;
  EXPECT_THROW(dest->load_state(bad, RestorePolicy::exact_only),
               CompatibilityError);
  if constexpr (TypeParam::segmentation) {
    bad = saved;
    bad.fields.at("slice.3.counts") = bad.fields.at("slice.0.counts");
    EXPECT_THROW(dest->load_state(bad, RestorePolicy::exact_only),
                 SerializationError);
  }
  EXPECT_EQ(snapshot(*dest).fields, before.fields);
  dest->load_state(saved, RestorePolicy::exact_only);
  same_result(dest->result(), s->result());
}
TYPED_TEST(SpatialEvaluationSession,
           ExpectedTotalsLimitsAndEmptyInputAreExplicit) {
  int calls = 0;
  auto model = TypeParam::model(false, calls);
  auto short_run =
      session<TypeParam>(TypeParam::options(), binding<TypeParam>(), 5);
  EXPECT_THROW(short_run->run(model, TypeParam::source()), Error);
  EXPECT_FALSE(short_run->progress().complete);
  auto incomplete = session<TypeParam>();
  auto source = TypeParam::source();
  EXPECT_THROW(
      incomplete->run(
          model,
          [&](auto i, auto stop) -> std::optional<typename TypeParam::Batch> {
            if (i == 2) return std::nullopt;
            return source(i, stop);
          }),
      InvalidArgument);
  EXPECT_FALSE(incomplete->progress().complete);
  auto o = TypeParam::options();
  TypeParam::shrink(o);
  EXPECT_THROW(
      {
        auto limited = session<TypeParam>(o);
        limited->run(model, source);
      },
      ResourceExhausted);
  auto empty =
      session<TypeParam>(TypeParam::options(), binding<TypeParam>(), 0);
  const auto result = empty->run(
      model, [](auto, auto) -> std::optional<typename TypeParam::Batch> {
        return std::nullopt;
      });
  EXPECT_EQ(result.batches, 0);
  EXPECT_EQ(empty->progress().images, 0);
  EXPECT_TRUE(empty->progress().complete);
  auto restored =
      session<TypeParam>(TypeParam::options(), binding<TypeParam>(), 0);
  restored->load_state(snapshot(*empty), RestorePolicy::exact_only);
  same_result(restored->result(), result);
}
TYPED_TEST(SpatialEvaluationSession,
           ScorePolicyFailureCannotPublishCompleteState) {
  int calls = 0;
  auto model = TypeParam::model(false, calls);
  auto o = TypeParam::options();
  if constexpr (TypeParam::segmentation)
    o.metric.absent = metrics::AbsentClassPolicy::error;
  else
    o.metric.no_ground_truth = metrics::DetectionNoGroundTruth::error;
  auto s = session<TypeParam>(o);
  EXPECT_THROW(s->run(model, TypeParam::source()), NumericalError);
  EXPECT_FALSE(s->progress().complete);
  EXPECT_EQ(s->progress().batches, 3);
  EXPECT_THROW(s->result(), NotReady);
  const auto before = snapshot(*s);
  EXPECT_THROW(s->run(model, TypeParam::source()), NumericalError);
  EXPECT_EQ(snapshot(*s).fields, before.fields);
}
TYPED_TEST(SpatialEvaluationSession, NoSourceStateReopensWithEquivalentCounts) {
  auto o = TypeParam::options();
  if constexpr (TypeParam::segmentation)
    o.sources.clear();
  else
    o.metric.sources.clear();
  auto original = TypeParam::source();
  const auto source = [&](auto i, auto stop) {
    auto batch = original(i, stop);
    if (batch) {
      if constexpr (TypeParam::segmentation)
        batch->source_indices.clear();
      else
        for (auto& image : batch->images) image.source_index.reset();
    }
    return batch;
  };
  int calls = 0;
  auto model = TypeParam::model(false, calls);
  auto s = session<TypeParam>(o);
  const auto actual = s->run(model, source);
  std::uint64_t index = 0;
  const auto expected = [&] {
    if constexpr (TypeParam::segmentation)
      return evaluate_semantic_segmentation(
          model, [&](auto stop) { return source(index++, stop); }, o);
    else
      return evaluate_detection(
          model, [&](auto stop) { return source(index++, stop); }, o);
  }();
  same_result(actual, expected);
  auto restored = session<TypeParam>(o);
  restored->load_state(snapshot(*s), RestorePolicy::exact_only);
  same_result(restored->result(), expected);
}

#if JYPPX_LIUDING_COMPONENT_INTEROP_ENABLED
class TempRoot final {
 public:
  TempRoot() {
    for (unsigned i = 0; i < 64; ++i) {
      auto path =
          std::filesystem::temp_directory_path() /
          ("liuding-eval-" +
           std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count()) +
           "-" + std::to_string(std::random_device{}()));
      if (std::filesystem::create_directory(path)) {
        path_ = std::move(path);
        return;
      }
    }
    throw std::runtime_error("could not reserve evaluation fixture root");
  }
  ~TempRoot() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

template <typename T>
void durable(bool cuda) {
  TempRoot root;
  const auto path = root.path() / "evaluation.state";
  {
    int calls = 0;
    auto model = T::model(cuda, calls);
    auto s = session<T>();
    CheckpointParticipantRegistry registry;
    registry.register_participant(s);
    CheckpointCoordinator coordinator(registry);
    FileCheckpointStore store(path);
    EXPECT_THROW(
        s->run(model, T::source(cuda),
               [&](const auto&) {
                 coordinator.save(store);
                 throw std::runtime_error("interrupt after first batch");
               }),
        std::runtime_error);
  }
  int calls = 0;
  auto model = T::model(cuda, calls);
  auto s = session<T>();
  CheckpointParticipantRegistry registry;
  registry.register_participant(s);
  CheckpointCoordinator coordinator(registry);
  FileCheckpointStore store(path);
  coordinator.restore(store);
  EXPECT_EQ(s->progress().batches, 1);
  same_result(s->run(model, T::source(cuda)), baseline<T>(cuda));
  EXPECT_EQ(calls, 2);
  coordinator.save(store);
}
TYPED_TEST(SpatialEvaluationSession, PhysicalCheckpointReopensCPU) {
  durable<TypeParam>(false);
}
TYPED_TEST(SpatialEvaluationSession, PhysicalCheckpointReopensCUDA) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA unavailable";
  durable<TypeParam>(true);
}
#if GTEST_HAS_DEATH_TEST
class ExitRoot final {
 public:
  ExitRoot() {
#ifdef _WIN32
    wchar_t* value = nullptr;
    std::size_t size = 0;
    if (_wdupenv_s(&value, &size, kEnvironment) != 0)
      throw std::runtime_error("fixture environment");
    std::unique_ptr<wchar_t, decltype(&std::free)> owner(value, &std::free);
#else
    const char* value = std::getenv(kEnvironment);
#endif
    if (value) previous_ = value;
    if (previous_ && !previous_->empty()) {
      path_ = *previous_;
      if (!std::filesystem::is_regular_file(path_ / ".fixture"))
        throw std::runtime_error("invalid borrowed fixture root");
      return;
    }
    root_ = std::make_unique<TempRoot>();
    path_ = root_->path();
    std::ofstream(path_ / ".fixture") << "spatial evaluation abrupt exit";
#ifdef _WIN32
    if (_wputenv_s(kEnvironment, path_.c_str()) != 0)
      throw std::runtime_error("fixture environment");
#else
    if (::setenv(kEnvironment, path_.c_str(), 1) != 0)
      throw std::runtime_error("fixture environment");
#endif
  }
  ~ExitRoot() {
    if (!root_) return;
#ifdef _WIN32
    (void)_wputenv_s(kEnvironment, previous_ ? previous_->c_str() : L"");
#else
    if (previous_)
      (void)::setenv(kEnvironment, previous_->c_str(), 1);
    else
      (void)::unsetenv(kEnvironment);
#endif
  }
  const std::filesystem::path& path() const { return path_; }

 private:
#ifdef _WIN32
  static constexpr auto kEnvironment =
      L"JYPPX_LIUDING_TEST_SPATIAL_RECOVERY_ROOT";
#else
  static constexpr auto kEnvironment =
      "JYPPX_LIUDING_TEST_SPATIAL_RECOVERY_ROOT";
#endif
  std::optional<std::filesystem::path::string_type> previous_;
  std::unique_ptr<TempRoot> root_;
  std::filesystem::path path_;
};

template <typename T>
class SpatialEvaluationSessionDeathTest : public ::testing::Test {};
TYPED_TEST_SUITE(SpatialEvaluationSessionDeathTest, SpatialTypes);
TYPED_TEST(SpatialEvaluationSessionDeathTest,
           AbruptExitAfterBatchCommitReplaysOnlySuffix) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ExitRoot root;
  const auto path = root.path() / "evaluation.state";
  EXPECT_EXIT(
      {
        int calls = 0;
        auto model = TypeParam::model(false, calls);
        auto s = session<TypeParam>();
        CheckpointParticipantRegistry registry;
        registry.register_participant(s);
        CheckpointCoordinator coordinator(registry);
        FileCheckpointStore store(path);
        s->run(model, TypeParam::source(), [&](const auto&) {
          coordinator.save(store);
          std::_Exit(73);
        });
        std::_Exit(74);
      },
      ::testing::ExitedWithCode(73), "");
  int calls = 0;
  auto model = TypeParam::model(false, calls);
  auto s = session<TypeParam>();
  CheckpointParticipantRegistry registry;
  registry.register_participant(s);
  CheckpointCoordinator coordinator(registry);
  FileCheckpointStore store(path);
  coordinator.restore(store);
  auto source = TypeParam::source();
  std::vector<std::uint64_t> indices;
  auto result = s->run(model, [&](auto i, auto stop) {
    indices.push_back(i);
    return source(i, stop);
  });
  EXPECT_EQ(indices, (std::vector<std::uint64_t>{1, 2, 3}));
  EXPECT_EQ(calls, 2);
  same_result(result, baseline<TypeParam>());
}
#if defined(JYPPX_LIUDING_EVALUATION_REPORT_TEST)
template <typename T, typename R>
std::string report_json(const R& result) {
  experiment::EvaluationSuite suite{
      .suite_id = "spatial.fixture",
      .task = T::segmentation ? "semantic_segmentation"
                              : "axis_aligned_box_detection",
      .dataset_digest = binding<T>().dataset_digest,
      .environment_id = "fixed-spatial-fixture/v1"};
  for (std::size_t i = 0; i < result.observations.size(); ++i) {
    const auto& point = result.observations[i];
    suite.rules.push_back({.rule_id = "r" + std::to_string(i),
                           .definition = point.definition,
                           .context = point.context,
                           .minimum_samples = 1,
                           .minimum = 0.});
  }
  const auto id = binding<T>();
  return experiment::quality_report_json(experiment::evaluate_observations(
      suite,
      {.evaluation_id = id.evaluation_id,
       .run_id = "run.fixture",
       .model_digest = id.model_digest,
       .build_fingerprint = id.build_fingerprint,
       .adapter_id = "training.spatial.session/v1",
       .config_digest = id.config_digest},
      result.observations));
}
experiment::RunArtifactReference publish_report(
    const std::filesystem::path& root, const std::string& bytes) {
  experiment::SQLiteRunStore store(root);
  store.start({.run_id = "run.fixture",
               .project_id = "project.fixture",
               .experiment_id = "experiment.fixture",
               .display_name = "spatial evaluation fixture",
               .created_at_unix_ms = 10,
               .updated_at_unix_ms = 10});
  return store.put_artifact(
      "run.fixture", "eval.fixture.report",
      experiment::RunArtifactKind::artifact, "evaluation.report",
      std::as_bytes(std::span(bytes)),
      {.logical_name = "eval.fixture.quality.json",
       .media_type = "application/vnd.jyppx.quality-report+json;version=1",
       .producer = "training.spatial.session/v1",
       .expected_size_bytes = bytes.size()},
      20);
}
template <typename T>
void completed_crash(bool published) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ExitRoot root;
  const auto path = root.path() / "evaluation.state",
             runs = root.path() / "runs";
  EXPECT_EXIT(
      {
        int calls = 0;
        auto model = T::model(false, calls);
        auto s = session<T>();
        CheckpointParticipantRegistry registry;
        registry.register_participant(s);
        CheckpointCoordinator coordinator(registry);
        FileCheckpointStore store(path);
        s->run(model, T::source(), [&](const auto& state) {
          coordinator.save(store);
          if (!state.progress().complete) return;
          if (published) publish_report(runs, report_json<T>(state.result()));
          std::_Exit(73);
        });
        std::_Exit(74);
      },
      ::testing::ExitedWithCode(73), "");
  int calls = 0;
  auto model = T::model(false, calls);
  auto s = session<T>();
  CheckpointParticipantRegistry registry;
  registry.register_participant(s);
  CheckpointCoordinator coordinator(registry);
  FileCheckpointStore store(path);
  coordinator.restore(store);
  ASSERT_TRUE(s->progress().complete);
  int hooks = 0;
  const auto result = s->run(
      model,
      [](auto, auto) -> std::optional<typename T::Batch> {
        throw std::runtime_error("must not re-read");
      },
      [&](const auto& state) {
        publish_report(runs, report_json<T>(state.result()));
        ++hooks;
      });
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(hooks, 1);
  const auto canonical = report_json<T>(result);
  EXPECT_EQ(canonical, report_json<T>(baseline<T>()));
  EXPECT_EQ(publish_report(runs, canonical).reference_id,
            "eval.fixture.report");
  experiment::SQLiteRunStore database(runs);
  const auto refs = database.references("run.fixture");
  ASSERT_EQ(refs.size(), 1);
  EXPECT_TRUE(database.integrity_check().ok());
  experiment::FileArtifactStore artifacts(runs / "artifacts");
  auto reader = artifacts.open(refs[0].artifact.digest);
  std::vector<std::byte> bytes(static_cast<std::size_t>(reader->size_bytes()));
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    auto n = reader->read(std::span(bytes).subspan(offset));
    ASSERT_NE(n, 0);
    offset += n;
  }
  EXPECT_EQ(
      std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()),
      canonical);
  auto conflict = experiment::parse_quality_report_json(canonical);
  conflict.identity.model_digest.hex[0] = 'f';
  EXPECT_THROW(publish_report(runs, experiment::quality_report_json(conflict)),
               InvalidArgument);
  EXPECT_EQ(database.references("run.fixture")[0].artifact.digest,
            refs[0].artifact.digest);
}
TYPED_TEST(SpatialEvaluationSessionDeathTest,
           AbruptExitAfterEOFBeforeReportRetriesOneVerdict) {
  completed_crash<TypeParam>(false);
}
TYPED_TEST(SpatialEvaluationSessionDeathTest,
           AbruptExitAfterReportCommitRetriesOneVerdict) {
  completed_crash<TypeParam>(true);
}
#endif
#endif
#endif
}  // namespace
