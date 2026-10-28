// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/checkpoint_coordinator.hpp>
#include <jyppx/liuding/training/classification_evaluation.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>

#include <torch/torch.h>
#if defined(JYPPX_LIUDING_EVALUATION_REPORT_TEST)
#include <jyppx/liuding/experiment/evaluation.hpp>
#include <jyppx/liuding/experiment/run_store.hpp>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::training;

ClassificationEvaluationIdentity binding() {
  return {.evaluation_id = "eval.fixture",
          .dataset_digest = {.hex = std::string(64, 'a')},
          .model_digest = {.hex = std::string(64, 'b')},
          .config_digest = {.hex = std::string(64, 'c')},
          .build_fingerprint = "fixed-linear-fixture/v1"};
}
ClassificationEvaluationOptions options() {
  return {.num_classes = 3,
          .ignore_index = -1,
          .sources = {"a", "b", "empty"},
          .max_batch_samples = 2};
}
nn::ModelAdapter fixed_model(bool cuda = false) {
  auto linear = std::make_shared<torch::nn::LinearImpl>(2, 3);
  {
    torch::NoGradGuard guard;
    linear->weight.copy_(torch::tensor({{1.f, 0.f}, {0.f, 1.f}, {-1.f, -1.f}}));
    linear->bias.zero_();
  }
  linear->to(cuda ? torch::kCUDA : torch::kCPU);
  return nn::adapt_module(
      linear,
      [](torch::nn::LinearImpl& module,
         const nn::Invocation& call) -> c10::IValue {
        if (module.is_training() || torch::GradMode::is_enabled())
          throw std::runtime_error("inference guard missing");
        return module.forward(call.positional[0].toTensor());
      },
      {.positional = {ValueSchema::tensor()}}, ValueSchema::tensor());
}
IndexedClassificationEvaluationSource indexed_source(bool cuda = false) {
  const auto device = torch::Device(cuda ? torch::kCUDA : torch::kCPU);
  const auto x = torch::tensor({{3.f, 1.f},
                                {0.f, 2.f},
                                {-1.f, -1.f},
                                {2.f, 2.f},
                                {9.f, 0.f},
                                {0.f, 0.f}})
                     .to(device);
  const auto y = torch::tensor({0, 1, 1, 1, -1, 2}, torch::kInt64).to(device);
  const std::array<std::size_t, 6> names{0, 1, 0, 1, 0, 0};
  return [=](std::uint64_t index,
             std::stop_token) -> std::optional<ClassificationEvaluationBatch> {
    if (index >= 3) return std::nullopt;
    const auto begin = static_cast<std::int64_t>(2 * index), end = begin + 2;
    return ClassificationEvaluationBatch{
        .invocation = {.positional = {x.slice(0, begin, end)}},
        .targets = y.slice(0, begin, end),
        .source_indices = {names.begin() + begin, names.begin() + end}};
  };
}
ClassificationEvaluationResult baseline(bool cuda = false) {
  auto model = fixed_model(cuda);
  auto source = indexed_source(cuda);
  std::uint64_t index = 0;
  return evaluate_classification(
      model, [&](auto stop) { return source(index++, stop); }, options());
}
void same_result(const ClassificationEvaluationResult& actual,
                 const ClassificationEvaluationResult& expected) {
  EXPECT_EQ(actual.batches, expected.batches);
  EXPECT_EQ(actual.rows_seen, expected.rows_seen);
  EXPECT_EQ(actual.overall.confusion, expected.overall.confusion);
  EXPECT_EQ(actual.overall.samples, expected.overall.samples);
  ASSERT_EQ(actual.by_source.size(), expected.by_source.size());
  for (std::size_t i = 0; i < actual.by_source.size(); ++i)
    EXPECT_EQ(actual.by_source[i].confusion, expected.by_source[i].confusion);
  ASSERT_EQ(actual.observations.size(), expected.observations.size());
  for (std::size_t i = 0; i < actual.observations.size(); ++i) {
    EXPECT_EQ(actual.observations[i].observation.value,
              expected.observations[i].observation.value);
    EXPECT_EQ(actual.observations[i].observation.status,
              expected.observations[i].observation.status);
    EXPECT_EQ(actual.observations[i].sample_count,
              expected.observations[i].sample_count);
    EXPECT_EQ(actual.observations[i].context, expected.observations[i].context);
  }
}

class Snapshot final : public StateWriter, public StateReader {
 public:
  explicit Snapshot(const CheckpointParticipant& owner)
      : descriptor(owner.describe_state()) {}
  void write_blob(std::string_view name,
                  std::span<const std::byte> bytes) override {
    fields[std::string(name)] = {bytes.begin(), bytes.end()};
  }
  const StateManifest& manifest() const noexcept override { return descriptor; }
  bool contains(std::string_view name) const override {
    return fields.contains(std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t limit) const override {
    if (!contains(name)) throw SerializationError("fixture missing field");
    const auto& value = fields.at(std::string(name));
    if (value.size() > limit) throw ResourceExhausted("fixture read bound");
    return value;
  }
  StateManifest descriptor;
  std::map<std::string, std::vector<std::byte>> fields;
};
Snapshot snapshot(const ClassificationEvaluationSession& session) {
  Snapshot result(session);
  session.save_state(result);
  return result;
}
void set_integer(std::vector<std::byte>& data, std::size_t offset,
                 std::uint64_t value) {
  for (unsigned i = 0; i < 8; ++i)
    data.at(offset + i) = static_cast<std::byte>((value >> (8U * i)) & 0xffU);
}

void recreate(bool cuda) {
  const auto expected = baseline(cuda);
  std::optional<Snapshot> saved;
  {
    auto model = fixed_model(cuda);
    ClassificationEvaluationSession session(binding(), options(), 6);
    EXPECT_THROW(session.result(), NotReady);
    EXPECT_THROW(session.run(model, indexed_source(cuda),
                             [&](const auto& state) {
                               saved = snapshot(state);
                               throw std::runtime_error(
                                   "pause after first committed batch");
                             }),
                 std::runtime_error);
    EXPECT_EQ(session.progress().batches, 1);
    EXPECT_TRUE(model.state().is_training());
  }
  auto model = fixed_model(cuda);
  ClassificationEvaluationSession resumed(binding(), options(), 6);
  resumed.load_state(*saved, RestorePolicy::exact_only);
  std::vector<std::uint64_t> requested;
  const auto source = indexed_source(cuda);
  const auto result = resumed.run(model, [&](auto index, auto stop) {
    requested.push_back(index);
    return source(index, stop);
  });
  EXPECT_EQ(requested, (std::vector<std::uint64_t>{1, 2, 3}));
  EXPECT_TRUE(resumed.progress().complete);
  EXPECT_TRUE(model.state().is_training());
  same_result(result, expected);
  same_result(resumed.result(), expected);
}
TEST(ClassificationEvaluationSession, RecreatesFromCommittedStatisticsCPU) {
  recreate(false);
}
TEST(ClassificationEvaluationSession, RecreatesFromCommittedStatisticsCUDA) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA unavailable";
  recreate(true);
}

TEST(ClassificationEvaluationSession,
     BadBatchAndCancellationRepeatUncommittedIndex) {
  auto model = fixed_model();
  ClassificationEvaluationSession session(binding(), options(), 6);
  auto before = snapshot(session);
  auto source = indexed_source();
  EXPECT_THROW(
      session.run(model,
                  [&](auto index, auto stop) {
                    auto batch = source(index, stop);
                    batch->targets = torch::tensor({0, 99}, torch::kInt64);
                    return batch;
                  }),
      InvalidArgument);
  EXPECT_EQ(snapshot(session).fields, before.fields);
  EXPECT_THROW(
      session.run(model, source,
                  [](const auto&) { throw std::runtime_error("hook"); }),
      std::runtime_error);
  before = snapshot(session);
  EXPECT_EQ(session.progress().batches, 1);
  EXPECT_THROW(session.run(model,
                           [&](auto index, auto stop) {
                             auto batch = source(index, stop);
                             batch->invocation.positional[0].toTensor()[0][0] =
                                 std::numeric_limits<float>::infinity();
                             return batch;
                           }),
               NumericalError);
  EXPECT_EQ(snapshot(session).fields, before.fields);
  source = indexed_source();  // Replace deliberately corrupted fixture input.
  std::stop_source cancellation;
  EXPECT_THROW(session.run(
                   model,
                   [&](auto index, auto stop) {
                     auto batch = source(index, stop);
                     cancellation.request_stop();
                     return batch;
                   },
                   {}, cancellation.get_token()),
               Cancelled);
  EXPECT_EQ(snapshot(session).fields, before.fields);
  same_result(session.run(model, indexed_source()), baseline());
}

TEST(ClassificationEvaluationSession,
     ReentrantMutationAndUnsafeSaveAreRejected) {
  auto model = fixed_model();
  ClassificationEvaluationSession session(binding(), options(), 6);
  const auto initial = snapshot(session);
  auto source = indexed_source();
  int saves = 0;
  session.run(
      model,
      [&](auto index, auto stop) {
        Snapshot unsafe(session);
        EXPECT_THROW(session.save_state(unsafe), NotReady);
        return source(index, stop);
      },
      [&](const auto& state) {
        EXPECT_THROW(session.run(model, source), NotReady);
        EXPECT_THROW(session.load_state(initial, RestorePolicy::exact_only),
                     NotReady);
        EXPECT_NO_THROW(snapshot(state));
        ++saves;
      });
  EXPECT_EQ(saves, 4);
}

TEST(ClassificationEvaluationSession,
     CompletedReplayNeedsNoForwardAndRetriesFinalHook) {
  auto model = fixed_model();
  ClassificationEvaluationSession session(binding(), options(), 6);
  EXPECT_THROW(session.run(model, indexed_source(),
                           [](const auto& state) {
                             if (state.progress().complete)
                               throw std::runtime_error("acknowledgement lost");
                           }),
               std::runtime_error);
  ClassificationEvaluationSession restored(binding(), options(), 6);
  restored.load_state(snapshot(session), RestorePolicy::exact_only);
  int callbacks = 0;
  same_result(
      restored.run(
          model,
          [](auto, auto) -> std::optional<ClassificationEvaluationBatch> {
            throw std::runtime_error("completed source must not run");
          },
          [&](const auto& state) {
            EXPECT_TRUE(state.progress().complete);
            ++callbacks;
          }),
      baseline());
  EXPECT_EQ(callbacks, 1);
}

TEST(ClassificationEvaluationSession,
     RejectsChangedIdentityTopologyAndResourceContractAtomically) {
  auto model = fixed_model();
  ClassificationEvaluationSession session(binding(), options(), 6);
  session.run(model, indexed_source());
  const auto saved = snapshot(session);
  for (unsigned changed = 0; changed < 9; ++changed) {
    auto id = binding();
    auto o = options();
    std::optional<std::uint64_t> expected = 6;
    switch (changed) {
      case 0:
        id.evaluation_id += "x";
        break;
      case 1:
        id.dataset_digest.hex[0] = 'f';
        break;
      case 2:
        id.model_digest.hex[0] = 'f';
        break;
      case 3:
        id.config_digest.hex[0] = 'f';
        break;
      case 4:
        id.build_fingerprint += "x";
        break;
      case 5:
        std::swap(o.sources[0], o.sources[1]);
        break;
      case 6:
        o.zero_division = metrics::ZeroDivisionPolicy::one;
        break;
      case 7:
        ++o.max_batches;
        break;
      case 8:
        expected.reset();
        break;
    }
    ClassificationEvaluationSession target(id, o, expected);
    const auto before = snapshot(target);
    EXPECT_THROW(target.load_state(saved, RestorePolicy::exact_only),
                 CompatibilityError);
    EXPECT_EQ(snapshot(target).fields, before.fields);
  }
}

TEST(ClassificationEvaluationSession,
     RejectsCorruptProgressLateSliceAndIncompleteFieldsAtomically) {
  auto model = fixed_model();
  ClassificationEvaluationSession session(binding(), options(), 6);
  session.run(model, indexed_source());
  const auto saved = snapshot(session);
  for (unsigned changed = 0; changed < 13; ++changed) {
    auto corrupt = saved;
    switch (changed) {
      case 0:
        corrupt.fields["progress"].pop_back();
        break;
      case 1:
        corrupt.fields["progress"][0] ^= std::byte{1};
        break;
      case 2:
        set_integer(corrupt.fields["progress"], 24, 2);
        break;
      case 3:
        set_integer(corrupt.fields["progress"], 16, 0);
        break;
      case 4:
        set_integer(corrupt.fields["progress"], 16, 7);
        break;
      case 5:
        set_integer(corrupt.fields["progress"], 8, 1);
        break;
      case 6:
        set_integer(corrupt.fields["slice.3.counts"], 0, 1);
        break;
      case 7:
        set_integer(corrupt.fields["slice.0.counts"], 0,
                    std::numeric_limits<std::uint64_t>::max());
        break;
      case 8:
        corrupt.fields.erase("slice.3.counts");
        break;
      case 9:
        corrupt.fields["slice.3.topology"].pop_back();
        break;
      case 10:
        ++corrupt.descriptor.schema_version;
        break;
      case 11:
        corrupt.descriptor.fields.back().required = false;
        break;
      case 12:
        set_integer(corrupt.fields["slice.1.counts"], 0, 0);
        break;
    }
    ClassificationEvaluationSession target(binding(), options(), 6);
    const auto before = snapshot(target);
    EXPECT_THROW(target.load_state(corrupt, RestorePolicy::exact_only), Error)
        << changed;
    EXPECT_EQ(snapshot(target).fields, before.fields) << changed;
  }
}

TEST(ClassificationEvaluationSession,
     ExpectedSizeAndScoreFailuresDoNotCommitEOF) {
  auto model = fixed_model();
  ClassificationEvaluationSession session(binding(), options(), 6);
  const auto source = indexed_source();
  EXPECT_THROW(session.run(model,
                           [&](auto index, auto stop) {
                             return index == 1 ? std::nullopt
                                               : source(index, stop);
                           }),
               InvalidArgument);
  EXPECT_EQ(session.progress().batches, 1);
  EXPECT_FALSE(session.progress().complete);
  EXPECT_THROW(session.result(), NotReady);
  same_result(session.run(model, source), baseline());
  auto o = options();
  o.zero_division = metrics::ZeroDivisionPolicy::error;
  ClassificationEvaluationSession invalid_score(binding(), o, 6);
  EXPECT_THROW(invalid_score.run(model, source), NumericalError);
  EXPECT_EQ(invalid_score.progress().batches, 3);
  EXPECT_FALSE(invalid_score.progress().complete);
}

TEST(ClassificationEvaluationSession,
     IdentityPayloadAndDeadlineBoundsPrecedeExecution) {
  auto id = binding();
  id.dataset_digest.hex = "bad";
  EXPECT_THROW(static_cast<void>(ClassificationEvaluationSession(id)),
               InvalidArgument);
  id = binding();
  id.evaluation_id = "";
  EXPECT_THROW(static_cast<void>(ClassificationEvaluationSession(id)),
               InvalidArgument);
  id = binding();
  id.build_fingerprint.assign(1025, 'b');
  EXPECT_THROW(static_cast<void>(ClassificationEvaluationSession(id)),
               InvalidArgument);
  auto o = options();
  o.max_confusion_bytes = 719;  // (2*4+2)*3*3*8.
  EXPECT_THROW(ClassificationEvaluationSession(binding(), o),
               ResourceExhausted);
  o.max_confusion_bytes = 720;
  EXPECT_NO_THROW(ClassificationEvaluationSession(binding(), o));
  o = {};
  o.max_sources = 300;
  for (unsigned i = 0; i < 300; ++i)
    o.sources.push_back(std::to_string(i) + std::string(250, 'x'));
  EXPECT_THROW(ClassificationEvaluationSession(binding(), o),
               ResourceExhausted);
  auto model = fixed_model();
  ClassificationEvaluationSession session(binding(), options(), 6);
  int calls = 0;
  const auto source =
      [&](auto, auto) -> std::optional<ClassificationEvaluationBatch> {
    ++calls;
    return std::nullopt;
  };
  EXPECT_THROW(
      session.run(model, source, {}, {}, std::chrono::steady_clock::now()),
      DeadlineExceeded);
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(session.run(model, source, {}, stop.get_token()), Cancelled);
  EXPECT_THROW(session.run(model, {}), InvalidArgument);
  EXPECT_EQ(calls, 0);
  const auto state = snapshot(session);
  EXPECT_EQ(state.fields.at("progress").size(), 32);
  EXPECT_EQ(state.fields.at("slice.0.counts").size(), 72);
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
void durable(bool cuda) {
  TempRoot root;
  const auto path = root.path() / "evaluation.state";
  {
    auto model = fixed_model(cuda);
    auto session = std::make_shared<ClassificationEvaluationSession>(
        binding(), options(), 6);
    CheckpointParticipantRegistry registry;
    registry.register_participant(session);
    CheckpointCoordinator coordinator(registry);
    FileCheckpointStore store(path);
    EXPECT_THROW(session->run(model, indexed_source(cuda),
                              [&](const auto&) {
                                coordinator.save(store);
                                throw std::runtime_error("interrupt");
                              }),
                 std::runtime_error);
  }
  auto model = fixed_model(cuda);
  auto session = std::make_shared<ClassificationEvaluationSession>(
      binding(), options(), 6);
  CheckpointParticipantRegistry registry;
  registry.register_participant(session);
  CheckpointCoordinator coordinator(registry);
  FileCheckpointStore reopened(path);
  coordinator.restore(reopened);
  EXPECT_EQ(session->progress().batches, 1);
  same_result(session->run(model, indexed_source(cuda)), baseline(cuda));
  coordinator.save(reopened);
  EXPECT_TRUE(session->progress().complete);
}
TEST(ClassificationEvaluationSession,
     PhysicalCheckpointReopensAndContinuesCPU) {
  durable(false);
}
TEST(ClassificationEvaluationSession,
     PhysicalCheckpointReopensAndContinuesCUDA) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA unavailable";
  durable(true);
}

#if GTEST_HAS_DEATH_TEST
// Windows death tests re-execute the fixture; the environment shares only the
// parent's reserved test root. A marker prevents borrowing an unrelated path.
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
    std::ofstream(path_ / ".fixture")
        << "classification evaluation abrupt exit";
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
      L"JYPPX_LIUDING_TEST_EVALUATION_RECOVERY_ROOT";
#else
  static constexpr auto kEnvironment =
      "JYPPX_LIUDING_TEST_EVALUATION_RECOVERY_ROOT";
#endif
  std::optional<std::filesystem::path::string_type> previous_;
  std::unique_ptr<TempRoot> root_;
  std::filesystem::path path_;
};
TEST(ClassificationEvaluationSessionDeathTest,
     AbruptExitAfterBatchCommitReplaysOnlySuffix) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ExitRoot root;
  const auto path = root.path() / "evaluation.state";
  EXPECT_EXIT(
      {
        auto model = fixed_model();
        auto session = std::make_shared<ClassificationEvaluationSession>(
            binding(), options(), 6);
        CheckpointParticipantRegistry registry;
        registry.register_participant(session);
        CheckpointCoordinator coordinator(registry);
        FileCheckpointStore store(path);
        session->run(model, indexed_source(), [&](const auto&) {
          coordinator.save(store);
          std::_Exit(73);
        });
        std::_Exit(74);
      },
      ::testing::ExitedWithCode(73), "");
  auto model = fixed_model();
  auto session = std::make_shared<ClassificationEvaluationSession>(
      binding(), options(), 6);
  CheckpointParticipantRegistry registry;
  registry.register_participant(session);
  CheckpointCoordinator coordinator(registry);
  FileCheckpointStore store(path);
  coordinator.restore(store);
  auto source = indexed_source();
  std::vector<std::uint64_t> requested;
  const auto result = session->run(model, [&](auto index, auto stop) {
    requested.push_back(index);
    return source(index, stop);
  });
  EXPECT_EQ(requested, (std::vector<std::uint64_t>{1, 2, 3}));
  same_result(result, baseline());
}
#if defined(JYPPX_LIUDING_EVALUATION_REPORT_TEST)
std::string report_json(const ClassificationEvaluationResult& result) {
  experiment::EvaluationSuite suite{
      .suite_id = "classification.fixture",
      .task = "classification",
      .dataset_digest = binding().dataset_digest,
      .environment_id = "fixed-linear-fixture/v1"};
  for (std::size_t i = 0; i < result.observations.size(); ++i) {
    const auto& point = result.observations[i];
    suite.rules.push_back({.rule_id = "r" + std::to_string(i),
                           .definition = point.definition,
                           .context = point.context,
                           .minimum_samples = 1,
                           .minimum = 0.});
  }
  const auto id = binding();
  return experiment::quality_report_json(experiment::evaluate_observations(
      suite,
      {.evaluation_id = id.evaluation_id,
       .run_id = "run.fixture",
       .model_digest = id.model_digest,
       .build_fingerprint = id.build_fingerprint,
       .adapter_id = "training.classification.session/v1",
       .config_digest = id.config_digest},
      result.observations));
}
experiment::RunArtifactReference publish_report(
    const std::filesystem::path& root, const std::string& bytes) {
  experiment::SQLiteRunStore store(root);
  store.start({.run_id = "run.fixture",
               .project_id = "project.fixture",
               .experiment_id = "experiment.fixture",
               .display_name = "evaluation fixture",
               .created_at_unix_ms = 10,
               .updated_at_unix_ms = 10});
  return store.put_artifact(
      "run.fixture", "eval.fixture.report",
      experiment::RunArtifactKind::artifact, "evaluation.report",
      std::as_bytes(std::span(bytes)),
      {.logical_name = "eval.fixture.quality.json",
       .media_type = "application/vnd.jyppx.quality-report+json;version=1",
       .producer = "training.classification.session/v1",
       .expected_size_bytes = bytes.size()},
      20);
}
void completed_crash(bool published) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ExitRoot root;
  const auto path = root.path() / "evaluation.state",
             runs = root.path() / "runs";
  EXPECT_EXIT(
      {
        auto model = fixed_model();
        auto session = std::make_shared<ClassificationEvaluationSession>(
            binding(), options(), 6);
        CheckpointParticipantRegistry registry;
        registry.register_participant(session);
        CheckpointCoordinator coordinator(registry);
        FileCheckpointStore store(path);
        session->run(model, indexed_source(), [&](const auto& state) {
          coordinator.save(store);
          if (!state.progress().complete) return;
          if (published) publish_report(runs, report_json(state.result()));
          std::_Exit(73);
        });
        std::_Exit(74);
      },
      ::testing::ExitedWithCode(73), "");
  auto model = fixed_model();
  auto session = std::make_shared<ClassificationEvaluationSession>(
      binding(), options(), 6);
  CheckpointParticipantRegistry registry;
  registry.register_participant(session);
  CheckpointCoordinator coordinator(registry);
  FileCheckpointStore store(path);
  coordinator.restore(store);
  ASSERT_TRUE(session->progress().complete);
  int acknowledgements = 0;
  const auto result = session->run(
      model,
      [](auto, auto) -> std::optional<ClassificationEvaluationBatch> {
        throw std::runtime_error(
            "completed evaluation must not re-read/forward");
      },
      [&](const auto& state) {
        publish_report(runs, report_json(state.result()));
        ++acknowledgements;
      });
  EXPECT_EQ(acknowledgements, 1);
  const auto canonical = report_json(result);
  EXPECT_EQ(canonical, report_json(baseline()));
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
    const auto n = reader->read(std::span(bytes).subspan(offset));
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
TEST(ClassificationEvaluationSessionDeathTest,
     AbruptExitAfterEOFBeforeReportRecoversOneVerdict) {
  completed_crash(false);
}
TEST(ClassificationEvaluationSessionDeathTest,
     AbruptExitAfterReportCommitReplaysOneVerdict) {
  completed_crash(true);
}
#endif
#endif
#endif
}  // namespace
