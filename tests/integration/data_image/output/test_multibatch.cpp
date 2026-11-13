// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");

#include "pipeline_fixture.hpp"

#include <functional>
#include <vector>

namespace {
using namespace image_output_pipeline_test;
namespace experiment = jyppx::liuding::experiment;

// Destruction opens the gate, including early test exits. The callback keeps
// its own shared state alive while the sink drains.
class EvidenceGate {
 public:
  EvidenceGate() : entered(state_->entered.get_future()) {}
  ~EvidenceGate() { open(); }
  void open() {
    if (!state_->opened.exchange(true)) state_->release.set_value();
  }
  std::function<void()> waiter() const {
    return [state = state_] {
      state->entered.set_value();
      if (state->released.wait_for(10s) != std::future_status::ready)
        throw std::runtime_error("Evidence gate timed out");
    };
  }

 private:
  struct State {
    std::promise<void> entered, release;
    std::shared_future<void> released = release.get_future().share();
    std::atomic<bool> opened{false};
  };
  std::shared_ptr<State> state_ = std::make_shared<State>();

 public:
  std::future<void> entered;
};

void Start(experiment::SQLiteRunStore& store) {
  store.start({.run_id = "pipeline",
               .project_id = "images",
               .experiment_id = "multi-batch",
               .display_name = "Bounded multi-batch output",
               .status = experiment::RunStatus::running,
               .created_at_unix_ms = 100,
               .updated_at_unix_ms = 100});
}

image::ImageOutputSinkOptions SinkOptions(Fixture& f) {
  return {.max_pending_items = 2,
          .max_pending_bytes = 2 * kBatchBytes,
          .max_item_bytes = kBatchBytes,
          .resource_budget = f.budget,
          .wait_for_aggregate_queue = true};
}

image::ImageSinkSubmission Queue(
    Fixture& f, image::ImageOutputSink& sink,
    experiment::SQLiteRunStore& store, const data::BatchEnvelope& envelope,
    std::uint64_t sequence, std::function<void()> before = {},
    std::uint64_t event_sequence = 0) {
  return sink.produce_and_submit(
      f.root / (std::to_string(sequence) + ".png"), kBatchBytes,
      [&](std::stop_token stop) { return f.preview(envelope, stop); }, {}, {},
      [&, sequence, event_sequence, before = std::move(before),
       lease = envelope.admission_lease](
          const image::ImageSinkWriteResult& file) {
        (void)lease;
        if (before) before();
        const auto id = std::to_string(sequence);
        experiment::PublishedFileEvidenceRequest request;
        request.run_id = "pipeline";
        request.reference_id = id;
        request.path = file.path;
        request.descriptor = {.logical_name = id + ".png",
                              .media_type = "image/png",
                              .producer = "multi-batch-output-test",
                              .expected_size_bytes = file.bytes};
        request.attached_at_unix_ms = 100 + sequence;
        request.expected_digest = {.hex = file.sha256};
        request.event_batch_key = id;
        request.event.identity = {.stream_id = "previews",
                                  .event_id = id,
                                  .idempotency_key = id,
                                  .sequence = event_sequence == 0
                                                  ? sequence
                                                  : event_sequence};
        request.event.correlation.project_id = "images";
        request.event.correlation.run_id = "pipeline";
        request.event.payload =
            TelemetryArtifactReference{.role = "image.preview"};
        const auto receipt =
            store.record_published_file_evidence_atomic(request);
        if (receipt.reference_replay || receipt.event.batch_replay)
          throw std::runtime_error("Unexpected evidence replay");
      });
}

void ExpectPressure(Fixture& f) {
  EXPECT_EQ(f.decoded, 4U);
  EXPECT_EQ(f.acquired, 2U);
  EXPECT_EQ(f.budget->limit(image::ImageResourceKind::pipeline_inflight_batches),
            4U);
  EXPECT_EQ(f.budget->usage(image::ImageResourceKind::pipeline_inflight_batches)
                .current,
            2U);
  EXPECT_EQ(f.budget->usage(image::ImageResourceKind::pipeline_inflight_bytes)
                .current,
            2 * kBatchBytes);
  EXPECT_EQ(f.budget->usage(image::ImageResourceKind::decoded_bytes).current,
            0U);
}

void ExpectDurable(Fixture& f,
                   const std::vector<image::ImageSinkWriteResult>& files,
                   const std::vector<std::uint64_t>& sequences) {
  experiment::SQLiteRunStore reopened(f.root / "store");
  EXPECT_TRUE(reopened.integrity_check().ok());
  // Each atomic observation owns the file reference and the event payload.
  EXPECT_EQ(reopened.references("pipeline").size(), 2 * files.size());
  EXPECT_EQ(reopened.events("pipeline").events.size(), files.size());
  ASSERT_EQ(files.size(), sequences.size());
  for (std::size_t i = 0; i < files.size(); ++i) {
    const auto ref =
        reopened.find_reference("pipeline", std::to_string(sequences[i]));
    ASSERT_TRUE(ref);
    EXPECT_EQ(ref->artifact.digest.hex, files[i].sha256);
    EXPECT_EQ(ref->artifact.size_bytes, files[i].bytes);
    EXPECT_EQ(std::filesystem::file_size(files[i].path), files[i].bytes);
  }
  EXPECT_EQ(f.budget->usage(image::ImageResourceKind::pipeline_inflight_batches)
                .peak,
            2U);
  EXPECT_EQ(f.budget->usage(image::ImageResourceKind::pipeline_inflight_bytes)
                .peak,
            2 * kBatchBytes);
}

TEST(ImageOutputMultiBatchTest, ByteCapSpansTwoDurableEvidenceOwners) {
  Fixture f(2, 8);
  experiment::SQLiteRunStore store(f.root / "store");
  Start(store);
  image::ImageOutputSink sink(SinkOptions(f));
  EvidenceGate first_gate, second_gate;
  auto first = f.pipeline->next(), second = f.pipeline->next();
  ASSERT_TRUE(first && second);
  std::vector<image::ImageSinkSubmission> tickets;
  tickets.push_back(Queue(f, sink, store, *first, 1, first_gate.waiter()));
  tickets.push_back(Queue(f, sink, store, *second, 2, second_gate.waiter()));
  first.reset();
  second.reset();
  std::stop_source stop;
  auto next = std::async(std::launch::async,
                         [&] { return f.pipeline->next(stop.get_token()); });
  EXPECT_EQ(first_gate.entered.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(f.blocked_attempt.get_future().wait_for(5s),
            std::future_status::ready);
  EXPECT_EQ(next.wait_for(0s), std::future_status::timeout);
  ExpectPressure(f);
  first_gate.open();
  tickets[0].evidence_completion.get();
  if (next.wait_for(5s) != std::future_status::ready) stop.request_stop();
  auto third = next.get();
  ASSERT_TRUE(third);
  EXPECT_EQ(second_gate.entered.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(f.acquired, 3U);
  EXPECT_EQ(f.decoded, 6U);
  tickets.push_back(Queue(f, sink, store, *third, 3));
  third.reset();
  second_gate.open();
  tickets[1].evidence_completion.get();
  tickets[2].evidence_completion.get();
  auto fourth = f.pipeline->next();
  ASSERT_TRUE(fourth);
  tickets.push_back(Queue(f, sink, store, *fourth, 4));
  fourth.reset();
  std::vector<image::ImageSinkWriteResult> files;
  for (auto& ticket : tickets) {
    files.push_back(ticket.completion.get());
    if (ticket.evidence_completion.valid()) ticket.evidence_completion.get();
  }
  EXPECT_FALSE(f.pipeline->next());
  sink.close();
  EXPECT_EQ(f.decoded, 8U);
  ExpectDurable(f, files, {1, 2, 3, 4});
  f.expect_released();
}

TEST(ImageOutputMultiBatchTest, CancelledThirdReadPreservesTwoDurableOutputs) {
  Fixture f(2, 8);
  experiment::SQLiteRunStore store(f.root / "store");
  Start(store);
  image::ImageOutputSink sink(SinkOptions(f));
  EvidenceGate gate;
  auto first = f.pipeline->next(), second = f.pipeline->next();
  ASSERT_TRUE(first && second);
  auto a = Queue(f, sink, store, *first, 1, gate.waiter());
  auto b = Queue(f, sink, store, *second, 2);
  first.reset();
  second.reset();
  EXPECT_EQ(gate.entered.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(f.blocked_attempt.get_future().wait_for(5s),
            std::future_status::ready);
  std::stop_source stop;
  auto next = std::async(std::launch::async,
                         [&] { return f.pipeline->next(stop.get_token()); });
  ExpectPressure(f);
  stop.request_stop();
  EXPECT_FALSE(next.get());
  f.pipeline->close();
  ExpectPressure(f);
  gate.open();
  a.evidence_completion.get();
  b.evidence_completion.get();
  sink.close();
  ExpectDurable(f, {a.completion.get(), b.completion.get()}, {1, 2});
  EXPECT_EQ(f.decoded, 4U);
  f.expect_released();
}

TEST(ImageOutputMultiBatchTest, FailedEvidenceReleasesOnlyItsOwnAdmission) {
  Fixture f(2, 8);
  experiment::SQLiteRunStore store(f.root / "store");
  Start(store);
  auto options = SinkOptions(f);
  options.continue_after_error = true;
  image::ImageOutputSink sink(options);
  EvidenceGate failed_gate, retained_gate;
  auto first = f.pipeline->next(), second = f.pipeline->next();
  ASSERT_TRUE(first && second);
  const auto wait = failed_gate.waiter();
  auto a = Queue(f, sink, store, *first, 1, [wait] {
    wait();
    throw std::runtime_error("Rejected evidence");
  });
  // Rejected evidence never consumed producer sequence 1.
  auto b = Queue(f, sink, store, *second, 2, retained_gate.waiter(), 1);
  first.reset();
  second.reset();
  EXPECT_EQ(failed_gate.entered.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(f.blocked_attempt.get_future().wait_for(5s),
            std::future_status::ready);
  ExpectPressure(f);
  failed_gate.open();
  EXPECT_THROW(a.evidence_completion.get(), std::runtime_error);
  const auto published = a.completion.get();
  EXPECT_TRUE(std::filesystem::is_regular_file(published.path));
  EXPECT_EQ(retained_gate.entered.wait_for(5s), std::future_status::ready);
  std::stop_source stop;
  auto next = std::async(std::launch::async,
                         [&] { return f.pipeline->next(stop.get_token()); });
  if (next.wait_for(5s) != std::future_status::ready) stop.request_stop();
  auto third = next.get();
  ASSERT_TRUE(third);
  EXPECT_EQ(f.acquired, 3U);
  EXPECT_EQ(f.decoded, 6U);
  f.pipeline->close();
  auto c = Queue(f, sink, store, *third, 3, {}, 2);
  third.reset();
  retained_gate.open();
  b.evidence_completion.get();
  c.evidence_completion.get();
  EXPECT_THROW(sink.close(), std::runtime_error);
  EXPECT_FALSE(store.find_reference("pipeline", "1"));
  EXPECT_EQ(sink.stats().failed, 1U);
  ExpectDurable(f, {b.completion.get(), c.completion.get()}, {2, 3});
  f.expect_released();
}
}  // namespace
