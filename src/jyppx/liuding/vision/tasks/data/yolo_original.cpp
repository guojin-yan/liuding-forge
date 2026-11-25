// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "yolo_original_detail.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <unordered_set>

namespace jyppx::liuding::vision::tasks {
namespace {
namespace im = data::image;
using J = nlohmann::json;
constexpr std::size_t kStateBytes = 8U << 20;
void Stop(std::stop_token stop) {
  if (stop.stop_requested())
    throw Cancelled("Original YOLO preparation cancelled");
}
void Probability(double value) {
  if (!std::isfinite(value) || value < 0 || value > 1)
    throw InvalidArgument("Original YOLO probability outside[0,1]");
}
std::string Hash(const std::string& text) {
  return sha256_hex(std::as_bytes(std::span(text)));
}
bool Closed(const YoloDataOptions& o, std::uint64_t epoch) {
  const auto& m = o.augmentation.multi_source;
  return m.close_epochs && epoch >= m.total_epochs - m.close_epochs;
}
void Loaded(std::size_t index, std::size_t limit,
            detail::OriginalYoloBuffer& buffer) {
  if (std::find(buffer.begin(), buffer.end(), index) != buffer.end()) return;
  buffer.push_back(index);
  if (buffer.size() > 1 && buffer.size() >= limit) buffer.pop_front();
}
struct Draws {
  im::SampleRandom random;
  std::uint64_t cursor = 0;
  double Unit() { return random.unit(cursor++); }
  std::size_t Index(std::size_t size) {
    return std::size_t(random.bounded(size, cursor++));
  }
  double Uniform(double a, double b) { return a + (b - a) * Unit(); }
};
detail::OriginalYoloView View(std::size_t index, std::size_t population,
                              std::size_t limit,
                              detail::OriginalYoloBuffer& buffer,
                              const YoloDataOptions& o, bool closed, Draws& d) {
  static_cast<void>(population);
  detail::OriginalYoloView result;
  result.indices.fill(index);
  Loaded(index, limit, buffer);
  const auto& a = o.augmentation;
  const auto& p = a.projective;
  auto& g = result.geometry;
  g.size = o.letterbox.canvas.width;
  g.mosaic = d.Unit() < (!closed ? a.multi_source.mosaic_probability : 0);
  if (g.mosaic) {
    for (std::size_t i = 1; i < 4; ++i)
      result.indices[i] = buffer[d.Index(buffer.size())];
    for (std::size_t i = 1; i < 4; ++i)
      Loaded(result.indices[i], limit, buffer);
    g.mosaic_center[1] =
        std::uint32_t(d.Uniform(double(g.size) / 2, double(g.size) * 1.5));
    g.mosaic_center[0] =
        std::uint32_t(d.Uniform(double(g.size) / 2, double(g.size) * 1.5));
  }
  g.perspective = p.perspective != 0;
  g.perspective_x = d.Uniform(-p.perspective, p.perspective);
  g.perspective_y = d.Uniform(-p.perspective, p.perspective);
  g.rotation_degrees = d.Uniform(-p.degrees, p.degrees);
  g.scale = d.Uniform(1 - p.scale_gain, 1 + p.scale_gain);
  g.shear_x_degrees = d.Uniform(-p.shear_degrees, p.shear_degrees);
  g.shear_y_degrees = d.Uniform(-p.shear_degrees, p.shear_degrees);
  g.translate_x =
      d.Uniform(.5 - p.translate_fraction, .5 + p.translate_fraction);
  g.translate_y =
      d.Uniform(.5 - p.translate_fraction, .5 + p.translate_fraction);
  return result;
}
}  // namespace

std::string yolo_original_data_identity(const YoloOriginalDataOptions& o) {
  if (o.max_plan_bytes < 1024 || o.max_plan_bytes > (1ULL << 32) ||
      o.decoded_cache_bytes > (1ULL << 34))
    throw InvalidArgument("Original YOLO plan/cache limits outside bounds");
  return Hash("yolo-original/opencv5.1-box-v1:" + std::to_string(o.enabled) +
              ":" + std::to_string(o.training) + ":" +
              std::to_string(o.max_plan_bytes) + ":" +
              std::to_string(o.decoded_cache_bytes));
}
double detail::original_yolo_beta32(const im::SampleRandom& random,
                                    std::uint64_t first) {
  if (first > std::numeric_limits<std::uint64_t>::max() - 63)
    throw InvalidArgument("Beta draw cursor overflow");
  double a = 0, b = 0;
  for (std::uint64_t i = 0; i < 32; ++i) {
    a -= std::log1p(-random.unit(first + i));
    b -= std::log1p(-random.unit(first + 32 + i));
  }
  if (!(a + b > 0)) throw NumericalError("Degenerate original YOLO Beta draw");
  return a / (a + b);
}
void detail::check_original_yolo(const YoloDataOptions& o) {
  static_cast<void>(yolo_original_data_identity(o.yolo_original));
  const auto& a = o.augmentation;
  const auto& p = a.projective;
  const auto& m = a.multi_source;
  if (!yolo_opencv_available())
    throw NotImplemented(
        "Original YOLO profile requires the explicit OpenCV backend");
  if (o.rtdetr.enabled || o.classification.enabled || o.rectangular ||
      o.multiscale || o.letterbox.canvas.width != o.letterbox.canvas.height ||
      o.letterbox.canvas.width < 32 || o.letterbox.canvas.width > 4096 ||
      o.letterbox.canvas.width % 32 || !o.letterbox.scale_up ||
      !o.letterbox.center || o.letterbox.padding_value != 114 ||
      a.spatial.scale_gain || a.spatial.translate_fraction ||
      a.photo.grayscale_probability || a.photo.blur_probability ||
      a.photo.median_blur_probability || a.photo.clahe_probability ||
      m.cutmix_probability || p.min_box_area != 4 || p.min_visibility != .1 ||
      !m.total_epochs || m.close_epochs > m.total_epochs)
    throw InvalidArgument(
        "Original YOLO box profile has incompatible geometry or augmentation "
        "options");
  for (auto value :
       {m.mosaic_probability, m.mixup_probability, m.copypaste_probability,
        a.horizontal_flip_probability, a.vertical_flip_probability,
        a.bgr_probability, a.photo.hue_shift, a.photo.saturation_gain,
        a.photo.value_gain})
    Probability(value);
}
detail::OriginalYoloSample detail::plan_original_yolo_sample(
    std::size_t primary, std::size_t population, std::size_t limit,
    OriginalYoloBuffer& buffer, const YoloDataOptions& o, std::uint64_t epoch,
    std::uint64_t occurrence) {
  if (!population || primary >= population || !limit)
    throw InvalidArgument("Invalid original YOLO planner population");
  Draws d{im::SampleRandom({.dataset_id = o.augmentation.dataset_id,
                            .sample_id = std::to_string(primary),
                            .transform_id = "yolo-original-sample/v1",
                            .seed = o.augmentation.seed,
                            .epoch = epoch,
                            .occurrence = occurrence})};
  OriginalYoloSample result;
  result.color.center_box_flips = true;
  result.primary =
      View(primary, population, limit, buffer, o, Closed(o, epoch), d);
  result.mixup =
      d.Unit() <
      (!Closed(o, epoch) ? o.augmentation.multi_source.mixup_probability : 0);
  if (result.mixup) {
    result.partner = View(d.Index(population), population, limit, buffer, o,
                          Closed(o, epoch), d);
    result.color.mixup_ratio = original_yolo_beta32(d.random, d.cursor);
    d.cursor += 64;
  }
  // Original cutmix has probability zero and box-only CopyPaste returns
  // before drawing. Photometry and flips continue after multi-source closure.
  static_cast<void>(d.Unit());
  result.color.hsv = o.augmentation.photo.hue_shift ||
                     o.augmentation.photo.saturation_gain ||
                     o.augmentation.photo.value_gain;
  result.color.hue_shift = d.Uniform(-o.augmentation.photo.hue_shift,
                                     o.augmentation.photo.hue_shift);
  result.color.saturation_gain =
      1 + d.Uniform(-o.augmentation.photo.saturation_gain,
                    o.augmentation.photo.saturation_gain);
  result.color.value_gain = 1 + d.Uniform(-o.augmentation.photo.value_gain,
                                          o.augmentation.photo.value_gain);
  result.color.vertical_flip =
      d.Unit() < o.augmentation.vertical_flip_probability;
  result.color.horizontal_flip =
      d.Unit() < o.augmentation.horizontal_flip_probability;
  result.bgr = d.Unit() < o.augmentation.bgr_probability;
  return result;
}

namespace {
struct EpochPlan {
  std::uint64_t epoch = 0;
  std::vector<std::size_t> order;
  std::vector<detail::OriginalYoloSample> samples;
  std::vector<detail::OriginalYoloBuffer> begin, end;
  std::size_t worker_offset = 0;
};
struct Plans {
  mutable std::mutex mutex;
  std::shared_ptr<const EpochPlan> current, previous;
  void Publish(std::shared_ptr<const EpochPlan> p) {
    std::lock_guard lock(mutex);
    if (current && current->epoch != p->epoch) previous = current;
    current = std::move(p);
  }
  std::shared_ptr<const EpochPlan> Find(std::uint64_t epoch) const {
    std::lock_guard lock(mutex);
    if (current && current->epoch == epoch) return current;
    if (previous && previous->epoch == epoch) return previous;
    throw NotReady("Original YOLO epoch plan is not prepared");
  }
};
class PlannedSampler final : public data::samplers::Sampler {
 public:
  PlannedSampler(std::shared_ptr<data::samplers::Sampler> source,
                 std::shared_ptr<Plans> plans, YoloDataOptions options,
                 std::size_t workers, std::size_t batch, bool drop)
      : source_(std::move(source)),
        plans_(std::move(plans)),
        options_(std::move(options)),
        workers_(workers),
        batch_(batch),
        drop_(drop) {
    if (!source_->supports_state() || !source_->supports_epoch_control() ||
        source_->configuration_identity().empty())
      throw InvalidArgument(
          "Original YOLO requires a checkpointable epoch-controlled "
          "permutation sampler");
    population_ = source_->size();
    limit_ = std::min({population_, batch_ * 8, std::size_t{1000}});
    const auto history_bytes =
        sizeof(EpochPlan) +
        workers_ * 2 *
            (sizeof(detail::OriginalYoloBuffer) +
             std::max(std::size_t{1}, limit_ ? limit_ - 1 : 0) *
                 sizeof(std::size_t));
    const auto budget = options_.yolo_original.max_plan_bytes;
    if (!population_ || history_bytes > budget ||
        population_ >
            (budget - history_bytes) /
                (sizeof(detail::OriginalYoloSample) + sizeof(std::size_t)))
      throw ResourceExhausted("Original YOLO epoch plan exceeds payload limit");
    identity_ =
        "yolo-original-permutation/v1:" +
        Hash(source_->configuration_identity() + ":" + source_->save_state() +
             ":" + yolo_data_identity(options_) + ":" +
             std::to_string(workers_) + ":" + std::to_string(batch_) + ":" +
             std::to_string(drop_));
  }
  std::size_t size() const override { return population_; }
  bool supports_state() const override { return true; }
  bool supports_epoch_control() const override { return true; }
  std::string configuration_identity() const override { return identity_; }
  void set_epoch(std::uint64_t epoch) override {
    if (epoch >= options_.augmentation.multi_source.total_epochs)
      throw InvalidArgument(
          "Original YOLO epoch exceeds configured training schedule");
    if (epoch &&
        (!prepared_ || (epoch != plan_->epoch && epoch != plan_->epoch + 1)))
      throw InvalidArgument(
          "Original YOLO history requires epoch0, the next epoch or restored "
          "state");
    pending_ = epoch;
    source_->set_epoch(epoch);
  }
  std::vector<std::size_t> indices() const override {
    auto order = source_->indices();
    CheckOrder(order);
    std::vector<detail::OriginalYoloBuffer> begin(workers_);
    if (prepared_ && pending_ == plan_->epoch)
      begin = plan_->begin;
    else if (prepared_ && pending_ == plan_->epoch + 1)
      begin = plan_->end;
    if (!pending_ ||
        (Closed(options_, pending_) && !Closed(options_, pending_ - 1)))
      for (auto& buffer : begin) buffer.clear();
    auto p = Build(pending_, std::move(order), std::move(begin));
    plans_->Publish(p);
    plan_ = std::move(p);
    prepared_ = true;
    return plan_->order;
  }
  std::string save_state() const override {
    J state{{"format", "yolo-original-plan/v1"},
            {"identity", identity_},
            {"source", source_->save_state()},
            {"prepared", prepared_},
            {"pending", pending_}};
    if (prepared_)
      state.update({{"epoch", plan_->epoch},
                    {"order", plan_->order},
                    {"begin", plan_->begin}});
    auto text = state.dump();
    if (text.size() > kStateBytes)
      throw ResourceExhausted("Original YOLO sampler state exceeds8MiB");
    return text;
  }
  void load_state(const std::string& text) override {
    if (text.size() > kStateBytes)
      throw ResourceExhausted("Original YOLO sampler state exceeds8MiB");
    const auto j = J::parse(text);
    if (j.at("format") != "yolo-original-plan/v1" ||
        j.at("identity") != identity_)
      throw CompatibilityError("Original YOLO sampler identity mismatch");
    const auto prepared = j.at("prepared").get<bool>();
    const auto pending = j.at("pending").get<std::uint64_t>();
    if (pending >= options_.augmentation.multi_source.total_epochs)
      throw SerializationError("Original YOLO restored epoch outside schedule");
    std::shared_ptr<const EpochPlan> p;
    if (prepared) {
      auto order = j.at("order").get<std::vector<std::size_t>>();
      CheckOrder(order);
      auto begin = j.at("begin").get<std::vector<detail::OriginalYoloBuffer>>();
      CheckBuffers(begin);
      const auto epoch = j.at("epoch").get<std::uint64_t>();
      if (epoch >= options_.augmentation.multi_source.total_epochs ||
          (pending != epoch && pending != epoch + 1 && pending != 0))
        throw SerializationError("Original YOLO restored plan epoch mismatch");
      p = Build(epoch, std::move(order), std::move(begin));
    }
    const auto old = source_->save_state();
    try {
      source_->load_state(j.at("source").get<std::string>());
    } catch (...) {
      source_->load_state(old);
      throw;
    }
    if (p) plans_->Publish(p);
    plan_ = std::move(p);
    prepared_ = prepared;
    pending_ = pending;
  }

 private:
  void CheckOrder(const std::vector<std::size_t>& order) const {
    if (order.size() != population_)
      throw SerializationError("Original YOLO requires a complete permutation");
    std::vector<bool> seen(population_);
    for (auto index : order) {
      if (index >= population_ || seen[index])
        throw SerializationError(
            "Original YOLO permutation contains repeated/outside indices");
      seen[index] = true;
    }
  }
  void CheckBuffers(
      const std::vector<detail::OriginalYoloBuffer>& buffers) const {
    if (buffers.size() != workers_)
      throw SerializationError("Original YOLO restored worker count mismatch");
    for (const auto& b : buffers) {
      if (b.size() > std::max(std::size_t{1}, limit_ - 1))
        throw SerializationError("Original YOLO buffer exceeds cap");
      std::unordered_set<std::size_t> seen;
      for (auto i : b)
        if (i >= population_ || !seen.insert(i).second)
          throw SerializationError("Invalid original YOLO buffer inventory");
    }
  }
  std::shared_ptr<const EpochPlan> Build(
      std::uint64_t epoch, std::vector<std::size_t> order,
      std::vector<detail::OriginalYoloBuffer> begin) const {
    auto p = std::make_shared<EpochPlan>();
    p->epoch = epoch;
    p->order = std::move(order);
    p->begin = begin;
    p->end = std::move(begin);
    const auto count = drop_ ? population_ / batch_ * batch_ : population_;
    p->samples.reserve(count);
    const auto batches = count / batch_ + (count % batch_ != 0);
    const auto close = options_.augmentation.multi_source.total_epochs -
                       options_.augmentation.multi_source.close_epochs;
    const auto since_reset = Closed(options_, epoch) ? epoch - close : epoch;
    p->worker_offset =
        ((since_reset % workers_) * (batches % workers_)) % workers_;
    for (std::size_t i = 0; i < count; ++i) {
      auto& buffer = p->end[(i / batch_ + p->worker_offset) % workers_];
      p->samples.push_back(detail::plan_original_yolo_sample(
          p->order[i], population_, limit_, buffer, options_, epoch, i));
    }
    return p;
  }
  std::shared_ptr<data::samplers::Sampler> source_;
  std::shared_ptr<Plans> plans_;
  YoloDataOptions options_;
  std::size_t workers_, batch_, population_ = 0, limit_ = 0;
  bool drop_;
  std::string identity_;
  std::uint64_t pending_ = 0;
  mutable bool prepared_ = false;
  mutable std::shared_ptr<const EpochPlan> plan_;
};
class PixelCache {
 public:
  PixelCache(std::size_t maximum, std::uint64_t bytes)
      : maximum_(maximum), maximum_bytes_(bytes) {}
  im::VisionSample Get(const im::ImageDataset& dataset, std::size_t index,
                       std::stop_token stop) {
    Stop(stop);
    std::lock_guard lock(mutex_);
    if (const auto found = images_.find(index); found != images_.end())
      return found->second;
    auto sample = dataset.get(index, stop);
    const auto bytes =
        std::uint64_t(sample.image.numel()) * sample.image.element_size();
    if (bytes > maximum_bytes_ || !maximum_) return sample;
    while (!fifo_.empty() &&
           (images_.size() >= maximum_ || used_ > maximum_bytes_ - bytes)) {
      const auto victim = fifo_.front();
      fifo_.pop_front();
      const auto& old = images_.at(victim);
      used_ -= std::uint64_t(old.image.numel()) * old.image.element_size();
      images_.erase(victim);
    }
    images_.emplace(index, sample);
    fifo_.push_back(index);
    used_ += bytes;
    return sample;
  }

 private:
  std::mutex mutex_;
  std::size_t maximum_;
  std::uint64_t maximum_bytes_, used_ = 0;
  std::deque<std::size_t> fifo_;
  std::unordered_map<std::size_t, im::VisionSample> images_;
};
im::ImageBatch Batch(std::span<const im::VisionSample> samples,
                     std::span<const im::ImageBatchKey> keys,
                     const YoloDataOptions& o, std::stop_token stop) {
  im::ImageCollateOptions collate;
  collate.limits = o.limits;
  collate.resource_budget = o.letterbox.geometry.resource_budget;
  auto batch = im::collate_images(samples, keys, collate, stop);
  if (o.numeric_on_consumer) return batch;
  im::ImageBatchTransformOptions numeric;
  numeric.limits = o.limits;
  numeric.resource_budget = collate.resource_budget;
  numeric.image.mean = o.normalization_mean;
  numeric.image.std = o.normalization_std;
  return im::transform_image_batch(batch, numeric, stop);
}
}  // namespace

im::ImageBatch detail::prepare_original_yolo_evaluation_batch(
    std::span<const im::VisionSample> samples,
    std::span<const im::ImageBatchKey> keys, const YoloDataOptions& o,
    bool defer_numeric, std::stop_token stop) {
  if (o.yolo_original.training)
    throw InvalidArgument(
        "Original YOLO training needs the dataset pipeline and its partner "
        "history");
  if (samples.empty() || samples.size() != keys.size() ||
      samples.size() > o.limits.max_samples)
    throw InvalidArgument(
        "Original YOLO evaluation requires bounded samples and keys");
  std::vector<im::VisionSample> prepared;
  prepared.reserve(samples.size());
  for (const auto& source : samples) {
    Stop(stop);
    prepared.push_back(prepare_yolo_opencv_geometry(
        std::span(&source, 1),
        {.size = o.letterbox.canvas.width, .warp = false}, o.letterbox.geometry,
        stop));
  }
  auto current = o;
  current.numeric_on_consumer = defer_numeric;
  return Batch(prepared, keys, current, stop);
}

std::shared_ptr<data::DataPipeline> detail::make_original_yolo_pipeline(
    std::shared_ptr<const im::ImageDataset> dataset,
    data::DataPipelineOptions pipeline, const YoloDataOptions& options,
    std::shared_ptr<data::samplers::Sampler> sampler) {
  check_original_yolo(options);
  const auto batch_size = pipeline.batch_size;
  if (!batch_size || batch_size > 100000 || pipeline.num_workers > 64)
    throw InvalidArgument("Original YOLO worker/batch bounds exceeded");
  const auto workers = std::max(pipeline.num_workers, std::size_t{1});
  std::shared_ptr<Plans> plans;
  if (options.yolo_original.training) {
    if (sampler->size() != dataset->size())
      throw InvalidArgument(
          "Original YOLO sampler must cover the complete dataset");
    plans = std::make_shared<Plans>();
    sampler = std::make_shared<PlannedSampler>(sampler, plans, options, workers,
                                               batch_size, pipeline.drop_last);
  }
  const auto limit =
      std::min({dataset->size(), batch_size * 8, std::size_t{1000}});
  std::vector<std::shared_ptr<PixelCache>> caches;
  for (std::size_t i = 0; i < workers; ++i)
    caches.push_back(std::make_shared<PixelCache>(
        limit, options.yolo_original.decoded_cache_bytes / workers));
  return std::make_shared<data::DataPipeline>(
      std::move(sampler), std::move(pipeline),
      [dataset = std::move(dataset), plans, caches = std::move(caches), options,
       batch_size, workers](std::span<const std::size_t> indices,
                            const data::DataWorkerContext& context,
                            std::stop_token stop) {
        Stop(stop);
        std::vector<im::VisionSample> samples;
        std::vector<im::ImageBatchKey> keys;
        auto plan = plans ? plans->Find(context.epoch) : nullptr;
        const auto offset = context.sequence * batch_size;
        const auto worker =
            plan ? (context.sequence + plan->worker_offset) % workers
                 : context.worker_id;
        auto& cache = *caches.at(worker);
        auto view = [&](const OriginalYoloView& p) {
          std::vector<im::VisionSample> source;
          const auto count = p.geometry.mosaic ? 4U : 1U;
          for (std::size_t j = 0; j < count; ++j)
            source.push_back(cache.Get(*dataset, p.indices[j], stop));
          return prepare_yolo_opencv_geometry(source, p.geometry,
                                              options.letterbox.geometry, stop);
        };
        for (std::size_t i = 0; i < indices.size(); ++i) {
          Stop(stop);
          im::VisionSample sample;
          if (plan) {
            if (offset + i >= plan->samples.size() ||
                plan->order[offset + i] != indices[i])
              throw CompatibilityError(
                  "Original YOLO batch and planned order differ");
            const auto& p = plan->samples[offset + i];
            sample = view(p.primary);
            std::optional<im::VisionSample> partner;
            if (p.mixup) partner = view(p.partner);
            sample = prepare_yolo_opencv_color(
                sample, p.color, partner ? &*partner : nullptr,
                options.letterbox.geometry, stop);
            if (p.bgr) sample.image = sample.image.flip({0}).contiguous();
          } else {
            auto source = cache.Get(*dataset, indices[i], stop);
            sample = prepare_yolo_opencv_geometry(
                std::span(&source, 1),
                {.size = options.letterbox.canvas.width, .warp = false},
                options.letterbox.geometry, stop);
          }
          samples.push_back(std::move(sample));
          keys.push_back(
              {.id = samples.back().image_path, .occurrence = offset + i});
        }
        return im::image_batch_value(Batch(samples, keys, options, stop),
                                     options.limits);
      });
}
}  // namespace jyppx::liuding::vision::tasks
