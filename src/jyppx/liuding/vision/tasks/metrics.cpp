// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/vision/ops/rotated.hpp>
#include <jyppx/liuding/vision/tasks/metrics.hpp>

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <numeric>
#include <set>
namespace jyppx::liuding::vision::tasks {
namespace {
using I = std::int64_t;
using Json = nlohmann::json;
struct Record {
  std::size_t category;
  std::uint64_t image;
  std::size_t row;
  double score;
  std::vector<int> matches;
};
std::string Identity(const TaskMetricOptions& o) {
  Json value = {{"task", static_cast<int>(o.task)},
                {"dataset", o.dataset_id},
                {"classes", o.classes_id},
                {"class_count", o.num_classes},
                {"thresholds", o.thresholds},
                {"sigmas", o.keypoint_sigmas},
                {"max_det", o.max_detections_per_image_class},
                {"empty", static_cast<int>(o.no_ground_truth)}};
  const auto bytes = value.dump();
  return sha256_hex(std::as_bytes(std::span(bytes)));
}
void Check(const ImageResult& r, std::size_t classes) {
  if (!r.boxes.defined() || !r.boxes.is_cpu() || r.boxes.dim() != 2 ||
      r.boxes.size(1) != 4 || !r.boxes.is_floating_point() ||
      !r.classes.defined() || !r.classes.is_cpu() || r.classes.dim() != 1 ||
      r.classes.scalar_type() != torch::kInt64 ||
      r.classes.size(0) != r.boxes.size(0) ||
      !torch::isfinite(r.boxes).all().item<bool>())
    throw InvalidArgument("Metric results require finite CPU box/class rows");
  if (r.classes.numel() &&
      (r.classes.min().item<I>() < 0 ||
       r.classes.max().item<I>() >= static_cast<I>(classes)))
    throw InvalidArgument("Metric class index out of range");
}
Tensor Flag(const Tensor& value, I size) {
  if (!value.defined()) return torch::zeros({size}, torch::kBool);
  if (!value.is_cpu() || value.dim() != 1 || value.size(0) != size ||
      value.scalar_type() != torch::kBool)
    throw InvalidArgument("Metric ignore/crowd requires CPU bool target rows");
  return value;
}
}  // namespace
struct TaskMetric::Impl {
  TaskMetricOptions options;
  std::string identity;
  std::set<std::uint64_t> images;
  std::vector<std::uint64_t> targets;
  std::vector<Record> records;
  explicit Impl(TaskMetricOptions o)
      : options(std::move(o)),
        identity(Identity(options)),
        targets(options.num_classes, 0) {
    if (options.task != TaskKind::instance_segment &&
        options.task != TaskKind::pose && options.task != TaskKind::obb)
      throw InvalidArgument(
          "TaskMetric covers mask/OKS/rotated AP; box AP uses the existing "
          "DetectionMetric");
    if (options.dataset_id.empty() || options.classes_id.empty() ||
        options.num_classes < 1 || options.num_classes > 10000 ||
        options.thresholds.empty() || options.thresholds.size() > 64 ||
        options.max_images < 1 || options.max_retained_predictions < 1 ||
        options.max_similarity_pairs < 1 ||
        options.max_detections_per_image_class < 1 ||
        options.max_mask_elements < 1)
      throw InvalidArgument("Invalid task metric identity/count/bounds");
    double previous = 0;
    for (auto threshold : options.thresholds) {
      if (!std::isfinite(threshold) || threshold <= previous || threshold > 1)
        throw InvalidArgument(
            "Metric thresholds must strictly increase in (0,1]");
      previous = threshold;
    }
    for (auto sigma : options.keypoint_sigmas)
      if (!std::isfinite(sigma) || sigma <= 0)
        throw InvalidArgument("OKS sigmas must be positive finite");
  }
};
TaskMetric::TaskMetric(TaskMetricOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
TaskMetric::~TaskMetric() = default;
void TaskMetric::update(std::uint64_t image_id, const ImageResult& prediction,
                        const ImageResult& target, const Tensor& input_areas,
                        const Tensor& input_ignore, const Tensor& input_crowd) {
  torch::NoGradGuard guard;
  auto& s = *impl_;
  const auto& o = s.options;
  if (s.images.contains(image_id))
    throw InvalidArgument("Duplicate image in task metric");
  if (s.images.size() >= o.max_images)
    throw ResourceExhausted("Task metric image bound exceeded");
  Check(prediction, o.num_classes);
  Check(target, o.num_classes);
  const I np = prediction.boxes.size(0), nt = target.boxes.size(0);
  if (nt > 0 && np > static_cast<I>(o.max_similarity_pairs) / nt)
    throw ResourceExhausted("Task metric similarity pair bound exceeded");
  if (!prediction.scores.defined() || !prediction.scores.is_cpu() ||
      prediction.scores.dim() != 1 || prediction.scores.size(0) != np ||
      !prediction.scores.is_floating_point() ||
      !torch::isfinite(prediction.scores).all().item<bool>())
    throw InvalidArgument("Task metric requires finite CPU prediction scores");
  auto ignored = Flag(input_ignore, nt).clone(), crowd = Flag(input_crowd, nt);
  auto areas = input_areas.defined()
                   ? input_areas.to(torch::kCPU, torch::kFloat64)
                   : ((target.boxes.select(1, 2) - target.boxes.select(1, 0)) *
                      (target.boxes.select(1, 3) - target.boxes.select(1, 1)))
                         .to(torch::kFloat64);
  if (areas.dim() != 1 || areas.size(0) != nt ||
      !torch::isfinite(areas).all().item<bool>() ||
      !(areas >= 0).all().item<bool>())
    throw InvalidArgument("Invalid task target areas");
  auto similarities = torch::zeros({np, nt}, torch::kFloat64);
  if (o.task == TaskKind::instance_segment) {
    for (const auto* r : {&prediction, &target})
      if (!r->masks.defined() || !r->masks.is_cpu() || r->masks.dim() != 3 ||
          r->masks.size(0) != r->boxes.size(0) ||
          r->masks.numel() > o.max_mask_elements)
        throw InvalidArgument("Mask AP requires bounded CPU [N,H,W] masks");
    if (prediction.masks.size(1) != target.masks.size(1) ||
        prediction.masks.size(2) != target.masks.size(2))
      throw InvalidArgument(
          "Mask AP requires one common original pixel canvas");
    const auto pixels = target.masks.size(1) * target.masks.size(2);
    if (pixels > 0 && np * nt > 1000000000LL / pixels)
      throw ResourceExhausted("Mask AP pair/pixel work bound exceeded");
    auto p = prediction.masks.to(torch::kBool),
         t = target.masks.to(torch::kBool);
    for (I i = 0; i < np; ++i)
      for (I j = 0; j < nt; ++j) {
        const double intersection = (p[i] & t[j]).sum().item<double>();
        const double denominator = crowd[j].item<bool>()
                                       ? p[i].sum().item<double>()
                                       : (p[i] | t[j]).sum().item<double>();
        similarities[i][j] = denominator > 0 ? intersection / denominator : 0;
      }
  } else if (o.task == TaskKind::obb) {
    similarities = vision_ops::pairwise_quadrilateral_iou(
        prediction.oriented_corners, target.oriented_corners,
        {.max_pairs = static_cast<I>(o.max_similarity_pairs)});
  } else {
    for (const auto* r : {&prediction, &target})
      if (!r->keypoints.defined() || !r->keypoints.is_cpu() ||
          r->keypoints.dim() != 3 || r->keypoints.size(0) != r->boxes.size(0) ||
          r->keypoints.size(2) != 3 ||
          !torch::isfinite(r->keypoints).all().item<bool>())
        throw InvalidArgument("OKS AP requires finite CPU [N,K,3] keypoints");
    const I k = target.keypoints.size(1);
    if (k < 1 || prediction.keypoints.size(1) != k ||
        (!o.keypoint_sigmas.empty() &&
         o.keypoint_sigmas.size() != static_cast<std::size_t>(k)))
      throw InvalidArgument("OKS keypoint/sigma topology mismatch");
    auto sigma = o.keypoint_sigmas;
    if (sigma.empty()) sigma.assign(static_cast<std::size_t>(k), 1. / k);
    auto sigmas = torch::tensor(sigma, torch::kFloat64);
    for (I j = 0; j < nt; ++j) {
      auto visible = target.keypoints[j].select(1, 2) > 0;
      if (!visible.any().item<bool>()) {
        ignored[j] = true;
        continue;
      }
      auto distances =
          (prediction.keypoints.slice(2, 0, 2).to(torch::kFloat64) -
           target.keypoints[j].slice(1, 0, 2).to(torch::kFloat64))
              .square()
              .sum(2);
      auto oks =
          torch::exp(-distances / (8 * sigmas.square() * (areas[j] + 1e-12)));
      similarities.select(1, j).copy_((oks * visible).sum(1) / visible.sum());
    }
  }
  std::vector<Record> staged;
  auto next_targets = s.targets;
  for (std::size_t category = 0; category < o.num_classes; ++category) {
    std::vector<I> gt, pred;
    for (I j = 0; j < nt; ++j)
      if (target.classes[j].item<I>() == static_cast<I>(category)) {
        gt.push_back(j);
        if (!ignored[j].item<bool>() && !crowd[j].item<bool>())
          ++next_targets[category];
      }
    std::stable_sort(gt.begin(), gt.end(), [&](I a, I b) {
      return (ignored[a].item<bool>() || crowd[a].item<bool>()) <
             (ignored[b].item<bool>() || crowd[b].item<bool>());
    });
    for (I i = 0; i < np; ++i)
      if (prediction.classes[i].item<I>() == static_cast<I>(category))
        pred.push_back(i);
    std::stable_sort(pred.begin(), pred.end(), [&](I a, I b) {
      return prediction.scores[a].item<double>() >
             prediction.scores[b].item<double>();
    });
    if (pred.size() > o.max_detections_per_image_class)
      pred.resize(o.max_detections_per_image_class);
    const auto start = staged.size();
    for (auto row : pred)
      staged.push_back({category, image_id, static_cast<std::size_t>(row),
                        prediction.scores[row].item<double>(),
                        std::vector<int>(o.thresholds.size(), 0)});
    for (std::size_t threshold = 0; threshold < o.thresholds.size();
         ++threshold) {
      std::vector<bool> matched(gt.size(), false);
      for (std::size_t pi = 0; pi < pred.size(); ++pi) {
        double best = o.thresholds[threshold];
        std::optional<std::size_t> selected;
        for (std::size_t gj = 0; gj < gt.size(); ++gj) {
          const I j = gt[gj];
          if (matched[gj] && !crowd[j].item<bool>()) continue;
          if (selected && !ignored[gt[*selected]].item<bool>() &&
              !crowd[gt[*selected]].item<bool>() &&
              (ignored[j].item<bool>() || crowd[j].item<bool>()))
            break;
          const double similarity = similarities[pred[pi]][j].item<double>();
          if (similarity >= best) {
            best = similarity;
            selected = gj;
          }
        }
        if (selected) {
          matched[*selected] = true;
          const I j = gt[*selected];
          staged[start + pi].matches[threshold] =
              (ignored[j].item<bool>() || crowd[j].item<bool>()) ? 2 : 1;
        }
      }
    }
  }
  if (staged.size() > o.max_retained_predictions - s.records.size())
    throw ResourceExhausted("Task metric retained prediction bound exceeded");
  s.records.reserve(s.records.size() + staged.size());
  s.images.insert(image_id);
  for (auto& record : staged) s.records.push_back(std::move(record));
  s.targets.swap(next_targets);
}
TaskMetricReport TaskMetric::compute() const {
  const auto& s = *impl_;
  const auto& o = s.options;
  TaskMetricReport report;
  report.images = s.images.size();
  report.classes.resize(o.num_classes);
  report.protocol =
      o.task == TaskKind::instance_segment ? "native.mask_ap.r101/v1"
      : o.task == TaskKind::pose           ? "native.oks_ap.r101/v1"
                                           : "native.quadrilateral_ap.r101/v1";
  double sum = 0, sum50 = 0;
  std::size_t count = 0, count50 = 0;
  for (std::size_t c = 0; c < o.num_classes; ++c) {
    auto& result = report.classes[c];
    result.targets = s.targets[c];
    std::vector<const Record*> rows;
    for (const auto& r : s.records)
      if (r.category == c) rows.push_back(&r);
    std::stable_sort(rows.begin(), rows.end(), [](auto a, auto b) {
      if (a->score != b->score) return a->score > b->score;
      if (a->image != b->image) return a->image < b->image;
      return a->row < b->row;
    });
    result.predictions = rows.size();
    if (!result.targets &&
        o.no_ground_truth == training::metrics::DetectionNoGroundTruth::error)
      throw InvalidArgument(
          "Task AP has a class with no eligible ground truth");
    if (!result.targets &&
        o.no_ground_truth == training::metrics::DetectionNoGroundTruth::exclude)
      continue;
    for (std::size_t t = 0; t < o.thresholds.size(); ++t) {
      double tp = 0, fp = 0;
      std::vector<double> precision, recall;
      for (auto row : rows) {
        if (row->matches[t] == 2) continue;
        if (row->matches[t] == 1)
          ++tp;
        else
          ++fp;
        precision.push_back(tp / (tp + fp));
        recall.push_back(result.targets ? tp / result.targets : 0);
      }
      auto envelope = precision;
      for (std::size_t i = envelope.size(); i > 1; --i)
        envelope[i - 2] = (std::max)(envelope[i - 2], envelope[i - 1]);
      double ap = 0;
      for (int point = 0; point <= 100; ++point) {
        auto position =
            std::lower_bound(recall.begin(), recall.end(), point / 100.);
        if (position != recall.end())
          ap += envelope[static_cast<std::size_t>(position - recall.begin())] /
                101.;
      }
      result.ap_by_threshold.push_back(ap);
      if (t == 0) {
        result.precision = std::move(precision);
        result.recall = std::move(recall);
      }
      if (std::abs(o.thresholds[t] - .5) < 1e-12) {
        sum50 += ap;
        ++count50;
      }
    }
    result.average_precision =
        std::accumulate(result.ap_by_threshold.begin(),
                        result.ap_by_threshold.end(), 0.) /
        o.thresholds.size();
    sum += *result.average_precision;
    ++count;
  }
  if (count) report.mean_average_precision = sum / count;
  if (count50) report.ap50 = sum50 / count50;
  return report;
}
std::string TaskMetric::save_state(std::size_t max_bytes) const {
  const auto& s = *impl_;
  Json value = {{"schema", "vision.task_metric/v1"},
                {"identity", s.identity},
                {"images", s.images},
                {"targets", s.targets},
                {"records", Json::array()}};
  for (const auto& r : s.records)
    value["records"].push_back(
        {r.category, r.image, r.row, r.score, r.matches});
  auto bytes = value.dump();
  if (bytes.size() > max_bytes)
    throw ResourceExhausted("Task metric checkpoint byte bound exceeded");
  return bytes;
}
void TaskMetric::load_state(std::string_view bytes) {
  if (bytes.size() > 256U << 20U)
    throw ResourceExhausted("Task metric checkpoint byte bound exceeded");
  const auto value = Json::parse(bytes);
  if (value.at("schema") != "vision.task_metric/v1" ||
      value.at("identity") != impl_->identity)
    throw CompatibilityError("Task metric restore identity mismatch");
  auto next = std::make_unique<Impl>(impl_->options);
  next->images = value.at("images").get<std::set<std::uint64_t>>();
  next->targets = value.at("targets").get<std::vector<std::uint64_t>>();
  if (next->images.size() > next->options.max_images ||
      next->targets.size() != next->options.num_classes ||
      value.at("records").size() > next->options.max_retained_predictions)
    throw ResourceExhausted(
        "Task metric restored topology/resource bound exceeded");
  for (const auto& r : value.at("records")) {
    if (!r.is_array() || r.size() != 5)
      throw SerializationError("Invalid task metric record");
    Record record{r[0].get<std::size_t>(), r[1].get<std::uint64_t>(),
                  r[2].get<std::size_t>(), r[3].get<double>(),
                  r[4].get<std::vector<int>>()};
    if (record.category >= next->options.num_classes ||
        !next->images.contains(record.image) || !std::isfinite(record.score) ||
        record.matches.size() != next->options.thresholds.size() ||
        std::any_of(record.matches.begin(), record.matches.end(),
                    [](int x) { return x < 0 || x > 2; }))
      throw SerializationError("Invalid task metric restored record");
    next->records.push_back(std::move(record));
  }
  impl_.swap(next);
}
}  // namespace jyppx::liuding::vision::tasks
