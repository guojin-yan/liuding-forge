// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_PRIVATE_EVALUATION_DIAGNOSTICS_HPP
#define JYPPX_LIUDING_VISION_TASKS_PRIVATE_EVALUATION_DIAGNOSTICS_HPP
#include <jyppx/liuding/vision/tasks/results.hpp>

#include <nlohmann/json.hpp>
namespace jyppx::liuding::vision::tasks::detail {
class EvaluationDiagnostics {
  std::map<std::pair<std::int64_t, std::int64_t>, std::uint64_t> cells_;
  nlohmann::json failures_ = nlohmann::json::array();

 public:
  void update(const ImageResult& prediction,
              const data::image::VisionAnnotationRecord& target) {
    using I = std::int64_t;
    std::vector<bool> matched(target.objects.size());
    std::uint64_t missed = 0, unmatched = 0, wrong = 0;
    if (target.objects.size() && static_cast<std::size_t>(prediction.boxes.size(
                                     0)) > 1048576 / target.objects.size())
      throw ResourceExhausted("Diagnostic matching pair budget exceeded");
    auto iou = [](const data::image::ImageBox& a,
                  const data::image::ImageBox& b) {
      auto intersect = (std::max)(0., (std::min)(a.x_max, b.x_max) -
                                          (std::max)(a.x_min, b.x_min)) *
                       (std::max)(0., (std::min)(a.y_max, b.y_max) -
                                          (std::max)(a.y_min, b.y_min));
      return intersect /
             ((a.x_max - a.x_min) * (a.y_max - a.y_min) +
              (b.x_max - b.x_min) * (b.y_max - b.y_min) - intersect + 1e-12);
    };
    for (I row = 0; row < prediction.boxes.size(0); ++row) {
      if (prediction.scores[row].item<double>() < .25) continue;
      auto box = prediction.boxes[row];
      data::image::ImageBox b{box[0].item<double>(), box[1].item<double>(),
                              box[2].item<double>(), box[3].item<double>()};
      double best = .5;
      std::optional<std::size_t> chosen;
      bool ignored = false;
      for (std::size_t k = 0; k < target.objects.size(); ++k) {
        const auto& object = target.objects[k];
        if (!object.box) continue;
        const auto similarity = iou(b, *object.box);
        if (object.crowd || object.ignore || object.difficult) {
          ignored = ignored || similarity >= .5;
          continue;
        }
        if (!matched[k] && similarity >= best) {
          best = similarity;
          chosen = k;
        }
      }
      const auto predicted_class = prediction.classes[row].item<I>();
      if (chosen) {
        matched[*chosen] = true;
        const auto actual = static_cast<I>(target.objects[*chosen].category);
        ++cells_[{actual, predicted_class}];
        wrong += actual != predicted_class;
      } else if (!ignored) {
        ++cells_[{-1, predicted_class}];
        ++unmatched;
      }
    }
    for (std::size_t k = 0; k < target.objects.size(); ++k) {
      const auto& object = target.objects[k];
      if (!matched[k] && !object.crowd && !object.ignore && !object.difficult) {
        ++cells_[{static_cast<I>(object.category), -1}];
        ++missed;
      }
    }
    if (cells_.size() > 100000)
      throw ResourceExhausted(
          "Diagnostic confusion matrix exceeds sparse cell budget");
    if (failures_.size() < 256 && (missed || unmatched || wrong))
      failures_.push_back({{"source", prediction.source_id},
                           {"missed_targets", missed},
                           {"unmatched_predictions", unmatched},
                           {"wrong_class", wrong}});
  }
  nlohmann::json report() const {
    auto cells = nlohmann::json::array();
    for (auto [pair, count] : cells_)
      cells.push_back({{"target", pair.first},
                       {"prediction", pair.second},
                       {"count", count}});
    return {{"protocol", "diagnostic.class_agnostic_box_iou50_confidence25/v1"},
            {"background_id", -1},
            {"confusion", cells},
            {"failure_samples", failures_}};
  }
};
}  // namespace jyppx::liuding::vision::tasks::detail
#endif
