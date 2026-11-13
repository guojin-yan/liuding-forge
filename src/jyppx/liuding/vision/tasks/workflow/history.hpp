// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_PRIVATE_HISTORY_HPP
#define JYPPX_LIUDING_VISION_TASKS_PRIVATE_HISTORY_HPP
#include "artifact_io.hpp"
#include "telemetry.hpp"

#include <jyppx/liuding/optim/optimizer_adapter.hpp>
#include <jyppx/liuding/training/callback_dispatcher.hpp>

#include <chrono>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
namespace jyppx::liuding::vision::tasks::detail {
// An observational log per output segment. It is never restored as training
// state; the full Trainer checkpoint remains the recovery source of truth.
class History final : public training::CallbackHandler {
  std::filesystem::path output_;
  std::shared_ptr<optim::OptimizerAdapter> optimizer_;
  training::CallbackDescriptor descriptor_{
      .type_id = "jyppx.vision.history.v1",
      .callback_id = "callback.vision_history",
      .priority = -100,
      .subscriptions = {training::CallbackEventKind::epoch_started,
                        training::CallbackEventKind::epoch_finished}};
  nlohmann::json rows_ = nlohmann::json::array();
  std::chrono::steady_clock::time_point started_;
  MetricObserver observer_;

 public:
  History(std::filesystem::path output,
          std::shared_ptr<optim::OptimizerAdapter> optimizer, bool tensorboard)
      : output_(std::move(output)),
        optimizer_(std::move(optimizer)),
        observer_(tensorboard_observer(output_, tensorboard)) {}
  const training::CallbackDescriptor& descriptor() const noexcept override {
    return descriptor_;
  }
  training::CallbackCommandRequest handle(
      const training::CallbackEvent& event) override {
    using J = nlohmann::json;
    if (event.kind == training::CallbackEventKind::epoch_started) {
      started_ = std::chrono::steady_clock::now();
      return {};
    }
    auto metrics = event.metrics;
    auto rates = optimizer_->learning_rates();
    const auto& groups = optimizer_->parameter_group_ids();
    for (std::size_t i = 0; i < rates.size(); ++i)
      metrics["lr." + groups[i]] = rates[i];
    metrics["epoch.seconds"] = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - started_)
                                   .count();
    if (observer_) observer_(event, metrics);
    J row = {{"epoch", event.epoch},
             {"step", event.global_step},
             {"metrics", metrics}};
    if (!rows_.empty() && rows_.back()["epoch"] == event.epoch)
      rows_.back() = row;
    else
      rows_.push_back(row);
    auto json = J{
        {"schema", "vision.history/v1"},
        {"run_id", event.run_id},
        {"segment", true},
        {"epochs",
         rows_}}.dump(2);
    if (json.size() > (64U << 20U))
      throw ResourceExhausted("Vision epoch history exceeds 64 MiB");
    publish_text(output_ / "metrics.json", json);
    std::set<std::string> columns;
    for (const auto& item : rows_)
      for (const auto& metric : item["metrics"].items())
        columns.insert(metric.key());
    std::ostringstream csv;
    csv << "epoch,step";
    for (const auto& key : columns) csv << ",\"" << key << "\"";
    csv << '\n';
    csv.precision(17);
    for (const auto& item : rows_) {
      csv << item["epoch"] << ',' << item["step"];
      for (const auto& key : columns) {
        csv << ',';
        if (item["metrics"].contains(key))
          csv << item["metrics"][key].get<double>();
      }
      csv << '\n';
    }
    publish_text(output_ / "metrics.csv", csv.str());
    std::ostringstream html;
    html << "<!doctype html><meta charset=utf-8><title>Training "
            "metrics</title><style>body{font:15px "
            "system-ui;margin:32px;max-width:1000px}svg{width:100%;border:1px "
            "solid #ddd}a{margin-right:16px}</style><h1>Training "
            "metrics</h1><p><a href=metrics.json>JSON</a><a "
            "href=metrics.csv>CSV</a></p><svg viewBox='0 0 960 300' role=img "
            "aria-label='Training and validation loss'><path d='M40 "
            "15V270H940' fill=none stroke='#777'/>";
    double upper = 0;
    for (const auto& item : rows_)
      for (auto key : {"training.loss", "validation.loss"})
        if (item["metrics"].contains(key))
          upper = (std::max)(upper, item["metrics"][key].get<double>());
    for (auto key : {"training.loss", "validation.loss"}) {
      html << "<polyline fill=none stroke='"
           << (std::string_view(key) == "training.loss" ? "#2563eb" : "#ea580c")
           << "' points='";
      for (std::size_t i = 0; i < rows_.size(); ++i)
        if (rows_[i]["metrics"].contains(key))
          html << 40 + 900. * i / (std::max)(std::size_t{1}, rows_.size() - 1)
               << ','
               << 270 - 245. * rows_[i]["metrics"][key].get<double>() /
                            (std::max)(upper, 1e-12)
               << ' ';
      html << "'/>";
    }
    html << "</svg><p style='color:#2563eb'>Training loss</p><p "
            "style='color:#ea580c'>Validation loss</p>";
    publish_text(output_ / "report.html", html.str());
    return {};
  }
};
}  // namespace jyppx::liuding::vision::tasks::detail
#endif
