// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_ADVANCED_METRICS_HPP
#define JYPPX_LIUDING_VISION_TASKS_ADVANCED_METRICS_HPP
/** @file advanced_metrics.hpp @brief Bounded native multilabel/retrieval metrics. */
#include <jyppx/liuding/core/types.hpp>
#include <map>
#include <string>
namespace jyppx::liuding::vision::tasks {
/** @brief Known-label AP/macro-F1/micro-F1; classes without known positives
 * are excluded from mAP. Threshold is a fixed validation-calibrated [C] or .5. */
std::map<std::string,double> multilabel_metrics(const Tensor& scores,
    const Tensor& values,const Tensor& known,const Tensor& thresholds={});
/** @brief Selects per-class F1 thresholds on an explicitly identified
 * validation split only; callers persist that split identity with the values. */
Tensor calibrate_multilabel_thresholds(const Tensor& scores,const Tensor& values,
    const Tensor& known,std::string_view validation_split_id);
/** @brief Leave-one-out retrieval; excludes self and same identity+camera.
 * Skips queries without valid positives; bounded at 8192 rows. */
std::map<std::string,double> reid_retrieval_metrics(const Tensor& embeddings,
    const Tensor& identities,const std::vector<std::string>& cameras);
}
#endif
