// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#pragma once
#include <jyppx/liuding/core/types.hpp>

#include <torch/torch.h>

#include <array>

namespace jyppx::liuding::vision::tasks::detail {
// Preserve Ultralytics 8.4.177 ProbIoU's operation order. The covariance floor
// applies to grid-unit regression boxes, while assignment uses pixel boxes
// with no floor. This is deliberately separate from the public geometry API.
inline Tensor yolo26_obb_similarity(const Tensor& left, const Tensor& right,
                                    double floor = 0.) {
  constexpr double epsilon = 1e-7;
  const auto covariance = [floor](const Tensor& boxes) {
    auto w = boxes.select(-1, 2).square() / 12 + floor;
    auto h = boxes.select(-1, 3).square() / 12 + floor;
    auto c = boxes.select(-1, 4).cos(), s = boxes.select(-1, 4).sin();
    return std::array<Tensor, 3>{w * c.square() + h * s.square(),
                                 w * s.square() + h * c.square(),
                                 (w - h) * c * s};
  };
  const auto a = covariance(left), b = covariance(right);
  auto dx = left.select(-1, 0) - right.select(-1, 0);
  auto dy = left.select(-1, 1) - right.select(-1, 1);
  auto aa = a[0] + b[0], bb = a[1] + b[1], cc = a[2] + b[2];
  auto determinant = aa * bb - cc.square();
  auto translation =
      (aa * dy.square() + bb * dx.square()) / (determinant + epsilon) * .25;
  auto cross = cc * (-dx) * dy / (determinant + epsilon) * .5;
  auto shape = (determinant / (4 * ((a[0] * a[1] - a[2].square()).clamp_min(0) *
                                    (b[0] * b[1] - b[2].square()).clamp_min(0))
                                       .sqrt() +
                               epsilon) +
                epsilon)
                   .log() *
               .5;
  auto distance = (translation + cross + shape).clamp(epsilon, 100.);
  return 1 - (1 - (-distance).exp() + epsilon).sqrt();
}

inline Tensor yolo26_obb_candidates(const Tensor& centers, const Tensor& gt,
                                    double minimum_side) {
  const auto ctr = gt.slice(1, 0, 2);
  const auto wh = gt.slice(1, 2, 4).clamp_min(minimum_side);
  const auto angle = gt.slice(1, 4, 5);
  const auto w = wh.slice(1, 0, 1), h = wh.slice(1, 1, 2);
  const auto c = angle.cos(), s = angle.sin();
  const auto u = torch::cat({w / 2 * c, w / 2 * s}, 1);
  const auto v = torch::cat({-h / 2 * s, h / 2 * c}, 1);
  const auto a = ctr + u + v;
  const auto ab = (ctr + u - v) - a, ad = (ctr - u + v) - a;
  const auto ap = centers.unsqueeze(0) - a.unsqueeze(1);
  const auto dot_ab = ap.select(2, 0) * ab.select(1, 0).unsqueeze(1) +
                      ap.select(2, 1) * ab.select(1, 1).unsqueeze(1);
  const auto dot_ad = ap.select(2, 0) * ad.select(1, 0).unsqueeze(1) +
                      ap.select(2, 1) * ad.select(1, 1).unsqueeze(1);
  return (dot_ab >= 0) & (dot_ab <= ab.square().sum(-1).unsqueeze(1)) &
         (dot_ad >= 0) & (dot_ad <= ad.square().sum(-1).unsqueeze(1));
}
}  // namespace jyppx::liuding::vision::tasks::detail
