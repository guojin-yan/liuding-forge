// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/vision/ops/rotated.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>
namespace jyppx::liuding::vision_ops {
namespace {
struct Point {
  double x, y;
};
double Cross(Point a, Point b, Point c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}
using Quad = std::array<Point, 4>;
Quad Corners(const double* b) {
  const double c = std::cos(b[4]), s = std::sin(b[4]);
  Quad q;
  const double dx[4]{-.5, .5, .5, -.5}, dy[4]{-.5, -.5, .5, .5};
  for (int i = 0; i < 4; ++i)
    q[i] = {b[0] + c * dx[i] * b[2] - s * dy[i] * b[3],
            b[1] + s * dx[i] * b[2] + c * dy[i] * b[3]};
  return q;
}
double Area(const std::vector<Point>& p) {
  double sum = 0;
  for (std::size_t i = 0; i < p.size(); ++i)
    sum += p[i].x * p[(i + 1) % p.size()].y - p[i].y * p[(i + 1) % p.size()].x;
  return std::abs(sum) * .5;
}
double QuadIoU(const Quad& qa, const Quad& qb) {
  std::vector<Point> polygon(qa.begin(), qa.end());
  for (int edge = 0; edge < 4 && !polygon.empty(); ++edge) {
    std::vector<Point> next;
    const auto first = qb[edge], second = qb[(edge + 1) % 4];
    auto previous = polygon.back();
    auto pside = Cross(first, second, previous);
    for (const auto& current : polygon) {
      const auto side = Cross(first, second, current);
      if ((side >= 0) != (pside >= 0)) {
        const auto t = pside / (pside - side);
        next.push_back({previous.x + t * (current.x - previous.x),
                        previous.y + t * (current.y - previous.y)});
      }
      if (side >= 0) next.push_back(current);
      previous = current;
      pside = side;
    }
    polygon = std::move(next);
  }
  const auto intersection = Area(polygon);
  return intersection / (Area(std::vector<Point>(qa.begin(), qa.end())) +
                         Area(std::vector<Point>(qb.begin(), qb.end())) -
                         intersection + 1e-12);
}
double IoU(const double* a, const double* b) {
  return QuadIoU(Corners(a), Corners(b));
}
Tensor Checked(const Tensor& boxes) {
  if (!boxes.defined() || !boxes.is_cpu() || !boxes.is_floating_point() ||
      boxes.dim() != 2 || boxes.size(1) != 5)
    throw InvalidArgument(
        "Exact rotated geometry requires floating CPU [N,5] boxes");
  auto x = boxes.detach().to(torch::kFloat64).contiguous();
  if (!torch::isfinite(x).all().item<bool>() ||
      (x.size(0) && !(x.slice(1, 2, 4) > 0).all().item<bool>()))
    throw InvalidArgument(
        "Rotated boxes require finite coordinates and positive width/height");
  return x;
}
}  // namespace
Tensor probabilistic_box_iou(const Tensor& left, const Tensor& right,
                             double eps) {
  if (!left.defined() || !right.defined() || left.dim() < 1 ||
      right.dim() < 1 || left.size(-1) != 5 || right.size(-1) != 5 ||
      !left.is_floating_point() || !right.is_floating_point() ||
      left.device() != right.device() || !std::isfinite(eps) || eps <= 0)
    throw InvalidArgument(
        "Probabilistic rotated IoU requires floating [...,5] boxes on one "
        "device");
  auto covariance = [](const Tensor& boxes) {
    auto w = boxes.select(-1, 2).square() / 12,
         h = boxes.select(-1, 3).square() / 12;
    auto c = boxes.select(-1, 4).cos(), s = boxes.select(-1, 4).sin();
    return std::array<Tensor, 3>{w * c.square() + h * s.square(),
                                 w * s.square() + h * c.square(),
                                 (w - h) * c * s};
  };
  const auto a = covariance(left), b = covariance(right);
  auto dx = left.select(-1, 0) - right.select(-1, 0),
       dy = left.select(-1, 1) - right.select(-1, 1);
  auto aa = a[0] + b[0], bb = a[1] + b[1], cc = a[2] + b[2];
  auto det = (aa * bb - cc.square()).clamp_min(eps);
  auto translation =
      (bb * dx.square() + aa * dy.square() - 2 * cc * dx * dy) / (4 * det);
  auto det_a = (a[0] * a[1] - a[2].square()).clamp_min(0),
       det_b = (b[0] * b[1] - b[2].square()).clamp_min(0);
  auto shape =
      .5 * torch::log(det / (4 * torch::sqrt(det_a * det_b) + eps) + eps);
  auto distance = (translation + shape).clamp(eps, 100);
  return 1 - torch::sqrt((1 - torch::exp(-distance)).clamp_min(0) + eps);
}
Tensor pairwise_rotated_iou(const Tensor& left, const Tensor& right,
                            const PairwiseBoxOptions& o) {
  auto a = Checked(left), b = Checked(right);
  if (o.max_pairs < 1 || (b.size(0) > 0 && a.size(0) > o.max_pairs / b.size(0)))
    throw ResourceExhausted("Rotated IoU pair bound exceeded");
  auto result = torch::empty({a.size(0), b.size(0)}, torch::kFloat64);
  auto* values = result.data_ptr<double>();
  for (std::int64_t i = 0; i < a.size(0); ++i)
    for (std::int64_t j = 0; j < b.size(0); ++j)
      values[i * b.size(0) + j] =
          IoU(a.data_ptr<double>() + 5 * i, b.data_ptr<double>() + 5 * j);
  return result;
}
Tensor pairwise_quadrilateral_iou(const Tensor& left, const Tensor& right,
                                  const PairwiseBoxOptions& o) {
  for (const auto& x : {left, right})
    if (!x.defined() || !x.is_cpu() || !x.is_floating_point() || x.dim() != 3 ||
        x.size(1) != 4 || x.size(2) != 2)
      throw InvalidArgument(
          "Quadrilateral IoU requires floating CPU [N,4,2] corners");
  if (o.max_pairs < 1 ||
      (right.size(0) > 0 && left.size(0) > o.max_pairs / right.size(0)))
    throw ResourceExhausted("Quadrilateral IoU pair bound exceeded");
  auto a = left.detach().to(torch::kFloat64).contiguous(),
       b = right.detach().to(torch::kFloat64).contiguous();
  if (!torch::isfinite(a).all().item<bool>() ||
      !torch::isfinite(b).all().item<bool>())
    throw InvalidArgument("Nonfinite quadrilateral coordinate");
  auto quad = [](const double* p) {
    Quad q;
    for (int i = 0; i < 4; ++i) q[i] = {p[2 * i], p[2 * i + 1]};
    double direction = 0;
    for (int i = 0; i < 4; ++i) {
      const double cross = Cross(q[i], q[(i + 1) % 4], q[(i + 2) % 4]);
      if (std::abs(cross) < 1e-12 || (direction && cross * direction < 0))
        throw InvalidArgument(
            "Quadrilateral must be convex with positive area");
      direction = cross;
    }
    if (direction < 0) std::reverse(q.begin(), q.end());
    return q;
  };
  std::vector<Quad> qa, qb;
  for (std::int64_t i = 0; i < a.size(0); ++i)
    qa.push_back(quad(a.data_ptr<double>() + i * 8));
  for (std::int64_t i = 0; i < b.size(0); ++i)
    qb.push_back(quad(b.data_ptr<double>() + i * 8));
  auto result = torch::empty({a.size(0), b.size(0)}, torch::kFloat64);
  auto* values = result.data_ptr<double>();
  for (std::size_t i = 0; i < qa.size(); ++i)
    for (std::size_t j = 0; j < qb.size(); ++j)
      values[i * qb.size() + j] = QuadIoU(qa[i], qb[j]);
  return result;
}
Tensor rotated_nms(const Tensor& boxes, const Tensor& scores,
                   const NmsOptions& o) {
  auto b = Checked(boxes);
  if (!scores.defined() || !scores.is_cpu() || scores.dim() != 1 ||
      scores.size(0) != b.size(0) || !scores.is_floating_point() ||
      !std::isfinite(o.iou_threshold) || o.iou_threshold < 0 ||
      o.iou_threshold > 1 || o.max_output < 1 || o.max_candidates < 1)
    throw InvalidArgument("Invalid rotated NMS score/options contract");
  if (b.size(0) > o.max_candidates)
    throw ResourceExhausted("Rotated NMS candidate bound exceeded");
  auto s = scores.detach().to(torch::kFloat64).contiguous();
  if (!torch::isfinite(s).all().item<bool>())
    throw InvalidArgument("Nonfinite rotated NMS score");
  std::vector<std::int64_t> order(static_cast<std::size_t>(b.size(0))), keep;
  std::iota(order.begin(), order.end(), 0);
  const auto* p = s.data_ptr<double>();
  std::stable_sort(order.begin(), order.end(),
                   [p](auto i, auto j) { return p[i] > p[j]; });
  for (auto i : order) {
    bool suppress = false;
    for (auto j : keep)
      if (IoU(b.data_ptr<double>() + i * 5, b.data_ptr<double>() + j * 5) >
          o.iou_threshold) {
        suppress = true;
        break;
      }
    if (!suppress) keep.push_back(i);
    if (static_cast<std::int64_t>(keep.size()) == o.max_output) break;
  }
  return torch::tensor(keep, torch::kInt64);
}
}  // namespace jyppx::liuding::vision_ops
