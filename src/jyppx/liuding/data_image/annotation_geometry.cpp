// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "annotation_detail.hpp"

#include <numeric>

namespace jyppx::liuding::data::image::detail {
namespace {

bool Equal(const ImagePoint& a, const ImagePoint& b) {
  return a.x == b.x && a.y == b.y;
}

long double Cross(const ImagePoint& a, const ImagePoint& b,
                  const ImagePoint& c) {
  return (static_cast<long double>(b.x) - a.x) *
             (static_cast<long double>(c.y) - a.y) -
         (static_cast<long double>(b.y) - a.y) *
             (static_cast<long double>(c.x) - a.x);
}

bool Within(const ImagePoint& a, const ImagePoint& b, const ImagePoint& p) {
  return p.x >= (std::min)(a.x, b.x) && p.x <= (std::max)(a.x, b.x) &&
         p.y >= (std::min)(a.y, b.y) && p.y <= (std::max)(a.y, b.y);
}

bool Intersects(const ImagePoint& a, const ImagePoint& b, const ImagePoint& c,
                const ImagePoint& d) {
  const auto ab_c = Cross(a, b, c), ab_d = Cross(a, b, d);
  const auto cd_a = Cross(c, d, a), cd_b = Cross(c, d, b);
  const auto opposite = [](long double x, long double y) {
    return (x < 0 && y > 0) || (x > 0 && y < 0);
  };
  return (opposite(ab_c, ab_d) && opposite(cd_a, cd_b)) ||
         (ab_c == 0 && Within(a, b, c)) || (ab_d == 0 && Within(a, b, d)) ||
         (cd_a == 0 && Within(c, d, a)) || (cd_b == 0 && Within(c, d, b));
}

}  // namespace

void ValidatePolygon(std::span<const ImagePoint> points, ParseState& state) {
  // Explicit closing points are legal and remain present in the owned value.
  if (points.size() > 1U && Equal(points.front(), points.back()))
    points = points.first(points.size() - 1U);
  if (points.size() < 3U) Malformed("polygon.point_count");
  long double twice_area = 0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    CheckStop(state.stop);
    const auto& a = points[i];
    const auto& b = points[(i + 1U) % points.size()];
    const auto& c = points[(i + 2U) % points.size()];
    if (Equal(a, b)) Malformed("polygon.repeated_edge_vertex");
    if (Cross(a, b, c) == 0 &&
        (static_cast<long double>(b.x) - a.x) * (c.x - b.x) +
                (static_cast<long double>(b.y) - a.y) * (c.y - b.y) <
            0)
      Malformed("polygon.overlapping_edges");
    twice_area += static_cast<long double>(a.x) * b.y -
                  static_cast<long double>(b.x) * a.y;
  }
  if (!std::isfinite(twice_area) || twice_area == 0)
    Malformed("polygon.zero_area");

  // Broad-phase sort avoids comparing separated edges. Degenerate or very
  // dense inputs still have a shared, explicit comparison budget.
  std::vector<std::size_t> order(points.size());
  std::iota(order.begin(), order.end(), 0U);
  const auto left = [&](std::size_t i) {
    return (std::min)(points[i].x, points[(i + 1U) % points.size()].x);
  };
  std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) { return left(a) < left(b); });
  for (std::size_t i = 0; i < order.size(); ++i) {
    CheckStop(state.stop);
    const auto a = order[i], a_next = (a + 1U) % points.size();
    const auto right = (std::max)(points[a].x, points[a_next].x);
    for (std::size_t j = i + 1U; j < order.size(); ++j) {
      CheckStop(state.stop);
      const auto b = order[j], b_next = (b + 1U) % points.size();
      if (left(b) > right) break;
      Add(state.geometry_checks, 1U, state.options.limits.max_geometry_checks,
          "polygon.comparison_budget");
      if (a_next == b || b_next == a) continue;
      if (Intersects(points[a], points[a_next], points[b], points[b_next]))
        Malformed("polygon.self_intersection");
    }
  }
}

}  // namespace jyppx::liuding::data::image::detail
