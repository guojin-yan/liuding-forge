// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/ops/assignment.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace jyppx::liuding::vision_ops {
Assignment linear_assignment(const Tensor& costs, const AssignmentOptions& o,
                             std::stop_token stop) {
  using I = std::int64_t;
  if (!costs.defined() || costs.dim() != 2 || !costs.is_floating_point() ||
      o.max_pairs < 1 || o.max_work < 1 ||
      (o.cost_limit && (!std::isfinite(*o.cost_limit) || *o.cost_limit <= 0)))
    throw InvalidArgument("Invalid assignment matrix/options");
  I nr = costs.size(0), nc = costs.size(1);
  const auto empty = [] { return Assignment{torch::empty({0}, torch::kInt64),
                                            torch::empty({0}, torch::kInt64)}; };
  if (stop.stop_requested()) throw Cancelled("Assignment cancelled");
  if (!nr || !nc) return empty();
  const bool partial = o.cost_limit.has_value();
  I n = partial ? nr + nc : (std::min)(nr, nc);
  I m = partial ? n : (std::max)(nr, nc);
  if (n > o.max_pairs / m)
    throw ResourceExhausted("Assignment pair limit exceeded");
  auto snapshot = costs.detach().to(torch::kCPU, torch::kFloat64).contiguous();
  if (!torch::isfinite(snapshot).all().item<bool>())
    throw InvalidArgument("Assignment requires finite costs");
  auto access = snapshot.accessor<double, 2>();
  const bool transpose = !partial && nr > nc;
  // Shift/scale finite costs to avoid overflow in potential arithmetic.
  double scale = (std::max)(1., snapshot.abs().max().item<double>());
  if (partial) scale = (std::max)(scale, *o.cost_limit);
  auto cost = [&](I row, I col) {
    if (!partial) return (transpose ? access[col][row] : access[row][col]) / scale;
    if (row < nr && col < nc)
      return access[row][col] > *o.cost_limit ? 4. : access[row][col] / scale;
    if (row < nr || col < nc) return *o.cost_limit / (2 * scale);
    return 0.;
  };
  std::vector<double> u(n + 1), v(m + 1), best(m + 1);
  std::vector<I> p(m + 1), way(m + 1);
  std::vector<bool> used(m + 1);
  std::uint64_t work = 0;
  for (I row = 1; row <= n; ++row) {
    p[0] = row;
    I column = 0;
    std::fill(best.begin(), best.end(), std::numeric_limits<double>::infinity());
    std::fill(used.begin(), used.end(), false);
    do {
      if (stop.stop_requested()) throw Cancelled("Assignment cancelled");
      used[column] = true;
      const I current = p[column];
      double delta = std::numeric_limits<double>::infinity();
      I next = 0;
      for (I j = 1; j <= m; ++j) {
        if (++work > o.max_work)
          throw ResourceExhausted("Assignment work limit exceeded");
        if (used[j]) continue;
        double candidate = cost(current - 1, j - 1) - u[current] - v[j];
        if (candidate < best[j]) { best[j] = candidate; way[j] = column; }
        if (best[j] < delta) { delta = best[j]; next = j; }
      }
      for (I j = 0; j <= m; ++j) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
        else best[j] -= delta;
      }
      column = next;
    } while (p[column] != 0);
    do { I next = way[column]; p[column] = p[next]; column = next; }
    while (column);
  }
  std::vector<std::pair<I, I>> pairs;
  for (I j = 1; j <= m; ++j) {
    if (!p[j]) continue;
    I row = p[j] - 1, col = j - 1;
    if (transpose) std::swap(row, col);
    if (row >= nr || col >= nc) continue;
    if (partial && access[row][col] > *o.cost_limit) continue;
    pairs.emplace_back(row, col);
  }
  std::sort(pairs.begin(), pairs.end());
  std::vector<I> rows, columns;
  for (auto [r, c] : pairs) { rows.push_back(r); columns.push_back(c); }
  return {torch::tensor(rows, torch::kInt64), torch::tensor(columns, torch::kInt64)};
}
}  // namespace jyppx::liuding::vision_ops
