// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/metrics/detection.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <utility>

namespace jyppx::liuding::training::metrics {
namespace {
constexpr std::uint64_t kNone = std::numeric_limits<std::uint64_t>::max();
constexpr char kTopologyMedia[] =
    "application/vnd.jyppx.liuding.metric.box-ap-topology-v1";
constexpr char kMatchMedia[] =
    "application/vnd.jyppx.liuding.metric.box-ap-matches-le-v1";
constexpr std::uint64_t kMagic = 0x315041584f42464cULL;
std::uint64_t add(std::uint64_t a, std::uint64_t b, std::uint64_t cap = kNone) {
  if (a > cap || b > cap - a)
    throw ResourceExhausted("Detection AP logical resource limit exceeded");
  return a + b;
}
std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
  if (a && b > kNone / a) throw ResourceExhausted("Detection AP size overflow");
  return a * b;
}
bool identifier(const std::string& s, bool opaque = false) {
  return !s.empty() && s.size() <= 256 &&
         std::all_of(s.begin(), s.end(), [opaque](unsigned char c) {
           return opaque ? c >= 33 && c <= 126
                         : (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                               (c >= '0' && c <= '9') || c == '.' || c == '_' ||
                               c == '-';
         });
}
double area(const DetectionBox& b) {
  if (!std::all_of(b.begin(), b.end(),
                   [](double v) { return std::isfinite(v); }) ||
      !(b[2] > b[0]) || !(b[3] > b[1]))
    throw InvalidArgument(
        "Detection boxes require finite positive-area continuous xyxy "
        "coordinates");
  const auto value = (b[2] - b[0]) * (b[3] - b[1]);
  if (!std::isfinite(value) || value <= 0)
    throw InvalidArgument("Detection box area is not finite and positive");
  return value;
}
bool outside(double a, const DetectionMetricOptions& o) {
  return a < o.min_area || a > o.max_area;
}
bool ignored(const DetectionTarget& t, const DetectionMetricOptions& o) {
  return t.crowd || t.ignore || outside(t.area.value_or(area(t.box)), o);
}
double overlap(const DetectionBox& d, const DetectionBox& g, bool crowd) {
  const long double w =
      std::max(0.L, static_cast<long double>(std::min(d[2], g[2])) -
                        std::max(d[0], g[0]));
  const long double h =
      std::max(0.L, static_cast<long double>(std::min(d[3], g[3])) -
                        std::max(d[1], g[1]));
  const long double a = static_cast<long double>(d[2] - d[0]) * (d[3] - d[1]);
  const long double b = static_cast<long double>(g[2] - g[0]) * (g[3] - g[1]);
  return static_cast<double>(w * h / (crowd ? a : a + b - w * h));
}
void put(std::vector<std::byte>& b, std::uint64_t v) {
  for (unsigned i = 0; i < 8; ++i)
    b.push_back(static_cast<std::byte>((v >> (8U * i)) & 255U));
}
void put_string(std::vector<std::byte>& b, const std::string& s) {
  put(b, s.size());
  const auto bytes = std::as_bytes(std::span(s));
  b.insert(b.end(), bytes.begin(), bytes.end());
}
std::uint64_t get(std::span<const std::byte> bytes, std::size_t& p) {
  if (p > bytes.size() || bytes.size() - p < 8)
    throw SerializationError("Truncated detection AP state");
  std::uint64_t v = 0;
  for (unsigned i = 0; i < 8; ++i)
    v |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(bytes[p++]))
         << (8U * i);
  return v;
}
std::vector<std::byte> topology(const DetectionMetricDescriptor& d) {
  const auto& o = d.options;
  std::vector<std::byte> b;
  put_string(b, d.type_id);
  put(b, d.schema_version);
  put_string(b, o.metric_id);
  put_string(b, o.dataset_id);
  put_string(b, o.class_vocabulary_id);
  put(b, o.num_classes);
  put(b, o.iou_thresholds.size());
  for (double t : o.iou_thresholds) put(b, std::bit_cast<std::uint64_t>(t));
  put(b, std::bit_cast<std::uint64_t>(o.min_area));
  put(b, std::bit_cast<std::uint64_t>(o.max_area));
  put(b, o.max_detections_per_image_class);
  put(b, static_cast<std::uint64_t>(o.no_ground_truth));
  put(b, o.sources.size());
  for (const auto& s : o.sources) put_string(b, s);
  return b;
}
struct Match {
  double score;
  std::uint64_t ordinal, tp, ignore;
};
struct ClassMatches {
  std::uint64_t positives = 0;
  std::vector<Match> predictions;
};
struct ImageMatches {
  std::optional<std::size_t> source;
  std::map<std::size_t, ClassMatches> classes;
};
using Images = std::map<std::uint64_t, ImageMatches>;
std::uint64_t bytes(const ImageMatches& image) {
  std::uint64_t n = 24;
  for (const auto& [id, c] : image.classes) {
    (void)id;
    n = add(n, add(24, mul(32, c.predictions.size())));
  }
  return n;
}
bool score_order(const Match& a, const Match& b) {
  return a.score != b.score ? a.score > b.score : a.ordinal < b.ordinal;
}
void source_valid(std::optional<std::size_t> s,
                  const DetectionMetricOptions& o) {
  if ((o.sources.empty() && s) ||
      (!o.sources.empty() && (!s || *s >= o.sources.size())))
    throw InvalidArgument(
        "Detection source ID must match the explicit source vocabulary");
}
}  // namespace

struct DetectionAveragePrecisionMetric::Impl {
  DetectionMetricDescriptor descriptor;
  DetectionMetricLimits limits;
  Images images;
  std::uint64_t encoded_bytes = 16, retained = 0;
};
DetectionAveragePrecisionMetric::DetectionAveragePrecisionMetric(
    DetectionMetricOptions o, DetectionMetricLimits l) {
  if (!identifier(o.metric_id) || !identifier(o.dataset_id, true) ||
      !identifier(o.class_vocabulary_id, true) || !o.num_classes ||
      o.iou_thresholds.empty() || o.iou_thresholds.size() > 64 ||
      !o.max_detections_per_image_class || !std::isfinite(o.min_area) ||
      !std::isfinite(o.max_area) || o.min_area < 0 || o.max_area < o.min_area ||
      (o.no_ground_truth != DetectionNoGroundTruth::exclude &&
       o.no_ground_truth != DetectionNoGroundTruth::zero &&
       o.no_ground_truth != DetectionNoGroundTruth::error) ||
      !l.max_classes || !l.max_sources || !l.max_images ||
      !l.max_predictions_per_image || !l.max_targets_per_image ||
      !l.max_retained_predictions || l.max_state_bytes < 16 ||
      !l.max_workspace_bytes || !l.max_match_operations_per_update)
    throw InvalidArgument("Invalid detection AP options or limits");
  double last = 0;
  for (double t : o.iou_thresholds) {
    if (!std::isfinite(t) || t <= last || t > 1)
      throw InvalidArgument(
          "Detection IoUs must be strictly increasing in (0,1]");
    last = t;
  }
  if (o.num_classes > l.max_classes || o.sources.size() > l.max_sources ||
      o.max_detections_per_image_class > l.max_predictions_per_image)
    throw ResourceExhausted("Detection AP topology exceeds limits");
  for (std::size_t i = 0; i < o.sources.size(); ++i)
    if (!identifier(o.sources[i]) ||
        std::find(o.sources.begin(),
                  o.sources.begin() + static_cast<std::ptrdiff_t>(i),
                  o.sources[i]) !=
            o.sources.begin() + static_cast<std::ptrdiff_t>(i))
      throw InvalidArgument("Invalid or duplicate detection source name");
  impl_ = std::make_unique<Impl>();
  impl_->descriptor.options = std::move(o);
  impl_->limits = l;
}
DetectionAveragePrecisionMetric::~DetectionAveragePrecisionMetric() = default;

void DetectionAveragePrecisionMetric::update(
    std::span<const DetectionImage> images) {
  auto& s = *impl_;
  const auto& o = s.descriptor.options;
  const auto& l = s.limits;
  add(s.images.size(), images.size(), l.max_images);
  Images staged;
  auto nbytes = s.encoded_bytes, retained = s.retained;
  std::uint64_t operations = 0;
  for (const auto& image : images) {
    if (s.images.contains(image.image_id) || staged.contains(image.image_id))
      throw InvalidArgument("Duplicate detection image ID");
    source_valid(image.source_index, o);
    if (image.predictions.size() > l.max_predictions_per_image ||
        image.targets.size() > l.max_targets_per_image)
      throw ResourceExhausted("Detection AP per-image input limit exceeded");
    for (const auto& p : image.predictions) {
      (void)area(p.box);
      if (p.class_index >= o.num_classes || !std::isfinite(p.score))
        throw InvalidArgument("Invalid detection class or score");
    }
    for (const auto& t : image.targets) {
      (void)area(t.box);
      if (t.class_index >= o.num_classes ||
          (t.area && (!std::isfinite(*t.area) || *t.area < 0)))
        throw InvalidArgument(
            "Invalid detection target class or annotation area");
    }
    // One pair of index arrays, plus matched bits and one reusable GT overlap
    // row.
    const auto work =
        add(mul(add(image.targets.size(), image.predictions.size()),
                sizeof(std::size_t)),
            mul(image.targets.size(), 16));
    if (work > l.max_workspace_bytes)
      throw ResourceExhausted("Detection matching workspace exceeds limit");
    std::vector<std::size_t> ds(image.predictions.size()),
        gs(image.targets.size());
    std::iota(ds.begin(), ds.end(), 0);
    std::iota(gs.begin(), gs.end(), 0);
    std::sort(ds.begin(), ds.end(), [&](auto a, auto b) {
      const auto& x = image.predictions[a];
      const auto& y = image.predictions[b];
      if (x.class_index != y.class_index) return x.class_index < y.class_index;
      return x.score != y.score ? x.score > y.score : a < b;
    });
    std::sort(gs.begin(), gs.end(), [&](auto a, auto b) {
      const auto& x = image.targets[a];
      const auto& y = image.targets[b];
      if (x.class_index != y.class_index) return x.class_index < y.class_index;
      const auto ix = ignored(x, o), iy = ignored(y, o);
      return ix != iy ? ix < iy : a < b;
    });
    ImageMatches output{.source = image.source_index, .classes = {}};
    // Charge the image header before allocating its compact state.
    nbytes = add(nbytes, 24, l.max_state_bytes);
    std::size_t db = 0, gb = 0;
    while (db < ds.size() || gb < gs.size()) {
      const auto cls = std::min(
          db < ds.size() ? image.predictions[ds[db]].class_index
                         : o.num_classes,
          gb < gs.size() ? image.targets[gs[gb]].class_index : o.num_classes);
      auto de = db, ge = gb;
      while (de < ds.size() && image.predictions[ds[de]].class_index == cls)
        ++de;
      while (ge < gs.size() && image.targets[gs[ge]].class_index == cls) ++ge;
      const auto end = db + std::min(de - db, o.max_detections_per_image_class);
      retained = add(retained, end - db, l.max_retained_predictions);
      nbytes = add(nbytes, add(24, mul(end - db, 32)), l.max_state_bytes);
      operations =
          add(operations, mul(mul(end - db, ge - gb), o.iou_thresholds.size()),
              l.max_match_operations_per_update);
      auto& c = output.classes[cls];
      c.predictions.reserve(end - db);
      for (auto j = gb; j < ge; ++j)
        if (!ignored(image.targets[gs[j]], o)) ++c.positives;
      std::vector<std::uint64_t> used(ge - gb);
      std::vector<double> overlaps(ge - gb);
      for (auto i = db; i < end; ++i) {
        const auto& p = image.predictions[ds[i]];
        Match record{.score = p.score, .ordinal = ds[i], .tp = 0, .ignore = 0};
        for (auto j = gb; j < ge; ++j)
          overlaps[j - gb] = overlap(p.box, image.targets[gs[j]].box,
                                     image.targets[gs[j]].crowd);
        for (std::size_t k = 0; k < o.iou_thresholds.size(); ++k) {
          const auto bit = std::uint64_t{1} << k;
          auto best = o.iou_thresholds[k];
          std::optional<std::size_t> found;
          for (auto j = gb; j < ge; ++j) {
            const auto& t = image.targets[gs[j]];
            if ((used[j - gb] & bit) && !t.crowd) continue;
            if (found && !ignored(image.targets[gs[*found]], o) &&
                ignored(t, o))
              break;
            if (overlaps[j - gb] < best) continue;
            best = overlaps[j - gb];
            found = j;
          }
          if (found) {
            used[*found - gb] |= bit;
            if (ignored(image.targets[gs[*found]], o))
              record.ignore |= bit;
            else
              record.tp |= bit;
          } else if (outside(area(p.box), o))
            record.ignore |= bit;
        }
        c.predictions.push_back(record);
      }
      db = de;
      gb = ge;
    }
    staged.emplace(image.image_id, std::move(output));
  }
  // All validation/allocation succeeds before no-allocation node transfer.
  s.images.merge(staged);
  s.encoded_bytes = nbytes;
  s.retained = retained;
}

DetectionSummary DetectionAveragePrecisionMetric::summarize(
    std::optional<std::size_t> source) const {
  const auto& s = *impl_;
  const auto& o = s.descriptor.options;
  if (source && *source >= o.sources.size())
    throw InvalidArgument("Unknown detection summary source");
  struct Ref {
    const Match* match;
    std::uint64_t image;
  };
  const auto result_bytes =
      mul(o.num_classes,
          add(sizeof(DetectionClassResult),
              mul(2 * sizeof(std::optional<double>), o.iou_thresholds.size())));
  if (add(result_bytes, mul(s.retained, sizeof(Ref))) >
      s.limits.max_workspace_bytes)
    throw ResourceExhausted("Detection summary workspace exceeds limit");
  DetectionSummary result;
  result.per_class.resize(o.num_classes);
  for (const auto& [id, image] : s.images) {
    (void)id;
    if (!source || source == image.source) ++result.images;
  }
  double total = 0;
  std::size_t included = 0;
  for (std::size_t cls = 0; cls < o.num_classes; ++cls) {
    auto& c = result.per_class[cls];
    c.ap_by_iou.resize(o.iou_thresholds.size());
    c.recall_by_iou.resize(o.iou_thresholds.size());
    std::vector<Ref> refs;
    // Count first so sort storage has exact requested capacity, without growth
    // copies.
    for (const auto& [id, image] : s.images) {
      (void)id;
      if (source && source != image.source) continue;
      const auto it = image.classes.find(cls);
      if (it == image.classes.end()) continue;
      c.ground_truth = add(c.ground_truth, it->second.positives);
      c.predictions = add(c.predictions, it->second.predictions.size());
    }
    if (!result.images) continue;
    if (!c.ground_truth && o.no_ground_truth == DetectionNoGroundTruth::error)
      throw NumericalError("Detection AP class has no eligible ground truth");
    if (!c.ground_truth && o.no_ground_truth == DetectionNoGroundTruth::exclude)
      continue;
    refs.reserve(static_cast<std::size_t>(c.predictions));
    for (const auto& [id, image] : s.images) {
      if (source && source != image.source) continue;
      const auto it = image.classes.find(cls);
      if (it == image.classes.end()) continue;
      for (const auto& m : it->second.predictions) refs.push_back({&m, id});
    }
    std::sort(refs.begin(), refs.end(), [](const Ref& a, const Ref& b) {
      if (a.match->score != b.match->score)
        return a.match->score > b.match->score;
      return a.image != b.image ? a.image < b.image
                                : a.match->ordinal < b.match->ordinal;
    });
    double ap_sum = 0;
    for (std::size_t k = 0; k < o.iou_thresholds.size(); ++k) {
      const auto bit = std::uint64_t{1} << k;
      std::uint64_t tp = 0, fp = 0;
      for (const auto& r : refs) {
        if (r.match->tp & bit)
          ++tp;
        else if (!(r.match->ignore & bit))
          ++fp;
      }
      c.recall_by_iou[k] =
          c.ground_truth
              ? static_cast<double>(tp) / static_cast<double>(c.ground_truth)
              : 0.;
      std::array<double, 101> precision{};
      int next = 100;
      // COCO uses the FP64 grid arange(101) * (1.0 / 100), then
      // searchsorted against FP64 recall. Rational comparisons differ at
      // representational boundaries such as grid[35] versus recall 7/20.
      const auto gt = static_cast<double>(c.ground_truth);
      while (next >= 0 &&
             static_cast<double>(next) * .01 >
                 (c.ground_truth ? static_cast<double>(tp) / gt : 0.))
        --next;
      double best = 0;
      if (c.ground_truth) {
        for (auto it = refs.rbegin(); it != refs.rend(); ++it) {
          if (it->match->ignore & bit) continue;
          if (tp + fp)
            best = std::max(
                best, static_cast<double>(tp) / static_cast<double>(tp + fp));
          if (it->match->tp & bit) {
            while (next >= 0 && static_cast<double>(next) * .01 >
                                    static_cast<double>(tp - 1) / gt)
              precision[static_cast<std::size_t>(next--)] = best;
            --tp;
          } else
            --fp;
        }
        while (next >= 0) precision[static_cast<std::size_t>(next--)] = best;
      }
      const auto ap =
          std::accumulate(precision.begin(), precision.end(), 0.) / 101.;
      c.ap_by_iou[k] = ap;
      ap_sum += ap;
    }
    c.average_precision = ap_sum / static_cast<double>(o.iou_thresholds.size());
    total += *c.average_precision;
    ++included;
  }
  if (included)
    result.mean_average_precision = total / static_cast<double>(included);
  return result;
}
double DetectionAveragePrecisionMetric::compute() const {
  const auto result = summarize();
  if (!result.mean_average_precision)
    throw NotReady("Detection AP has no included dataset denominator");
  return *result.mean_average_precision;
}
void DetectionAveragePrecisionMetric::merge(
    const DetectionAveragePrecisionMetric& other) {
  auto& s = *impl_;
  const auto& r = *other.impl_;
  if (topology(s.descriptor) != topology(r.descriptor))
    throw CompatibilityError("Detection AP task/data identities differ");
  add(s.images.size(), r.images.size(), s.limits.max_images);
  const auto n =
      add(s.encoded_bytes, r.encoded_bytes - 16, s.limits.max_state_bytes);
  const auto kept =
      add(s.retained, r.retained, s.limits.max_retained_predictions);
  for (const auto& [id, image] : r.images) {
    if (s.images.contains(id))
      throw InvalidArgument("Detection AP merge contains duplicate image IDs");
    std::uint64_t positives = 0;
    for (const auto& [cls, c] : image.classes) {
      (void)cls;
      positives = add(positives, c.positives, s.limits.max_targets_per_image);
      for (const auto& m : c.predictions)
        if (m.ordinal >= s.limits.max_predictions_per_image)
          throw ResourceExhausted(
              "Detection merge prediction ordinal exceeds local input limit");
    }
  }
  Images staged(r.images);
  s.images.merge(staged);
  s.encoded_bytes = n;
  s.retained = kept;
}
void DetectionAveragePrecisionMetric::reset() noexcept {
  impl_->images.clear();
  impl_->encoded_bytes = 16;
  impl_->retained = 0;
}
std::uint64_t DetectionAveragePrecisionMetric::image_count() const noexcept {
  return impl_->images.size();
}
std::uint64_t DetectionAveragePrecisionMetric::state_bytes() const noexcept {
  return impl_->encoded_bytes;
}
const DetectionMetricDescriptor& DetectionAveragePrecisionMetric::descriptor()
    const noexcept {
  return impl_->descriptor;
}
std::string DetectionAveragePrecisionMetric::participant_id() const {
  return impl_->descriptor.options.metric_id;
}
StateManifest DetectionAveragePrecisionMetric::describe_state() const {
  StateManifest m{.participant_id = participant_id(),
                  .schema_version = 1,
                  .restore_level = RestoreLevel::exact,
                  .fields = {{.name = "topology",
                              .media_type = kTopologyMedia,
                              .max_bytes = topology(impl_->descriptor).size(),
                              .required = true},
                             {.name = "matches",
                              .media_type = kMatchMedia,
                              .max_bytes = impl_->limits.max_state_bytes,
                              .required = true}},
                  .degradation_reasons = {}};
  validate_state_manifest(m);
  return m;
}
void DetectionAveragePrecisionMetric::save_state(StateWriter& writer) const {
  writer.write_blob("topology", topology(impl_->descriptor));
  std::vector<std::byte> b;
  b.reserve(static_cast<std::size_t>(impl_->encoded_bytes));
  put(b, kMagic);
  put(b, impl_->images.size());
  for (const auto& [id, image] : impl_->images) {
    put(b, id);
    put(b, image.source.value_or(kNone));
    put(b, image.classes.size());
    for (const auto& [cls, c] : image.classes) {
      put(b, cls);
      put(b, c.positives);
      put(b, c.predictions.size());
      for (const auto& m : c.predictions) {
        put(b, std::bit_cast<std::uint64_t>(m.score));
        put(b, m.ordinal);
        put(b, m.tp);
        put(b, m.ignore);
      }
    }
  }
  writer.write_blob("matches", b);
}
void DetectionAveragePrecisionMetric::load_state(const StateReader& reader,
                                                 RestorePolicy policy) {
  auto& s = *impl_;
  const auto& o = s.descriptor.options;
  const auto& l = s.limits;
  auto expected = describe_state();
  const auto& stored = reader.manifest();
  validate_state_manifest(stored);
  // Resource ceilings belong to the destination, not to metric identity.
  for (auto& f : expected.fields)
    for (const auto& old : stored.fields)
      if (f.name == old.name) f.max_bytes = old.max_bytes;
  if (stored.schema_version != 1 || stored.participant_id != participant_id() ||
      stored.restore_level != RestoreLevel::exact ||
      !restore_policy_accepts(policy, stored.restore_level) ||
      !stored.degradation_reasons.empty() ||
      !state_fields_equal(stored.fields, expected.fields))
    throw CompatibilityError("Detection AP checkpoint manifest differs");
  const auto top = topology(s.descriptor);
  const auto actual = reader.read_blob("topology", top.size());
  if (actual.size() != top.size() ||
      !std::equal(actual.begin(), actual.end(), top.begin()))
    throw CompatibilityError("Detection AP checkpoint topology differs");
  const auto b = reader.read_blob("matches", l.max_state_bytes);
  std::size_t p = 0;
  if (get(b, p) != kMagic)
    throw SerializationError("Unknown detection AP state format");
  const auto n = get(b, p);
  if (n > l.max_images)
    throw ResourceExhausted("Detection checkpoint image limit exceeded");
  if (n > (b.size() - p) / 24)
    throw SerializationError("Invalid detection image count");
  Images candidate;
  std::uint64_t retained = 0, nbytes = 16;
  const auto mask = o.iou_thresholds.size() == 64
                        ? kNone
                        : (std::uint64_t{1} << o.iou_thresholds.size()) - 1;
  for (std::uint64_t i = 0; i < n; ++i) {
    const auto id = get(b, p), source = get(b, p), classes = get(b, p);
    if ((!candidate.empty() && id <= candidate.rbegin()->first) ||
        classes > o.num_classes || classes > (b.size() - p) / 24)
      throw SerializationError(
          "Invalid detection image/class ordering or count");
    ImageMatches image;
    if (source != kNone) {
      if (source >= o.sources.size())
        throw SerializationError("Invalid detection checkpoint source ID");
      image.source = static_cast<std::size_t>(source);
    }
    if (o.sources.empty() == image.source.has_value())
      throw SerializationError("Detection checkpoint source coverage differs");
    std::uint64_t total_gt = 0;
    std::vector<std::uint64_t> ordinals;
    for (std::uint64_t j = 0; j < classes; ++j) {
      const auto cls = get(b, p), gt = get(b, p), count = get(b, p);
      if (cls >= o.num_classes ||
          (!image.classes.empty() && cls <= image.classes.rbegin()->first) ||
          count > o.max_detections_per_image_class ||
          count > (b.size() - p) / 32)
        throw SerializationError("Invalid detection class record");
      total_gt = add(total_gt, gt, l.max_targets_per_image);
      retained = add(retained, count, l.max_retained_predictions);
      if (mul(add(ordinals.size(), count), sizeof(std::uint64_t)) >
          l.max_workspace_bytes)
        throw ResourceExhausted("Detection restore workspace exceeds limit");
      auto& c = image.classes[static_cast<std::size_t>(cls)];
      c.positives = gt;
      c.predictions.reserve(static_cast<std::size_t>(count));
      std::array<std::uint64_t, 64> tps{};
      for (std::uint64_t k = 0; k < count; ++k) {
        Match m{.score = std::bit_cast<double>(get(b, p)),
                .ordinal = get(b, p),
                .tp = get(b, p),
                .ignore = get(b, p)};
        if (!std::isfinite(m.score) ||
            m.ordinal >= l.max_predictions_per_image ||
            ((m.tp | m.ignore) & ~mask) || (m.tp & m.ignore) ||
            (!c.predictions.empty() && !score_order(c.predictions.back(), m)))
          throw SerializationError("Invalid detection match flags/score/order");
        for (std::size_t t = 0; t < o.iou_thresholds.size(); ++t)
          if ((m.tp >> t) & 1U) {
            if (++tps[t] > gt)
              throw SerializationError(
                  "Detection true positives exceed ground truth");
          }
        ordinals.push_back(m.ordinal);
        c.predictions.push_back(m);
      }
    }
    std::sort(ordinals.begin(), ordinals.end());
    if (std::adjacent_find(ordinals.begin(), ordinals.end()) != ordinals.end())
      throw SerializationError(
          "Duplicate prediction identity in detection state");
    nbytes = add(nbytes, bytes(image), l.max_state_bytes);
    candidate.emplace(id, std::move(image));
  }
  if (p != b.size() || nbytes != b.size())
    throw SerializationError("Trailing or inconsistent detection AP state");
  s.images.swap(candidate);
  s.encoded_bytes = nbytes;
  s.retained = retained;
}
}  // namespace jyppx::liuding::training::metrics
