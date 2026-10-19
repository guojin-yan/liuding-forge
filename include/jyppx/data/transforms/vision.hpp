#ifndef JYPPX_DATA_TRANSFORMS_VISION_HPP
#define JYPPX_DATA_TRANSFORMS_VISION_HPP

/**
 * @file vision.hpp
 * @brief Defines tensor-native image preprocessing and augmentation transforms.
 */

#include <jyppx/data/detail/error.hpp>
#include <jyppx/data/transforms/transform.hpp>

#include <torch/nn/functional.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace jyppx {
namespace data {
namespace transforms {

namespace detail {

/**
 * @brief Throws a structured invalid-argument error for a vision transform.
 * @param operation Transform and field or invocation being validated.
 * @param expected Required range, shape, or configuration.
 * @param actual Observed invalid value or shape.
 * @param hint Actionable remediation guidance for the caller.
 * @throws jyppx::InvalidArgument Always.
 */
[[noreturn]] inline void throw_transform_invalid(std::string operation,
                                                 std::string expected,
                                                 std::string actual,
                                                 std::string hint) {
  ::jyppx::data::detail::throw_invalid("data.transforms.vision", operation,
                                       std::move(expected), std::move(actual),
                                       std::move(hint));
}

/**
 * @brief Describes whether a tensor is undefined or reports its rank.
 * @param input Tensor whose definition state and rank are inspected.
 * @return Stable text suitable for structured error details.
 */
inline std::string describe_tensor_rank(const Tensor& input) {
  if (!input.defined()) {
    return "undefined tensor";
  }
  return std::to_string(input.dim()) + " dimensions";
}

/**
 * @brief Normalizes a one- or two-value crop size.
 * @param size `{n}` for a square or `{height, width}` for a rectangle.
 * @return Normalized height and width pair.
 * @throws jyppx::InvalidArgument The vector does not contain one or two values.
 */
inline std::pair<int64_t, int64_t> crop_size(const std::vector<int64_t>& size) {
  if (size.size() == 1) {
    return {size[0], size[0]};
  }
  if (size.size() == 2) {
    return {size[0], size[1]};
  }
  throw_transform_invalid("crop_size", "one or two dimensions",
                          std::to_string(size.size()) + " dimensions",
                          "Use {size} or {height, width}.");
}

/** Pi used for degree-to-radian conversion without platform macros. */
constexpr double kPi = 3.14159265358979323846;

/**
 * @brief Warps an image with an inverse affine matrix.
 * @param input Image tensor shaped `{C, H, W}`.
 * @param t00 Matrix row-zero, column-zero coefficient.
 * @param t01 Matrix row-zero, column-one coefficient.
 * @param t02 Matrix row-zero normalized translation.
 * @param t10 Matrix row-one, column-zero coefficient.
 * @param t11 Matrix row-one, column-one coefficient.
 * @param t12 Matrix row-one normalized translation.
 * @return Float32 image sampled with bilinear, zero-padded grid sampling.
 */
inline Tensor affine_warp(const Tensor& input, double t00, double t01,
                          double t02, double t10, double t11, double t12) {
  const int64_t h = input.size(-2);
  const int64_t w = input.size(-1);
  auto theta =
      torch::tensor(
          {{t00, t01, t02}, {t10, t11, t12}},
          torch::TensorOptions().dtype(torch::kFloat32).device(input.device()))
          .unsqueeze(0);
  auto grid = torch::nn::functional::affine_grid(
      theta, {1, input.size(-3), h, w}, /*align_corners=*/false);
  auto expanded = input.unsqueeze(0).to(torch::kFloat32);
  return torch::nn::functional::grid_sample(
             expanded, grid,
             torch::nn::functional::GridSampleFuncOptions()
                 .mode(torch::kBilinear)
                 .padding_mode(torch::kZeros)
                 .align_corners(false))
      .squeeze(0);
}

/**
 * @brief Applies centered rotation, uniform scale, and pixel translation.
 * @param input Image tensor shaped `{C, H, W}`.
 * @param degrees Counter-clockwise rotation in degrees.
 * @param scale Uniform inverse-grid scale factor.
 * @param tx_px Horizontal translation in input pixels.
 * @param ty_px Vertical translation in input pixels.
 * @return Float32 affine-warped image.
 */
inline Tensor affine_transform(const Tensor& input, double degrees,
                               double scale, double tx_px, double ty_px) {
  const int64_t h = input.size(-2);
  const int64_t w = input.size(-1);
  const double rad = degrees * kPi / 180.0;
  const double c = std::cos(rad);
  const double s = std::sin(rad);
  const double nx = 2.0 * tx_px / static_cast<double>(w);
  const double ny = 2.0 * ty_px / static_cast<double>(h);
  return affine_warp(input, scale * c, scale * s, nx, -scale * s, scale * c,
                     ny);
}

/**
 * @brief Rotates an image around its center without scaling or translation.
 * @param input Image tensor shaped `{C, H, W}`.
 * @param degrees Counter-clockwise rotation in degrees.
 * @return Float32 rotated image.
 */
inline Tensor affine_rotate(const Tensor& input, double degrees) {
  return affine_transform(input, degrees, 1.0, 0.0, 0.0);
}

/**
 * @brief Converts RGB to one ITU-R 601-2 luma channel.
 * @param input Tensor shaped `{1, H, W}` or `{3, H, W}`.
 * @return Original one-channel tensor or a Float32 luma tensor.
 * @throws jyppx::InvalidArgument Shape or channel count is unsupported.
 */
inline Tensor rgb_to_grayscale(const Tensor& input) {
  if (!input.defined() || input.dim() != 3) {
    throw_transform_invalid("grayscale.input_shape", "{C, H, W}",
                            describe_tensor_rank(input),
                            "Provide a channel-first image tensor.");
  }
  if (input.size(0) == 1) {
    return input;
  }
  if (input.size(0) != 3) {
    throw_transform_invalid("grayscale.channels", "1 or 3 channels",
                            std::to_string(input.size(0)) + " channels",
                            "Provide a grayscale or RGB image.");
  }
  auto floats = input.to(torch::kFloat32);
  auto weights =
      torch::tensor(
          {0.299f, 0.587f, 0.114f},
          torch::TensorOptions().dtype(torch::kFloat32).device(input.device()))
          .view({3, 1, 1});
  return (floats * weights).sum(0, /*keepdim=*/true);
}

/**
 * @brief Shifts RGB hue on a Float32 image.
 * @param rgb Three-channel tensor shaped `{3, H, W}`.
 * @param delta Signed hue offset in turns, where one turn is a full cycle.
 * @return Three-channel hue-shifted tensor.
 */
inline Tensor shift_hue(const Tensor& rgb, double delta) {
  auto r = rgb.select(0, 0);
  auto g = rgb.select(0, 1);
  auto b = rgb.select(0, 2);
  const auto maxc = torch::maximum(torch::maximum(r, g), b);
  const auto minc = torch::minimum(torch::minimum(r, g), b);
  const auto range = maxc - minc + 1e-12f;

  auto hue = torch::where(
      maxc == r, (g - b) / range,
      torch::where(maxc == g, (b - r) / range + 2.0f, (r - g) / range + 4.0f));
  hue = hue / 6.0f;
  hue = hue - torch::floor(hue);  // wrap into [0, 1)
  hue = torch::fmod(hue + static_cast<float>(delta) + 1.0f, 1.0f);

  const auto value = maxc;
  const auto sat = (maxc - minc + 1e-12f) / (maxc + 1e-12f);
  const auto sector = hue * 6.0f;
  const auto i = torch::floor(sector);
  const auto f = sector - i;
  const auto p = value * (1.0f - sat);
  const auto q = value * (1.0f - sat * f);
  const auto t = value * (1.0f - sat * (1.0f - f));

  auto rr = torch::where(
      i < 1.0f, value,
      torch::where(
          i < 2.0f, q,
          torch::where(
              i < 3.0f, p,
              torch::where(i < 4.0f, p, torch::where(i < 5.0f, t, value)))));
  auto gg = torch::where(
      i < 1.0f, t,
      torch::where(i < 2.0f, value,
                   torch::where(i < 3.0f, value,
                                torch::where(i < 4.0f, q,
                                             torch::where(i < 5.0f, p, p)))));
  auto bb = torch::where(
      i < 1.0f, p,
      torch::where(
          i < 2.0f, p,
          torch::where(i < 3.0f, t,
                       torch::where(i < 4.0f, value,
                                    torch::where(i < 5.0f, value, q)))));
  return torch::stack({rr, gg, bb}, /*dim=*/0);
}

}  // namespace detail

/**
 * @brief Normalizes a {C, H, W} tensor: (input - mean) / std
 *
 * Integral inputs are promoted to float32 first. mean/std must have one
 * entry per channel.
 */
class Normalize : public TensorTransform {
 public:
  /**
   * @brief Creates a per-channel affine normalization.
   * @param mean Finite mean value for each input channel.
   * @param std Positive standard deviation for each input channel.
   * @throws jyppx::InvalidArgument Vectors are empty, differ in length, or a
   * standard deviation is not positive.
   */
  Normalize(std::vector<double> mean, std::vector<double> std)
      : mean_(
            torch::tensor(mean, torch::TensorOptions().dtype(torch::kFloat32))),
        std_(
            torch::tensor(std, torch::TensorOptions().dtype(torch::kFloat32))) {
    if (mean.empty() || mean.size() != std.size()) {
      detail::throw_transform_invalid(
          "normalize.constructor", "non-empty equal-length mean and std",
          "mean=" + std::to_string(mean.size()) +
              ", std=" + std::to_string(std.size()),
          "Provide one positive standard deviation per channel.");
    }
    for (const double value : mean) {
      if (!std::isfinite(value)) {
        detail::throw_transform_invalid(
            "normalize.mean", "finite channel means", std::to_string(value),
            "Replace NaN or infinite channel means.");
      }
    }
    for (const double s : std) {
      if (!std::isfinite(s) || s <= 0.0) {
        detail::throw_transform_invalid(
            "normalize.std", "positive standard deviations", std::to_string(s),
            "Replace zero or negative standard deviations.");
      }
    }
    // Keep the broadcast shape ready for {C, H, W} inputs.
    const auto channels = static_cast<int64_t>(mean.size());
    mean_ = mean_.view({channels, 1, 1});
    std_ = std_.view({channels, 1, 1});
  }

  /**
   * @brief Normalizes one channel-first tensor.
   * @param input Tensor shaped `{C, ...}` with configured channel count.
   * @return Floating tensor containing `(input - mean) / std`.
   * @throws jyppx::InvalidArgument Input channel count is incompatible.
   */
  Tensor operator()(const Tensor& input) const override {
    if (!input.defined() || input.dim() < 1 || input.size(0) != mean_.size(0)) {
      detail::throw_transform_invalid(
          "normalize.input_shape",
          "a {C, ...} tensor with C=" + std::to_string(mean_.size(0)),
          !input.defined() || input.dim() < 1
              ? "undefined or scalar tensor"
              : "C=" + std::to_string(input.size(0)),
          "Match the input channels to the configured statistics.");
    }
    Tensor float_input = input;
    if (!at::isFloatingType(input.scalar_type())) {
      float_input = input.to(torch::kFloat32);
    }
    const auto mean = mean_.to(float_input.options());
    const auto std = std_.to(float_input.options());
    return (float_input - mean) / std;
  }

  /**
   * @brief Returns the stable transform name.
   * @return `Normalize`.
   */
  std::string name() const override { return "Normalize"; }

 private:
  Tensor mean_;  ///< Float32 per-channel means shaped `{C, 1, 1}`.
  Tensor std_;   ///< Float32 positive deviations shaped `{C, 1, 1}`.
};

/**
 * @brief Resizes a {C, H, W} tensor to {C, size_h, size_w}
 *
 * size = {n} resizes to an n x n square. mode is "bilinear" (default) or
 * "nearest"; bilinear aligns corners like torchvision.
 */
class Resize : public TensorTransform {
 public:
  /**
   * @brief Creates a spatial resize operation.
   * @param size `{n}` for square output or `{height, width}`.
   * @param mode Either `bilinear` or `nearest`.
   * @throws jyppx::InvalidArgument Size or interpolation mode is invalid.
   */
  explicit Resize(std::vector<int64_t> size, std::string mode = "bilinear")
      : size_(std::move(size)), mode_(std::move(mode)) {
    if (size_.empty() || size_.size() > 2) {
      detail::throw_transform_invalid(
          "resize.size", "one or two dimensions",
          std::to_string(size_.size()) + " dimensions",
          "Use {size} or {height, width}.");
    }
    if (mode_ != "bilinear" && mode_ != "nearest") {
      detail::throw_transform_invalid("resize.mode", "bilinear or nearest",
                                      mode_,
                                      "Choose a supported interpolation mode.");
    }
    for (const int64_t dim : size_) {
      if (dim <= 0) {
        detail::throw_transform_invalid(
            "resize.size", "positive dimensions", std::to_string(dim),
            "Use output dimensions greater than zero.");
      }
    }
  }

  /**
   * @brief Resizes the trailing two spatial dimensions.
   * @param input Image tensor shaped `{C, H, W}`.
   * @return Tensor with configured output height and width.
   * @throws jyppx::InvalidArgument Input is not a channel-first image.
   */
  Tensor operator()(const Tensor& input) const override {
    if (!input.defined() || input.dim() != 3) {
      detail::throw_transform_invalid(
          "resize.input_shape", "{C, H, W}",
          detail::describe_tensor_rank(input),
          "Provide an image tensor with spatial dimensions.");
    }
    // interpolate() needs an explicit (h, w) pair; a single size means a
    // square output.
    std::vector<int64_t> output_size =
        size_.size() == 1 ? std::vector<int64_t>{size_[0], size_[0]} : size_;
    auto expanded = input.unsqueeze(0);
    auto options =
        torch::nn::functional::InterpolateFuncOptions().size(output_size);
    // torch's interpolation enums are distinct types, so select per branch;
    // align_corners is only valid for the interpolating modes.
    if (mode_ == "nearest") {
      options.mode(torch::kNearest);
    } else {
      options.mode(torch::kBilinear).align_corners(false);
    }
    return torch::nn::functional::interpolate(expanded, options).squeeze(0);
  }

  /**
   * @brief Returns the transform name with configured output size.
   * @return Human-readable resize description.
   */
  std::string name() const override {
    std::string sizes;
    for (size_t i = 0; i < size_.size(); ++i) {
      if (i > 0) sizes += ", ";
      sizes += std::to_string(size_[i]);
    }
    return "Resize(" + sizes + ")";
  }

 private:
  std::vector<int64_t> size_;  ///< One- or two-value positive output size.
  std::string mode_;           ///< Validated interpolation mode name.
};

/**
 * @brief Crops the central {h, w} region of a {C, H, W} tensor
 */
class CenterCrop : public TensorTransform {
 public:
  /**
   * @brief Creates a fixed center crop.
   * @param size `{n}` for a square or `{height, width}`.
   * @throws jyppx::InvalidArgument Shape count or dimensions are invalid.
   */
  explicit CenterCrop(std::vector<int64_t> size)
      : size_(detail::crop_size(size)) {
    if (size_.first <= 0 || size_.second <= 0) {
      detail::throw_transform_invalid(
          "center_crop.size", "positive height and width",
          std::to_string(size_.first) + "x" + std::to_string(size_.second),
          "Use crop dimensions greater than zero.");
    }
  }

  /**
   * @brief Extracts the configured region from the image center.
   * @param input Image tensor shaped `{C, H, W}`.
   * @return Spatial tensor view of the centered crop.
   * @throws jyppx::InvalidArgument Input shape or crop extent is incompatible.
   */
  Tensor operator()(const Tensor& input) const override {
    if (!input.defined() || input.dim() != 3) {
      detail::throw_transform_invalid("center_crop.input_shape", "{C, H, W}",
                                      detail::describe_tensor_rank(input),
                                      "Provide a channel-first image tensor.");
    }
    const int64_t h = input.size(1);
    const int64_t w = input.size(2);
    if (h < size_.first || w < size_.second) {
      detail::throw_transform_invalid(
          "center_crop.extent", "crop no larger than input",
          "crop=" + std::to_string(size_.first) + "x" +
              std::to_string(size_.second) + ", input=" + std::to_string(h) +
              "x" + std::to_string(w),
          "Reduce the crop or provide a larger image.");
    }
    const int64_t top = (h - size_.first) / 2;
    const int64_t left = (w - size_.second) / 2;
    return input.slice(1, top, top + size_.first)
        .slice(2, left, left + size_.second);
  }

  /**
   * @brief Returns the transform name with configured crop size.
   * @return Human-readable center-crop description.
   */
  std::string name() const override {
    return "CenterCrop(" + std::to_string(size_.first) + ", " +
           std::to_string(size_.second) + ")";
  }

 private:
  std::pair<int64_t, int64_t> size_;  ///< Positive height and width.
};

/**
 * @brief Crops a random {h, w} region of a {C, H, W} tensor
 *
 * The crop position is drawn from the global torch RNG, so sequences are
 * reproducible via torch::manual_seed and safe under the multi-worker
 * DataPipeline batch factory (ATen guards the default generator).
 */
class RandomCrop : public TensorTransform {
 public:
  /**
   * @brief Creates a random fixed-size crop.
   * @param size `{n}` for a square or `{height, width}`.
   * @throws jyppx::InvalidArgument Shape count or dimensions are invalid.
   */
  explicit RandomCrop(std::vector<int64_t> size)
      : size_(detail::crop_size(size)) {
    if (size_.first <= 0 || size_.second <= 0) {
      detail::throw_transform_invalid(
          "random_crop.size", "positive height and width",
          std::to_string(size_.first) + "x" + std::to_string(size_.second),
          "Use crop dimensions greater than zero.");
    }
  }

  /**
   * @brief Extracts one uniformly positioned crop.
   * @param input Image tensor shaped `{C, H, W}`.
   * @return Spatial tensor view selected using the global Torch RNG.
   * @throws jyppx::InvalidArgument Input shape or crop extent is incompatible.
   */
  Tensor operator()(const Tensor& input) const override {
    if (!input.defined() || input.dim() != 3) {
      detail::throw_transform_invalid("random_crop.input_shape", "{C, H, W}",
                                      detail::describe_tensor_rank(input),
                                      "Provide a channel-first image tensor.");
    }
    const int64_t h = input.size(1);
    const int64_t w = input.size(2);
    if (h < size_.first || w < size_.second) {
      detail::throw_transform_invalid(
          "random_crop.extent", "crop no larger than input",
          "crop=" + std::to_string(size_.first) + "x" +
              std::to_string(size_.second) + ", input=" + std::to_string(h) +
              "x" + std::to_string(w),
          "Reduce the crop or provide a larger image.");
    }

    const int64_t top =
        torch::randint(0, h - size_.first + 1, {1}).item<int64_t>();
    const int64_t left =
        torch::randint(0, w - size_.second + 1, {1}).item<int64_t>();
    return input.slice(1, top, top + size_.first)
        .slice(2, left, left + size_.second);
  }

  /**
   * @brief Returns the transform name with configured crop size.
   * @return Human-readable random-crop description.
   */
  std::string name() const override {
    return "RandomCrop(" + std::to_string(size_.first) + ", " +
           std::to_string(size_.second) + ")";
  }

 private:
  std::pair<int64_t, int64_t> size_;  ///< Positive crop height and width.
};

namespace detail {

/**
 * @brief Applies a probabilistic flip along one image dimension.
 * @param input Image tensor shaped `{C, H, W}`.
 * @param dim Spatial dimension to reverse.
 * @param p Flip probability in the closed range `[0, 1]`.
 * @param name Transform name included in diagnostic messages.
 * @return Flipped tensor or the original handle when the draw is rejected.
 * @throws jyppx::InvalidArgument Probability or input shape is invalid.
 */
inline Tensor random_flip(const Tensor& input, int64_t dim, double p,
                          const char* name) {
  if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
    throw_transform_invalid(std::string(name) + ".probability", "[0, 1]",
                            std::to_string(p),
                            "Use a finite probability in the closed interval.");
  }
  if (!input.defined() || input.dim() != 3) {
    throw_transform_invalid(std::string(name) + ".input_shape", "{C, H, W}",
                            describe_tensor_rank(input),
                            "Provide a channel-first image tensor.");
  }
  const double draw = torch::rand({1}).item<double>();
  if (draw < p) {
    return torch::flip(input, {dim});
  }
  return input;
}

}  // namespace detail

/**
 * @brief Horizontally flips a {C, H, W} tensor with probability p
 */
class RandomHorizontalFlip : public TensorTransform {
 public:
  /**
   * @brief Creates a probabilistic horizontal flip.
   * @param p Flip probability in the closed range `[0, 1]`.
   * @throws jyppx::InvalidArgument @p p is outside the valid range.
   */
  explicit RandomHorizontalFlip(double p = 0.5) : p_(p) {
    if (!std::isfinite(p_) || p_ < 0.0 || p_ > 1.0) {
      detail::throw_transform_invalid(
          "random_horizontal_flip.probability", "[0, 1]", std::to_string(p_),
          "Use a finite probability in the closed interval.");
    }
  }

  /**
   * @brief Optionally reverses image width.
   * @param input Image tensor shaped `{C, H, W}`.
   * @return Flipped tensor or the original handle.
   */
  Tensor operator()(const Tensor& input) const override {
    return detail::random_flip(input, /*dim=*/2, p_, "RandomHorizontalFlip");
  }

  /**
   * @brief Returns the stable transform name.
   * @return `RandomHorizontalFlip`.
   */
  std::string name() const override { return "RandomHorizontalFlip"; }

 private:
  double p_;  ///< Validated flip probability in `[0, 1]`.
};

/**
 * @brief Vertically flips a {C, H, W} tensor with probability p
 */
class RandomVerticalFlip : public TensorTransform {
 public:
  /**
   * @brief Creates a probabilistic vertical flip.
   * @param p Flip probability in the closed range `[0, 1]`.
   * @throws jyppx::InvalidArgument @p p is outside the valid range.
   */
  explicit RandomVerticalFlip(double p = 0.5) : p_(p) {
    if (!std::isfinite(p_) || p_ < 0.0 || p_ > 1.0) {
      detail::throw_transform_invalid(
          "random_vertical_flip.probability", "[0, 1]", std::to_string(p_),
          "Use a finite probability in the closed interval.");
    }
  }

  /**
   * @brief Optionally reverses image height.
   * @param input Image tensor shaped `{C, H, W}`.
   * @return Flipped tensor or the original handle.
   */
  Tensor operator()(const Tensor& input) const override {
    return detail::random_flip(input, /*dim=*/1, p_, "RandomVerticalFlip");
  }

  /**
   * @brief Returns the stable transform name.
   * @return `RandomVerticalFlip`.
   */
  std::string name() const override { return "RandomVerticalFlip"; }

 private:
  double p_;  ///< Validated flip probability in `[0, 1]`.
};

/**
 * @brief Pads a {C, H, W} tensor with a constant value
 *
 * padding = {n} pads all sides by n; {left, right, top, bottom} pads each
 * side independently (torchvision order).
 */
class Pad : public TensorTransform {
 public:
  /**
   * @brief Creates a constant padding operation.
   * @param padding One, two, or four side widths in torchvision order.
   * @param fill Scalar value written outside the source image.
   * @throws jyppx::InvalidArgument Padding count is unsupported.
   */
  explicit Pad(std::vector<int64_t> padding, double fill = 0.0)
      : padding_(std::move(padding)), fill_(fill) {
    if (padding_.size() != 1 && padding_.size() != 2 && padding_.size() != 4) {
      detail::throw_transform_invalid(
          "pad.padding", "one, two, or four values",
          std::to_string(padding_.size()) + " values",
          "Use torchvision-compatible padding arity.");
    }
  }

  /**
   * @brief Pads the two trailing spatial dimensions.
   * @param input Tensor shaped at least `{H, W}`, normally `{C, H, W}`.
   * @return Constant-padded tensor.
   * @throws jyppx::InvalidArgument Input has fewer than two dimensions.
   */
  Tensor operator()(const Tensor& input) const override {
    if (!input.defined() || input.dim() < 2) {
      detail::throw_transform_invalid(
          "pad.input_shape", "at least two dimensions",
          detail::describe_tensor_rank(input),
          "Provide a tensor with spatial dimensions.");
    }
    // Expand {n} -> {n, n, n, n} and {l, r} -> {l, r, l, r} (torchvision).
    std::vector<int64_t> pad;
    if (padding_.size() == 1) {
      pad = {padding_[0], padding_[0], padding_[0], padding_[0]};
    } else if (padding_.size() == 2) {
      pad = {padding_[0], padding_[1], padding_[0], padding_[1]};
    } else {
      pad = padding_;
    }
    // constant_pad_nd expects the last-dimension-first order: (w_left,
    // w_right, h_top, h_bottom) for a {C, H, W} tensor.
    return torch::constant_pad_nd(input, {pad[0], pad[1], pad[2], pad[3]},
                                  fill_);
  }

  /**
   * @brief Returns the stable transform name.
   * @return `Pad`.
   */
  std::string name() const override { return "Pad"; }

 private:
  std::vector<int64_t> padding_;  ///< Validated side widths.
  double fill_;                   ///< Constant padding value.
};

/**
 * @brief Rotates a {C, H, W} tensor by a random angle in [-degrees, degrees]
 *
 * The image rotates around its center with bilinear sampling; areas outside
 * the source image are filled with zeros. The angle is drawn from the global
 * torch RNG.
 */
class RandomRotation : public TensorTransform {
 public:
  /**
   * @brief Creates a symmetric random rotation range.
   * @param degrees Non-negative maximum absolute angle in degrees.
   * @throws jyppx::InvalidArgument @p degrees is negative.
   */
  explicit RandomRotation(double degrees) : degrees_(degrees) {
    if (!std::isfinite(degrees_) || degrees_ < 0.0) {
      detail::throw_transform_invalid(
          "random_rotation.degrees", "non-negative degrees",
          std::to_string(degrees_), "Use a non-negative rotation extent.");
    }
  }

  /**
   * @brief Rotates an image by one uniformly drawn angle.
   * @param input Image tensor shaped `{C, H, W}`.
   * @return Float32 rotated image with zero fill.
   * @throws jyppx::InvalidArgument Input is not a channel-first image.
   */
  Tensor operator()(const Tensor& input) const override {
    if (!input.defined() || input.dim() != 3) {
      detail::throw_transform_invalid("random_rotation.input_shape",
                                      "{C, H, W}",
                                      detail::describe_tensor_rank(input),
                                      "Provide a channel-first image tensor.");
    }
    const double angle =
        (torch::rand({1}).item<double>() * 2.0 - 1.0) * degrees_;
    return detail::affine_rotate(input, angle);
  }

  /**
   * @brief Returns the stable transform name.
   * @return `RandomRotation`.
   */
  std::string name() const override { return "RandomRotation"; }

 private:
  double degrees_;  ///< Non-negative maximum absolute rotation.
};

/**
 * @brief Random affine transform: rotation, translation and scaling
 *
 * The angle is drawn from [-degrees, degrees]; each translation axis from
 * [-translate * size, translate * size]; the scale from
 * [scale[0], scale[1]]. Applied in the order rotate -> scale -> translate
 * (torchvision order), bilinear sampling, zero fill.
 */
class RandomAffine : public TensorTransform {
 public:
  /**
   * @brief Creates a random affine augmentation range.
   * @param degrees Non-negative maximum absolute rotation in degrees.
   * @param translate Maximum absolute width and height fractions.
   * @param scale Positive inclusive scale interval.
   * @throws jyppx::InvalidArgument Any configured range is invalid.
   */
  RandomAffine(double degrees, std::pair<double, double> translate = {0.0, 0.0},
               std::pair<double, double> scale = {1.0, 1.0})
      : degrees_(degrees), translate_(translate), scale_(scale) {
    if (!std::isfinite(degrees_) || degrees_ < 0.0) {
      detail::throw_transform_invalid(
          "random_affine.degrees", "non-negative degrees",
          std::to_string(degrees_), "Use a non-negative rotation extent.");
    }
    if (!std::isfinite(translate_.first) || !std::isfinite(translate_.second) ||
        translate_.first < 0.0 || translate_.second < 0.0 ||
        translate_.first > 1.0 || translate_.second > 1.0) {
      detail::throw_transform_invalid(
          "random_affine.translate", "both fractions in [0, 1]",
          std::to_string(translate_.first) + "," +
              std::to_string(translate_.second),
          "Use bounded width and height translation fractions.");
    }
    if (!std::isfinite(scale_.first) || !std::isfinite(scale_.second) ||
        scale_.first <= 0.0 || scale_.second < scale_.first) {
      detail::throw_transform_invalid(
          "random_affine.scale", "0 < minimum <= maximum",
          std::to_string(scale_.first) + "," + std::to_string(scale_.second),
          "Use a positive ordered scale interval.");
    }
  }

  /**
   * @brief Draws and applies one affine transform.
   * @param input Image tensor shaped `{C, H, W}`.
   * @return Float32 affine-warped image with zero fill.
   * @throws jyppx::InvalidArgument Input is not a channel-first image.
   */
  Tensor operator()(const Tensor& input) const override {
    if (!input.defined() || input.dim() != 3) {
      detail::throw_transform_invalid(
          "random_affine.input_shape", "{C, H, W}",
          detail::describe_tensor_rank(input),
          "Provide an image tensor with spatial dimensions.");
    }
    const int64_t h = input.size(input.dim() - 2);
    const int64_t w = input.size(input.dim() - 1);

    const double angle =
        (torch::rand({1}).item<double>() * 2.0 - 1.0) * degrees_;
    const double tx =
        (torch::rand({1}).item<double>() * 2.0 - 1.0) * translate_.first * w;
    const double ty =
        (torch::rand({1}).item<double>() * 2.0 - 1.0) * translate_.second * h;
    const double lo = std::min(scale_.first, scale_.second);
    const double hi = std::max(scale_.first, scale_.second);
    const double scale = lo + torch::rand({1}).item<double>() * (hi - lo);

    return detail::affine_transform(input, angle, scale, tx, ty);
  }

  /**
   * @brief Returns the stable transform name.
   * @return `RandomAffine`.
   */
  std::string name() const override { return "RandomAffine"; }

 private:
  double degrees_;  ///< Non-negative maximum absolute rotation.
  std::pair<double, double>
      translate_;  ///< Maximum width and height translation fractions.
  std::pair<double, double> scale_;  ///< Positive inclusive scale interval.
};

/**
 * @brief Converts RGB {3, H, W} (or pass-through {1, H, W}) to grayscale
 *
 * Uses the ITU-R 601-2 luma transform: 0.299 R + 0.587 G + 0.114 B.
 * num_output_channels = 1 returns a single channel; 3 replicates the luma.
 */
class Grayscale : public TensorTransform {
 public:
  /**
   * @brief Creates a deterministic grayscale conversion.
   * @param num_output_channels One luma channel or three replicated channels.
   * @throws jyppx::InvalidArgument Output channel count is not one or three.
   */
  explicit Grayscale(int64_t num_output_channels = 1)
      : channels_(num_output_channels) {
    if (channels_ != 1 && channels_ != 3) {
      detail::throw_transform_invalid(
          "grayscale.output_channels", "1 or 3", std::to_string(channels_),
          "Select one luma or three replicated channels.");
    }
  }

  /**
   * @brief Converts one image to grayscale.
   * @param input Tensor shaped `{1, H, W}` or `{3, H, W}`.
   * @return Float32 luma with the configured output channel count.
   */
  Tensor operator()(const Tensor& input) const override {
    Tensor gray = detail::rgb_to_grayscale(input);
    if (channels_ == 3) {
      gray = gray.expand({3, gray.size(1), gray.size(2)}).contiguous();
    }
    return gray;
  }

  /**
   * @brief Returns the stable transform name.
   * @return `Grayscale`.
   */
  std::string name() const override { return "Grayscale"; }

 private:
  int64_t channels_;  ///< Validated output channel count of one or three.
};

/**
 * @brief Converts to grayscale with probability p (output keeps 3 channels)
 */
class RandomGrayscale : public TensorTransform {
 public:
  /**
   * @brief Creates a probabilistic three-channel grayscale conversion.
   * @param p Conversion probability in the closed range `[0, 1]`.
   * @throws jyppx::InvalidArgument @p p is outside the valid range.
   */
  explicit RandomGrayscale(double p = 0.1) : p_(p) {
    if (!std::isfinite(p_) || p_ < 0.0 || p_ > 1.0) {
      detail::throw_transform_invalid(
          "random_grayscale.probability", "[0, 1]", std::to_string(p_),
          "Use a finite probability in the closed interval.");
    }
  }

  /**
   * @brief Optionally replaces RGB channels with replicated luma.
   * @param input Image tensor shaped `{C, H, W}`.
   * @return Three-channel grayscale or the original tensor handle.
   * @throws jyppx::InvalidArgument Input is not a channel-first image.
   */
  Tensor operator()(const Tensor& input) const override {
    if (!input.defined() || input.dim() != 3) {
      detail::throw_transform_invalid("random_grayscale.input_shape",
                                      "{C, H, W}",
                                      detail::describe_tensor_rank(input),
                                      "Provide a channel-first image tensor.");
    }
    const double draw = torch::rand({1}).item<double>();
    if (draw < p_ && input.size(0) == 3) {
      const auto gray = detail::rgb_to_grayscale(input);
      return gray.expand({3, gray.size(1), gray.size(2)}).contiguous();
    }
    return input;
  }

  /**
   * @brief Returns the stable transform name.
   * @return `RandomGrayscale`.
   */
  std::string name() const override { return "RandomGrayscale"; }

 private:
  double p_;  ///< Validated conversion probability in `[0, 1]`.
};

/**
 * @brief Randomly adjusts brightness, contrast, saturation and hue
 *
 * Each factor is drawn uniformly: brightness from
 * [max(0, 1-b), 1+b], contrast from [max(0, 1-c), 1+c], saturation from
 * [max(0, 1-s), 1+s], hue from [-h, h]. The operations are applied in a
 * random order per call (torchvision behavior); zero factors are skipped.
 */
class ColorJitter : public TensorTransform {
 public:
  /**
   * @brief Creates independent color adjustment ranges.
   * @param brightness Non-negative symmetric brightness range around one.
   * @param contrast Non-negative symmetric contrast range around one.
   * @param saturation Non-negative symmetric saturation range around one.
   * @param hue Maximum hue offset in turns, within `[0, 0.5]`.
   * @throws jyppx::InvalidArgument Any range is outside its valid domain.
   */
  ColorJitter(double brightness = 0.0, double contrast = 0.0,
              double saturation = 0.0, double hue = 0.0)
      : brightness_(brightness),
        contrast_(contrast),
        saturation_(saturation),
        hue_(hue) {
    if (!std::isfinite(brightness_) || !std::isfinite(contrast_) ||
        !std::isfinite(saturation_) || brightness_ < 0.0 || contrast_ < 0.0 ||
        saturation_ < 0.0) {
      detail::throw_transform_invalid(
          "color_jitter.factors", "non-negative factors",
          std::to_string(brightness_) + "," + std::to_string(contrast_) + "," +
              std::to_string(saturation_),
          "Use non-negative brightness, contrast, and saturation ranges.");
    }
    if (!std::isfinite(hue_) || hue_ < 0.0 || hue_ > 0.5) {
      detail::throw_transform_invalid(
          "color_jitter.hue", "[0, 0.5]", std::to_string(hue_),
          "Use a hue range no larger than half a turn.");
    }
  }

  /**
   * @brief Applies enabled adjustments in randomized order.
   * @param input Image tensor shaped `{C, H, W}`.
   * @return Float32 adjusted image.
   * @throws jyppx::InvalidArgument Input is not a channel-first image.
   */
  Tensor operator()(const Tensor& input) const override {
    if (!input.defined() || input.dim() != 3) {
      detail::throw_transform_invalid("color_jitter.input_shape", "{C, H, W}",
                                      detail::describe_tensor_rank(input),
                                      "Provide a channel-first image tensor.");
    }
    Tensor result = input.to(torch::kFloat32);

    // Random application order over the enabled operations.
    std::vector<int> order = {0, 1, 2, 3};
    std::shuffle(order.begin(), order.end(), generator_);

    for (const int op : order) {
      switch (op) {
        case 0:
          if (brightness_ > 0.0) {
            result = result * jitter_factor(brightness_);
          }
          break;
        case 1: {
          if (contrast_ > 0.0) {
            const auto gray = detail::rgb_to_grayscale(result);
            const double mean = gray.mean().item<double>();
            const double factor = jitter_factor(contrast_);
            result = result * factor + mean * (1.0 - factor);
          }
          break;
        }
        case 2: {
          if (saturation_ > 0.0 && result.size(0) == 3) {
            const auto gray = detail::rgb_to_grayscale(result);
            const auto gray3 = gray.expand({3, gray.size(1), gray.size(2)});
            result = gray3.lerp(result, jitter_factor(saturation_));
          }
          break;
        }
        case 3:
          if (hue_ > 0.0 && result.size(0) == 3) {
            result = detail::shift_hue(
                result, (torch::rand({1}).item<double>() * 2.0 - 1.0) * hue_);
          }
          break;
        default:
          break;
      }
    }
    return result;
  }

  /**
   * @brief Returns the stable transform name.
   * @return `ColorJitter`.
   */
  std::string name() const override { return "ColorJitter"; }

 private:
  /**
   * @brief Draws a multiplicative factor centered on one.
   * @param factor Non-negative symmetric range width.
   * @return Uniform draw from `[max(0, 1-factor), 1+factor]`.
   */
  double jitter_factor(double factor) const {
    const double lo = std::max(0.0, 1.0 - factor);
    return lo + torch::rand({1}).item<double>() * (1.0 + factor - lo);
  }

  double brightness_;  ///< Non-negative brightness range width.
  double contrast_;    ///< Non-negative contrast range width.
  double saturation_;  ///< Non-negative saturation range width.
  double hue_;         ///< Maximum hue offset in `[0, 0.5]` turns.
  mutable std::mt19937 generator_{
      std::random_device{}()};  ///< Operation-order engine.
};

}  // namespace transforms
}  // namespace data
}  // namespace jyppx

#endif  // JYPPX_DATA_TRANSFORMS_VISION_HPP
