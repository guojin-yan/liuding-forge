#ifndef JYPPX_DATA_TRANSFORMS_VISION_HPP
#define JYPPX_DATA_TRANSFORMS_VISION_HPP

#include <jyppx/data/transforms/transform.hpp>

#include <torch/nn/functional.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace jyppx {
namespace data {
namespace transforms {

namespace detail {

// Normalizes a size argument: {n} means a square, {h, w} is used as-is.
inline std::pair<int64_t, int64_t> crop_size(const std::vector<int64_t>& size) {
    if (size.size() == 1) {
        return {size[0], size[0]};
    }
    if (size.size() == 2) {
        return {size[0], size[1]};
    }
    throw std::invalid_argument("size must contain one (square) or two (h, w) values");
}

constexpr double kPi = 3.14159265358979323846;

// Warps `input` {C, H, W} with the inverse affine matrix
// [t00 t01 t02; t10 t11 t12] in normalized (align_corners=False) coords.
inline Tensor affine_warp(const Tensor& input, double t00, double t01, double t02,
                          double t10, double t11, double t12) {
    const int64_t h = input.size(-2);
    const int64_t w = input.size(-1);
    auto theta = torch::tensor({{t00, t01, t02}, {t10, t11, t12}},
                               torch::TensorOptions().dtype(torch::kFloat32))
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

inline Tensor affine_transform(const Tensor& input, double degrees, double scale,
                               double tx_px, double ty_px) {
    const int64_t h = input.size(-2);
    const int64_t w = input.size(-1);
    const double rad = degrees * kPi / 180.0;
    const double c = std::cos(rad);
    const double s = std::sin(rad);
    const double nx = 2.0 * tx_px / static_cast<double>(w);
    const double ny = 2.0 * ty_px / static_cast<double>(h);
    return affine_warp(input, scale * c, scale * s, nx, -scale * s, scale * c, ny);
}

// Rotation about the image center (normalized coords put (0, 0) there).
inline Tensor affine_rotate(const Tensor& input, double degrees) {
    return affine_transform(input, degrees, 1.0, 0.0, 0.0);
}

// ITU-R 601-2 luma for {3, H, W}; 1-channel input passes through.
inline Tensor rgb_to_grayscale(const Tensor& input) {
    if (input.dim() < 3) {
        throw std::invalid_argument("grayscale expects a {C, H, W} tensor");
    }
    if (input.size(0) == 1) {
        return input;
    }
    if (input.size(0) != 3) {
        throw std::invalid_argument("grayscale expects 1 or 3 channels");
    }
    auto floats = input.to(torch::kFloat32);
    auto weights = torch::tensor({0.299f, 0.587f, 0.114f},
                                 torch::TensorOptions().dtype(torch::kFloat32))
                       .view({3, 1, 1});
    return (floats * weights).sum(0, /*keepdim=*/true);
}

// Shifts RGB hue by delta turns on a {3, H, W} float tensor.
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
    hue = hue - torch::floor(hue);          // wrap into [0, 1)
    hue = torch::fmod(hue + static_cast<float>(delta) + 1.0f, 1.0f);

    const auto value = maxc;
    const auto sat = (maxc - minc + 1e-12f) / (maxc + 1e-12f);
    const auto sector = hue * 6.0f;
    const auto i = torch::floor(sector);
    const auto f = sector - i;
    const auto p = value * (1.0f - sat);
    const auto q = value * (1.0f - sat * f);
    const auto t = value * (1.0f - sat * (1.0f - f));

    auto rr = torch::where(i < 1.0f, value,
               torch::where(i < 2.0f, q,
                 torch::where(i < 3.0f, p,
                   torch::where(i < 4.0f, p,
                     torch::where(i < 5.0f, t, value)))));
    auto gg = torch::where(i < 1.0f, t,
               torch::where(i < 2.0f, value,
                 torch::where(i < 3.0f, value,
                   torch::where(i < 4.0f, q,
                     torch::where(i < 5.0f, p, p)))));
    auto bb = torch::where(i < 1.0f, p,
               torch::where(i < 2.0f, p,
                 torch::where(i < 3.0f, t,
                   torch::where(i < 4.0f, value,
                     torch::where(i < 5.0f, value, q)))));
    return torch::stack({rr, gg, bb}, /*dim=*/0);
}

} // namespace detail

/**
 * @brief Normalizes a {C, H, W} tensor: (input - mean) / std
 *
 * Integral inputs are promoted to float32 first. mean/std must have one
 * entry per channel.
 */
class Normalize : public TensorTransform {
public:
    Normalize(std::vector<double> mean, std::vector<double> std)
        : mean_(torch::tensor(mean, torch::TensorOptions().dtype(torch::kFloat32))),
          std_(torch::tensor(std, torch::TensorOptions().dtype(torch::kFloat32))) {
        if (mean.empty() || mean.size() != std.size()) {
            throw std::invalid_argument("Normalize mean and std must be non-empty and equal length");
        }
        for (const double s : std) {
            if (s <= 0.0) {
                throw std::invalid_argument("Normalize std values must be positive");
            }
        }
        // Keep the broadcast shape ready for {C, H, W} inputs.
        const auto channels = static_cast<int64_t>(mean.size());
        mean_ = mean_.view({channels, 1, 1});
        std_ = std_.view({channels, 1, 1});
    }

    Tensor operator()(const Tensor& input) const override {
        if (input.dim() < 1 || input.size(0) != mean_.size(0)) {
            throw std::invalid_argument(
                "Normalize expects a {C, ...} tensor with C = " +
                std::to_string(mean_.size(0)));
        }
        Tensor float_input = input;
        if (!at::isFloatingType(input.scalar_type())) {
            float_input = input.to(torch::kFloat32);
        }
        return (float_input - mean_) / std_;
    }

    std::string name() const override { return "Normalize"; }

private:
    Tensor mean_;
    Tensor std_;
};

/**
 * @brief Resizes a {C, H, W} tensor to {C, size_h, size_w}
 *
 * size = {n} resizes to an n x n square. mode is "bilinear" (default) or
 * "nearest"; bilinear aligns corners like torchvision.
 */
class Resize : public TensorTransform {
public:
    explicit Resize(std::vector<int64_t> size, std::string mode = "bilinear")
        : size_(std::move(size)), mode_(std::move(mode)) {
        if (size_.empty() || size_.size() > 2) {
            throw std::invalid_argument("Resize size must contain one or two values");
        }
        if (mode_ != "bilinear" && mode_ != "nearest") {
            throw std::invalid_argument("Resize mode must be 'bilinear' or 'nearest'");
        }
        for (const int64_t dim : size_) {
            if (dim <= 0) {
                throw std::invalid_argument("Resize size values must be positive");
            }
        }
    }

    Tensor operator()(const Tensor& input) const override {
        if (input.dim() < 2) {
            throw std::invalid_argument("Resize expects a {C, H, W} tensor");
        }
        // interpolate() needs an explicit (h, w) pair; a single size means a
        // square output.
        std::vector<int64_t> output_size =
            size_.size() == 1 ? std::vector<int64_t>{size_[0], size_[0]} : size_;
        auto expanded = input.unsqueeze(0);
        auto options = torch::nn::functional::InterpolateFuncOptions().size(output_size);
        // torch's interpolation enums are distinct types, so select per branch;
        // align_corners is only valid for the interpolating modes.
        if (mode_ == "nearest") {
            options.mode(torch::kNearest);
        } else {
            options.mode(torch::kBilinear).align_corners(false);
        }
        return torch::nn::functional::interpolate(expanded, options).squeeze(0);
    }

    std::string name() const override {
        std::string sizes;
        for (size_t i = 0; i < size_.size(); ++i) {
            if (i > 0) sizes += ", ";
            sizes += std::to_string(size_[i]);
        }
        return "Resize(" + sizes + ")";
    }

private:
    std::vector<int64_t> size_;
    std::string mode_;
};

/**
 * @brief Crops the central {h, w} region of a {C, H, W} tensor
 */
class CenterCrop : public TensorTransform {
public:
    explicit CenterCrop(std::vector<int64_t> size) : size_(detail::crop_size(size)) {
        if (size_.first <= 0 || size_.second <= 0) {
            throw std::invalid_argument("CenterCrop size values must be positive");
        }
    }

    Tensor operator()(const Tensor& input) const override {
        if (input.dim() < 3) {
            throw std::invalid_argument("CenterCrop expects a {C, H, W} tensor");
        }
        const int64_t h = input.size(1);
        const int64_t w = input.size(2);
        if (h < size_.first || w < size_.second) {
            throw std::invalid_argument(
                "CenterCrop size is larger than the input image");
        }
        const int64_t top = (h - size_.first) / 2;
        const int64_t left = (w - size_.second) / 2;
        return input.slice(1, top, top + size_.first)
            .slice(2, left, left + size_.second);
    }

    std::string name() const override {
        return "CenterCrop(" + std::to_string(size_.first) + ", " +
               std::to_string(size_.second) + ")";
    }

private:
    std::pair<int64_t, int64_t> size_;
};

/**
 * @brief Crops a random {h, w} region of a {C, H, W} tensor
 *
 * The crop position is drawn from the global torch RNG, so sequences are
 * reproducible via torch::manual_seed and safe under the multi-worker
 * DataLoader (ATen guards the default generator).
 */
class RandomCrop : public TensorTransform {
public:
    explicit RandomCrop(std::vector<int64_t> size)
        : size_(detail::crop_size(size)) {
        if (size_.first <= 0 || size_.second <= 0) {
            throw std::invalid_argument("RandomCrop size values must be positive");
        }
    }

    Tensor operator()(const Tensor& input) const override {
        if (input.dim() < 3) {
            throw std::invalid_argument("RandomCrop expects a {C, H, W} tensor");
        }
        const int64_t h = input.size(1);
        const int64_t w = input.size(2);
        if (h < size_.first || w < size_.second) {
            throw std::invalid_argument("RandomCrop size is larger than the input image");
        }

        const int64_t top =
            torch::randint(0, h - size_.first + 1, {1}).item<int64_t>();
        const int64_t left =
            torch::randint(0, w - size_.second + 1, {1}).item<int64_t>();
        return input.slice(1, top, top + size_.first)
            .slice(2, left, left + size_.second);
    }

    std::string name() const override {
        return "RandomCrop(" + std::to_string(size_.first) + ", " +
               std::to_string(size_.second) + ")";
    }

private:
    std::pair<int64_t, int64_t> size_;
};

namespace detail {

// Shared flip logic: flips dim 2 (width) or dim 1 (height) with probability p.
inline Tensor random_flip(const Tensor& input, int64_t dim, double p, const char* name) {
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument(std::string(name) +
                                    " probability must be in [0, 1]");
    }
    if (input.dim() < 3) {
        throw std::invalid_argument(std::string(name) +
                                    " expects a {C, H, W} tensor");
    }
    const double draw = torch::rand({1}).item<double>();
    if (draw < p) {
        return torch::flip(input, {dim});
    }
    return input;
}

} // namespace detail

/**
 * @brief Horizontally flips a {C, H, W} tensor with probability p
 */
class RandomHorizontalFlip : public TensorTransform {
public:
    explicit RandomHorizontalFlip(double p = 0.5) : p_(p) {
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument(
                "RandomHorizontalFlip probability must be in [0, 1]");
        }
    }

    Tensor operator()(const Tensor& input) const override {
        return detail::random_flip(input, /*dim=*/2, p_, "RandomHorizontalFlip");
    }

    std::string name() const override { return "RandomHorizontalFlip"; }

private:
    double p_;
};

/**
 * @brief Vertically flips a {C, H, W} tensor with probability p
 */
class RandomVerticalFlip : public TensorTransform {
public:
    explicit RandomVerticalFlip(double p = 0.5) : p_(p) {
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument(
                "RandomVerticalFlip probability must be in [0, 1]");
        }
    }

    Tensor operator()(const Tensor& input) const override {
        return detail::random_flip(input, /*dim=*/1, p_, "RandomVerticalFlip");
    }

    std::string name() const override { return "RandomVerticalFlip"; }

private:
    double p_;
};

/**
 * @brief Pads a {C, H, W} tensor with a constant value
 *
 * padding = {n} pads all sides by n; {left, right, top, bottom} pads each
 * side independently (torchvision order).
 */
class Pad : public TensorTransform {
public:
    explicit Pad(std::vector<int64_t> padding, double fill = 0.0)
        : padding_(std::move(padding)), fill_(fill) {
        if (padding_.empty() || padding_.size() > 4) {
            throw std::invalid_argument("Pad padding must contain 1, 2 or 4 values");
        }
    }

    Tensor operator()(const Tensor& input) const override {
        if (input.dim() < 2) {
            throw std::invalid_argument("Pad expects a {C, H, W} tensor");
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
        return torch::constant_pad_nd(input, {pad[0], pad[1], pad[2], pad[3]}, fill_);
    }

    std::string name() const override { return "Pad"; }

private:
    std::vector<int64_t> padding_;
    double fill_;
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
    explicit RandomRotation(double degrees) : degrees_(degrees) {
        if (degrees_ < 0.0) {
            throw std::invalid_argument("RandomRotation degrees must be non-negative");
        }
    }

    Tensor operator()(const Tensor& input) const override {
        const double angle = (torch::rand({1}).item<double>() * 2.0 - 1.0) * degrees_;
        return detail::affine_rotate(input, angle);
    }

    std::string name() const override { return "RandomRotation"; }

private:
    double degrees_;
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
    RandomAffine(double degrees, std::pair<double, double> translate = {0.0, 0.0},
                 std::pair<double, double> scale = {1.0, 1.0})
        : degrees_(degrees), translate_(translate), scale_(scale) {
        if (degrees_ < 0.0) {
            throw std::invalid_argument("RandomAffine degrees must be non-negative");
        }
        if (translate_.first < 0.0 || translate_.second < 0.0 ||
            translate_.first > 1.0 || translate_.second > 1.0) {
            throw std::invalid_argument("RandomAffine translate must be within [0, 1]");
        }
        if (scale_.first <= 0.0 || scale_.second < scale_.first) {
            throw std::invalid_argument(
                "RandomAffine scale must be positive and increasing");
        }
    }

    Tensor operator()(const Tensor& input) const override {
        if (input.dim() < 2) {
            throw std::invalid_argument("RandomAffine expects a {C, H, W} tensor");
        }
        const int64_t h = input.size(input.dim() - 2);
        const int64_t w = input.size(input.dim() - 1);

        const double angle = (torch::rand({1}).item<double>() * 2.0 - 1.0) * degrees_;
        const double tx =
            (torch::rand({1}).item<double>() * 2.0 - 1.0) * translate_.first * w;
        const double ty =
            (torch::rand({1}).item<double>() * 2.0 - 1.0) * translate_.second * h;
        const double lo = std::min(scale_.first, scale_.second);
        const double hi = std::max(scale_.first, scale_.second);
        const double scale = lo + torch::rand({1}).item<double>() * (hi - lo);

        return detail::affine_transform(input, angle, scale, tx, ty);
    }

    std::string name() const override { return "RandomAffine"; }

private:
    double degrees_;
    std::pair<double, double> translate_;
    std::pair<double, double> scale_;
};

/**
 * @brief Converts RGB {3, H, W} (or pass-through {1, H, W}) to grayscale
 *
 * Uses the ITU-R 601-2 luma transform: 0.299 R + 0.587 G + 0.114 B.
 * num_output_channels = 1 returns a single channel; 3 replicates the luma.
 */
class Grayscale : public TensorTransform {
public:
    explicit Grayscale(int64_t num_output_channels = 1)
        : channels_(num_output_channels) {
        if (channels_ != 1 && channels_ != 3) {
            throw std::invalid_argument("Grayscale num_output_channels must be 1 or 3");
        }
    }

    Tensor operator()(const Tensor& input) const override {
        Tensor gray = detail::rgb_to_grayscale(input);
        if (channels_ == 3) {
            gray = gray.expand({3, gray.size(1), gray.size(2)}).contiguous();
        }
        return gray;
    }

    std::string name() const override { return "Grayscale"; }

private:
    int64_t channels_;
};

/**
 * @brief Converts to grayscale with probability p (output keeps 3 channels)
 */
class RandomGrayscale : public TensorTransform {
public:
    explicit RandomGrayscale(double p = 0.1) : p_(p) {
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument(
                "RandomGrayscale probability must be in [0, 1]");
        }
    }

    Tensor operator()(const Tensor& input) const override {
        if (input.dim() < 3) {
            throw std::invalid_argument("RandomGrayscale expects a {C, H, W} tensor");
        }
        const double draw = torch::rand({1}).item<double>();
        if (draw < p_ && input.size(0) == 3) {
            const auto gray = detail::rgb_to_grayscale(input);
            return gray.expand({3, gray.size(1), gray.size(2)}).contiguous();
        }
        return input;
    }

    std::string name() const override { return "RandomGrayscale"; }

private:
    double p_;
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
    ColorJitter(double brightness = 0.0, double contrast = 0.0,
                double saturation = 0.0, double hue = 0.0)
        : brightness_(brightness), contrast_(contrast),
          saturation_(saturation), hue_(hue) {
        if (brightness_ < 0.0 || contrast_ < 0.0 || saturation_ < 0.0) {
            throw std::invalid_argument(
                "ColorJitter factors must be non-negative");
        }
        if (hue_ < 0.0 || hue_ > 0.5) {
            throw std::invalid_argument("ColorJitter hue must be in [0, 0.5]");
        }
    }

    Tensor operator()(const Tensor& input) const override {
        if (input.dim() < 3) {
            throw std::invalid_argument("ColorJitter expects a {C, H, W} tensor");
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
                        const auto gray3 =
                            gray.expand({3, gray.size(1), gray.size(2)});
                        result = gray3.lerp(result, jitter_factor(saturation_));
                    }
                    break;
                }
                case 3:
                    if (hue_ > 0.0 && result.size(0) == 3) {
                        result = detail::shift_hue(
                            result,
                            (torch::rand({1}).item<double>() * 2.0 - 1.0) * hue_);
                    }
                    break;
                default:
                    break;
            }
        }
        return result;
    }

    std::string name() const override { return "ColorJitter"; }

private:
    double jitter_factor(double factor) const {
        const double lo = std::max(0.0, 1.0 - factor);
        return lo + torch::rand({1}).item<double>() * (1.0 + factor - lo);
    }

    double brightness_;
    double contrast_;
    double saturation_;
    double hue_;
    mutable std::mt19937 generator_{std::random_device{}()};
};

} // namespace transforms
} // namespace data
} // namespace jyppx

#endif // JYPPX_DATA_TRANSFORMS_VISION_HPP
