#ifndef JYPPX_DATA_TRANSFORMS_VISION_HPP
#define JYPPX_DATA_TRANSFORMS_VISION_HPP

#include <jyppx/data/transforms/transform.hpp>

#include <torch/nn/functional.h>

#include <algorithm>
#include <cmath>
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

} // namespace transforms
} // namespace data
} // namespace jyppx

#endif // JYPPX_DATA_TRANSFORMS_VISION_HPP
