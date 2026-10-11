#include <jyppx/training/metrics/average_meter.hpp>

#include <stdexcept>

namespace jyppx {
namespace training {
namespace metrics {

void AverageMeter::update(double value, int64_t count) {
    if (count <= 0) {
        throw std::invalid_argument("AverageMeter count must be positive");
    }
    sum_ += value * static_cast<double>(count);
    count_ += count;
    last_value_ = value;
}

void AverageMeter::update(const Tensor& value, int64_t count) {
    if (!value.defined() || value.numel() == 0) {
        throw std::invalid_argument("AverageMeter value must be a non-empty tensor");
    }
    update(value.mean().item<double>(), count);
}

double AverageMeter::average() const {
    return count_ == 0 ? 0.0 : sum_ / static_cast<double>(count_);
}

void AverageMeter::reset() {
    sum_ = 0.0;
    count_ = 0;
    last_value_ = 0.0;
}

} // namespace metrics
} // namespace training
} // namespace jyppx
