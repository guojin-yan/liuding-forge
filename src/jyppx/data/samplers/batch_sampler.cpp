#include <jyppx/data/samplers/batch_sampler.hpp>

#include <stdexcept>

namespace jyppx::data::samplers {

BatchSampler::BatchSampler(std::shared_ptr<Sampler> sampler, size_t batch_size,
                           bool drop_last)
    : sampler_(std::move(sampler)),
      batch_size_(batch_size),
      drop_last_(drop_last) {
  if (!sampler_) {
    throw std::invalid_argument("BatchSampler sampler cannot be null");
  }
  if (batch_size_ == 0) {
    throw std::invalid_argument("BatchSampler batch_size must be positive");
  }
}

std::vector<size_t> BatchSampler::indices() const {
  auto result = sampler_->indices();
  if (drop_last_) {
    result.resize((result.size() / batch_size_) * batch_size_);
  }
  return result;
}

size_t BatchSampler::size() const {
  const size_t sample_count = sampler_->size();
  return drop_last_ ? sample_count / batch_size_
                    : (sample_count + batch_size_ - 1) / batch_size_;
}

}  // namespace jyppx::data::samplers
