#include <jyppx/data/samplers/batch_sampler.hpp>
#include <jyppx/data/detail/error.hpp>

namespace jyppx::data::samplers {

BatchSampler::BatchSampler(std::shared_ptr<Sampler> sampler, size_t batch_size,
                           bool drop_last)
    : sampler_(std::move(sampler)),
      batch_size_(batch_size),
      drop_last_(drop_last) {
  if (!sampler_) {
    ::jyppx::data::detail::throw_invalid(
        "data.batch_sampler", "constructor.sampler", "non-null sampler",
        "null sampler", "Provide a valid sampler handle.");
  }
  if (batch_size_ == 0) {
    ::jyppx::data::detail::throw_invalid(
        "data.batch_sampler", "constructor.batch_size", "positive batch size",
        "zero", "Set batch_size to at least one.");
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
  const size_t complete_batches = sample_count / batch_size_;
  const size_t remainder = sample_count % batch_size_;
  return drop_last_ || remainder == 0 ? complete_batches : complete_batches + 1;
}

}  // namespace jyppx::data::samplers
