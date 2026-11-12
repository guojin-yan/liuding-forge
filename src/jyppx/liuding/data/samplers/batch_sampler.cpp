// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
#include <jyppx/liuding/data/detail/error.hpp>
#include <jyppx/liuding/data/samplers/batch_sampler.hpp>

namespace jyppx::liuding::data::samplers {
namespace {

size_t flat_sample_count(const Sampler& sampler) {
  const auto* batches = dynamic_cast<const BatchSampler*>(&sampler);
  return batches ? batches->sample_count() : sampler.size();
}

}  // namespace

BatchSampler::BatchSampler(std::shared_ptr<Sampler> sampler, size_t batch_size,
                           bool drop_last)
    : sampler_(std::move(sampler)),
      batch_size_(batch_size),
      drop_last_(drop_last) {
  if (!sampler_) {
    ::jyppx::liuding::data::detail::throw_invalid(
        "data.batch_sampler", "constructor.sampler", "non-null sampler",
        "null sampler", "Provide a valid sampler handle.");
  }
  if (batch_size_ == 0) {
    ::jyppx::liuding::data::detail::throw_invalid(
        "data.batch_sampler", "constructor.batch_size", "positive batch size",
        "zero", "Set batch_size to at least one.");
  }
}

std::vector<size_t> BatchSampler::indices() const {
  auto result = sampler_->indices();
  const auto expected = flat_sample_count(*sampler_);
  if (result.size() != expected) {
    ::jyppx::liuding::data::detail::throw_invalid(
        "data.batch_sampler", "indices", std::to_string(expected),
        std::to_string(result.size()),
        "Fix the underlying sampler size contract.");
  }
  if (drop_last_) {
    result.resize((result.size() / batch_size_) * batch_size_);
  }
  return result;
}

size_t BatchSampler::size() const {
  const auto count = sample_count();
  const size_t complete_batches = count / batch_size_;
  const size_t remainder = count % batch_size_;
  return drop_last_ || remainder == 0 ? complete_batches : complete_batches + 1;
}

size_t BatchSampler::sample_count() const {
  const auto count = flat_sample_count(*sampler_);
  return drop_last_ ? (count / batch_size_) * batch_size_ : count;
}

std::string BatchSampler::configuration_identity() const {
  const auto base = sampler_->configuration_identity();
  if (base.empty()) return {};
  return "jyppx.sampler.batch/v1:batch_size=" + std::to_string(batch_size_) +
         ":drop_last=" + (drop_last_ ? "1" : "0") + ":base=" + base;
}

}  // namespace jyppx::liuding::data::samplers
