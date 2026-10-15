#include <jyppx/data/samplers/sequential_sampler.hpp>

#include <numeric>

namespace jyppx::data::samplers {

SequentialSampler::SequentialSampler(size_t size) : size_(size) {}

std::vector<size_t> SequentialSampler::indices() const {
  std::vector<size_t> result(size_);
  std::iota(result.begin(), result.end(), size_t{0});
  return result;
}

size_t SequentialSampler::size() const { return size_; }

}  // namespace jyppx::data::samplers
