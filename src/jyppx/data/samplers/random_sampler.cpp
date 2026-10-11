#include <jyppx/data/samplers/random_sampler.hpp>

#include <algorithm>
#include <numeric>
#include <random>

namespace jyppx::data::samplers {

namespace {

uint64_t random_seed() {
    std::random_device device;
    return (static_cast<uint64_t>(device()) << 32) ^ static_cast<uint64_t>(device());
}

} // namespace

RandomSampler::RandomSampler(size_t size, Optional<int64_t> seed)
    : size_(size), generator_(seed.has_value()
          ? static_cast<std::mt19937_64::result_type>(*seed)
          : static_cast<std::mt19937_64::result_type>(random_seed())) {}

std::vector<size_t> RandomSampler::indices() const {
    std::vector<size_t> result(size_);
    std::iota(result.begin(), result.end(), size_t{0});

    std::shuffle(result.begin(), result.end(), generator_);
    return result;
}

size_t RandomSampler::size() const {
    return size_;
}

} // namespace jyppx::data::samplers
