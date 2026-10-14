#include <jyppx/data/samplers/random_sampler.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace jyppx::data::samplers {

namespace {

uint64_t random_seed() {
  std::random_device device;
  return (static_cast<uint64_t>(device()) << 32) ^
         static_cast<uint64_t>(device());
}

}  // namespace

RandomSampler::RandomSampler(size_t size, Optional<int64_t> seed)
    : size_(size),
      generator_(
          seed.has_value()
              ? static_cast<std::mt19937_64::result_type>(*seed)
              : static_cast<std::mt19937_64::result_type>(random_seed())) {
  if (size_ == 0) {
    throw std::invalid_argument("RandomSampler size must be positive");
  }
}

std::vector<size_t> RandomSampler::indices() const {
  std::vector<size_t> result(size_);
  std::iota(result.begin(), result.end(), size_t{0});

  std::shuffle(result.begin(), result.end(), generator_);
  return result;
}

size_t RandomSampler::size() const { return size_; }

std::string RandomSampler::save_state() const {
  std::ostringstream stream;
  stream << generator_;
  return stream.str();
}

void RandomSampler::load_state(const std::string& state) {
  auto restored = generator_;
  std::istringstream stream(state);
  if (!(stream >> restored)) {
    throw std::invalid_argument("Invalid RandomSampler state");
  }
  stream >> std::ws;
  if (!stream.eof()) {
    throw std::invalid_argument("RandomSampler state contains trailing data");
  }
  generator_ = std::move(restored);
}

}  // namespace jyppx::data::samplers
