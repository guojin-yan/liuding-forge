#ifndef JYPPX_FOUNDATION_RANDOM_HPP
#define JYPPX_FOUNDATION_RANDOM_HPP

#include <jyppx/core/types.hpp>

#include <cstdint>
#include <vector>

namespace jyppx::core {

inline constexpr std::uint32_t kRngStateFormatVersion = 1;

struct RngState {
    Tensor cpu;
    std::vector<Tensor> cuda;
    std::uint32_t format_version = kRngStateFormatVersion;
};

RngState get_rng_state(bool include_cuda = true);
void validate_rng_state(const RngState& state, bool include_cuda = true);
void set_rng_state(const RngState& state, bool include_cuda = true);
void manual_seed(std::uint64_t seed, bool include_cuda = true);

/** Stable SplitMix64-based derivation for epochs, workers, and user streams. */
std::uint64_t derive_seed(std::uint64_t base_seed,
                          std::uint64_t stream_id,
                          std::uint64_t epoch = 0,
                          std::uint64_t worker_id = 0) noexcept;

class RngGuard {
public:
    explicit RngGuard(std::uint64_t seed, bool include_cuda = true);
    ~RngGuard() noexcept;

    RngGuard(const RngGuard&) = delete;
    RngGuard& operator=(const RngGuard&) = delete;
    RngGuard(RngGuard&&) = delete;
    RngGuard& operator=(RngGuard&&) = delete;

private:
    RngState previous_;
    bool include_cuda_;
};

} // namespace jyppx::core

#endif // JYPPX_FOUNDATION_RANDOM_HPP
