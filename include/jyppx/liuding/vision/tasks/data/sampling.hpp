// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_DATA_SAMPLING_HPP
#define JYPPX_LIUDING_VISION_TASKS_DATA_SAMPLING_HPP
/** @file sampling.hpp @brief Checkpointable identity-balanced sampling. */
#include <jyppx/liuding/data/samplers/sampler.hpp>
#include <memory>
namespace jyppx::liuding::vision::tasks {
/** @brief P distinct identities, K>=2 images per identity, contiguous batches.
 * Requires two distinct source images per identity; extra K draws repeat only
 * after using the shuffled available source images. State includes epoch and
 * draw generation, with the complete source label inventory in its identity. */
std::shared_ptr<data::samplers::Sampler> make_identity_sampler(
    const std::vector<std::int64_t>& identities,std::size_t batch_size,
    std::size_t images_per_identity=2,std::uint64_t seed=0);
/** @brief Training-only inverse-frequency weights for a single-label inventory.
 * Power in [0,1], bounded at one million records; no validation counts used. */
std::vector<double> long_tail_weights(const std::vector<std::int64_t>& labels,
    double power=.5);
}
#endif
