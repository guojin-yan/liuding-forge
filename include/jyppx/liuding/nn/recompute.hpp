// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_NN_RECOMPUTE_HPP
#define JYPPX_LIUDING_NN_RECOMPUTE_HPP
/** @file recompute.hpp @brief Native activation recomputation with autograd. */
#include <jyppx/liuding/nn/model_adapter.hpp>
namespace jyppx::liuding::nn {
/** @brief Recomputes a pure one-tensor region during backward. Parameters are
 * explicit autograd inputs, including when the data input requires no grad.
 * Saves/restores CPU/CUDA RNG and CPU/CUDA autocast; normal forward runs under
 * NoGrad. The callback and shared module survive until backward completes.
 * Training BatchNorm/InstanceNorm running stats and buffer/parameter mutation
 * are rejected. Mode/topology/state cannot change between forward/backward.
 * No higher-order gradients; process RNG users require external synchronization.
 * Disabled or NoGrad invocations call the region directly. This is unrelated
 * to training checkpoint files or resume. */
Tensor recompute(const Tensor& input,
    std::shared_ptr<torch::nn::Module> region,
    std::function<Tensor(torch::nn::Module&,const Tensor&)> forward,
    bool enabled=true);
}
#endif
