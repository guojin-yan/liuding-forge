#ifndef JYPPX_JYPPX_HPP
#define JYPPX_JYPPX_HPP

/**
 * @file jyppx.hpp
 * @brief Main header file for JYPPX Training Framework
 *
 * Include this file to use all JYPPX functionality.
 */

// Foundation
#include <jyppx/foundation/foundation.hpp>

// Core
#include <jyppx/core/types.hpp>
#include <jyppx/core/tensor_utils.hpp>
#include <jyppx/core/device.hpp>

// Optimizers
#include <jyppx/optim/optimizer.hpp>
#include <jyppx/optim/optimizer_rate_control.hpp>
#include <jyppx/optim/lr_scheduler.hpp>
#include <jyppx/optim/sgd.hpp>
#include <jyppx/optim/sgd_adapter.hpp>
#include <jyppx/optim/step_lr_scheduler.hpp>
#include <jyppx/optim/composite_lr_scheduler.hpp>
#include <jyppx/optim/adam.hpp>
#include <jyppx/optim/adamw.hpp>
#include <jyppx/optim/rmsprop.hpp>
#include <jyppx/optim/adagrad.hpp>
#include <jyppx/optim/nadam.hpp>
#include <jyppx/optim/lamb.hpp>

// Learning Rate Schedulers
#include <jyppx/optim/schedulers/lr_scheduler.hpp>
#include <jyppx/optim/schedulers/step_lr.hpp>
#include <jyppx/optim/schedulers/exponential_lr.hpp>
#include <jyppx/optim/schedulers/cosine_annealing_lr.hpp>
#include <jyppx/optim/schedulers/multi_step_lr.hpp>
#include <jyppx/optim/schedulers/reduce_lr_on_plateau.hpp>
#include <jyppx/optim/schedulers/one_cycle_lr.hpp>
#include <jyppx/optim/schedulers/cosine_annealing_warm_restarts.hpp>
#include <jyppx/optim/schedulers/linear_lr.hpp>
#include <jyppx/optim/schedulers/p1_schedulers.hpp>
#include <jyppx/optim/schedulers/warmup_scheduler.hpp>

// Data
#include <jyppx/data/dataset.hpp>
#include <jyppx/data/tensor_dataset.hpp>
#include <jyppx/data/dataloader.hpp>
#include <jyppx/data/thread_safe_queue.hpp>
#include <jyppx/data/subset.hpp>
#include <jyppx/data/concat_dataset.hpp>
#include <jyppx/data/chain_dataset.hpp>
#include <jyppx/data/stack_dataset.hpp>

// Transforms
#include <jyppx/data/transforms/transform.hpp>
#include <jyppx/data/transforms/compose.hpp>
#include <jyppx/data/transforms/vision.hpp>

// Samplers
#include <jyppx/data/samplers/sampler.hpp>
#include <jyppx/data/samplers/sequential_sampler.hpp>
#include <jyppx/data/samplers/random_sampler.hpp>
#include <jyppx/data/samplers/batch_sampler.hpp>
#include <jyppx/data/samplers/weighted_random_sampler.hpp>

// Training
#include <jyppx/training/checkpoint_coordinator.hpp>
#include <jyppx/training/trainer.hpp>
#include <jyppx/training/callback.hpp>

// Mixed precision
#include <jyppx/training/amp/autocast.hpp>
#include <jyppx/training/amp/grad_scaler.hpp>
#include <jyppx/training/amp/grad_scaler_checkpoint_participant.hpp>

// Callbacks
#include <jyppx/training/callbacks/model_checkpoint.hpp>
#include <jyppx/training/callbacks/early_stopping.hpp>
#include <jyppx/training/callbacks/lr_scheduler_callback.hpp>
#include <jyppx/training/callbacks/progress_callback.hpp>
#include <jyppx/training/callbacks/logging_callback.hpp>
#include <jyppx/training/callbacks/tensorboard_callback.hpp>

// Metrics
#include <jyppx/training/metric.hpp>
#include <jyppx/training/metrics/accuracy.hpp>
#include <jyppx/training/metrics/topk_accuracy.hpp>
#include <jyppx/training/metrics/classification.hpp>
#include <jyppx/training/metrics/regression.hpp>
#include <jyppx/training/metrics/average_meter.hpp>
#include <jyppx/training/metrics/metric_collection.hpp>

// Utils
#include <jyppx/utils/logger.hpp>
#include <jyppx/utils/progress_bar.hpp>
#include <jyppx/utils/tensorboard.hpp>
#include <jyppx/utils/model_summary.hpp>
#include <jyppx/utils/profiler.hpp>

// Distributed (Phase 5)
#include <jyppx/distributed/distributed.hpp>
#include <jyppx/distributed/ddp.hpp>

// NN
#include <jyppx/nn/invocation.hpp>
#include <jyppx/nn/loss/contrastive_loss.hpp>
#include <jyppx/nn/loss/dice_loss.hpp>
#include <jyppx/nn/loss/focal_loss.hpp>
#include <jyppx/nn/loss/label_smoothing_cross_entropy.hpp>
#include <jyppx/nn/model_adapter.hpp>
#include <jyppx/nn/model_checkpoint_participant.hpp>

#endif // JYPPX_JYPPX_HPP
