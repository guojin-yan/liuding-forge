# Add implemented source files to the library. New modules are added here as
# their implementations land, keeping the project buildable between phases.
target_sources(jyppx_training_framework PRIVATE
    # Core
    jyppx/core/device.cpp
    jyppx/core/tensor_utils.cpp

    # Optim
    jyppx/optim/optimizer.cpp
    jyppx/optim/sgd.cpp
    jyppx/optim/adam.cpp
    jyppx/optim/adamw.cpp
    jyppx/optim/rmsprop.cpp
    jyppx/optim/adagrad.cpp

    # Optim schedulers
    jyppx/optim/schedulers/lr_scheduler.cpp
    jyppx/optim/schedulers/step_lr.cpp
    jyppx/optim/schedulers/exponential_lr.cpp
    jyppx/optim/schedulers/cosine_annealing_lr.cpp
    jyppx/optim/schedulers/multi_step_lr.cpp
    jyppx/optim/schedulers/reduce_lr_on_plateau.cpp

    # Data
    jyppx/data/tensor_dataset.cpp
    jyppx/data/samplers/sequential_sampler.cpp
    jyppx/data/samplers/random_sampler.cpp
    jyppx/data/samplers/batch_sampler.cpp

    # NN
    jyppx/nn/init.cpp
    jyppx/nn/losses.cpp

    # Training
    jyppx/training/trainer.cpp
    jyppx/training/amp/grad_scaler.cpp
    jyppx/training/metrics/accuracy.cpp
    jyppx/training/metrics/topk_accuracy.cpp
    jyppx/training/metrics/regression.cpp
    jyppx/training/metrics/average_meter.cpp
    jyppx/training/callbacks/early_stopping.cpp
    jyppx/training/callbacks/lr_scheduler_callback.cpp
    jyppx/training/callbacks/model_checkpoint.cpp
    jyppx/training/callbacks/progress_callback.cpp

    # Utils
    jyppx/utils/logger.cpp
)
