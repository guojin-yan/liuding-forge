# Add implemented source files to the library. New modules are added here as
# their implementations land, keeping the project buildable between phases.
target_sources(jyppx_training_framework PRIVATE
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
    jyppx/optim/schedulers/reduce_lr_on_plateau.cpp

    # Optim (extended)
    jyppx/optim/nadam.cpp
    jyppx/optim/lamb.cpp

    # Training
    jyppx/training/amp/grad_scaler.cpp
    jyppx/training/amp/grad_scaler_checkpoint_participant.cpp
    jyppx/training/metrics/accuracy.cpp
    jyppx/training/metrics/topk_accuracy.cpp
    jyppx/training/metrics/regression.cpp
    jyppx/training/metrics/average_meter.cpp
    # Utils
    jyppx/utils/logger.cpp
    jyppx/utils/tensorboard.cpp

    # Distributed (Phase 5, single-process reference)
    jyppx/distributed/distributed.cpp
)
