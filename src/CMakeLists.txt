# Add implemented source files to the library. New modules are added here as
# their implementations land, keeping the project buildable between phases.
target_sources(jyppx_training_framework PRIVATE
    # Optim
    jyppx/optim/optimizer.cpp
    # Optim (extended)
    jyppx/optim/lamb.cpp

    # Legacy SDK-only utilities
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
