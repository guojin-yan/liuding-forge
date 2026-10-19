# Compiler Options for JYPPX Training Framework
#
# Per-configuration flags use generator expressions so they work with both
# single-config (Ninja/Make) and multi-config (Visual Studio) generators.

# ============================================
# Warning Flags
# ============================================
set(JYPPX_COMPILED_TARGETS
    jyppx_foundation
    jyppx_core
    jyppx_data
    jyppx_nn
    jyppx_training
    jyppx_optim
    jyppx_training_framework
)
set(JYPPX_STRICT_TARGETS
    jyppx_foundation
    jyppx_core
    jyppx_data
    jyppx_nn
    jyppx_training
    jyppx_optim
)
foreach(JYPPX_TARGET IN LISTS JYPPX_COMPILED_TARGETS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${JYPPX_TARGET} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wno-unused-parameter
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:RelWithDebInfo>:-O2>
            $<$<CONFIG:Debug>:-O0;-g>
        )
    elseif(MSVC)
        target_compile_options(${JYPPX_TARGET} PRIVATE
            /W4
            /wd4100  # Unused parameter
        )
    endif()
endforeach()

if(JYPPX_ENABLE_WARNINGS_AS_ERRORS)
    if(MSVC)
        message(FATAL_ERROR
            "JYPPX_ENABLE_WARNINGS_AS_ERRORS is not available with MSVC: "
            "LibTorch template instantiations emit upstream C4267 warnings. "
            "The project does not globally suppress that diagnostic.")
    endif()
    foreach(JYPPX_TARGET IN LISTS JYPPX_STRICT_TARGETS)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${JYPPX_TARGET} PRIVATE -Werror)
        endif()
    endforeach()
endif()

# The strict quality gates expand one migrated component at a time. Foundation
# core, data, migrated nn, migrated optim, and the current training slices are
# completed ownership boundaries; adding a source to the migrated list requires
# its declarations, implementation, and tests to pass the same gate first.
if(JYPPX_ENABLE_CLANG_TIDY)
    if(NOT JYPPX_CLANG_TIDY_EXECUTABLE)
        find_program(JYPPX_CLANG_TIDY_EXECUTABLE
            NAMES clang-tidy clang-tidy-20 clang-tidy-19 clang-tidy-18
        )
    endif()
    if(NOT JYPPX_CLANG_TIDY_EXECUTABLE)
        message(FATAL_ERROR
            "JYPPX_ENABLE_CLANG_TIDY=ON requires a clang-tidy executable")
    endif()
    foreach(JYPPX_TARGET IN LISTS JYPPX_STRICT_TARGETS)
        set_target_properties(${JYPPX_TARGET} PROPERTIES
            CXX_CLANG_TIDY
                "${JYPPX_CLANG_TIDY_EXECUTABLE};--config-file=${PROJECT_SOURCE_DIR}/.clang-tidy"
        )
    endforeach()
endif()

if(JYPPX_ENABLE_FORMAT_CHECK)
    if(NOT JYPPX_CLANG_FORMAT_EXECUTABLE)
        find_program(JYPPX_CLANG_FORMAT_EXECUTABLE
            NAMES clang-format clang-format-20 clang-format-19 clang-format-18
        )
    endif()
    if(NOT JYPPX_CLANG_FORMAT_EXECUTABLE)
        message(FATAL_ERROR
            "JYPPX_ENABLE_FORMAT_CHECK=ON requires a clang-format executable")
    endif()
    set(JYPPX_MIGRATED_FORMAT_FILES
        ${PROJECT_SOURCE_DIR}/include/jyppx/foundation/capability.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/foundation/checkpoint.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/foundation/error.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/foundation/foundation.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/foundation/random.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/foundation/random_checkpoint_participant.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/foundation/version.hpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/foundation/capability.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/foundation/checkpoint.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/foundation/error.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/foundation/random.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/foundation/random_checkpoint_participant.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/foundation/version.cpp
        ${PROJECT_SOURCE_DIR}/tests/install/migrated_consumer/main.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/foundation/test_checkpoint.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/foundation/test_foundation.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/foundation/test_rng_checkpoint_participant.cpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/training/amp/grad_scaler.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/training/amp/grad_scaler_checkpoint_participant.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/training/amp/autocast.hpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/training/amp/grad_scaler.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/training/amp/grad_scaler_checkpoint_participant.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/training/test_amp.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/training/test_grad_scaler_checkpoint_participant.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/training/test_nan_inf.cpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/training/checkpoint_coordinator.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/training/callback_dispatcher.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/training/callbacks/early_stopping_callback.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/training/event_sink.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/training/metrics/classification.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/training/metrics/ranking.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/training/scheduler_adapter.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/training/single_device_strategy.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/training/strategy.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/training/trainer.hpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/training/checkpoint_coordinator.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/training/callback_dispatcher.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/training/callbacks/early_stopping_callback.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/training/detail/memory_participant_state.hpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/training/event_sink.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/training/metrics/classification.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/training/metrics/ranking.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/training/scheduler_adapter.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/training/single_device_strategy.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/training/trainer.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/training/test_callback_dispatcher.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/training/test_checkpoint_coordinator.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/training/test_event_sink.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/training/test_classification_metric.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/training/test_ranking_metric.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/training/test_scheduler_adapter.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/training/test_single_device_strategy.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/training/test_trainer.cpp
        ${PROJECT_SOURCE_DIR}/examples/basic_trainer.cpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/core/device.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/core/tensor_utils.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/core/types.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/core/value_schema.hpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/core/device.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/core/tensor_utils.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/core/value_schema.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/core/test_core.cpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/chain_dataset.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/batch_memory.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/batch_schema.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/concat_dataset.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/data_pipeline.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/dataset.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/detail/error.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/stack_dataset.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/subset.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/tensor_dataset.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/iterable_source.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/iterable_data_pipeline.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/samplers/batch_sampler.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/samplers/random_sampler.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/samplers/sampler.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/samplers/sequential_sampler.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/samplers/weighted_random_sampler.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/transforms/compose.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/transforms/transform.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/data/transforms/vision.hpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/data/tensor_dataset.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/data/batch_memory.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/data/batch_schema.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/data/data_capability.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/data/iterable_data_pipeline.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/data/data_pipeline.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/data/samplers/batch_sampler.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/data/samplers/random_sampler.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/data/samplers/sequential_sampler.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/data/test_batch_memory.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/data/test_value_schema.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/data/test_data_pipeline.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/data/test_iterable_data_pipeline.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/data/test_dataset_utils.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/data/test_samplers.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/data/test_tensor_dataset.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/data/test_transforms.cpp
        ${PROJECT_SOURCE_DIR}/tests/fuzz/data_pipeline_state_fuzz.cpp
        ${PROJECT_SOURCE_DIR}/tests/fuzz/iterable_pipeline_state_fuzz.cpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/utils/logger.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/utils/tensorboard.hpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/utils/logger.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/utils/tensorboard.cpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/distributed/ddp.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/distributed/distributed.hpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/distributed/distributed.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/utils/test_logger.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/test_plan_completion.cpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/nn/invocation.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/nn/model_adapter.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/nn/model_checkpoint_participant.hpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/nn/invocation.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/nn/model_adapter.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/nn/model_checkpoint_participant.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/nn/test_model_adapter.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/nn/test_model_checkpoint_participant.cpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/nn/loss/contrastive_loss.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/nn/loss/dice_loss.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/nn/loss/focal_loss.hpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/nn/loss/contrastive_loss.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/nn/loss/dice_loss.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/nn/loss/focal_loss.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/nn/loss/test_contrastive_loss.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/nn/loss/test_dice_loss.cpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/nn/loss/label_smoothing_cross_entropy.hpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/nn/loss/label_smoothing_cross_entropy.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/nn/loss/test_label_smoothing_cross_entropy.cpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/optim/optimizer_rate_control.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/optim/lr_scheduler.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/optim/composite_lr_scheduler.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/optim/optimizer_adapter.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/optim/adadelta_adapter.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/optim/lamb_adapter.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/optim/adagrad_adapter.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/optim/epoch_lr_scheduler.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/optim/event_lr_scheduler.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/optim/linear_lr_scheduler.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/optim/sgd_adapter.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/optim/adam_adapter.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/optim/adamw_adapter.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/optim/nadam_adapter.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/optim/rmsprop_adapter.hpp
        ${PROJECT_SOURCE_DIR}/include/jyppx/optim/step_lr_scheduler.hpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/optim/composite_lr_scheduler.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/optim/epoch_lr_scheduler.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/optim/event_lr_scheduler.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/optim/linear_lr_scheduler.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/optim/adadelta_adapter.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/optim/lamb_adapter.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/optim/adagrad_adapter.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/optim/sgd_adapter.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/optim/adam_adapter.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/optim/adamw_adapter.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/optim/nadam_adapter.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/optim/rmsprop_adapter.cpp
        ${PROJECT_SOURCE_DIR}/src/jyppx/optim/detail/stateful_optimizer_adapter.inc
        ${PROJECT_SOURCE_DIR}/src/jyppx/optim/step_lr_scheduler.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/optim/test_composite_lr_scheduler.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/optim/test_epoch_lr_scheduler.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/optim/test_event_lr_scheduler.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/optim/test_linear_lr_scheduler.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/optim/test_adadelta_adapter.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/optim/test_lamb_adapter.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/optim/test_adagrad_adapter.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/optim/test_sgd_adapter.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/optim/test_adam_adapter.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/optim/test_adamw_adapter.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/optim/test_nadam_adapter.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/optim/test_rmsprop_adapter.cpp
        ${PROJECT_SOURCE_DIR}/tests/unit/optim/test_step_lr_scheduler.cpp
        ${PROJECT_SOURCE_DIR}/tests/benchmarks/bench_optimizer_convergence.cpp
        ${PROJECT_SOURCE_DIR}/tests/benchmarks/bench_adadelta_native_throughput.cpp
        ${PROJECT_SOURCE_DIR}/tests/benchmarks/bench_lamb_native_throughput.cpp
        ${PROJECT_SOURCE_DIR}/tests/benchmarks/bench_adagrad_adapter_overhead.cpp
        ${PROJECT_SOURCE_DIR}/tests/benchmarks/bench_sgd_adapter_overhead.cpp
        ${PROJECT_SOURCE_DIR}/tests/benchmarks/bench_adam_adapter_overhead.cpp
        ${PROJECT_SOURCE_DIR}/tests/benchmarks/bench_adamw_adapter_overhead.cpp
        ${PROJECT_SOURCE_DIR}/tests/benchmarks/bench_nadam_native_throughput.cpp
        ${PROJECT_SOURCE_DIR}/tests/benchmarks/bench_rmsprop_adapter_overhead.cpp
        ${PROJECT_SOURCE_DIR}/tests/benchmarks/bench_cuda_training.cpp
    )
    add_custom_target(jyppx_migrated_format_check ALL
        COMMAND ${JYPPX_CLANG_FORMAT_EXECUTABLE}
            --dry-run --Werror ${JYPPX_MIGRATED_FORMAT_FILES}
        COMMENT "Checking migrated sources with clang-format"
        VERBATIM
    )
endif()

# ============================================
# Debug Definitions
# ============================================
foreach(JYPPX_TARGET IN LISTS JYPPX_COMPILED_TARGETS)
    target_compile_definitions(${JYPPX_TARGET} PRIVATE
        $<$<CONFIG:Debug>:JYPPX_DEBUG>
    )
endforeach()

# ============================================
# Instrumentation (Optional)
# ============================================
set(JYPPX_INSTRUMENTATION_REQUESTED OFF)
if(JYPPX_ENABLE_ASAN OR JYPPX_ENABLE_UBSAN OR JYPPX_ENABLE_TSAN
   OR JYPPX_ENABLE_COVERAGE)
    set(JYPPX_INSTRUMENTATION_REQUESTED ON)
endif()

if(JYPPX_INSTRUMENTATION_REQUESTED
   AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR
        "Coverage and sanitizer options require a GNU or Clang compiler")
endif()

if(JYPPX_ENABLE_ASAN AND JYPPX_ENABLE_TSAN)
    message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer cannot be combined")
endif()

if(JYPPX_ENABLE_COVERAGE AND JYPPX_ENABLE_TSAN)
    message(FATAL_ERROR "Coverage and ThreadSanitizer cannot be combined")
endif()

function(jyppx_apply_instrumentation target_name)
    get_target_property(target_type "${target_name}" TYPE)
    if(target_type STREQUAL "STATIC_LIBRARY")
        # Static archives do not have a link step of their own. Propagate the
        # runtime flag through the exported target so installed consumers link
        # the sanitizer or gcov runtime required by instrumented objects.
        set(link_scope INTERFACE)
    else()
        set(link_scope PRIVATE)
    endif()

    if(JYPPX_ENABLE_ASAN)
        target_compile_options(${target_name} PRIVATE -fsanitize=address)
        target_link_options(${target_name} ${link_scope} -fsanitize=address)
    endif()
    if(JYPPX_ENABLE_UBSAN)
        target_compile_options(${target_name} PRIVATE -fsanitize=undefined)
        target_link_options(${target_name} ${link_scope} -fsanitize=undefined)
    endif()
    if(JYPPX_ENABLE_TSAN)
        target_compile_options(${target_name} PRIVATE -fsanitize=thread)
        target_link_options(${target_name} ${link_scope} -fsanitize=thread)
    endif()
    if(JYPPX_ENABLE_COVERAGE)
        target_compile_options(${target_name} PRIVATE --coverage)
        target_link_options(${target_name} ${link_scope} --coverage)
    endif()
endfunction()

foreach(JYPPX_TARGET IN LISTS JYPPX_COMPILED_TARGETS)
    jyppx_apply_instrumentation(${JYPPX_TARGET})
endforeach()
