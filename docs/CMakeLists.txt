# API documentation is gated one migrated component at a time. Only components
# that pass the complete strict gate belong in this input set.
set(DOXYGEN_PROJECT_NAME "JYPPX Training Framework")
set(DOXYGEN_PROJECT_NUMBER "${PROJECT_VERSION}")
set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(DOXYGEN_EXTRACT_ALL NO)
set(DOXYGEN_EXTRACT_PRIVATE YES)
set(DOXYGEN_EXTRACT_STATIC YES)
set(DOXYGEN_HIDE_UNDOC_MEMBERS NO)
set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
set(DOXYGEN_WARN_IF_DOC_ERROR YES)
set(DOXYGEN_WARN_IF_INCOMPLETE_DOC YES)
set(DOXYGEN_WARN_NO_PARAMDOC YES)
set(DOXYGEN_WARN_IF_UNDOC_ENUM_VAL YES)
set(DOXYGEN_WARN_AS_ERROR FAIL_ON_WARNINGS)
set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_GENERATE_LATEX NO)
set(DOXYGEN_QUIET YES)
set(DOXYGEN_RECURSIVE YES)
set(DOXYGEN_STRIP_FROM_PATH "${PROJECT_SOURCE_DIR}/include")

doxygen_add_docs(jyppx_migrated_api_docs
    "${PROJECT_SOURCE_DIR}/include/jyppx/foundation"
    "${PROJECT_SOURCE_DIR}/include/jyppx/core"
    "${PROJECT_SOURCE_DIR}/include/jyppx/data"
    "${PROJECT_SOURCE_DIR}/include/jyppx/nn"
    "${PROJECT_SOURCE_DIR}/include/jyppx/optim/optimizer_rate_control.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/optim/lr_scheduler.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/optim/composite_lr_scheduler.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/optim/epoch_lr_scheduler.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/optim/event_lr_scheduler.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/optim/linear_lr_scheduler.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/optim/sgd_adapter.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/optim/adam_adapter.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/optim/adamw_adapter.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/optim/adadelta_adapter.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/optim/lamb_adapter.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/optim/nadam_adapter.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/optim/rmsprop_adapter.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/optim/adagrad_adapter.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/optim/step_lr_scheduler.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/training/amp/autocast.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/training/amp/grad_scaler.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/training/amp/grad_scaler_checkpoint_participant.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/training/callback_dispatcher.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/training/callbacks/early_stopping_callback.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/training/checkpoint_coordinator.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/training/event_sink.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/training/metrics/classification.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/training/metrics/ranking.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/training/scheduler_adapter.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/training/single_device_strategy.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/training/strategy.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/training/trainer.hpp"
    "${PROJECT_SOURCE_DIR}/include/jyppx/optim/optimizer_adapter.hpp"
    COMMENT "Validating and generating migrated API documentation"
)

# Doxygen 1.18 on Windows can crash while replacing an existing HTML search
# index. Documentation is a reproducible build artifact, so always start this
# target from its own empty output directory on every generator and platform.
add_custom_target(jyppx_migrated_api_docs_clean
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${DOXYGEN_OUTPUT_DIRECTORY}"
    COMMENT "Cleaning generated migrated API documentation"
    VERBATIM
)
add_dependencies(jyppx_migrated_api_docs jyppx_migrated_api_docs_clean)
