if(NOT DEFINED JYPPX_SOURCE_DIR)
    message(FATAL_ERROR "JYPPX_SOURCE_DIR is required")
endif()

file(REAL_PATH "${JYPPX_SOURCE_DIR}" source_root)
set(violations)

# Exact paths retired by completed migration slices. Reintroducing any one of
# them would create a forwarding surface or restore an obsolete owner.
set(retired_source_paths
    cmake/jyppx_liuding_training_frameworkConfig.cmake.in
    docs/api/distributed.md
    include/jyppx/liuding/data/dataloader.hpp
    include/jyppx/liuding/data/thread_safe_queue.hpp
    include/jyppx/liuding/distributed/ddp.hpp
    include/jyppx/liuding/distributed/distributed.hpp
    include/jyppx/liuding/nn/init.hpp
    include/jyppx/liuding/nn/losses.hpp
    include/jyppx/liuding/nn/quantization.hpp
    include/jyppx/liuding/optim/adagrad.hpp
    include/jyppx/liuding/optim/adam.hpp
    include/jyppx/liuding/optim/adamw.hpp
    include/jyppx/liuding/optim/lamb.hpp
    include/jyppx/liuding/optim/nadam.hpp
    include/jyppx/liuding/optim/optimizer.hpp
    include/jyppx/liuding/optim/rmsprop.hpp
    include/jyppx/liuding/optim/schedulers/chained_scheduler.hpp
    include/jyppx/liuding/optim/schedulers/composite_schedulers.hpp
    include/jyppx/liuding/optim/schedulers/constant_lr.hpp
    include/jyppx/liuding/optim/schedulers/cosine_annealing_lr.hpp
    include/jyppx/liuding/optim/schedulers/cosine_annealing_warm_restarts.hpp
    include/jyppx/liuding/optim/schedulers/cyclic_lr.hpp
    include/jyppx/liuding/optim/schedulers/exponential_lr.hpp
    include/jyppx/liuding/optim/schedulers/lambda_lr.hpp
    include/jyppx/liuding/optim/schedulers/linear_lr.hpp
    include/jyppx/liuding/optim/schedulers/lr_scheduler.hpp
    include/jyppx/liuding/optim/schedulers/multi_step_lr.hpp
    include/jyppx/liuding/optim/schedulers/one_cycle_lr.hpp
    include/jyppx/liuding/optim/schedulers/p1_schedulers.hpp
    include/jyppx/liuding/optim/schedulers/polynomial_lr.hpp
    include/jyppx/liuding/optim/schedulers/reduce_lr_on_plateau.hpp
    include/jyppx/liuding/optim/schedulers/sequential_lr.hpp
    include/jyppx/liuding/optim/schedulers/step_lr.hpp
    include/jyppx/liuding/optim/schedulers/warmup_scheduler.hpp
    include/jyppx/liuding/optim/sgd.hpp
    include/jyppx/liuding/training/callback.hpp
    include/jyppx/liuding/training/callbacks/early_stopping.hpp
    include/jyppx/liuding/training/callbacks/logging_callback.hpp
    include/jyppx/liuding/training/callbacks/lr_scheduler_callback.hpp
    include/jyppx/liuding/training/callbacks/model_checkpoint.hpp
    include/jyppx/liuding/training/callbacks/progress_callback.hpp
    include/jyppx/liuding/training/callbacks/tensorboard_callback.hpp
    include/jyppx/liuding/utils/logger.hpp
    include/jyppx/liuding/utils/tensorboard.hpp
    include/jyppx/liuding/version.hpp
    src/CMakeLists.txt
    src/jyppx/liuding/distributed/distributed.cpp
    src/jyppx/liuding/nn/init.cpp
    src/jyppx/liuding/nn/losses.cpp
    src/jyppx/liuding/optim/adagrad.cpp
    src/jyppx/liuding/optim/adam.cpp
    src/jyppx/liuding/optim/adamw.cpp
    src/jyppx/liuding/optim/lamb.cpp
    src/jyppx/liuding/optim/nadam.cpp
    src/jyppx/liuding/optim/optimizer.cpp
    src/jyppx/liuding/optim/rmsprop.cpp
    src/jyppx/liuding/optim/schedulers/composite_schedulers.cpp
    src/jyppx/liuding/optim/schedulers/cosine_annealing_lr.cpp
    src/jyppx/liuding/optim/schedulers/cosine_annealing_warm_restarts.cpp
    src/jyppx/liuding/optim/schedulers/exponential_lr.cpp
    src/jyppx/liuding/optim/schedulers/linear_lr.cpp
    src/jyppx/liuding/optim/schedulers/lr_scheduler.cpp
    src/jyppx/liuding/optim/schedulers/multi_step_lr.cpp
    src/jyppx/liuding/optim/schedulers/one_cycle_lr.cpp
    src/jyppx/liuding/optim/schedulers/p1_schedulers.cpp
    src/jyppx/liuding/optim/schedulers/reduce_lr_on_plateau.cpp
    src/jyppx/liuding/optim/schedulers/step_lr.cpp
    src/jyppx/liuding/optim/schedulers/warmup_scheduler.cpp
    src/jyppx/liuding/optim/sgd.cpp
    src/jyppx/liuding/training/callbacks/early_stopping.cpp
    src/jyppx/liuding/training/callbacks/logging_callback.cpp
    src/jyppx/liuding/training/callbacks/lr_scheduler_callback.cpp
    src/jyppx/liuding/training/callbacks/model_checkpoint.cpp
    src/jyppx/liuding/training/callbacks/progress_callback.cpp
    src/jyppx/liuding/training/callbacks/tensorboard_callback.cpp
    src/jyppx/liuding/utils/logger.cpp
    src/jyppx/liuding/utils/tensorboard.cpp
)

foreach(retired_path IN LISTS retired_source_paths)
    if(EXISTS "${source_root}/${retired_path}")
        list(APPEND violations "retired source surface exists: ${retired_path}")
    endif()
endforeach()

file(GLOB_RECURSE active_build_files LIST_DIRECTORIES false
    "${source_root}/CMakeLists.txt"
    "${source_root}/cmake/*.cmake"
    "${source_root}/cmake/*.cmake.in"
    "${source_root}/examples/CMakeLists.txt"
    "${source_root}/tests/CMakeLists.txt"
    "${source_root}/tests/fuzz/CMakeLists.txt"
    "${source_root}/tools/*.ps1"
    "${source_root}/scripts/*.ps1"
    "${source_root}/scripts/*.sh"
)
foreach(active_file IN LISTS active_build_files)
    if(active_file STREQUAL "${source_root}/cmake/CheckLegacySurface.cmake" OR
       active_file MATCHES "/tests/install/")
        continue()
    endif()
    file(READ "${active_file}" active_text)
    if(active_text MATCHES "jyppx_liuding_training_framework")
        file(RELATIVE_PATH relative_file "${source_root}" "${active_file}")
        list(APPEND violations
            "active build surface uses retired target/package name: ${relative_file}")
    endif()
endforeach()

if(DEFINED JYPPX_LIUDING_INSTALL_DIR)
    get_filename_component(install_root "${JYPPX_LIUDING_INSTALL_DIR}" ABSOLUTE)
    foreach(retired_path IN LISTS retired_source_paths)
        if(retired_path MATCHES "^include/(.+)$")
            set(retired_header "${CMAKE_MATCH_1}")
            if(EXISTS "${install_root}/include/${retired_header}")
                list(APPEND violations
                    "retired public header is installed: include/${retired_header}")
            endif()
        endif()
    endforeach()

    if(EXISTS "${install_root}/lib/cmake/jyppx_liuding_training_framework")
        list(APPEND violations
            "retired package directory is installed: lib/cmake/jyppx_liuding_training_framework")
    endif()
    file(GLOB_RECURSE retired_install_entries LIST_DIRECTORIES false
        "${install_root}/*jyppx_liuding_training_framework*")
    foreach(retired_entry IN LISTS retired_install_entries)
        file(RELATIVE_PATH relative_entry "${install_root}" "${retired_entry}")
        list(APPEND violations
            "installed artifact uses retired target/package name: ${relative_entry}")
    endforeach()
endif()

if(violations)
    list(JOIN violations "\n  - " violation_text)
    message(FATAL_ERROR "Legacy surface violations:\n  - ${violation_text}")
endif()

list(LENGTH retired_source_paths retired_path_count)
message(STATUS
    "Legacy surface check passed: ${retired_path_count} retired paths absent")
