cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx VERSION 0.3.0 LANGUAGES CXX)

# ============================================
# Project Options
# ============================================
option(JYPPX_BUILD_TESTS "Build tests" ON)
option(JYPPX_BUILD_EXAMPLES "Build examples" ON)
option(JYPPX_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(JYPPX_BUILD_FUZZERS "Build libFuzzer targets" OFF)
option(JYPPX_BUILD_DOCS "Build documentation" OFF)
option(JYPPX_BUILD_CORE "Build the LibTorch value and tensor core" ON)
option(JYPPX_BUILD_INTEROP "Build interoperability formats and manifests" ON)
option(JYPPX_BUILD_DATA "Build data pipelines, datasets, and samplers" ON)
option(JYPPX_BUILD_NN "Build neural-network adapters and losses" ON)
option(JYPPX_BUILD_OPTIM "Build optimizer and scheduler adapters" ON)
option(JYPPX_BUILD_TRAINING "Build the training engine" ON)
option(JYPPX_BUILD_EXPERIMENT "Build local experiment configuration" ON)
set(JYPPX_TOOLS_DEFAULT OFF)
if(JYPPX_BUILD_CORE AND JYPPX_BUILD_INTEROP AND JYPPX_BUILD_DATA AND
   JYPPX_BUILD_NN AND JYPPX_BUILD_OPTIM AND JYPPX_BUILD_TRAINING AND
   JYPPX_BUILD_EXPERIMENT)
    set(JYPPX_TOOLS_DEFAULT ON)
endif()
option(JYPPX_BUILD_TOOLS "Build the user-facing jyppx command-line tool" ${JYPPX_TOOLS_DEFAULT})
option(JYPPX_INSTALL_TELEMETRY_SIDECAR "Install the optional Python telemetry runtime and dependency lock" OFF)
option(JYPPX_TEST_OFFICIAL_TELEMETRY "Run mandatory official SDK and real collector acceptance" OFF)
option(JYPPX_TEST_CLI_CUDA "Run product CLI workflows on a real CUDA device" OFF)
set(JYPPX_TELEMETRY_PYTHON "" CACHE FILEPATH "Locked CPython 3.11 interpreter for optional telemetry acceptance")
set(JYPPX_OTELCOL_EXECUTABLE "" CACHE FILEPATH "Pinned otelcol-contrib executable for acceptance")
set(JYPPX_SAFETENSORS_PYTHON "" CACHE FILEPATH "Locked CPython 3.11 interpreter for official safetensors interchange")
option(JYPPX_ENABLE_COVERAGE "Enable code coverage" OFF)
option(JYPPX_ENABLE_ASAN "Enable Address Sanitizer" OFF)
option(JYPPX_ENABLE_UBSAN "Enable Undefined Behavior Sanitizer" OFF)
option(JYPPX_ENABLE_TSAN "Enable Thread Sanitizer" OFF)
option(JYPPX_ENABLE_CLANG_TIDY "Run clang-tidy on all framework targets" OFF)
option(JYPPX_ENABLE_FORMAT_CHECK "Check all project C/C++ sources with clang-format" OFF)
option(JYPPX_ENABLE_WARNINGS_AS_ERRORS
       "Treat warnings as errors on all framework GNU/Clang targets" OFF)

if(JYPPX_BUILD_FUZZERS AND NOT JYPPX_BUILD_TESTS)
    message(FATAL_ERROR "JYPPX_BUILD_FUZZERS=ON requires JYPPX_BUILD_TESTS=ON")
endif()

foreach(JYPPX_CORE_DEPENDENT IN ITEMS DATA NN OPTIM EXPERIMENT)
    if(JYPPX_BUILD_${JYPPX_CORE_DEPENDENT} AND NOT JYPPX_BUILD_CORE)
        message(FATAL_ERROR
            "JYPPX_BUILD_${JYPPX_CORE_DEPENDENT}=ON requires JYPPX_BUILD_CORE=ON")
    endif()
endforeach()
if(JYPPX_BUILD_TRAINING AND
   (NOT JYPPX_BUILD_DATA OR NOT JYPPX_BUILD_NN OR NOT JYPPX_BUILD_OPTIM))
    message(FATAL_ERROR
        "JYPPX_BUILD_TRAINING=ON requires DATA, NN, and OPTIM components")
endif()

# ============================================
# C++ Standard
# ============================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================
# Build Type
# ============================================
get_property(JYPPX_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT JYPPX_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

# ============================================
# CMake Modules
# ============================================
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# ============================================
# Find LibTorch
# ============================================
set(JYPPX_BUNDLED_LIBTORCH "${PROJECT_SOURCE_DIR}/third_party/install/libtorch")
if(EXISTS "${JYPPX_BUNDLED_LIBTORCH}/share/cmake/Torch/TorchConfig.cmake"
   AND NOT Torch_DIR AND NOT CMAKE_PREFIX_PATH)
    list(PREPEND CMAKE_PREFIX_PATH "${JYPPX_BUNDLED_LIBTORCH}")
    message(STATUS "Using bundled LibTorch: ${JYPPX_BUNDLED_LIBTORCH}")
endif()
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(MSVC)
    # Source and header comments are UTF-8; keep MSVC from decoding them with
    # the active Windows code page (and emitting C4819 warnings).
    add_compile_options(/utf-8)
    # LibTorch's official Windows binaries are Release-built with /MD. Every
    # config of this project must use the same CRT, otherwise STL objects and
    # heap blocks crossing the torch DLL boundary corrupt the heap
    # (STATUS_HEAP_CORRUPTION / access violations in Debug builds).
    # CMP0091 moves runtime selection out of CMAKE_CXX_FLAGS_<CONFIG>, so set
    # the abstraction variable instead of string-replacing flags.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    # CMake still defines _DEBUG for Debug configs, which flips the STL back
    # to iterator-debug level 2 (different std layout than the DLLs use).
    # Pin level 0 so both sides agree.
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " /D_ITERATOR_DEBUG_LEVEL=0")
endif()

message(STATUS "LibTorch version: ${Torch_VERSION}")
message(STATUS "LibTorch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "LibTorch libraries: ${TORCH_LIBRARIES}")

# Windows test and example targets share a runtime directory. The helper
# serializes one copy per binary directory so parallel MSBuild cannot race on
# the same DLL destination.
include(cmake/CopyTorchRuntime.cmake)

# ============================================
# Bounded JSON Parser Implementation Dependency
# ============================================
if(JYPPX_BUILD_INTEROP OR JYPPX_BUILD_EXPERIMENT)
    include(FetchContent)
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()
    set(JYPPX_NLOHMANN_JSON_VERSION 3.12.0)
    set(JYPPX_NLOHMANN_JSON_ARCHIVE
        "${PROJECT_SOURCE_DIR}/third_party/downloads/nlohmann-json-${JYPPX_NLOHMANN_JSON_VERSION}.tar.xz")
    if(EXISTS "${JYPPX_NLOHMANN_JSON_ARCHIVE}")
        set(JYPPX_NLOHMANN_JSON_URL "${JYPPX_NLOHMANN_JSON_ARCHIVE}")
    else()
        set(JYPPX_NLOHMANN_JSON_URL
            "https://github.com/nlohmann/json/releases/download/v${JYPPX_NLOHMANN_JSON_VERSION}/json.tar.xz")
    endif()
    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    set(JSON_Install OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(jyppx_nlohmann_json
        URL "${JYPPX_NLOHMANN_JSON_URL}"
        URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
    )
    FetchContent_MakeAvailable(jyppx_nlohmann_json)
endif()

# SQLite stays an implementation dependency of experiment. Compile its pinned
# amalgamation separately so project warning/tidy gates do not scan upstream C,
# and embed the object in the installed experiment archive.
if(JYPPX_BUILD_EXPERIMENT)
    enable_language(C)
    find_package(Threads REQUIRED)
    set(JYPPX_SQLITE_ARCHIVE
        "${PROJECT_SOURCE_DIR}/third_party/downloads/sqlite-amalgamation-3540100.zip")
    if(EXISTS "${JYPPX_SQLITE_ARCHIVE}")
        set(JYPPX_SQLITE_URL "${JYPPX_SQLITE_ARCHIVE}")
    else()
        set(JYPPX_SQLITE_URL "https://sqlite.org/2026/sqlite-amalgamation-3540100.zip")
    endif()
    FetchContent_Declare(jyppx_sqlite
        URL "${JYPPX_SQLITE_URL}"
        URL_HASH SHA256=0a9c8e34352dad46006aa82836d5383bff9dc6bc9ce5a60553c05aa30e6ad1f9
    )
    FetchContent_MakeAvailable(jyppx_sqlite)
    add_library(sqlite3_amalgamation OBJECT "${jyppx_sqlite_SOURCE_DIR}/sqlite3.c")
    set_target_properties(sqlite3_amalgamation PROPERTIES
        C_STANDARD 99
        C_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_definitions(sqlite3_amalgamation PRIVATE
        SQLITE_THREADSAFE=1
        SQLITE_DQS=0
        SQLITE_OMIT_LOAD_EXTENSION
    )
endif()

# ============================================
# Generated Foundation Build Metadata
# ============================================
execute_process(
    COMMAND git rev-parse --verify HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE JYPPX_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT JYPPX_GIT_COMMIT)
    set(JYPPX_GIT_COMMIT "unknown")
endif()
execute_process(
    COMMAND git status --porcelain --untracked-files=normal
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE JYPPX_GIT_STATUS_RESULT
    OUTPUT_VARIABLE JYPPX_GIT_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT JYPPX_GIT_STATUS_RESULT EQUAL 0 OR JYPPX_GIT_STATUS)
    string(APPEND JYPPX_GIT_COMMIT "+dirty")
endif()

set(JYPPX_OS_ARCH "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_GENERATOR_PLATFORM)
    set(JYPPX_OS_ARCH "${CMAKE_SYSTEM_NAME}-${CMAKE_GENERATOR_PLATFORM}")
endif()
if(TARGET torch_cuda OR TORCH_CUDA_VERSION)
    set(JYPPX_ACCELERATOR_BUILD "cuda")
else()
    set(JYPPX_ACCELERATOR_BUILD "cpu")
endif()

if(MSVC)
    set(JYPPX_LIBTORCH_ABI "msvc-md-iterator0")
elseif(APPLE)
    set(JYPPX_LIBTORCH_ABI "libcxx")
elseif(TORCH_CXX_FLAGS MATCHES "_GLIBCXX_USE_CXX11_ABI=([01])")
    set(JYPPX_LIBTORCH_ABI "cxx11-abi${CMAKE_MATCH_1}")
else()
    set(JYPPX_LIBTORCH_ABI "unknown-abi")
endif()
set(JYPPX_LIBTORCH_PACKAGE_TAG "")
if(DEFINED TORCH_INSTALL_PREFIX
   AND EXISTS "${TORCH_INSTALL_PREFIX}/build-version")
    file(STRINGS "${TORCH_INSTALL_PREFIX}/build-version"
         JYPPX_LIBTORCH_PACKAGE_VERSION LIMIT_COUNT 1)
    string(STRIP "${JYPPX_LIBTORCH_PACKAGE_VERSION}"
           JYPPX_LIBTORCH_PACKAGE_VERSION)
    if(JYPPX_LIBTORCH_PACKAGE_VERSION MATCHES
       "\\+([A-Za-z0-9_.-]+)$")
        set(JYPPX_LIBTORCH_PACKAGE_TAG "${CMAKE_MATCH_1}")
    endif()
endif()
if(JYPPX_ACCELERATOR_BUILD STREQUAL "cuda")
    if(JYPPX_LIBTORCH_PACKAGE_TAG MATCHES "^cu[0-9]+$")
        set(JYPPX_LIBTORCH_CUDA_TAG "${JYPPX_LIBTORCH_PACKAGE_TAG}")
    elseif(TORCH_CUDA_VERSION)
        set(JYPPX_LIBTORCH_CUDA_TAG "${TORCH_CUDA_VERSION}")
    else()
        set(JYPPX_LIBTORCH_CUDA_TAG "unknown")
    endif()
    set(JYPPX_LIBTORCH_BUILD
        "cuda-${JYPPX_LIBTORCH_CUDA_TAG}-${JYPPX_LIBTORCH_ABI}")
else()
    set(JYPPX_LIBTORCH_BUILD "cpu-${JYPPX_LIBTORCH_ABI}")
endif()
message(STATUS "LibTorch build identity: ${JYPPX_LIBTORCH_BUILD}")

set(JYPPX_DEPENDENCY_MANIFEST
    "${CMAKE_CURRENT_SOURCE_DIR}/dependency-manifest.json")
if(EXISTS "${JYPPX_DEPENDENCY_MANIFEST}")
    file(SHA256 "${JYPPX_DEPENDENCY_MANIFEST}"
         JYPPX_DEPENDENCY_LOCK_DIGEST)
else()
    set(JYPPX_DEPENDENCY_LOCK_DIGEST "unlocked")
endif()
set(JYPPX_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(JYPPX_GENERATED_PRIVATE_INCLUDE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated-private")
set(JYPPX_COMPONENT_CORE_ENABLED ${JYPPX_BUILD_CORE})
set(JYPPX_COMPONENT_INTEROP_ENABLED ${JYPPX_BUILD_INTEROP})
set(JYPPX_COMPONENT_DATA_ENABLED ${JYPPX_BUILD_DATA})
set(JYPPX_COMPONENT_NN_ENABLED ${JYPPX_BUILD_NN})
set(JYPPX_COMPONENT_OPTIM_ENABLED ${JYPPX_BUILD_OPTIM})
set(JYPPX_COMPONENT_TRAINING_ENABLED ${JYPPX_BUILD_TRAINING})
set(JYPPX_COMPONENT_EXPERIMENT_ENABLED ${JYPPX_BUILD_EXPERIMENT})
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/jyppx_build_config.hpp.in"
    "${JYPPX_GENERATED_INCLUDE_DIR}/jyppx/foundation/build_config.hpp"
    @ONLY
)
# Volatile provenance belongs only to version.cpp. Keeping it outside the
# public include graph avoids recompiling every consumer after each commit.
# configure_file preserves the output timestamp when the content is unchanged.
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/jyppx_build_metadata.hpp.in"
    "${JYPPX_GENERATED_PRIVATE_INCLUDE_DIR}/jyppx/foundation/build_metadata.hpp"
    @ONLY
)
set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/jyppx/foundation/version.cpp"
    APPEND PROPERTY INCLUDE_DIRECTORIES "${JYPPX_GENERATED_PRIVATE_INCLUDE_DIR}")

# ============================================
# JYPPX Libraries
# ============================================
set(JYPPX_ENABLED_TARGETS)
set(JYPPX_ENABLED_COMPONENTS)
set(JYPPX_ENABLED_HEADER_DIRS)

add_library(jyppx_foundation STATIC
    src/jyppx/foundation/capability.cpp
    src/jyppx/foundation/checkpoint.cpp
    src/jyppx/foundation/error.cpp
    src/jyppx/foundation/random.cpp
    src/jyppx/foundation/random_checkpoint_participant.cpp
    src/jyppx/foundation/sha256.cpp
    src/jyppx/foundation/telemetry.cpp
    src/jyppx/foundation/version.cpp
)
add_library(jyppx::foundation ALIAS jyppx_foundation)
set_target_properties(jyppx_foundation PROPERTIES
    OUTPUT_NAME jyppx_foundation
    EXPORT_NAME foundation
)
target_include_directories(jyppx_foundation
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${JYPPX_GENERATED_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
)
foreach(JYPPX_TORCH_INCLUDE_DIR IN LISTS TORCH_INCLUDE_DIRS)
    target_include_directories(jyppx_foundation SYSTEM PUBLIC
        $<BUILD_INTERFACE:${JYPPX_TORCH_INCLUDE_DIR}>
    )
endforeach()
target_link_libraries(jyppx_foundation PUBLIC ${TORCH_LIBRARIES})
target_compile_definitions(jyppx_foundation
    PUBLIC
        JYPPX_CMAKE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        JYPPX_CMAKE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        JYPPX_CMAKE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    PRIVATE
        JYPPX_BUILD_TYPE="$<CONFIG>"
)
list(APPEND JYPPX_ENABLED_TARGETS jyppx_foundation)
list(APPEND JYPPX_ENABLED_COMPONENTS foundation)
list(APPEND JYPPX_ENABLED_HEADER_DIRS foundation)

if(JYPPX_BUILD_CORE)
    add_library(jyppx_core STATIC
        src/jyppx/core/device.cpp
        src/jyppx/core/tensor_utils.cpp
        src/jyppx/core/value_schema.cpp
    )
    add_library(jyppx::core ALIAS jyppx_core)
    set_target_properties(jyppx_core PROPERTIES
        OUTPUT_NAME jyppx_core
        EXPORT_NAME core
    )
    target_include_directories(jyppx_core
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    foreach(JYPPX_TORCH_INCLUDE_DIR IN LISTS TORCH_INCLUDE_DIRS)
        target_include_directories(jyppx_core SYSTEM PUBLIC
            $<BUILD_INTERFACE:${JYPPX_TORCH_INCLUDE_DIR}>
        )
    endforeach()
    target_link_libraries(jyppx_core PUBLIC ${TORCH_LIBRARIES} jyppx_foundation)
    if(MSVC)
        # Official Windows LibTorch binaries use the release DLL runtime and
        # STL iterator ABI in every configuration.
        target_compile_options(jyppx_core PUBLIC /MD)
        target_compile_definitions(jyppx_core PUBLIC _ITERATOR_DEBUG_LEVEL=0)
    endif()
    list(APPEND JYPPX_ENABLED_TARGETS jyppx_core)
    list(APPEND JYPPX_ENABLED_COMPONENTS core)
    list(APPEND JYPPX_ENABLED_HEADER_DIRS core utils)
endif()

if(JYPPX_BUILD_INTEROP)
    add_library(jyppx_interop STATIC
        src/jyppx/interop/checkpoint_manifest.cpp
        src/jyppx/interop/file_state_archive.cpp
        src/jyppx/interop/safetensors.cpp
        src/jyppx/interop/weight_import_report.cpp
        src/jyppx/interop/export_backend.cpp
        src/jyppx/interop/model_bundle.cpp
    )
    add_library(jyppx::interop ALIAS jyppx_interop)
    set_target_properties(jyppx_interop PROPERTIES
        OUTPUT_NAME jyppx_interop
        EXPORT_NAME interop
    )
    target_include_directories(jyppx_interop
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${jyppx_nlohmann_json_SOURCE_DIR}/single_include
    )
    target_link_libraries(jyppx_interop PUBLIC jyppx_foundation)
    list(APPEND JYPPX_ENABLED_TARGETS jyppx_interop)
    list(APPEND JYPPX_ENABLED_COMPONENTS interop)
    list(APPEND JYPPX_ENABLED_HEADER_DIRS interop)
endif()

if(JYPPX_BUILD_DATA)
add_library(jyppx_data STATIC
    src/jyppx/data/batch_memory.cpp
    src/jyppx/data/batch_schema.cpp
    src/jyppx/data/data_capability.cpp
    src/jyppx/data/data_pipeline.cpp
    src/jyppx/data/iterable_data_pipeline.cpp
    src/jyppx/data/tensor_dataset.cpp
    src/jyppx/data/samplers/batch_sampler.cpp
    src/jyppx/data/samplers/random_sampler.cpp
    src/jyppx/data/samplers/sequential_sampler.cpp
)
add_library(jyppx::data ALIAS jyppx_data)
set_target_properties(jyppx_data PROPERTIES
    OUTPUT_NAME jyppx_data
    EXPORT_NAME data
)
target_include_directories(jyppx_data
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(jyppx_data
    PUBLIC
        jyppx_foundation
        jyppx_core
)
list(APPEND JYPPX_ENABLED_TARGETS jyppx_data)
list(APPEND JYPPX_ENABLED_COMPONENTS data)
list(APPEND JYPPX_ENABLED_HEADER_DIRS data)
endif()

if(JYPPX_BUILD_NN)
add_library(jyppx_nn STATIC
    src/jyppx/nn/invocation.cpp
    src/jyppx/nn/model_adapter.cpp
    src/jyppx/nn/model_checkpoint_participant.cpp
    src/jyppx/nn/loss/contrastive_loss.cpp
    src/jyppx/nn/loss/dice_loss.cpp
    src/jyppx/nn/loss/focal_loss.cpp
    src/jyppx/nn/loss/label_smoothing_cross_entropy.cpp
)
add_library(jyppx::nn ALIAS jyppx_nn)
set_target_properties(jyppx_nn PROPERTIES
    OUTPUT_NAME jyppx_nn
    EXPORT_NAME nn
)
target_include_directories(jyppx_nn
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(jyppx_nn
    PUBLIC
        jyppx_foundation
        jyppx_core
)
list(APPEND JYPPX_ENABLED_TARGETS jyppx_nn)
list(APPEND JYPPX_ENABLED_COMPONENTS nn)
list(APPEND JYPPX_ENABLED_HEADER_DIRS nn)
endif()

if(JYPPX_BUILD_TRAINING)
add_library(jyppx_training STATIC
    src/jyppx/training/callback_dispatcher.cpp
    src/jyppx/training/callbacks/early_stopping_callback.cpp
    src/jyppx/training/checkpoint_coordinator.cpp
    src/jyppx/training/event_sink.cpp
    src/jyppx/training/telemetry_adapter.cpp
    src/jyppx/training/amp/grad_scaler.cpp
    src/jyppx/training/amp/grad_scaler_checkpoint_participant.cpp
    src/jyppx/training/metrics/classification.cpp
    src/jyppx/training/metrics/accuracy.cpp
    src/jyppx/training/metrics/average_meter.cpp
    src/jyppx/training/metrics/ranking.cpp
    src/jyppx/training/metrics/regression.cpp
    src/jyppx/training/metrics/topk_accuracy.cpp
    src/jyppx/training/scheduler_adapter.cpp
    src/jyppx/training/single_device_strategy.cpp
    src/jyppx/training/trainer.cpp
)
add_library(jyppx::training ALIAS jyppx_training)
set_target_properties(jyppx_training PROPERTIES
    OUTPUT_NAME jyppx_training
    EXPORT_NAME training
)
target_include_directories(jyppx_training
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(jyppx_training PUBLIC jyppx_foundation)
if(JYPPX_BUILD_INTEROP)
    target_sources(jyppx_training PRIVATE
        src/jyppx/training/file_checkpoint_store.cpp
    )
    target_link_libraries(jyppx_training PRIVATE jyppx_interop)
endif()
list(APPEND JYPPX_ENABLED_TARGETS jyppx_training)
list(APPEND JYPPX_ENABLED_COMPONENTS training)
list(APPEND JYPPX_ENABLED_HEADER_DIRS training)
endif()

if(JYPPX_BUILD_OPTIM)
add_library(jyppx_optim STATIC
    src/jyppx/optim/adadelta_adapter.cpp
    src/jyppx/optim/adagrad_adapter.cpp
    src/jyppx/optim/adam_adapter.cpp
    src/jyppx/optim/adamw_adapter.cpp
    src/jyppx/optim/lamb_adapter.cpp
    src/jyppx/optim/nadam_adapter.cpp
    src/jyppx/optim/rmsprop_adapter.cpp
    src/jyppx/optim/composite_lr_scheduler.cpp
    src/jyppx/optim/epoch_lr_scheduler.cpp
    src/jyppx/optim/event_lr_scheduler.cpp
    src/jyppx/optim/linear_lr_scheduler.cpp
    src/jyppx/optim/sgd_adapter.cpp
    src/jyppx/optim/step_lr_scheduler.cpp
)
add_library(jyppx::optim ALIAS jyppx_optim)
set_target_properties(jyppx_optim PROPERTIES
    OUTPUT_NAME jyppx_optim
    EXPORT_NAME optim
)
target_include_directories(jyppx_optim
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(jyppx_optim
    PUBLIC
        jyppx_foundation
        jyppx_core
)
list(APPEND JYPPX_ENABLED_TARGETS jyppx_optim)
list(APPEND JYPPX_ENABLED_COMPONENTS optim)
list(APPEND JYPPX_ENABLED_HEADER_DIRS optim)
endif()

if(JYPPX_BUILD_EXPERIMENT)
    add_library(jyppx_experiment STATIC
        src/jyppx/experiment/artifact_store.cpp
        src/jyppx/experiment/detail/artifact_publish.cpp
        src/jyppx/experiment/detail/native_artifact_publish.cpp
        src/jyppx/experiment/config.cpp
        src/jyppx/experiment/run.cpp
        src/jyppx/experiment/run_store.cpp
        src/jyppx/experiment/run_telemetry_sink.cpp
        src/jyppx/experiment/telemetry_exporters.cpp
        src/jyppx/experiment/evaluation.cpp
        $<TARGET_OBJECTS:sqlite3_amalgamation>
    )
    # Keep the native publication forwarding seam in its own archive member
    # for real process-boundary integration supervision.
    set_source_files_properties(
        src/jyppx/experiment/detail/artifact_publish.cpp
        src/jyppx/experiment/detail/native_artifact_publish.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
    add_library(jyppx::experiment ALIAS jyppx_experiment)
    set_target_properties(jyppx_experiment PROPERTIES
        OUTPUT_NAME jyppx_experiment
        EXPORT_NAME experiment
    )
    target_include_directories(jyppx_experiment
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${jyppx_nlohmann_json_SOURCE_DIR}/single_include
    )
    target_include_directories(jyppx_experiment SYSTEM PRIVATE
        "${jyppx_sqlite_SOURCE_DIR}"
    )
    target_link_libraries(jyppx_experiment
        PUBLIC
            jyppx_foundation
            jyppx_core
        PRIVATE
            Threads::Threads
    )
    list(APPEND JYPPX_ENABLED_TARGETS jyppx_experiment)
    list(APPEND JYPPX_ENABLED_COMPONENTS experiment)
    list(APPEND JYPPX_ENABLED_HEADER_DIRS experiment)
endif()

if(JYPPX_BUILD_TRAINING)
    target_link_libraries(jyppx_training PUBLIC
        jyppx_core
        jyppx_data
        jyppx_optim
        jyppx_nn
    )
endif()

if(JYPPX_BUILD_INTEROP AND JYPPX_BUILD_NN)
    target_sources(jyppx_interop PRIVATE
        src/jyppx/interop/model_weight_import.cpp
        src/jyppx/interop/torchscript_compatibility.cpp)
    target_link_libraries(jyppx_interop PUBLIC jyppx_nn)
endif()

add_library(jyppx_all INTERFACE)
add_library(jyppx::all ALIAS jyppx_all)
set_target_properties(jyppx_all PROPERTIES EXPORT_NAME all)
target_include_directories(jyppx_all
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
foreach(JYPPX_TORCH_INCLUDE_DIR IN LISTS TORCH_INCLUDE_DIRS)
    target_include_directories(jyppx_all SYSTEM INTERFACE
        $<BUILD_INTERFACE:${JYPPX_TORCH_INCLUDE_DIR}>
    )
endforeach()
target_link_libraries(jyppx_all INTERFACE ${JYPPX_ENABLED_TARGETS})
list(APPEND JYPPX_ENABLED_TARGETS jyppx_all)
list(APPEND JYPPX_ENABLED_COMPONENTS all)

if(JYPPX_BUILD_CORE AND JYPPX_BUILD_INTEROP AND JYPPX_BUILD_DATA AND
   JYPPX_BUILD_NN AND JYPPX_BUILD_OPTIM AND JYPPX_BUILD_TRAINING AND
   JYPPX_BUILD_EXPERIMENT)
    add_library(jyppx_sdk INTERFACE)
    add_library(jyppx::sdk ALIAS jyppx_sdk)
    set_target_properties(jyppx_sdk PROPERTIES EXPORT_NAME sdk)
    target_include_directories(jyppx_sdk
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    foreach(JYPPX_TORCH_INCLUDE_DIR IN LISTS TORCH_INCLUDE_DIRS)
        target_include_directories(jyppx_sdk SYSTEM INTERFACE
            $<BUILD_INTERFACE:${JYPPX_TORCH_INCLUDE_DIR}>
        )
    endforeach()
    target_link_libraries(jyppx_sdk INTERFACE jyppx_all)
    list(APPEND JYPPX_ENABLED_TARGETS jyppx_sdk)
    list(APPEND JYPPX_ENABLED_COMPONENTS sdk)
endif()

# ============================================
# Component Dependency Graph Gate
# ============================================
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(JYPPX_COMPONENT_DEPENDENCY_MANIFEST
    "${CMAKE_CURRENT_SOURCE_DIR}/component-dependencies.json")
set(JYPPX_COMPONENT_TARGET_GRAPH
    "${CMAKE_CURRENT_BINARY_DIR}/component-target-graph.tsv")
set(JYPPX_COMPONENT_GRAPH_REPORT
    "${CMAKE_CURRENT_BINARY_DIR}/component-dependency-report.json")
include(cmake/WriteComponentGraph.cmake)
jyppx_write_component_graph("${JYPPX_COMPONENT_TARGET_GRAPH}")

file(GLOB_RECURSE JYPPX_COMPONENT_GRAPH_SCAN_INPUTS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/jyppx/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/jyppx/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jyppx/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jyppx/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jyppx/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jyppx/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jyppx/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jyppx/*.inc"
)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${JYPPX_COMPONENT_DEPENDENCY_MANIFEST}"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/WriteComponentGraph.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/validation/component_graph.py"
    ${JYPPX_COMPONENT_GRAPH_SCAN_INPUTS}
)
execute_process(
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/validation/component_graph.py"
            --manifest "${JYPPX_COMPONENT_DEPENDENCY_MANIFEST}"
            --source-root "${CMAKE_CURRENT_SOURCE_DIR}"
            --target-graph "${JYPPX_COMPONENT_TARGET_GRAPH}"
            --output "${JYPPX_COMPONENT_GRAPH_REPORT}"
    RESULT_VARIABLE JYPPX_COMPONENT_GRAPH_RESULT
    OUTPUT_VARIABLE JYPPX_COMPONENT_GRAPH_OUTPUT
    ERROR_VARIABLE JYPPX_COMPONENT_GRAPH_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)
if(NOT JYPPX_COMPONENT_GRAPH_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Component dependency graph gate failed:\n"
        "${JYPPX_COMPONENT_GRAPH_OUTPUT}\n${JYPPX_COMPONENT_GRAPH_ERROR}")
endif()
message(STATUS "${JYPPX_COMPONENT_GRAPH_OUTPUT}")

# Compiler options must be applied after the target has been created.
if(JYPPX_BUILD_TOOLS)
    if(NOT TARGET jyppx::sdk)
        message(FATAL_ERROR "JYPPX_BUILD_TOOLS=ON requires the complete jyppx::sdk component closure")
    endif()
    add_subdirectory(tools/cli)
elseif(JYPPX_INSTALL_TELEMETRY_SIDECAR)
    message(FATAL_ERROR "JYPPX_INSTALL_TELEMETRY_SIDECAR=ON requires JYPPX_BUILD_TOOLS=ON")
endif()
include(cmake/CompilerOptions.cmake)

# ============================================
# Tests
# ============================================
if(JYPPX_BUILD_TESTS)
    enable_testing()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        add_subdirectory(tests)
    else()
        message(STATUS "Skipping tests: tests/CMakeLists.txt is not present yet")
    endif()
endif()

# ============================================
# Examples
# ============================================
if(JYPPX_BUILD_EXAMPLES)
    if(NOT TARGET jyppx::sdk)
        message(STATUS "Skipping examples: the complete jyppx::sdk is disabled")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
        add_subdirectory(examples)
    else()
        message(STATUS "Skipping examples: examples/CMakeLists.txt is not present yet")
    endif()
endif()

# ============================================
# Documentation
# ============================================
if(JYPPX_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    add_subdirectory(docs)
endif()

# ============================================
# Installation
# ============================================
install(TARGETS ${JYPPX_ENABLED_TARGETS}
    EXPORT jyppxTargetsExport
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

foreach(JYPPX_HEADER_DIR IN LISTS JYPPX_ENABLED_HEADER_DIRS)
    if(JYPPX_HEADER_DIR STREQUAL "interop" AND NOT JYPPX_BUILD_NN)
        install(DIRECTORY "include/jyppx/interop"
            DESTINATION include/jyppx
            PATTERN "model_weight_import.hpp" EXCLUDE
            PATTERN "torchscript_compatibility.hpp" EXCLUDE
        )
        continue()
    endif()
    if(JYPPX_HEADER_DIR STREQUAL "training" AND NOT JYPPX_BUILD_INTEROP)
        install(DIRECTORY "include/jyppx/training"
            DESTINATION include/jyppx
            PATTERN "file_checkpoint_store.hpp" EXCLUDE
        )
        continue()
    endif()
    install(DIRECTORY "include/jyppx/${JYPPX_HEADER_DIR}"
        DESTINATION include/jyppx
    )
endforeach()

if(TARGET jyppx::sdk)
    install(FILES include/jyppx/jyppx.hpp DESTINATION include/jyppx)
endif()

install(FILES
    "${JYPPX_GENERATED_INCLUDE_DIR}/jyppx/foundation/build_config.hpp"
    DESTINATION include/jyppx/foundation
)

install(EXPORT jyppxTargetsExport
    FILE jyppxTargets.cmake
    NAMESPACE jyppx::
    DESTINATION lib/cmake/jyppx
)

# ============================================
# Package Configuration
# ============================================
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/jyppxConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/jyppxConfig.cmake
    INSTALL_DESTINATION lib/cmake/jyppx
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/jyppxConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/jyppxConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/jyppxConfigVersion.cmake
    DESTINATION lib/cmake/jyppx
)

install(FILES
    LICENSE
    NOTICE
    THIRD_PARTY_NOTICES.md
    DESTINATION share/doc/jyppx
    COMPONENT licenses
)

# ============================================
# Summary
# ============================================
message(STATUS "")
message(STATUS "JYPPX Training Framework Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build tests: ${JYPPX_BUILD_TESTS}")
message(STATUS "  Build examples: ${JYPPX_BUILD_EXAMPLES}")
message(STATUS "  Build benchmarks: ${JYPPX_BUILD_BENCHMARKS}")
message(STATUS "  Build fuzzers: ${JYPPX_BUILD_FUZZERS}")
message(STATUS "  Build docs: ${JYPPX_BUILD_DOCS}")
message(STATUS "  Components: ${JYPPX_ENABLED_COMPONENTS}")
message(STATUS "  Clang-tidy: ${JYPPX_ENABLE_CLANG_TIDY}")
message(STATUS "  Format check: ${JYPPX_ENABLE_FORMAT_CHECK}")
message(STATUS "  Warnings as errors: ${JYPPX_ENABLE_WARNINGS_AS_ERRORS}")
message(STATUS "")
