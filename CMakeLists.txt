cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx VERSION 0.3.0 LANGUAGES CXX)

# Public brand metadata is deliberately separate from the stable jyppx::liuding
# namespace, jyppx package configuration, target names, and protocol identifiers.
set(LIUDING_FORGE_DISPLAY_NAME_ZH "六丁炼算")
set(LIUDING_FORGE_DISPLAY_NAME_EN "Liuding Forge")
set(LIUDING_FORGE_ARTIFACT_PREFIX "liuding-forge")
set(LIUDING_FORGE_PROJECT_ALIAS "JYPPX Training Framework")

# Several MSBuild targets can independently request regeneration after a source
# scanned by the component gate changes. Serialize the configure processes so
# their generated graph, reports and generation stamps cannot interleave.
file(LOCK "${CMAKE_BINARY_DIR}/jyppx-configure.lock" GUARD PROCESS TIMEOUT 120)

# ============================================
# Project Options
# ============================================
option(JYPPX_LIUDING_BUILD_TESTS "Build tests" ON)
option(JYPPX_LIUDING_BUILD_EXAMPLES "Build examples" ON)
option(JYPPX_LIUDING_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(JYPPX_LIUDING_BUILD_FUZZERS "Build libFuzzer targets" OFF)
option(JYPPX_LIUDING_BUILD_DOCS "Build documentation" OFF)
option(JYPPX_LIUDING_BUILD_CORE "Build the LibTorch value and tensor core" ON)
option(JYPPX_LIUDING_BUILD_INTEROP "Build interoperability formats and manifests" ON)
option(JYPPX_LIUDING_BUILD_DATA "Build data pipelines, datasets, and samplers" ON)
option(JYPPX_LIUDING_BUILD_DATA_IMAGE "Build the optional built-in image/annotation adapters" OFF)
option(JYPPX_LIUDING_BUILD_NN "Build neural-network adapters and losses" ON)
option(JYPPX_LIUDING_BUILD_OPTIM "Build optimizer and scheduler adapters" ON)
option(JYPPX_LIUDING_BUILD_TRAINING "Build the training engine" ON)
option(JYPPX_LIUDING_BUILD_EXPERIMENT "Build local experiment configuration" ON)
option(JYPPX_LIUDING_BUILD_MODELS "Build the independent models Preview" OFF)
option(JYPPX_LIUDING_BUILD_QUANTIZATION "Build the independent storage quantization Preview" OFF)
option(JYPPX_LIUDING_BUILD_SERVING "Build the independent in-process serving Preview" OFF)
set(JYPPX_LIUDING_MODELS_REFERENCE_DIR "" CACHE PATH
    "Pre-generated official models reference fixtures; enables the optional differential test")
set(JYPPX_TOOLS_DEFAULT OFF)
if(JYPPX_LIUDING_BUILD_CORE AND JYPPX_LIUDING_BUILD_INTEROP AND JYPPX_LIUDING_BUILD_DATA AND
   JYPPX_LIUDING_BUILD_NN AND JYPPX_LIUDING_BUILD_OPTIM AND JYPPX_LIUDING_BUILD_TRAINING AND
   JYPPX_LIUDING_BUILD_EXPERIMENT)
    set(JYPPX_TOOLS_DEFAULT ON)
endif()
option(JYPPX_LIUDING_BUILD_TOOLS "Build the user-facing Liuding Forge command-line tool" ${JYPPX_TOOLS_DEFAULT})
option(JYPPX_LIUDING_INSTALL_TELEMETRY_SIDECAR "Install the optional Python telemetry runtime and dependency lock" OFF)
option(JYPPX_LIUDING_TEST_OFFICIAL_TELEMETRY "Run mandatory official SDK and real collector acceptance" OFF)
option(JYPPX_LIUDING_TEST_CLI_CUDA "Run product CLI workflows on a real CUDA device" OFF)
set(JYPPX_TELEMETRY_PYTHON "" CACHE FILEPATH "Locked CPython 3.11 interpreter for optional telemetry acceptance")
set(JYPPX_OTELCOL_EXECUTABLE "" CACHE FILEPATH "Pinned otelcol-contrib executable for acceptance")
set(JYPPX_SAFETENSORS_PYTHON "" CACHE FILEPATH "Locked CPython 3.11 interpreter for official safetensors interchange")
option(JYPPX_LIUDING_ENABLE_COVERAGE "Enable code coverage" OFF)
option(JYPPX_LIUDING_ENABLE_ASAN "Enable Address Sanitizer" OFF)
option(JYPPX_LIUDING_ENABLE_UBSAN "Enable Undefined Behavior Sanitizer" OFF)
option(JYPPX_LIUDING_ENABLE_TSAN "Enable Thread Sanitizer" OFF)
option(JYPPX_LIUDING_ENABLE_CLANG_TIDY "Run clang-tidy on all framework targets" OFF)
option(JYPPX_LIUDING_ENABLE_FORMAT_CHECK "Check all project C/C++ sources with clang-format" OFF)
option(JYPPX_LIUDING_ENABLE_WARNINGS_AS_ERRORS
       "Treat warnings as errors on all framework GNU/Clang targets" OFF)

if(JYPPX_LIUDING_BUILD_FUZZERS AND NOT JYPPX_LIUDING_BUILD_TESTS)
    message(FATAL_ERROR "JYPPX_LIUDING_BUILD_FUZZERS=ON requires JYPPX_LIUDING_BUILD_TESTS=ON")
endif()

foreach(JYPPX_CORE_DEPENDENT IN ITEMS DATA NN OPTIM EXPERIMENT)
    if(JYPPX_LIUDING_BUILD_${JYPPX_CORE_DEPENDENT} AND NOT JYPPX_LIUDING_BUILD_CORE)
        message(FATAL_ERROR
            "JYPPX_LIUDING_BUILD_${JYPPX_CORE_DEPENDENT}=ON requires JYPPX_LIUDING_BUILD_CORE=ON")
    endif()
endforeach()
if(JYPPX_LIUDING_BUILD_TRAINING AND
   (NOT JYPPX_LIUDING_BUILD_DATA OR NOT JYPPX_LIUDING_BUILD_NN OR NOT JYPPX_LIUDING_BUILD_OPTIM))
    message(FATAL_ERROR
        "JYPPX_LIUDING_BUILD_TRAINING=ON requires DATA, NN, and OPTIM components")
endif()
if(JYPPX_LIUDING_BUILD_DATA_IMAGE AND NOT JYPPX_LIUDING_BUILD_DATA)
    message(FATAL_ERROR "JYPPX_LIUDING_BUILD_DATA_IMAGE=ON requires JYPPX_LIUDING_BUILD_DATA=ON")
endif()

# Preview model/serving adapters require the explicit lower component closure.
# Quantization is tensor-neutral and deliberately needs foundation only.
foreach(JYPPX_PREVIEW_COMPONENT IN ITEMS MODELS SERVING)
    foreach(JYPPX_PREVIEW_DEPENDENCY IN ITEMS CORE INTEROP NN)
        if(JYPPX_LIUDING_BUILD_${JYPPX_PREVIEW_COMPONENT} AND
           NOT JYPPX_LIUDING_BUILD_${JYPPX_PREVIEW_DEPENDENCY})
            message(FATAL_ERROR
                "JYPPX_LIUDING_BUILD_${JYPPX_PREVIEW_COMPONENT}=ON requires JYPPX_LIUDING_BUILD_${JYPPX_PREVIEW_DEPENDENCY}=ON")
        endif()
    endforeach()
endforeach()
if(JYPPX_LIUDING_MODELS_REFERENCE_DIR AND
   (NOT JYPPX_LIUDING_BUILD_MODELS OR NOT JYPPX_LIUDING_BUILD_TESTS))
    message(FATAL_ERROR
        "JYPPX_LIUDING_MODELS_REFERENCE_DIR requires MODELS and TESTS components")
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
if(JYPPX_LIUDING_BUILD_INTEROP OR JYPPX_LIUDING_BUILD_EXPERIMENT OR JYPPX_LIUDING_BUILD_DATA_IMAGE)
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

if(JYPPX_LIUDING_BUILD_DATA_IMAGE)
    # New adapter dependencies must be staged explicitly in the verified local
    # cache. Configuring an optional reader never silently downloads a parser.
    include(cmake/DataImageDependencies.cmake)
endif()

if(JYPPX_LIUDING_BUILD_SERVING)
    find_package(Threads REQUIRED)
endif()

# SQLite stays an implementation dependency of experiment. Compile its pinned
# amalgamation separately so project warning/tidy gates do not scan upstream C,
# and embed the object in the installed experiment archive.
if(JYPPX_LIUDING_BUILD_EXPERIMENT)
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

include(cmake/DetectLibTorchAbi.cmake)
jyppx_detect_libtorch_abi(JYPPX_LIBTORCH_ABI)
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
set(JYPPX_LIUDING_COMPONENT_CORE_ENABLED ${JYPPX_LIUDING_BUILD_CORE})
set(JYPPX_LIUDING_COMPONENT_INTEROP_ENABLED ${JYPPX_LIUDING_BUILD_INTEROP})
set(JYPPX_LIUDING_COMPONENT_DATA_ENABLED ${JYPPX_LIUDING_BUILD_DATA})
set(JYPPX_LIUDING_COMPONENT_DATA_IMAGE_ENABLED ${JYPPX_LIUDING_BUILD_DATA_IMAGE})
set(JYPPX_LIUDING_COMPONENT_NN_ENABLED ${JYPPX_LIUDING_BUILD_NN})
set(JYPPX_LIUDING_COMPONENT_OPTIM_ENABLED ${JYPPX_LIUDING_BUILD_OPTIM})
set(JYPPX_LIUDING_COMPONENT_TRAINING_ENABLED ${JYPPX_LIUDING_BUILD_TRAINING})
set(JYPPX_LIUDING_COMPONENT_EXPERIMENT_ENABLED ${JYPPX_LIUDING_BUILD_EXPERIMENT})
set(JYPPX_LIUDING_COMPONENT_MODELS_ENABLED ${JYPPX_LIUDING_BUILD_MODELS})
set(JYPPX_LIUDING_COMPONENT_QUANTIZATION_ENABLED ${JYPPX_LIUDING_BUILD_QUANTIZATION})
set(JYPPX_LIUDING_COMPONENT_SERVING_ENABLED ${JYPPX_LIUDING_BUILD_SERVING})
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/jyppx_build_config.hpp.in"
    "${JYPPX_GENERATED_INCLUDE_DIR}/jyppx/liuding/foundation/build_config.hpp"
    @ONLY
)
# Volatile provenance belongs only to version.cpp. Keeping it outside the
# public include graph avoids recompiling every consumer after each commit.
# configure_file preserves the output timestamp when the content is unchanged.
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/jyppx_build_metadata.hpp.in"
    "${JYPPX_GENERATED_PRIVATE_INCLUDE_DIR}/jyppx/liuding/foundation/build_metadata.hpp"
    @ONLY
)
set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/jyppx/liuding/foundation/version.cpp"
    APPEND PROPERTY INCLUDE_DIRECTORIES "${JYPPX_GENERATED_PRIVATE_INCLUDE_DIR}")

# ============================================
# JYPPX Libraries
# ============================================
set(JYPPX_ENABLED_TARGETS)
set(JYPPX_ENABLED_COMPONENTS)
set(JYPPX_ENABLED_HEADER_DIRS)

add_library(jyppx_liuding_foundation STATIC
    src/jyppx/liuding/foundation/capability.cpp
    src/jyppx/liuding/foundation/checkpoint.cpp
    src/jyppx/liuding/foundation/error.cpp
    src/jyppx/liuding/foundation/random.cpp
    src/jyppx/liuding/foundation/random_checkpoint_participant.cpp
    src/jyppx/liuding/foundation/sha256.cpp
    src/jyppx/liuding/foundation/telemetry.cpp
    src/jyppx/liuding/foundation/version.cpp
)
add_library(jyppx::liuding::foundation ALIAS jyppx_liuding_foundation)
set_target_properties(jyppx_liuding_foundation PROPERTIES
    OUTPUT_NAME liuding-forge-foundation
    EXPORT_NAME foundation
)
target_include_directories(jyppx_liuding_foundation
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${JYPPX_GENERATED_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
)
foreach(JYPPX_TORCH_INCLUDE_DIR IN LISTS TORCH_INCLUDE_DIRS)
    target_include_directories(jyppx_liuding_foundation SYSTEM PUBLIC
        $<BUILD_INTERFACE:${JYPPX_TORCH_INCLUDE_DIR}>
    )
endforeach()
target_link_libraries(jyppx_liuding_foundation PUBLIC ${TORCH_LIBRARIES})
target_compile_definitions(jyppx_liuding_foundation
    PUBLIC
        JYPPX_LIUDING_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        JYPPX_LIUDING_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        JYPPX_LIUDING_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    PRIVATE
        JYPPX_LIUDING_BUILD_TYPE="$<CONFIG>"
)
list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_foundation)
list(APPEND JYPPX_ENABLED_COMPONENTS foundation)
list(APPEND JYPPX_ENABLED_HEADER_DIRS foundation)

if(JYPPX_LIUDING_BUILD_CORE)
    add_library(jyppx_liuding_core STATIC
        src/jyppx/liuding/core/device.cpp
        src/jyppx/liuding/core/tensor_utils.cpp
        src/jyppx/liuding/core/value_schema.cpp
    )
    add_library(jyppx::liuding::core ALIAS jyppx_liuding_core)
    set_target_properties(jyppx_liuding_core PROPERTIES
        OUTPUT_NAME liuding-forge-core
        EXPORT_NAME core
    )
    target_include_directories(jyppx_liuding_core
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    foreach(JYPPX_TORCH_INCLUDE_DIR IN LISTS TORCH_INCLUDE_DIRS)
        target_include_directories(jyppx_liuding_core SYSTEM PUBLIC
            $<BUILD_INTERFACE:${JYPPX_TORCH_INCLUDE_DIR}>
        )
    endforeach()
    target_link_libraries(jyppx_liuding_core PUBLIC ${TORCH_LIBRARIES} jyppx_liuding_foundation)
    if(MSVC)
        # Official Windows LibTorch binaries use the release DLL runtime and
        # STL iterator ABI in every configuration.
        target_compile_options(jyppx_liuding_core PUBLIC /MD)
        target_compile_definitions(jyppx_liuding_core PUBLIC _ITERATOR_DEBUG_LEVEL=0)
    endif()
    list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_core)
    list(APPEND JYPPX_ENABLED_COMPONENTS core)
    list(APPEND JYPPX_ENABLED_HEADER_DIRS core utils)
endif()

if(JYPPX_LIUDING_BUILD_INTEROP)
    add_library(jyppx_liuding_interop STATIC
        src/jyppx/liuding/interop/checkpoint_manifest.cpp
        src/jyppx/liuding/interop/file_state_archive.cpp
        src/jyppx/liuding/interop/safetensors.cpp
        src/jyppx/liuding/interop/weight_import_report.cpp
        src/jyppx/liuding/interop/export_backend.cpp
        src/jyppx/liuding/interop/model_bundle.cpp
    )
    add_library(jyppx::liuding::interop ALIAS jyppx_liuding_interop)
    set_target_properties(jyppx_liuding_interop PROPERTIES
        OUTPUT_NAME liuding-forge-interop
        EXPORT_NAME interop
    )
    target_include_directories(jyppx_liuding_interop
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${jyppx_nlohmann_json_SOURCE_DIR}/single_include
    )
    target_link_libraries(jyppx_liuding_interop PUBLIC jyppx_liuding_foundation)
    list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_interop)
    list(APPEND JYPPX_ENABLED_COMPONENTS interop)
    list(APPEND JYPPX_ENABLED_HEADER_DIRS interop)
endif()

if(JYPPX_LIUDING_BUILD_DATA)
add_library(jyppx_liuding_data STATIC
    src/jyppx/liuding/data/batch_memory.cpp
    src/jyppx/liuding/data/batch_schema.cpp
    src/jyppx/liuding/data/data_capability.cpp
    src/jyppx/liuding/data/data_pipeline.cpp
    src/jyppx/liuding/data/iterable_data_pipeline.cpp
    src/jyppx/liuding/data/tensor_dataset.cpp
    src/jyppx/liuding/data/samplers/batch_sampler.cpp
    src/jyppx/liuding/data/samplers/random_sampler.cpp
    src/jyppx/liuding/data/samplers/sequential_sampler.cpp
)
add_library(jyppx::liuding::data ALIAS jyppx_liuding_data)
set_target_properties(jyppx_liuding_data PROPERTIES
    OUTPUT_NAME liuding-forge-data
    EXPORT_NAME data
)
target_include_directories(jyppx_liuding_data
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(jyppx_liuding_data
    PUBLIC
        jyppx_liuding_foundation
        jyppx_liuding_core
)
list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_data)
list(APPEND JYPPX_ENABLED_COMPONENTS data)
list(APPEND JYPPX_ENABLED_HEADER_DIRS data)
endif()

if(JYPPX_LIUDING_BUILD_DATA_IMAGE)
    add_library(jyppx_liuding_data_image STATIC
        src/jyppx/liuding/data_image/annotations.cpp
        src/jyppx/liuding/data_image/annotation_geometry.cpp
        src/jyppx/liuding/data_image/annotation_files.cpp
        src/jyppx/liuding/data_image/coco_annotations.cpp
        src/jyppx/liuding/data_image/text_annotations.cpp
        src/jyppx/liuding/data_image/extended_annotations.cpp
        src/jyppx/liuding/data_image/manifests.cpp
        src/jyppx/liuding/data_image/yolo_dataset_yaml.cpp
        src/jyppx/liuding/data_image/image_decoder.cpp
        src/jyppx/liuding/data_image/image_dataset.cpp
        src/jyppx/liuding/data_image/stb_image_impl.cpp
        $<TARGET_OBJECTS:data_image_xml_backend>
        $<TARGET_OBJECTS:data_image_yaml_backend>)
    add_library(jyppx::liuding::data_image ALIAS jyppx_liuding_data_image)
    set_target_properties(jyppx_liuding_data_image PROPERTIES
        OUTPUT_NAME liuding-forge-data-image EXPORT_NAME data_image)
    target_include_directories(jyppx_liuding_data_image PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_include_directories(jyppx_liuding_data_image SYSTEM PRIVATE
        ${jyppx_nlohmann_json_SOURCE_DIR}/single_include
        ${jyppx_image_tinyxml2_SOURCE_DIR}
        ${jyppx_image_yaml_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/third_party/stb)
    target_compile_definitions(jyppx_liuding_data_image PRIVATE YAML_CPP_STATIC_DEFINE)
    target_link_libraries(jyppx_liuding_data_image
        PUBLIC jyppx_liuding_foundation jyppx_liuding_core jyppx_liuding_data)
    list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_data_image)
    list(APPEND JYPPX_ENABLED_COMPONENTS data_image)
    list(APPEND JYPPX_ENABLED_HEADER_DIRS data_image)
endif()

if(JYPPX_LIUDING_BUILD_NN)
add_library(jyppx_liuding_nn STATIC
    src/jyppx/liuding/nn/invocation.cpp
    src/jyppx/liuding/nn/model_adapter.cpp
    src/jyppx/liuding/nn/model_checkpoint_participant.cpp
    src/jyppx/liuding/nn/loss/contrastive_loss.cpp
    src/jyppx/liuding/nn/loss/dice_loss.cpp
    src/jyppx/liuding/nn/loss/focal_loss.cpp
    src/jyppx/liuding/nn/loss/label_smoothing_cross_entropy.cpp
)
add_library(jyppx::liuding::nn ALIAS jyppx_liuding_nn)
set_target_properties(jyppx_liuding_nn PROPERTIES
    OUTPUT_NAME liuding-forge-nn
    EXPORT_NAME nn
)
target_include_directories(jyppx_liuding_nn
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(jyppx_liuding_nn
    PUBLIC
        jyppx_liuding_foundation
        jyppx_liuding_core
)
list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_nn)
list(APPEND JYPPX_ENABLED_COMPONENTS nn)
list(APPEND JYPPX_ENABLED_HEADER_DIRS nn)
endif()

if(JYPPX_LIUDING_BUILD_TRAINING)
add_library(jyppx_liuding_training STATIC
    src/jyppx/liuding/training/callback_dispatcher.cpp
    src/jyppx/liuding/training/callbacks/early_stopping_callback.cpp
    src/jyppx/liuding/training/checkpoint_coordinator.cpp
    src/jyppx/liuding/training/event_sink.cpp
    src/jyppx/liuding/training/telemetry_adapter.cpp
    src/jyppx/liuding/training/amp/grad_scaler.cpp
    src/jyppx/liuding/training/amp/grad_scaler_checkpoint_participant.cpp
    src/jyppx/liuding/training/metrics/classification.cpp
    src/jyppx/liuding/training/metrics/accuracy.cpp
    src/jyppx/liuding/training/metrics/average_meter.cpp
    src/jyppx/liuding/training/metrics/ranking.cpp
    src/jyppx/liuding/training/metrics/regression.cpp
    src/jyppx/liuding/training/metrics/topk_accuracy.cpp
    src/jyppx/liuding/training/scheduler_adapter.cpp
    src/jyppx/liuding/training/single_device_strategy.cpp
    src/jyppx/liuding/training/trainer.cpp
)
add_library(jyppx::liuding::training ALIAS jyppx_liuding_training)
set_target_properties(jyppx_liuding_training PROPERTIES
    OUTPUT_NAME liuding-forge-training
    EXPORT_NAME training
)
target_include_directories(jyppx_liuding_training
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(jyppx_liuding_training PUBLIC jyppx_liuding_foundation)
if(JYPPX_LIUDING_BUILD_INTEROP)
    target_sources(jyppx_liuding_training PRIVATE
        src/jyppx/liuding/training/file_checkpoint_store.cpp
    )
    target_link_libraries(jyppx_liuding_training PRIVATE jyppx_liuding_interop)
endif()
list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_training)
list(APPEND JYPPX_ENABLED_COMPONENTS training)
list(APPEND JYPPX_ENABLED_HEADER_DIRS training)
endif()

if(JYPPX_LIUDING_BUILD_OPTIM)
add_library(jyppx_liuding_optim STATIC
    src/jyppx/liuding/optim/adadelta_adapter.cpp
    src/jyppx/liuding/optim/adagrad_adapter.cpp
    src/jyppx/liuding/optim/adam_adapter.cpp
    src/jyppx/liuding/optim/adamw_adapter.cpp
    src/jyppx/liuding/optim/lamb_adapter.cpp
    src/jyppx/liuding/optim/nadam_adapter.cpp
    src/jyppx/liuding/optim/rmsprop_adapter.cpp
    src/jyppx/liuding/optim/composite_lr_scheduler.cpp
    src/jyppx/liuding/optim/epoch_lr_scheduler.cpp
    src/jyppx/liuding/optim/event_lr_scheduler.cpp
    src/jyppx/liuding/optim/linear_lr_scheduler.cpp
    src/jyppx/liuding/optim/sgd_adapter.cpp
    src/jyppx/liuding/optim/step_lr_scheduler.cpp
)
add_library(jyppx::liuding::optim ALIAS jyppx_liuding_optim)
set_target_properties(jyppx_liuding_optim PROPERTIES
    OUTPUT_NAME liuding-forge-optim
    EXPORT_NAME optim
)
target_include_directories(jyppx_liuding_optim
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(jyppx_liuding_optim
    PUBLIC
        jyppx_liuding_foundation
        jyppx_liuding_core
)
list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_optim)
list(APPEND JYPPX_ENABLED_COMPONENTS optim)
list(APPEND JYPPX_ENABLED_HEADER_DIRS optim)
endif()

if(JYPPX_LIUDING_BUILD_EXPERIMENT)
    add_library(jyppx_liuding_experiment STATIC
        src/jyppx/liuding/experiment/artifact_store.cpp
        src/jyppx/liuding/experiment/detail/artifact_publish.cpp
        src/jyppx/liuding/experiment/detail/native_artifact_publish.cpp
        src/jyppx/liuding/experiment/config.cpp
        src/jyppx/liuding/experiment/run.cpp
        src/jyppx/liuding/experiment/run_store.cpp
        src/jyppx/liuding/experiment/run_telemetry_sink.cpp
        src/jyppx/liuding/experiment/telemetry_exporters.cpp
        src/jyppx/liuding/experiment/evaluation.cpp
        $<TARGET_OBJECTS:sqlite3_amalgamation>
    )
    # Keep the native publication forwarding seam in its own archive member
    # for real process-boundary integration supervision.
    set_source_files_properties(
        src/jyppx/liuding/experiment/detail/artifact_publish.cpp
        src/jyppx/liuding/experiment/detail/native_artifact_publish.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
    add_library(jyppx::liuding::experiment ALIAS jyppx_liuding_experiment)
    set_target_properties(jyppx_liuding_experiment PROPERTIES
        OUTPUT_NAME liuding-forge-experiment
        EXPORT_NAME experiment
    )
    target_include_directories(jyppx_liuding_experiment
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${jyppx_nlohmann_json_SOURCE_DIR}/single_include
    )
    target_include_directories(jyppx_liuding_experiment SYSTEM PRIVATE
        "${jyppx_sqlite_SOURCE_DIR}"
    )
    target_link_libraries(jyppx_liuding_experiment
        PUBLIC
            jyppx_liuding_foundation
            jyppx_liuding_core
        PRIVATE
            Threads::Threads
    )
    list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_experiment)
    list(APPEND JYPPX_ENABLED_COMPONENTS experiment)
    list(APPEND JYPPX_ENABLED_HEADER_DIRS experiment)
endif()

if(JYPPX_LIUDING_BUILD_TRAINING)
    target_link_libraries(jyppx_liuding_training PUBLIC
        jyppx_liuding_core
        jyppx_liuding_data
        jyppx_liuding_optim
        jyppx_liuding_nn
    )
endif()

if(JYPPX_LIUDING_BUILD_INTEROP AND JYPPX_LIUDING_BUILD_NN)
    target_sources(jyppx_liuding_interop PRIVATE
        src/jyppx/liuding/interop/model_weight_import.cpp
        src/jyppx/liuding/interop/torchscript_compatibility.cpp)
    target_link_libraries(jyppx_liuding_interop PUBLIC jyppx_liuding_nn)
endif()

# Independent, default-OFF Preview components do not add reverse edges to core.
if(JYPPX_LIUDING_BUILD_MODELS)
    add_library(jyppx_liuding_models STATIC
        src/jyppx/liuding/models/registry.cpp
        src/jyppx/liuding/models/transformer.cpp
        src/jyppx/liuding/models/vision.cpp
        src/jyppx/liuding/models/weights.cpp)
    add_library(jyppx::liuding::models ALIAS jyppx_liuding_models)
    set_target_properties(jyppx_liuding_models PROPERTIES
        OUTPUT_NAME liuding-forge-models EXPORT_NAME models)
    target_include_directories(jyppx_liuding_models PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_link_libraries(jyppx_liuding_models
        PUBLIC jyppx_liuding_interop jyppx_liuding_nn
        PRIVATE jyppx_liuding_core jyppx_liuding_foundation)
    list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_models)
    list(APPEND JYPPX_ENABLED_COMPONENTS models)
    list(APPEND JYPPX_ENABLED_HEADER_DIRS models)
endif()

if(JYPPX_LIUDING_BUILD_QUANTIZATION)
    add_library(jyppx_liuding_quantization STATIC
        src/jyppx/liuding/quantization/quantization.cpp)
    add_library(jyppx::liuding::quantization ALIAS jyppx_liuding_quantization)
    set_target_properties(jyppx_liuding_quantization PROPERTIES
        OUTPUT_NAME liuding-forge-quantization EXPORT_NAME quantization)
    target_include_directories(jyppx_liuding_quantization PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_link_libraries(jyppx_liuding_quantization PUBLIC jyppx_liuding_foundation)
    list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_quantization)
    list(APPEND JYPPX_ENABLED_COMPONENTS quantization)
    list(APPEND JYPPX_ENABLED_HEADER_DIRS quantization)
endif()

if(JYPPX_LIUDING_BUILD_SERVING)
    add_library(jyppx_liuding_serving STATIC
        src/jyppx/liuding/serving/runtime.cpp
        src/jyppx/liuding/serving/batcher.cpp
        src/jyppx/liuding/serving/service.cpp)
    add_library(jyppx::liuding::serving ALIAS jyppx_liuding_serving)
    set_target_properties(jyppx_liuding_serving PROPERTIES
        OUTPUT_NAME liuding-forge-serving EXPORT_NAME serving)
    target_include_directories(jyppx_liuding_serving PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_link_libraries(jyppx_liuding_serving
        PUBLIC jyppx_liuding_foundation jyppx_liuding_interop jyppx_liuding_nn
        PRIVATE jyppx_liuding_core Threads::Threads)
    list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_serving)
    list(APPEND JYPPX_ENABLED_COMPONENTS serving)
    list(APPEND JYPPX_ENABLED_HEADER_DIRS serving)
endif()

add_library(jyppx_liuding_all INTERFACE)
add_library(jyppx::liuding::all ALIAS jyppx_liuding_all)
set_target_properties(jyppx_liuding_all PROPERTIES EXPORT_NAME all)
target_include_directories(jyppx_liuding_all
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
foreach(JYPPX_TORCH_INCLUDE_DIR IN LISTS TORCH_INCLUDE_DIRS)
    target_include_directories(jyppx_liuding_all SYSTEM INTERFACE
        $<BUILD_INTERFACE:${JYPPX_TORCH_INCLUDE_DIR}>
    )
endforeach()
target_link_libraries(jyppx_liuding_all INTERFACE ${JYPPX_ENABLED_TARGETS})
list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_all)
list(APPEND JYPPX_ENABLED_COMPONENTS all)

if(JYPPX_LIUDING_BUILD_CORE AND JYPPX_LIUDING_BUILD_INTEROP AND JYPPX_LIUDING_BUILD_DATA AND
   JYPPX_LIUDING_BUILD_NN AND JYPPX_LIUDING_BUILD_OPTIM AND JYPPX_LIUDING_BUILD_TRAINING AND
   JYPPX_LIUDING_BUILD_EXPERIMENT)
    add_library(jyppx_liuding_sdk INTERFACE)
    add_library(jyppx::liuding::sdk ALIAS jyppx_liuding_sdk)
    set_target_properties(jyppx_liuding_sdk PROPERTIES EXPORT_NAME sdk)
    target_include_directories(jyppx_liuding_sdk
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    foreach(JYPPX_TORCH_INCLUDE_DIR IN LISTS TORCH_INCLUDE_DIRS)
        target_include_directories(jyppx_liuding_sdk SYSTEM INTERFACE
            $<BUILD_INTERFACE:${JYPPX_TORCH_INCLUDE_DIR}>
        )
    endforeach()
    # The SDK is the stable eight-component Core profile.  Do not route it
    # through `all`: that aggregate intentionally includes every component
    # enabled in the build, including independent Preview components such as
    # models, quantization, and serving.
    target_link_libraries(jyppx_liuding_sdk INTERFACE
        jyppx_liuding_foundation
        jyppx_liuding_core
        jyppx_liuding_interop
        jyppx_liuding_data
        jyppx_liuding_nn
        jyppx_liuding_optim
        jyppx_liuding_training
        jyppx_liuding_experiment)
    list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_sdk)
    list(APPEND JYPPX_ENABLED_COMPONENTS sdk)
endif()

# ============================================
# Component Dependency Graph Gate
# ============================================
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(JYPPX_LIUDING_COMPONENT_DEPENDENCY_MANIFEST
    "${CMAKE_CURRENT_SOURCE_DIR}/component-dependencies.json")
set(JYPPX_LIUDING_COMPONENT_TARGET_GRAPH
    "${CMAKE_CURRENT_BINARY_DIR}/component-target-graph.tsv")
set(JYPPX_LIUDING_COMPONENT_GRAPH_REPORT
    "${CMAKE_CURRENT_BINARY_DIR}/component-dependency-report.json")
include(cmake/WriteComponentGraph.cmake)
jyppx_write_component_graph("${JYPPX_LIUDING_COMPONENT_TARGET_GRAPH}")

file(GLOB_RECURSE JYPPX_LIUDING_COMPONENT_GRAPH_SCAN_INPUTS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/jyppx/liuding/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/jyppx/liuding/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jyppx/liuding/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jyppx/liuding/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jyppx/liuding/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jyppx/liuding/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jyppx/liuding/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jyppx/liuding/*.inc"
)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${JYPPX_LIUDING_COMPONENT_DEPENDENCY_MANIFEST}"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/WriteComponentGraph.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/validation/component_graph.py"
    ${JYPPX_LIUDING_COMPONENT_GRAPH_SCAN_INPUTS}
)
execute_process(
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/validation/component_graph.py"
            --manifest "${JYPPX_LIUDING_COMPONENT_DEPENDENCY_MANIFEST}"
            --source-root "${CMAKE_CURRENT_SOURCE_DIR}"
            --target-graph "${JYPPX_LIUDING_COMPONENT_TARGET_GRAPH}"
            --output "${JYPPX_LIUDING_COMPONENT_GRAPH_REPORT}"
    RESULT_VARIABLE JYPPX_LIUDING_COMPONENT_GRAPH_RESULT
    OUTPUT_VARIABLE JYPPX_LIUDING_COMPONENT_GRAPH_OUTPUT
    ERROR_VARIABLE JYPPX_LIUDING_COMPONENT_GRAPH_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)
if(NOT JYPPX_LIUDING_COMPONENT_GRAPH_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Component dependency graph gate failed:\n"
        "${JYPPX_LIUDING_COMPONENT_GRAPH_OUTPUT}\n${JYPPX_LIUDING_COMPONENT_GRAPH_ERROR}")
endif()
message(STATUS "${JYPPX_LIUDING_COMPONENT_GRAPH_OUTPUT}")

# Compiler options must be applied after the target has been created.
if(JYPPX_LIUDING_BUILD_TOOLS)
    if(NOT TARGET jyppx::liuding::sdk)
        message(FATAL_ERROR "JYPPX_LIUDING_BUILD_TOOLS=ON requires the complete jyppx::liuding::sdk component closure")
    endif()
    add_subdirectory(tools/cli)
elseif(JYPPX_LIUDING_INSTALL_TELEMETRY_SIDECAR)
    message(FATAL_ERROR "JYPPX_LIUDING_INSTALL_TELEMETRY_SIDECAR=ON requires JYPPX_LIUDING_BUILD_TOOLS=ON")
endif()
include(cmake/CompilerOptions.cmake)

# ============================================
# Tests
# ============================================
if(JYPPX_LIUDING_BUILD_TESTS)
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
if(JYPPX_LIUDING_BUILD_EXAMPLES)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
        add_subdirectory(examples)
    else()
        message(STATUS "Skipping examples: examples/CMakeLists.txt is not present yet")
    endif()
endif()

# ============================================
# Documentation
# ============================================
if(JYPPX_LIUDING_BUILD_DOCS)
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
    jyppx_public_headers("${JYPPX_HEADER_DIR}" JYPPX_LIUDING_INSTALL_HEADERS)
    foreach(JYPPX_LIUDING_INSTALL_HEADER IN LISTS JYPPX_LIUDING_INSTALL_HEADERS)
        file(RELATIVE_PATH JYPPX_HEADER_RELATIVE "${PROJECT_SOURCE_DIR}/include"
            "${JYPPX_LIUDING_INSTALL_HEADER}")
        get_filename_component(JYPPX_HEADER_PARENT "${JYPPX_HEADER_RELATIVE}" DIRECTORY)
        install(FILES "${JYPPX_LIUDING_INSTALL_HEADER}"
            DESTINATION "include/${JYPPX_HEADER_PARENT}")
    endforeach()
endforeach()

if(TARGET jyppx::liuding::sdk)
    install(FILES include/jyppx/liuding/liuding.hpp DESTINATION include/jyppx/liuding)
endif()

install(FILES
    "${JYPPX_GENERATED_INCLUDE_DIR}/jyppx/liuding/foundation/build_config.hpp"
    DESTINATION include/jyppx/liuding/foundation
)

install(EXPORT jyppxTargetsExport
    FILE jyppxTargets.cmake
    NAMESPACE jyppx::liuding::
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
if(JYPPX_LIUDING_BUILD_INTEROP OR JYPPX_LIUDING_BUILD_EXPERIMENT OR JYPPX_LIUDING_BUILD_DATA_IMAGE)
    # nlohmann/json is compiled into these archives. Its original MIT notice
    # must accompany the binary SDK as well as the repository summary.
    install(FILES third_party/licenses/nlohmann-json-3.12.0-LICENSE.MIT
        DESTINATION share/doc/jyppx/licenses COMPONENT licenses)
endif()
if(JYPPX_LIUDING_BUILD_DATA_IMAGE)
    install(FILES
        third_party/licenses/tinyxml2-11.0.0-LICENSE.txt
        third_party/licenses/yaml-cpp-0.9.0-LICENSE.txt
        third_party/licenses/stb_image-2.30-LICENSE.txt
        DESTINATION share/doc/jyppx/licenses COMPONENT licenses)
endif()
if(JYPPX_LIUDING_BUILD_EXPERIMENT)
    install(FILES
        third_party/licenses/sqlite-3.54.1-copyright-blessing.txt
        third_party/licenses/sqlite-copyright-page.txt
        DESTINATION share/doc/jyppx/licenses COMPONENT licenses)
endif()

# ============================================
# Summary
# ============================================
message(STATUS "")
message(STATUS "${LIUDING_FORGE_DISPLAY_NAME_ZH} (${LIUDING_FORGE_DISPLAY_NAME_EN}) Configuration:")
message(STATUS "  Internal project alias: ${LIUDING_FORGE_PROJECT_ALIAS}")
message(STATUS "  Artifact prefix: ${LIUDING_FORGE_ARTIFACT_PREFIX}")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build tests: ${JYPPX_LIUDING_BUILD_TESTS}")
message(STATUS "  Build examples: ${JYPPX_LIUDING_BUILD_EXAMPLES}")
message(STATUS "  Build benchmarks: ${JYPPX_LIUDING_BUILD_BENCHMARKS}")
message(STATUS "  Build fuzzers: ${JYPPX_LIUDING_BUILD_FUZZERS}")
message(STATUS "  Build docs: ${JYPPX_LIUDING_BUILD_DOCS}")
message(STATUS "  Components: ${JYPPX_ENABLED_COMPONENTS}")
message(STATUS "  Clang-tidy: ${JYPPX_LIUDING_ENABLE_CLANG_TIDY}")
message(STATUS "  Format check: ${JYPPX_LIUDING_ENABLE_FORMAT_CHECK}")
message(STATUS "  Warnings as errors: ${JYPPX_LIUDING_ENABLE_WARNINGS_AS_ERRORS}")
message(STATUS "")

# CPack uses the public artifact identity while the installed CMake package
# remains jyppx for source and binary compatibility.
set(CPACK_PACKAGE_NAME "liuding-forge")
set(CPACK_PACKAGE_DISPLAY_NAME "六丁炼算 (Liuding Forge)")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "六丁炼算 (Liuding Forge) - C++20/LibTorch training framework")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "Liuding Forge")
set(CPACK_NSIS_DISPLAY_NAME "六丁炼算 (Liuding Forge)")
set(CPACK_PACKAGE_VENDOR "Liuding Forge")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
include(CPack)
