# Compiler Options for JYPPX Training Framework
#
# Per-configuration flags use generator expressions so they work with both
# single-config (Ninja/Make) and multi-config (Visual Studio) generators.

# ============================================
# Warning Flags
# ============================================
set(JYPPX_COMPILED_TARGETS
    jyppx_liuding_foundation
    jyppx_liuding_core
    jyppx_liuding_interop
    jyppx_liuding_data
    jyppx_liuding_data_image
    jyppx_liuding_vision_render
    jyppx_liuding_nn
    jyppx_liuding_training
    jyppx_liuding_optim
    jyppx_liuding_experiment
    jyppx_liuding_telemetry_native
    jyppx_liuding_telemetry_http
    jyppx_liuding_models
    jyppx_liuding_onnx_export
    jyppx_liuding_onnx_runtime
    jyppx_liuding_onnx_serving
    jyppx_liuding_bundle_signing
    jyppx_liuding_quantization
    jyppx_liuding_serving
    jyppx_dataset_preparation
    jyppx_dinov2_conversion
    jyppx_convert_dinov2_hf
    jyppx_cli_transport
    jyppx_cli_objects
    jyppx_cli
    jyppx_cli_recipe_gate
    jyppx_int8_prepare
    jyppx_int8_export
)
foreach(JYPPX_TARGET IN LISTS JYPPX_COMPILED_TARGETS)
    if(NOT TARGET ${JYPPX_TARGET})
        continue()
    endif()
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

if(JYPPX_LIUDING_ENABLE_WARNINGS_AS_ERRORS)
    if(MSVC)
        message(FATAL_ERROR
            "JYPPX_LIUDING_ENABLE_WARNINGS_AS_ERRORS is not available with MSVC: "
            "LibTorch template instantiations emit upstream C4267 warnings. "
            "The project does not globally suppress that diagnostic.")
    endif()
    foreach(JYPPX_TARGET IN LISTS JYPPX_COMPILED_TARGETS)
        if(NOT TARGET ${JYPPX_TARGET})
            continue()
        endif()
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${JYPPX_TARGET} PRIVATE -Werror)
        endif()
    endforeach()
endif()

# Every compiled framework target is held to the same static-analysis policy.
# Third-party targets and test executables remain outside this target list.
if(JYPPX_LIUDING_ENABLE_CLANG_TIDY)
    if(NOT JYPPX_CLANG_TIDY_EXECUTABLE)
        find_program(JYPPX_CLANG_TIDY_EXECUTABLE
            NAMES clang-tidy clang-tidy-20 clang-tidy-19 clang-tidy-18
        )
    endif()
    if(NOT JYPPX_CLANG_TIDY_EXECUTABLE)
        message(FATAL_ERROR
            "JYPPX_LIUDING_ENABLE_CLANG_TIDY=ON requires a clang-tidy executable")
    endif()
    foreach(JYPPX_TARGET IN LISTS JYPPX_COMPILED_TARGETS)
        if(NOT TARGET ${JYPPX_TARGET})
            continue()
        endif()
        set_target_properties(${JYPPX_TARGET} PROPERTIES
            CXX_CLANG_TIDY
                "${JYPPX_CLANG_TIDY_EXECUTABLE};--config-file=${PROJECT_SOURCE_DIR}/.clang-tidy"
        )
    endforeach()
endif()

if(JYPPX_LIUDING_ENABLE_FORMAT_CHECK)
    if(NOT JYPPX_CLANG_FORMAT_EXECUTABLE)
        find_program(JYPPX_CLANG_FORMAT_EXECUTABLE
            NAMES clang-format clang-format-20 clang-format-19 clang-format-18
        )
    endif()
    if(NOT JYPPX_CLANG_FORMAT_EXECUTABLE)
        message(FATAL_ERROR
            "JYPPX_LIUDING_ENABLE_FORMAT_CHECK=ON requires a clang-format executable")
    endif()
    set(JYPPX_FORMAT_FILES)
    foreach(JYPPX_FORMAT_ROOT IN ITEMS include src tests examples tools/cli tools/datasets tools/models)
        file(GLOB_RECURSE JYPPX_FORMAT_ROOT_FILES CONFIGURE_DEPENDS
            "${PROJECT_SOURCE_DIR}/${JYPPX_FORMAT_ROOT}/*.c"
            "${PROJECT_SOURCE_DIR}/${JYPPX_FORMAT_ROOT}/*.cc"
            "${PROJECT_SOURCE_DIR}/${JYPPX_FORMAT_ROOT}/*.cpp"
            "${PROJECT_SOURCE_DIR}/${JYPPX_FORMAT_ROOT}/*.cxx"
            "${PROJECT_SOURCE_DIR}/${JYPPX_FORMAT_ROOT}/*.cu"
            "${PROJECT_SOURCE_DIR}/${JYPPX_FORMAT_ROOT}/*.cuh"
            "${PROJECT_SOURCE_DIR}/${JYPPX_FORMAT_ROOT}/*.h"
            "${PROJECT_SOURCE_DIR}/${JYPPX_FORMAT_ROOT}/*.hh"
            "${PROJECT_SOURCE_DIR}/${JYPPX_FORMAT_ROOT}/*.hpp"
            "${PROJECT_SOURCE_DIR}/${JYPPX_FORMAT_ROOT}/*.hxx"
            "${PROJECT_SOURCE_DIR}/${JYPPX_FORMAT_ROOT}/*.inc"
            "${PROJECT_SOURCE_DIR}/${JYPPX_FORMAT_ROOT}/*.ipp"
            "${PROJECT_SOURCE_DIR}/${JYPPX_FORMAT_ROOT}/*.tpp"
        )
        list(APPEND JYPPX_FORMAT_FILES ${JYPPX_FORMAT_ROOT_FILES})
    endforeach()
    list(REMOVE_DUPLICATES JYPPX_FORMAT_FILES)
    list(SORT JYPPX_FORMAT_FILES)
    add_custom_target(jyppx_format_check ALL
        COMMAND ${JYPPX_CLANG_FORMAT_EXECUTABLE}
            --dry-run --Werror ${JYPPX_FORMAT_FILES}
        COMMENT "Checking all project C/C++ sources with clang-format"
        VERBATIM
    )
endif()

# ============================================
# Debug Definitions
# ============================================
foreach(JYPPX_TARGET IN LISTS JYPPX_COMPILED_TARGETS)
    if(NOT TARGET ${JYPPX_TARGET})
        continue()
    endif()
    target_compile_definitions(${JYPPX_TARGET} PRIVATE
        $<$<CONFIG:Debug>:JYPPX_DEBUG>
    )
endforeach()

# ============================================
# Instrumentation (Optional)
# ============================================
set(JYPPX_INSTRUMENTATION_REQUESTED OFF)
if(JYPPX_LIUDING_ENABLE_ASAN OR JYPPX_LIUDING_ENABLE_UBSAN OR JYPPX_LIUDING_ENABLE_TSAN
   OR JYPPX_LIUDING_ENABLE_COVERAGE)
    set(JYPPX_INSTRUMENTATION_REQUESTED ON)
endif()

if(JYPPX_INSTRUMENTATION_REQUESTED
   AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR
        "Coverage and sanitizer options require a GNU or Clang compiler")
endif()

if(JYPPX_LIUDING_ENABLE_ASAN AND JYPPX_LIUDING_ENABLE_TSAN)
    message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer cannot be combined")
endif()

if(JYPPX_LIUDING_ENABLE_COVERAGE AND JYPPX_LIUDING_ENABLE_TSAN)
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

    if(JYPPX_LIUDING_ENABLE_ASAN)
        target_compile_options(${target_name} PRIVATE -fsanitize=address)
        target_link_options(${target_name} ${link_scope} -fsanitize=address)
    endif()
    if(JYPPX_LIUDING_ENABLE_UBSAN)
        target_compile_options(${target_name} PRIVATE -fsanitize=undefined)
        target_link_options(${target_name} ${link_scope} -fsanitize=undefined)
    endif()
    if(JYPPX_LIUDING_ENABLE_TSAN)
        target_compile_options(${target_name} PRIVATE -fsanitize=thread)
        target_link_options(${target_name} ${link_scope} -fsanitize=thread)
    endif()
    if(JYPPX_LIUDING_ENABLE_COVERAGE)
        set(coverage_compile_options --coverage)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            # Data pipelines exercise instrumented code from multiple worker
            # threads. Atomic counters keep gcov results deterministic.
            list(APPEND coverage_compile_options -fprofile-update=atomic)
        endif()
        target_compile_options(${target_name} PRIVATE ${coverage_compile_options})
        target_link_options(${target_name} ${link_scope} --coverage)
    endif()
endfunction()

# Test tools created before this module need instrumentation without entering
# the framework-only warning and static-analysis target lists.
set(JYPPX_INSTRUMENTATION_TARGETS
    ${JYPPX_COMPILED_TARGETS}
    jyppx_int8_frozen_compare
    jyppx_native_telemetry_codec
)
foreach(JYPPX_TARGET IN LISTS JYPPX_INSTRUMENTATION_TARGETS)
    if(NOT TARGET ${JYPPX_TARGET})
        continue()
    endif()
    jyppx_apply_instrumentation(${JYPPX_TARGET})
endforeach()
