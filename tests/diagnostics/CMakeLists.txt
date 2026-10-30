# Explicit next-slice numerical diagnostic, absent from default builds/CTest.
# No public model/optimizer interface or normal CLI behavior changes.
if(TARGET jyppx::liuding::models AND TARGET jyppx::liuding::data_image)
    set(JYPPX_DINOV2_DIAGNOSTIC_SOURCES
        tests/diagnostics/dinov2_first_step.cpp
        tests/diagnostics/cli_input_support.inc
        tests/diagnostics/recipe_capture_support.inc
        tests/diagnostics/CMakeLists.txt
        tools/cli/transfer_recipe.cpp
        tools/cli/recipe_support.hpp
        tools/cli/cli.hpp
        tests/unit/models/reference_file.hpp
        src/jyppx/liuding/models/dino.cpp
        include/jyppx/liuding/models/dino.hpp
        tests/unit/models/reference/convert_dinov2_hf_reference.py
    )
    # The extracted fragments are compiled; their containing production files
    # are only observed by the external driver. Unrelated CLI integration must
    # not invalidate this target's compiled-source binding.
    set(JYPPX_DINOV2_DIAGNOSTIC_BINDINGS "{")
    set(JYPPX_DINOV2_DIAGNOSTIC_SEPARATOR "")
    foreach(JYPPX_DINOV2_DIAGNOSTIC_SOURCE IN LISTS JYPPX_DINOV2_DIAGNOSTIC_SOURCES)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${PROJECT_SOURCE_DIR}/${JYPPX_DINOV2_DIAGNOSTIC_SOURCE}")
        file(SHA256 "${PROJECT_SOURCE_DIR}/${JYPPX_DINOV2_DIAGNOSTIC_SOURCE}"
            JYPPX_DINOV2_DIAGNOSTIC_SHA)
        string(APPEND JYPPX_DINOV2_DIAGNOSTIC_BINDINGS
            "${JYPPX_DINOV2_DIAGNOSTIC_SEPARATOR}\"${JYPPX_DINOV2_DIAGNOSTIC_SOURCE}\":\"${JYPPX_DINOV2_DIAGNOSTIC_SHA}\"")
        set(JYPPX_DINOV2_DIAGNOSTIC_SEPARATOR ",")
    endforeach()
    string(APPEND JYPPX_DINOV2_DIAGNOSTIC_BINDINGS "}")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dinov2_compiled_sources.hpp"
        "#pragma once\n#include <string_view>\ninline constexpr std::string_view kDinoV2CompiledSources = R\"bindings(${JYPPX_DINOV2_DIAGNOSTIC_BINDINGS})bindings\";\n")
    add_executable(jyppx_dinov2_first_step_gate EXCLUDE_FROM_ALL
        dinov2_first_step.cpp "${PROJECT_SOURCE_DIR}/tools/cli/transfer_recipe.cpp")
    set_target_properties(jyppx_dinov2_first_step_gate PROPERTIES
        OUTPUT_NAME liuding-forge-dinov2-first-step-gate
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tools/cli")
    target_include_directories(jyppx_dinov2_first_step_gate PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/tools/cli"
        "${PROJECT_SOURCE_DIR}/tests/unit/models")
    target_compile_definitions(jyppx_dinov2_first_step_gate PRIVATE
        JYPPX_LIUDING_CLI_IMAGE_RECIPE=1)
    target_link_libraries(jyppx_dinov2_first_step_gate PRIVATE
        jyppx::liuding::sdk jyppx::liuding::models jyppx::liuding::data_image
        nlohmann_json::nlohmann_json)
    jyppx_apply_instrumentation(jyppx_dinov2_first_step_gate)
    jyppx_add_torch_runtime_dependency(jyppx_dinov2_first_step_gate)
endif()
