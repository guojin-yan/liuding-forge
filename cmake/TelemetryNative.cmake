# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.

# This optional component embeds unmodified official nanopb runtime objects.
# No generator, Python or protobuf SDK is needed by a product consumer.
if(NOT JYPPX_LIUDING_BUILD_EXPERIMENT)
    message(FATAL_ERROR "TELEMETRY_NATIVE requires EXPERIMENT")
endif()
set(JYPPX_LIUDING_NANOPB_SOURCE_DIR "" CACHE PATH
    "Explicit verified nanopb 0.4.9.2 source root; configure never downloads it")
if(NOT IS_DIRECTORY "${JYPPX_LIUDING_NANOPB_SOURCE_DIR}")
    message(FATAL_ERROR
        "TELEMETRY_NATIVE requires an explicit nanopb 0.4.9.2 source root")
endif()
get_filename_component(_jyppx_nanopb_root
    "${JYPPX_LIUDING_NANOPB_SOURCE_DIR}" REALPATH)
set(_jyppx_nanopb_files
    "pb.h|e0db84a27e0d41a2d2d347b8c879e30ceb856d36dc192cce0f1124f833c67bc2"
    "pb_common.h|6495a691aca68d6973f2274b5dd54b74fbb57f6b019c45fff255a857fe1abcfd"
    "pb_common.c|8d2ec28baaaf2b7a5e90e4cb2fa9700d21cef7f826f051a637c30b7a1e6a0516"
    "pb_encode.h|9aa00fee4ff08adf0da16e33a55be08810ea657800a648dc78f82e89c60c10cf"
    "pb_encode.c|debb0714dff8d1515b9724eae45531cad912f12028a0311fc3e2c694689e1fed"
    "pb_decode.h|fcac5f7680fe6e870157e4bcf34d5162bdd4fff0d7db3cad1122f2ad24a6da87"
    "pb_decode.c|f5b425beaa207251e531c8ce2c86c9b6867e2920ed59cc1b125332af0c147632"
    "LICENSE.txt|e2f2fc8fe3faa7dcb09dbe995db48c6ec5c1f72705db915101e4a83fed44f66d")
foreach(_jyppx_nanopb_entry IN LISTS _jyppx_nanopb_files)
    string(REPLACE "|" ";" _jyppx_nanopb_pair "${_jyppx_nanopb_entry}")
    list(GET _jyppx_nanopb_pair 0 _jyppx_nanopb_name)
    list(GET _jyppx_nanopb_pair 1 _jyppx_nanopb_expected)
    if(NOT EXISTS "${_jyppx_nanopb_root}/${_jyppx_nanopb_name}")
        message(FATAL_ERROR "Pinned nanopb source file is missing: ${_jyppx_nanopb_name}")
    endif()
    file(SHA256 "${_jyppx_nanopb_root}/${_jyppx_nanopb_name}" _jyppx_nanopb_actual)
    if(NOT _jyppx_nanopb_actual STREQUAL _jyppx_nanopb_expected)
        message(FATAL_ERROR "Pinned nanopb source digest mismatch: ${_jyppx_nanopb_name}")
    endif()
endforeach()

# C is enabled only for the mature dependency; product code remains C++20.
enable_language(C)
set(_jyppx_telemetry_root
    "${PROJECT_SOURCE_DIR}/src/jyppx/liuding/telemetry_native")
set(_jyppx_telemetry_proto "${_jyppx_telemetry_root}/detail/telemetry_proto")
file(GLOB_RECURSE _jyppx_tensorboard_generated CONFIGURE_DEPENDS
    "${_jyppx_telemetry_proto}/tensorboard/*.pb.c")
list(LENGTH _jyppx_tensorboard_generated _jyppx_tensorboard_generated_count)
if(NOT _jyppx_tensorboard_generated_count EQUAL 8)
    message(FATAL_ERROR "Native TensorBoard requires the complete pinned schema closure")
endif()
file(GLOB_RECURSE _jyppx_otlp_generated CONFIGURE_DEPENDS
    "${_jyppx_telemetry_proto}/opentelemetry/*.pb.c")
list(LENGTH _jyppx_otlp_generated _jyppx_otlp_generated_count)
if(NOT _jyppx_otlp_generated_count EQUAL 8)
    message(FATAL_ERROR "Native OTLP requires the complete pinned schema closure")
endif()
add_library(jyppx_native_telemetry_codec OBJECT
    "${_jyppx_nanopb_root}/pb_common.c"
    "${_jyppx_nanopb_root}/pb_encode.c"
    "${_jyppx_nanopb_root}/pb_decode.c"
    ${_jyppx_tensorboard_generated}
    ${_jyppx_otlp_generated})
set_target_properties(jyppx_native_telemetry_codec PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED YES C_EXTENSIONS NO
    POSITION_INDEPENDENT_CODE ON C_VISIBILITY_PRESET hidden)
target_include_directories(jyppx_native_telemetry_codec PRIVATE
    "${_jyppx_nanopb_root}" "${_jyppx_telemetry_proto}")
target_compile_definitions(jyppx_native_telemetry_codec PRIVATE
    PB_FIELD_32BIT=1 PB_VALIDATE_UTF8=1 PB_MESSAGE_NESTING_MAX=32)
if(MSVC)
    target_compile_options(jyppx_native_telemetry_codec PRIVATE
        "/FI${_jyppx_telemetry_root}/detail/nanopb_prefix.h")
else()
    target_compile_options(jyppx_native_telemetry_codec PRIVATE
        "-include" "${_jyppx_telemetry_root}/detail/nanopb_prefix.h")
endif()

add_library(jyppx_liuding_telemetry_native STATIC
    src/jyppx/liuding/telemetry_native/tensorboard.cpp
    src/jyppx/liuding/telemetry_native/otlp_projection.cpp
    src/jyppx/liuding/telemetry_native/detail/otlp_mapping.cpp
    src/jyppx/liuding/telemetry_native/detail/otlp_preflight.cpp
    $<TARGET_OBJECTS:jyppx_native_telemetry_codec>)
add_library(jyppx::liuding::telemetry_native ALIAS jyppx_liuding_telemetry_native)
set_target_properties(jyppx_liuding_telemetry_native PROPERTIES
    OUTPUT_NAME liuding-forge-telemetry-native EXPORT_NAME telemetry_native)
target_include_directories(jyppx_liuding_telemetry_native PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_include_directories(jyppx_liuding_telemetry_native PRIVATE
    "${_jyppx_nanopb_root}" "${_jyppx_telemetry_proto}")
target_compile_definitions(jyppx_liuding_telemetry_native PRIVATE
    PB_FIELD_32BIT=1 PB_VALIDATE_UTF8=1 PB_MESSAGE_NESTING_MAX=32)
target_link_libraries(jyppx_liuding_telemetry_native
    PUBLIC jyppx_liuding_foundation
    PRIVATE jyppx_liuding_experiment
        $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)
list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_telemetry_native)
list(APPEND JYPPX_ENABLED_COMPONENTS telemetry_native)
list(APPEND JYPPX_ENABLED_HEADER_DIRS telemetry_native)

# The private runtime is inside the product archive. No nanopb target, header,
# absolute source-root path, code generator or macro leaks into the export.
install(FILES "${_jyppx_nanopb_root}/LICENSE.txt"
    DESTINATION share/doc/jyppx/licenses RENAME nanopb-0.4.9.2-LICENSE.txt
    COMPONENT licenses)
install(FILES
    "${PROJECT_SOURCE_DIR}/third_party/licenses/tensorboard-2.20.0-LICENSE"
    "${PROJECT_SOURCE_DIR}/third_party/licenses/opentelemetry-proto-v1.7.0-LICENSE"
    DESTINATION share/doc/jyppx/licenses COMPONENT licenses)
