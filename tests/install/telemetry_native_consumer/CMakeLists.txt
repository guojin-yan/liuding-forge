cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx_liuding_native_telemetry_consumer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(JYPPX_EXPECT_NATIVE_TELEMETRY "Expect the optional component in this installation" ON)
# Torch may already define protobuf targets, including a dummy alias in bundled
# builds. Preserve that dependency baseline; telemetry must not add or alter it.
find_package(Torch REQUIRED)
set(_jyppx_torch_protobuf_targets protobuf::libprotobuf protobuf::libprotobuf-lite
    protobuf::protoc)
set(_jyppx_torch_properties NAME TYPE IMPORTED IMPORTED_GLOBAL ALIASED_TARGET
    ALIAS_GLOBAL SOURCE_DIR BINARY_DIR SOURCES INCLUDE_DIRECTORIES
    COMPILE_DEFINITIONS COMPILE_OPTIONS LINK_LIBRARIES LINK_OPTIONS
    INTERFACE_INCLUDE_DIRECTORIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
    INTERFACE_COMPILE_DEFINITIONS INTERFACE_COMPILE_OPTIONS
    INTERFACE_COMPILE_FEATURES INTERFACE_LINK_LIBRARIES INTERFACE_LINK_OPTIONS
    INTERFACE_LINK_DIRECTORIES INTERFACE_SOURCES IMPORTED_CONFIGURATIONS
    IMPORTED_LOCATION IMPORTED_IMPLIB IMPORTED_LIBNAME IMPORTED_SONAME
    IMPORTED_NO_SONAME IMPORTED_LINK_INTERFACE_LIBRARIES
    IMPORTED_LINK_DEPENDENT_LIBRARIES)
set(_jyppx_torch_configurations DEBUG RELEASE RELWITHDEBINFO MINSIZEREL
    ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE})
foreach(_jyppx_target IN LISTS _jyppx_torch_protobuf_targets)
    if(TARGET ${_jyppx_target})
        get_target_property(_jyppx_configs ${_jyppx_target} IMPORTED_CONFIGURATIONS)
        if(_jyppx_configs)
            list(APPEND _jyppx_torch_configurations ${_jyppx_configs})
        endif()
    endif()
endforeach()
list(REMOVE_DUPLICATES _jyppx_torch_configurations)
foreach(_jyppx_config IN LISTS _jyppx_torch_configurations)
    string(TOUPPER "${_jyppx_config}" _jyppx_config)
    foreach(_jyppx_property IN ITEMS IMPORTED_LOCATION IMPORTED_IMPLIB
            IMPORTED_LIBNAME IMPORTED_SONAME IMPORTED_NO_SONAME
            IMPORTED_LINK_INTERFACE_LIBRARIES IMPORTED_LINK_DEPENDENT_LIBRARIES
            MAP_IMPORTED_CONFIG)
        list(APPEND _jyppx_torch_properties "${_jyppx_property}_${_jyppx_config}")
    endforeach()
endforeach()
function(_jyppx_torch_dependency_state _jyppx_output)
    set(_jyppx_state "")
    foreach(_jyppx_target IN LISTS _jyppx_torch_protobuf_targets)
        if(TARGET ${_jyppx_target})
            string(APPEND _jyppx_state "${_jyppx_target}=present\n")
            foreach(_jyppx_property IN LISTS _jyppx_torch_properties)
                get_target_property(_jyppx_value ${_jyppx_target} ${_jyppx_property})
                get_property(_jyppx_set TARGET ${_jyppx_target}
                    PROPERTY ${_jyppx_property} SET)
                string(LENGTH "${_jyppx_value}" _jyppx_length)
                string(APPEND _jyppx_state
                    "${_jyppx_property}:${_jyppx_set}:${_jyppx_length}:${_jyppx_value}\n")
            endforeach()
        else()
            string(APPEND _jyppx_state "${_jyppx_target}=absent\n")
        endif()
    endforeach()
    set(${_jyppx_output} "${_jyppx_state}" PARENT_SCOPE)
endfunction()
_jyppx_torch_dependency_state(_jyppx_torch_before)
find_package(jyppx CONFIG QUIET COMPONENTS telemetry_native)
_jyppx_torch_dependency_state(_jyppx_torch_after)
if(NOT "${_jyppx_torch_before}" STREQUAL "${_jyppx_torch_after}")
    message(FATAL_ERROR "Installed jyppx changed the Torch protobuf dependency baseline")
endif()
foreach(_jyppx_private_target IN ITEMS nanopb::protobuf-nanopb-static
        jyppx_native_telemetry_codec jyppx::liuding::telemetry_codec)
    if(TARGET ${_jyppx_private_target})
        message(FATAL_ERROR
            "Installed native telemetry leaked private target: ${_jyppx_private_target}")
    endif()
endforeach()
if(NOT JYPPX_EXPECT_NATIVE_TELEMETRY)
    if(jyppx_FOUND OR jyppx_telemetry_native_FOUND OR TARGET jyppx::liuding::telemetry_native)
        message(FATAL_ERROR "Disabled installation exposed native telemetry")
    endif()
    get_filename_component(_jyppx_prefix "${jyppx_DIR}/../../.." ABSOLUTE)
    if(EXISTS "${_jyppx_prefix}/include/jyppx/liuding/telemetry_native/tensorboard.hpp")
        message(FATAL_ERROR "Disabled installation retained a native telemetry header")
    endif()
    add_custom_target(native_telemetry_off_consumer ALL COMMAND ${CMAKE_COMMAND} -E true)
    enable_testing()
    add_test(NAME Runs COMMAND ${CMAKE_COMMAND} -E true)
    return()
endif()
if(NOT jyppx_FOUND OR NOT jyppx_telemetry_native_FOUND OR
   NOT TARGET jyppx::liuding::telemetry_native)
    message(FATAL_ERROR "Installed package did not satisfy telemetry_native")
endif()
set(_jyppx_private_pattern
    "protobuf::|nanopb|PB_FIELD_32BIT|PB_VALIDATE_UTF8|PB_MESSAGE_NESTING_MAX|telemetry_proto|jyppx_native_telemetry_codec|jyppx::liuding::telemetry_codec")
foreach(_jyppx_property IN ITEMS INTERFACE_INCLUDE_DIRECTORIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES INTERFACE_COMPILE_DEFINITIONS
        INTERFACE_COMPILE_OPTIONS INTERFACE_COMPILE_FEATURES
        INTERFACE_LINK_LIBRARIES INTERFACE_LINK_OPTIONS INTERFACE_LINK_DIRECTORIES
        INTERFACE_SOURCES)
    get_target_property(_jyppx_interface jyppx::liuding::telemetry_native ${_jyppx_property})
    if(_jyppx_interface MATCHES "${_jyppx_private_pattern}")
        message(FATAL_ERROR
            "Installed native telemetry leaked private SDK configuration in ${_jyppx_property}")
    endif()
endforeach()
file(GLOB _jyppx_export_files "${jyppx_DIR}/jyppxTargets*.cmake")
if(NOT _jyppx_export_files)
    message(FATAL_ERROR "Installed package is missing its target exports")
endif()
foreach(_jyppx_export_file IN LISTS _jyppx_export_files)
    file(READ "${_jyppx_export_file}" _jyppx_export_text)
    if(_jyppx_export_text MATCHES "${_jyppx_private_pattern}")
        message(FATAL_ERROR "Installed package leaked private SDK in ${_jyppx_export_file}")
    endif()
endforeach()
add_executable(jyppx_liuding_native_telemetry_consumer main.cpp)
target_link_libraries(jyppx_liuding_native_telemetry_consumer PRIVATE jyppx::liuding::telemetry_native)
if(WIN32)
    if(NOT JYPPX_TORCH_RUNTIME_DIR)
        message(FATAL_ERROR "Windows telemetry consumer requires JYPPX_TORCH_RUNTIME_DIR")
    endif()
    if(JYPPX_RUNTIME_STAGER AND EXISTS "${JYPPX_RUNTIME_STAGER}")
        add_custom_command(TARGET jyppx_liuding_native_telemetry_consumer POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                "-DJYPPX_RUNTIME_SOURCE_DIR=${JYPPX_TORCH_RUNTIME_DIR}"
                "-DJYPPX_RUNTIME_OUTPUT_DIR=$<TARGET_FILE_DIR:jyppx_liuding_native_telemetry_consumer>"
                -P "${JYPPX_RUNTIME_STAGER}" VERBATIM)
    else()
        file(GLOB JYPPX_CONSUMER_RUNTIME_DLLS "${JYPPX_TORCH_RUNTIME_DIR}/*.dll")
        add_custom_command(TARGET jyppx_liuding_native_telemetry_consumer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${JYPPX_CONSUMER_RUNTIME_DLLS}
                "$<TARGET_FILE_DIR:jyppx_liuding_native_telemetry_consumer>" VERBATIM)
    endif()
endif()
enable_testing()
add_test(NAME Runs COMMAND jyppx_liuding_native_telemetry_consumer)
set_tests_properties(Runs PROPERTIES TIMEOUT 60)
