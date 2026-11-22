# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.

# Test-only external engine: no product target or installed dependency.
function(jyppx_add_migraphx_interchange_tests)
    set(JYPPX_LIUDING_MIGRAPHX_TEST_ROOT "" CACHE PATH
        "Explicit native MIGraphX 2.18.0 SDK for optional interchange tests")
    set(JYPPX_LIUDING_MIGRAPHX_TEST_FIXTURE_DIR "" CACHE PATH
        "Original independently checked native ONNX Linear/ResNet18 fixtures")
    if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8 OR
       NOT (WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Linux") OR
       NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64)$")
        message(FATAL_ERROR "Native MIGraphX interchange tests require Windows/Linux x64")
    endif()
    foreach(variable IN ITEMS JYPPX_LIUDING_MIGRAPHX_TEST_ROOT JYPPX_LIUDING_MIGRAPHX_TEST_FIXTURE_DIR)
        if(NOT IS_ABSOLUTE "${${variable}}" OR NOT IS_DIRECTORY "${${variable}}")
            message(FATAL_ERROR "TEST_MIGRAPHX requires an explicit existing ${variable}")
        endif()
    endforeach()
    get_filename_component(_jyppx_mgx_root "${JYPPX_LIUDING_MIGRAPHX_TEST_ROOT}" REALPATH)
    foreach(header IN ITEMS migraphx.h version.h)
        if(NOT EXISTS "${_jyppx_mgx_root}/include/migraphx/${header}")
            message(FATAL_ERROR "Incomplete MIGraphX test SDK: ${header}")
        endif()
    endforeach()
    file(STRINGS "${_jyppx_mgx_root}/include/migraphx/version.h" _jyppx_mgx_version
        REGEX "^#define MIGRAPHX_VERSION_(MAJOR|MINOR|PATCH) ")
    foreach(part IN ITEMS MAJOR MINOR PATCH)
        if(part STREQUAL "MAJOR")
            set(expected 2)
        elseif(part STREQUAL "MINOR")
            set(expected 18)
        else()
            set(expected 0)
        endif()
        if(NOT "${_jyppx_mgx_version}" MATCHES "MIGRAPHX_VERSION_${part} +${expected}(;|$)")
            message(FATAL_ERROR "Native interchange tests require MIGraphX 2.18.0")
        endif()
    endforeach()
    foreach(model IN ITEMS linear resnet18)
        foreach(asset IN ITEMS model.onnx case-0-input.f32 case-0-output.f32 case-1-input.f32 case-1-output.f32)
            if(NOT EXISTS "${JYPPX_LIUDING_MIGRAPHX_TEST_FIXTURE_DIR}/${model}/${asset}")
                message(FATAL_ERROR "Missing fixed MIGraphX interchange fixture: ${model}/${asset}")
            endif()
        endforeach()
    endforeach()

    add_library(jyppx_test_migraphx_c SHARED IMPORTED)
    set_target_properties(jyppx_test_migraphx_c PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_jyppx_mgx_root}/include")
    if(WIN32)
        foreach(artifact IN ITEMS bin/migraphx_c.dll lib/migraphx_c.lib)
            if(NOT EXISTS "${_jyppx_mgx_root}/${artifact}")
                message(FATAL_ERROR "Incomplete native MIGraphX SDK: ${artifact}")
            endif()
        endforeach()
        set_target_properties(jyppx_test_migraphx_c PROPERTIES
            IMPORTED_LOCATION "${_jyppx_mgx_root}/bin/migraphx_c.dll"
            IMPORTED_IMPLIB "${_jyppx_mgx_root}/lib/migraphx_c.lib")
    else()
        if(NOT EXISTS "${_jyppx_mgx_root}/lib/libmigraphx_c.so")
            message(FATAL_ERROR "Incomplete native MIGraphX SDK: lib/libmigraphx_c.so")
        endif()
        set_target_properties(jyppx_test_migraphx_c PROPERTIES
            IMPORTED_LOCATION "${_jyppx_mgx_root}/lib/libmigraphx_c.so")
    endif()
    add_executable(jyppx_test_migraphx_interchange
        "${PROJECT_SOURCE_DIR}/tests/integration/deployment/migraphx_interchange.cpp")
    target_link_libraries(jyppx_test_migraphx_interchange PRIVATE
        jyppx::liuding::foundation jyppx_test_migraphx_c ${CMAKE_DL_LIBS})
    if(WIN32)
        target_compile_definitions(jyppx_test_migraphx_interchange PRIVATE NOMINMAX)
        target_link_libraries(jyppx_test_migraphx_interchange PRIVATE psapi)
    endif()
    jyppx_apply_instrumentation(jyppx_test_migraphx_interchange)
    foreach(model IN ITEMS linear resnet18)
        foreach(index RANGE 0 1)
            set(name "MIGraphXInterchange.${model}_${index}")
            # The activated SDK supplies transitive ROCm dependencies and device data.
            # The explicit native SDK precedes any system MIGraphX DLL on Windows.
            if(WIN32)
                add_test(NAME "${name}" COMMAND "${CMAKE_COMMAND}" -E env
                    "PATH=${_jyppx_mgx_root}/bin;$ENV{PATH}"
                    "MIGRAPHX_GPU_COMPILE_PARALLEL=1"
                    $<TARGET_FILE:jyppx_test_migraphx_interchange>
                    "${JYPPX_LIUDING_MIGRAPHX_TEST_FIXTURE_DIR}" "${model}" "${index}"
                    "${CMAKE_BINARY_DIR}/evidence/migraphx-interchange")
            else()
                add_test(NAME "${name}" COMMAND "${CMAKE_COMMAND}" -E env
                    "MIGRAPHX_GPU_COMPILE_PARALLEL=1"
                    $<TARGET_FILE:jyppx_test_migraphx_interchange>
                    "${JYPPX_LIUDING_MIGRAPHX_TEST_FIXTURE_DIR}" "${model}" "${index}"
                    "${CMAKE_BINARY_DIR}/evidence/migraphx-interchange")
            endif()
            set_tests_properties("${name}" PROPERTIES
                LABELS "deployment;onnx_export;migraphx;gpu;interchange"
                TIMEOUT 600 RUN_SERIAL TRUE)
        endforeach()
    endforeach()
endfunction()

jyppx_add_migraphx_interchange_tests()
