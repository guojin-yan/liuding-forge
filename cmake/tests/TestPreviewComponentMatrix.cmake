# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0.

cmake_minimum_required(VERSION 3.18)
foreach(required IN ITEMS JYPPX_SOURCE_DIR JYPPX_TEST_ROOT
        JYPPX_CONSUMER_SOURCE_DIR JYPPX_TORCH_DIR JYPPX_CMAKE_GENERATOR)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "Preview component matrix requires ${required}")
    endif()
endforeach()

get_filename_component(source_dir "${JYPPX_SOURCE_DIR}" REALPATH)
get_filename_component(test_root "${JYPPX_TEST_ROOT}" ABSOLUTE)
get_filename_component(consumer_source_dir "${JYPPX_CONSUMER_SOURCE_DIR}" REALPATH)
file(RELATIVE_PATH matrix_relative "${source_dir}/out" "${test_root}")
if(matrix_relative MATCHES "^\\.\\." OR IS_ABSOLUTE "${matrix_relative}")
    message(FATAL_ERROR "Preview matrix staging must stay inside source out/")
endif()
file(MAKE_DIRECTORY "${test_root}")
# Reconfigure one persistent framework tree; only install staging is refreshed.
set(build_dir "${test_root}/build")
set(install_dir "${test_root}/install")
set(generator_args -G "${JYPPX_CMAKE_GENERATOR}")
if(JYPPX_CMAKE_GENERATOR_PLATFORM)
    list(APPEND generator_args -A "${JYPPX_CMAKE_GENERATOR_PLATFORM}")
endif()
if(JYPPX_CMAKE_GENERATOR_TOOLSET)
    list(APPEND generator_args -T "${JYPPX_CMAKE_GENERATOR_TOOLSET}")
endif()
set(config_args)
set(build_args --parallel 2)
set(install_args)
set(test_args)
if(JYPPX_TEST_CONFIG)
    list(APPEND config_args "-DCMAKE_BUILD_TYPE=${JYPPX_TEST_CONFIG}")
    list(APPEND build_args --config "${JYPPX_TEST_CONFIG}")
    list(APPEND install_args --config "${JYPPX_TEST_CONFIG}")
    list(APPEND test_args -C "${JYPPX_TEST_CONFIG}")
endif()
foreach(setting IN ITEMS CUDA_STANDARD CUDA_ARCHITECTURES)
    if(JYPPX_${setting})
        list(APPEND config_args "-DCMAKE_${setting}=${JYPPX_${setting}}")
    endif()
endforeach()
if(JYPPX_TORCH_CUDA_ARCH_LIST)
    list(APPEND config_args "-DTORCH_CUDA_ARCH_LIST=${JYPPX_TORCH_CUDA_ARCH_LIST}")
endif()
if(JYPPX_CUDA_COMPILER)
    list(APPEND config_args "-DCMAKE_CUDA_COMPILER=${JYPPX_CUDA_COMPILER}")
    list(APPEND config_args "-DCUDA_NVCC_EXECUTABLE=${JYPPX_CUDA_COMPILER}")
endif()
if(JYPPX_CUDA_TOOLKIT_ROOT)
    list(APPEND config_args "-DCUDAToolkit_ROOT=${JYPPX_CUDA_TOOLKIT_ROOT}")
    list(APPEND config_args "-DCUDA_TOOLKIT_ROOT_DIR=${JYPPX_CUDA_TOOLKIT_ROOT}")
    if(NOT JYPPX_CUDA_STANDARD)
        list(APPEND config_args "-DCMAKE_CUDA_STANDARD=17")
    endif()
endif()
if(WIN32 AND JYPPX_TORCH_RUNTIME_DIR)
    set(ENV{PATH} "${JYPPX_TORCH_RUNTIME_DIR};$ENV{PATH}")
endif()

set(optional_components core interop data nn optim training experiment
    models quantization serving)
set(preview_components models quantization serving)
set(common_options
    -DJYPPX_BUILD_TESTS=OFF -DJYPPX_BUILD_EXAMPLES=OFF
    -DJYPPX_BUILD_TOOLS=OFF -DJYPPX_BUILD_BENCHMARKS=OFF
    -DJYPPX_BUILD_FUZZERS=OFF -DJYPPX_BUILD_DOCS=OFF
    -DJYPPX_ENABLE_CLANG_TIDY=OFF -DJYPPX_ENABLE_FORMAT_CHECK=OFF
    -DJYPPX_ENABLE_WARNINGS_AS_ERRORS=OFF
    -DJYPPX_MODELS_REFERENCE_DIR=
)

# Defined global symbols prove disabled archives do not hide Preview code.
set(symbol_tool "${JYPPX_NM}")
if(NOT EXISTS "${symbol_tool}")
    if(WIN32)
        get_filename_component(linker_dir "${JYPPX_LINKER}" DIRECTORY)
        find_program(symbol_tool_found NAMES dumpbin llvm-nm nm
            HINTS "${linker_dir}")
    else()
        find_program(symbol_tool_found NAMES llvm-nm nm)
    endif()
    set(symbol_tool "${symbol_tool_found}")
endif()
if(NOT EXISTS "${symbol_tool}")
    message(FATAL_ERROR "Preview symbol isolation requires dumpbin or nm")
endif()
get_filename_component(symbol_tool_name "${symbol_tool}" NAME)
string(TOLOWER "${symbol_tool_name}" symbol_tool_name)
if(symbol_tool_name MATCHES "dumpbin")
    set(symbol_args /symbols)
    set(symbol_mode dumpbin)
else()
    set(symbol_args -g --defined-only -C)
    set(symbol_mode nm)
endif()

function(check_preview_symbols profile)
    file(GLOB installed_archives "${install_dir}/lib/*jyppx_*.a"
         "${install_dir}/lib/*jyppx_*.lib")
    if(NOT installed_archives)
        message(FATAL_ERROR "No installed archives for symbol check: ${profile}")
    endif()
    foreach(component IN LISTS preview_components)
        set(found_${component} FALSE)
    endforeach()
    foreach(archive IN LISTS installed_archives)
        execute_process(COMMAND "${symbol_tool}" ${symbol_args} "${archive}"
            RESULT_VARIABLE symbol_result OUTPUT_VARIABLE symbols
            ERROR_VARIABLE symbol_error)
        if(NOT symbol_result EQUAL 0)
            message(FATAL_ERROR "Cannot inspect ${archive}: ${symbol_error}")
        endif()
        if(symbol_mode STREQUAL "dumpbin")
            # Exclude UNDEF references: only defined external symbols count.
            string(REGEX MATCHALL "[^\n]*SECT[0-9A-Fa-f]+[^\n]*External[^\n]*"
                   defined_symbols "${symbols}")
            list(JOIN defined_symbols "\n" symbols)
        endif()
        foreach(component IN LISTS preview_components)
            if(symbols MATCHES "jyppx::${component}::|@${component}@jyppx@")
                if(NOT component IN_LIST profile_components)
                    message(FATAL_ERROR
                        "Disabled ${component} symbols in ${profile}: ${archive}")
                endif()
                set(found_${component} TRUE)
            endif()
        endforeach()
    endforeach()
    foreach(component IN LISTS preview_components)
        if(component IN_LIST profile_components AND NOT found_${component})
            message(FATAL_ERROR
                "Enabled ${component} has no installed definitions in ${profile}")
        endif()
    endforeach()
endfunction()

# Begin and end with the same core-only profile to exercise ON -> OFF cleanup.
foreach(profile IN ITEMS core_before quantization models serving all_previews core_after)
    set(profile_components)
    set(consumer_components foundation all)
    if(profile MATCHES "^core_")
        set(profile_components core)
        list(APPEND consumer_components core)
    elseif(profile STREQUAL "quantization")
        set(profile_components quantization)
        list(APPEND consumer_components quantization)
    elseif(profile STREQUAL "models" OR profile STREQUAL "serving")
        set(profile_components core interop nn ${profile})
        list(APPEND consumer_components ${profile})
    elseif(profile STREQUAL "all_previews")
        set(profile_components core interop nn models quantization serving)
        list(APPEND consumer_components models quantization serving)
    endif()
    set(component_options)
    set(absent_components sdk)
    foreach(component IN LISTS optional_components)
        string(TOUPPER "${component}" component_upper)
        if(component IN_LIST profile_components)
            list(APPEND component_options "-DJYPPX_BUILD_${component_upper}=ON")
        else()
            list(APPEND component_options "-DJYPPX_BUILD_${component_upper}=OFF")
            list(APPEND absent_components "${component}")
        endif()
    endforeach()
    file(REMOVE_RECURSE "${install_dir}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -S "${source_dir}" -B "${build_dir}"
            ${generator_args} "-DTorch_DIR=${JYPPX_TORCH_DIR}"
            "-DCMAKE_INSTALL_PREFIX=${install_dir}"
            ${common_options} ${component_options} ${config_args}
        RESULT_VARIABLE configure_result)
    if(NOT configure_result EQUAL 0)
        message(FATAL_ERROR "Configuring Preview profile failed: ${profile}")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" --build "${build_dir}" ${build_args}
        RESULT_VARIABLE build_result)
    if(NOT build_result EQUAL 0)
        message(FATAL_ERROR "Building Preview profile failed: ${profile}")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" --install "${build_dir}"
            --prefix "${install_dir}" ${install_args}
        RESULT_VARIABLE install_result)
    if(NOT install_result EQUAL 0)
        message(FATAL_ERROR "Installing Preview profile failed: ${profile}")
    endif()
    foreach(component IN LISTS optional_components)
        file(GLOB component_libraries
            "${install_dir}/lib/*jyppx_${component}*")
        if(component IN_LIST profile_components)
            if(NOT EXISTS "${install_dir}/include/jyppx/${component}" OR
               NOT component_libraries)
                message(FATAL_ERROR "Enabled ${component} install is incomplete")
            endif()
        else()
            if(EXISTS "${install_dir}/include/jyppx/${component}" OR
               component_libraries)
                message(FATAL_ERROR "Disabled ${component} surface remains installed")
            endif()
        endif()
    endforeach()
    if(EXISTS "${install_dir}/include/jyppx/jyppx.hpp")
        message(FATAL_ERROR "Stable SDK umbrella leaked into partial Preview profile")
    endif()
    if(NOT core IN_LIST profile_components AND
       EXISTS "${install_dir}/include/jyppx/utils")
        message(FATAL_ERROR "Core utils remain with tensor core disabled")
    endif()
    check_preview_symbols("${profile}")

    set(expected_components foundation all ${profile_components})
    list(JOIN expected_components "," expected_components_arg)
    list(JOIN absent_components "," absent_components_arg)
    foreach(component IN LISTS consumer_components)
        set(consumer_build_dir "${test_root}/consumers/${component}")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env "CMAKE_PREFIX_PATH="
                "${CMAKE_COMMAND}" -S "${consumer_source_dir}"
                -B "${consumer_build_dir}" ${generator_args}
                "-UTorch_*" "-UTORCH_*" "-UCaffe2_*" "-Uc10_*" "-Ukineto_*"
                "-DCMAKE_PREFIX_PATH=${install_dir}"
                "-DTorch_DIR=${JYPPX_TORCH_DIR}"
                "-DJYPPX_TORCH_RUNTIME_DIR=${JYPPX_TORCH_RUNTIME_DIR}"
                "-DJYPPX_RUNTIME_STAGER=${source_dir}/cmake/StageTorchRuntime.cmake"
                "-DJYPPX_CONSUMER_COMPONENT=${component}"
                "-DJYPPX_EXPECTED_COMPONENTS=${expected_components_arg}"
                "-DJYPPX_ABSENT_COMPONENTS=${absent_components_arg}" ${config_args}
            RESULT_VARIABLE consumer_configure_result)
        if(NOT consumer_configure_result EQUAL 0)
            message(FATAL_ERROR "Configuring ${profile}/${component} consumer failed")
        endif()
        execute_process(
            COMMAND "${CMAKE_COMMAND}" --build "${consumer_build_dir}" ${build_args}
            RESULT_VARIABLE consumer_build_result)
        if(NOT consumer_build_result EQUAL 0)
            message(FATAL_ERROR "Building ${profile}/${component} consumer failed")
        endif()
        execute_process(
            COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${consumer_build_dir}"
                ${test_args} --output-on-failure
            RESULT_VARIABLE consumer_test_result)
        if(NOT consumer_test_result EQUAL 0)
            message(FATAL_ERROR "Running ${profile}/${component} consumer failed")
        endif()
    endforeach()

    foreach(component IN LISTS preview_components)
        if(component IN_LIST profile_components)
            continue()
        endif()
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env "CMAKE_PREFIX_PATH="
                "${CMAKE_COMMAND}"
                -S "${source_dir}/cmake/tests/absent_component_consumer"
                -B "${test_root}/absent/${component}" ${generator_args}
                "-UTorch_*" "-UTORCH_*" "-UCaffe2_*" "-Uc10_*" "-Ukineto_*"
                "-DCMAKE_PREFIX_PATH=${install_dir}"
                "-DTorch_DIR=${JYPPX_TORCH_DIR}"
                "-DJYPPX_ABSENT_COMPONENT=${component}" ${config_args}
            RESULT_VARIABLE absent_result OUTPUT_VARIABLE absent_output
            ERROR_VARIABLE absent_error)
        string(CONCAT absent_log "${absent_output}" "${absent_error}")
        string(REGEX REPLACE "[ \t\r\n]+" " " absent_log "${absent_log}")
        if(absent_result EQUAL 0 OR NOT absent_log MATCHES
           "Requested jyppx component was disabled at configure time: ${component}")
            message(FATAL_ERROR
                "Required disabled package check failed for ${component}: ${absent_log}")
        endif()
    endforeach()
endforeach()

# Reject the exact missing dependency; an unrelated configure failure is a failure.
foreach(component IN ITEMS MODELS SERVING)
    foreach(missing IN ITEMS CORE INTEROP NN)
        set(invalid_options ${common_options})
        foreach(optional IN LISTS optional_components)
            string(TOUPPER "${optional}" optional_upper)
            if(optional_upper STREQUAL "CORE" OR optional_upper STREQUAL "INTEROP" OR
               optional_upper STREQUAL "NN" OR optional_upper STREQUAL component)
                list(APPEND invalid_options "-DJYPPX_BUILD_${optional_upper}=ON")
            else()
                list(APPEND invalid_options "-DJYPPX_BUILD_${optional_upper}=OFF")
            endif()
        endforeach()
        list(APPEND invalid_options "-DJYPPX_BUILD_${missing}=OFF")
        # NN has its own CORE edge. Disable it for CORE rejection so the
        # diagnostic belongs to the Preview adapter being exercised.
        if(missing STREQUAL "CORE")
            list(APPEND invalid_options -DJYPPX_BUILD_NN=OFF)
        endif()
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -S "${source_dir}"
                -B "${test_root}/invalid" ${generator_args}
                "-DTorch_DIR=${JYPPX_TORCH_DIR}" ${invalid_options} ${config_args}
            RESULT_VARIABLE invalid_result OUTPUT_VARIABLE invalid_output
            ERROR_VARIABLE invalid_error)
        string(CONCAT invalid_log "${invalid_output}" "${invalid_error}")
        string(REGEX REPLACE "[ \t\r\n]+" " " invalid_log "${invalid_log}")
        set(expected_error "JYPPX_BUILD_${component}=ON requires JYPPX_BUILD_${missing}=ON")
        string(FIND "${invalid_log}" "${expected_error}" expected_position)
        if(invalid_result EQUAL 0 OR expected_position EQUAL -1)
            message(FATAL_ERROR
                "Dependency rejection failed: ${expected_error}; ${invalid_log}")
        endif()
    endforeach()
endforeach()
message(STATUS "Preview matrix passed with one persistent framework build directory")
