cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED JYPPX_SOURCE_DIR OR NOT DEFINED JYPPX_TEST_ROOT OR
   NOT DEFINED JYPPX_CONSUMER_SOURCE_DIR OR NOT DEFINED JYPPX_TORCH_DIR OR
   NOT DEFINED JYPPX_CMAKE_GENERATOR)
    message(FATAL_ERROR "Optional component matrix is missing required paths")
endif()

get_filename_component(source_dir "${JYPPX_SOURCE_DIR}" REALPATH)
get_filename_component(test_root "${JYPPX_TEST_ROOT}" ABSOLUTE)
get_filename_component(consumer_source_dir
                       "${JYPPX_CONSUMER_SOURCE_DIR}" REALPATH)
file(MAKE_DIRECTORY "${test_root}")

set(generator_args -G "${JYPPX_CMAKE_GENERATOR}")
if(JYPPX_CMAKE_GENERATOR_PLATFORM)
    list(APPEND generator_args -A "${JYPPX_CMAKE_GENERATOR_PLATFORM}")
endif()
if(JYPPX_CMAKE_GENERATOR_TOOLSET)
    list(APPEND generator_args -T "${JYPPX_CMAKE_GENERATOR_TOOLSET}")
endif()

set(config_args)
set(build_args --parallel 2)
set(test_args)
if(JYPPX_TEST_CONFIG)
    list(APPEND config_args "-DCMAKE_BUILD_TYPE=${JYPPX_TEST_CONFIG}")
    list(APPEND build_args --config "${JYPPX_TEST_CONFIG}")
    list(APPEND test_args -C "${JYPPX_TEST_CONFIG}")
endif()
if(JYPPX_CUDA_STANDARD)
    list(APPEND config_args "-DCMAKE_CUDA_STANDARD=${JYPPX_CUDA_STANDARD}")
endif()
if(JYPPX_CUDA_ARCHITECTURES)
    list(APPEND config_args
         "-DCMAKE_CUDA_ARCHITECTURES=${JYPPX_CUDA_ARCHITECTURES}")
endif()
if(JYPPX_TORCH_CUDA_ARCH_LIST)
    list(APPEND config_args
         "-DTORCH_CUDA_ARCH_LIST=${JYPPX_TORCH_CUDA_ARCH_LIST}")
endif()

if(WIN32 AND JYPPX_TORCH_RUNTIME_DIR)
    set(ENV{PATH} "${JYPPX_TORCH_RUNTIME_DIR};$ENV{PATH}")
endif()

set(optional_components core interop data nn optim training experiment)
set(profiles foundation core interop data nn optim training experiment)

foreach(profile IN LISTS profiles)
    set(profile_components)
    if(profile STREQUAL "core")
        set(profile_components core)
    elseif(profile STREQUAL "interop")
        set(profile_components interop)
    elseif(profile STREQUAL "data")
        set(profile_components core data)
    elseif(profile STREQUAL "nn")
        set(profile_components core nn)
    elseif(profile STREQUAL "optim")
        set(profile_components core optim)
    elseif(profile STREQUAL "training")
        set(profile_components core data nn optim training)
    elseif(profile STREQUAL "experiment")
        set(profile_components core experiment)
    endif()

    set(component_options)
    foreach(component IN LISTS optional_components)
        string(TOUPPER "${component}" component_upper)
        if(component IN_LIST profile_components)
            list(APPEND component_options "-DJYPPX_BUILD_${component_upper}=ON")
        else()
            list(APPEND component_options "-DJYPPX_BUILD_${component_upper}=OFF")
        endif()
    endforeach()

    set(profile_root "${test_root}/${profile}")
    set(build_dir "${profile_root}/build")
    set(install_dir "${profile_root}/install")
    set(consumer_build_dir "${profile_root}/consumer")
    file(REMOVE_RECURSE "${install_dir}")

    execute_process(
        COMMAND "${CMAKE_COMMAND}" -S "${source_dir}" -B "${build_dir}"
                ${generator_args}
                "-DTorch_DIR=${JYPPX_TORCH_DIR}"
                "-DCMAKE_INSTALL_PREFIX=${install_dir}"
                -DJYPPX_BUILD_TESTS=OFF
                -DJYPPX_BUILD_EXAMPLES=OFF
                -DJYPPX_BUILD_BENCHMARKS=OFF
                -DJYPPX_BUILD_FUZZERS=OFF
                -DJYPPX_BUILD_DOCS=OFF
                -DJYPPX_ENABLE_CLANG_TIDY=OFF
                -DJYPPX_ENABLE_FORMAT_CHECK=OFF
                -DJYPPX_ENABLE_WARNINGS_AS_ERRORS=OFF
                ${component_options} ${config_args}
        RESULT_VARIABLE configure_result
    )
    if(NOT configure_result EQUAL 0)
        message(FATAL_ERROR "Configuring ${profile} component profile failed")
    endif()

    execute_process(
        COMMAND "${CMAKE_COMMAND}" --build "${build_dir}" --target install
                ${build_args}
        RESULT_VARIABLE build_result
    )
    if(NOT build_result EQUAL 0)
        message(FATAL_ERROR "Building ${profile} component profile failed")
    endif()

    set(expected_components foundation all ${profile_components})
    set(absent_components sdk)
    foreach(component IN LISTS optional_components)
        if(NOT component IN_LIST profile_components)
            list(APPEND absent_components "${component}")
            if(EXISTS "${install_dir}/include/jyppx/${component}")
                message(FATAL_ERROR
                    "Disabled ${component} headers remain in ${profile} profile")
            endif()
            file(GLOB disabled_libraries
                 "${install_dir}/lib/*jyppx_${component}*")
            if(disabled_libraries)
                message(FATAL_ERROR
                    "Disabled ${component} library remains in ${profile} profile")
            endif()
        endif()
    endforeach()
    if(interop IN_LIST profile_components AND NOT nn IN_LIST profile_components)
        foreach(adapter IN ITEMS model_weight_import torchscript_compatibility)
            if(EXISTS "${install_dir}/include/jyppx/interop/${adapter}.hpp")
                message(FATAL_ERROR
                    "NN adapter header remains installed with nn disabled in ${profile} profile")
            endif()
        endforeach()
    endif()
    if(NOT core IN_LIST profile_components AND
       EXISTS "${install_dir}/include/jyppx/utils")
        message(FATAL_ERROR "Core-owned utils headers remain with core disabled")
    endif()
    if(EXISTS "${install_dir}/include/jyppx/jyppx.hpp")
        message(FATAL_ERROR "SDK umbrella header remains in partial profile")
    endif()

    list(JOIN expected_components "," expected_components_arg)
    list(JOIN absent_components "," absent_components_arg)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env "CMAKE_PREFIX_PATH="
                "${CMAKE_COMMAND}"
                -S "${consumer_source_dir}"
                -B "${consumer_build_dir}"
                "-UTorch_*" "-UTORCH_*" "-UCaffe2_*" "-Uc10_*" "-Ukineto_*"
                "-DCMAKE_PREFIX_PATH=${install_dir}"
                "-DTorch_DIR=${JYPPX_TORCH_DIR}"
                "-DJYPPX_TORCH_RUNTIME_DIR=${JYPPX_TORCH_RUNTIME_DIR}"
                "-DJYPPX_CONSUMER_COMPONENT=${profile}"
                "-DJYPPX_EXPECTED_COMPONENTS=${expected_components_arg}"
                "-DJYPPX_ABSENT_COMPONENTS=${absent_components_arg}"
                ${config_args}
        RESULT_VARIABLE consumer_configure_result
    )
    if(NOT consumer_configure_result EQUAL 0)
        message(FATAL_ERROR "Configuring ${profile} profile consumer failed")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" --build "${consumer_build_dir}" ${build_args}
        RESULT_VARIABLE consumer_build_result
    )
    if(NOT consumer_build_result EQUAL 0)
        message(FATAL_ERROR "Building ${profile} profile consumer failed")
    endif()
    execute_process(
        COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${consumer_build_dir}"
                ${test_args} --output-on-failure
        RESULT_VARIABLE consumer_test_result
    )
    if(NOT consumer_test_result EQUAL 0)
        message(FATAL_ERROR "Running ${profile} profile consumer failed")
    endif()
endforeach()

set(invalid_cases
    core_off_data_on
    core_off_nn_on
    core_off_optim_on
    core_off_experiment_on
    training_without_data
    training_without_nn
    training_without_optim
)
foreach(invalid_case IN LISTS invalid_cases)
    set(invalid_options
        -DJYPPX_BUILD_CORE=ON
        -DJYPPX_BUILD_INTEROP=OFF
        -DJYPPX_BUILD_DATA=OFF
        -DJYPPX_BUILD_NN=OFF
        -DJYPPX_BUILD_OPTIM=OFF
        -DJYPPX_BUILD_TRAINING=OFF
        -DJYPPX_BUILD_EXPERIMENT=OFF
    )
    if(invalid_case MATCHES "^core_off_([a-z]+)_on$")
        string(TOUPPER "${CMAKE_MATCH_1}" dependent_component)
        list(APPEND invalid_options
             -DJYPPX_BUILD_CORE=OFF
             "-DJYPPX_BUILD_${dependent_component}=ON")
    elseif(invalid_case MATCHES "^training_without_([a-z]+)$")
        string(TOUPPER "${CMAKE_MATCH_1}" missing_component)
        list(APPEND invalid_options
             -DJYPPX_BUILD_DATA=ON
             -DJYPPX_BUILD_NN=ON
             -DJYPPX_BUILD_OPTIM=ON
             -DJYPPX_BUILD_TRAINING=ON
             "-DJYPPX_BUILD_${missing_component}=OFF")
    else()
        message(FATAL_ERROR "Unknown invalid component case: ${invalid_case}")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -S "${source_dir}"
                -B "${test_root}/invalid/${invalid_case}"
                ${generator_args}
                "-DTorch_DIR=${JYPPX_TORCH_DIR}"
                -DJYPPX_BUILD_TESTS=OFF -DJYPPX_BUILD_EXAMPLES=OFF
                ${invalid_options} ${config_args}
        RESULT_VARIABLE invalid_result
    )
    if(invalid_result EQUAL 0)
        message(FATAL_ERROR "Invalid component profile was accepted: ${invalid_case}")
    endif()
endforeach()

message(STATUS "Optional component matrix passed: ${profiles}")
