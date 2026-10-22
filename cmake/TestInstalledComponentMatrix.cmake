if(NOT DEFINED JYPPX_SOURCE_DIR OR NOT DEFINED JYPPX_BUILD_DIR OR
   NOT DEFINED JYPPX_CONSUMER_SOURCE_DIR OR NOT DEFINED JYPPX_TEST_ROOT OR
   NOT DEFINED JYPPX_TORCH_DIR)
    message(FATAL_ERROR "Installed component matrix is missing required paths")
endif()

get_filename_component(build_dir "${JYPPX_BUILD_DIR}" REALPATH)
get_filename_component(source_dir "${JYPPX_SOURCE_DIR}" REALPATH)
get_filename_component(consumer_source_dir
                       "${JYPPX_CONSUMER_SOURCE_DIR}" REALPATH)
get_filename_component(test_root "${JYPPX_TEST_ROOT}" ABSOLUTE)
file(RELATIVE_PATH test_root_relative "${build_dir}" "${test_root}")
if(test_root_relative MATCHES "^\\.\\." OR IS_ABSOLUTE "${test_root_relative}")
    message(FATAL_ERROR
        "Installed component matrix root must stay inside the build tree")
endif()

set(install_dir "${test_root}/i")
set(consumer_build_root "${test_root}/b")
# Refresh only the staging prefix so removed install files cannot survive.
# Per-component consumer build trees remain available for incremental builds.
file(REMOVE_RECURSE "${install_dir}")
file(MAKE_DIRECTORY "${install_dir}" "${consumer_build_root}")

set(config_args)
set(build_args)
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
if(JYPPX_CUDA_COMPILER)
    list(APPEND config_args
         "-DCMAKE_CUDA_COMPILER=${JYPPX_CUDA_COMPILER}")
endif()
if(JYPPX_CUDA_TOOLKIT_ROOT)
    list(APPEND config_args
         "-DCUDAToolkit_ROOT=${JYPPX_CUDA_TOOLKIT_ROOT}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${build_dir}"
            --prefix "${install_dir}" ${build_args}
    RESULT_VARIABLE install_result
)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "Installing JYPPX for the component matrix failed")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}"
            "-DJYPPX_SOURCE_DIR=${source_dir}"
            "-DJYPPX_INSTALL_DIR=${install_dir}"
            -P "${source_dir}/cmake/CheckLegacySurface.cmake"
    RESULT_VARIABLE legacy_surface_result
)
if(NOT legacy_surface_result EQUAL 0)
    message(FATAL_ERROR "Installed package retained a legacy surface")
endif()

if(WIN32 AND DEFINED JYPPX_TORCH_RUNTIME_DIR)
    set(ENV{PATH} "${JYPPX_TORCH_RUNTIME_DIR};$ENV{PATH}")
endif()

set(components foundation core interop data nn optim training experiment all sdk)
foreach(component IN LISTS components)
    set(consumer_build_dir "${consumer_build_root}/${component}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env "CMAKE_PREFIX_PATH="
                "${CMAKE_COMMAND}"
                -S "${consumer_source_dir}"
                -B "${consumer_build_dir}"
                "-UTorch_*"
                "-UTORCH_*"
                "-UCaffe2_*"
                "-Uc10_*"
                "-Ukineto_*"
                "-DCMAKE_PREFIX_PATH=${install_dir}"
                "-DTorch_DIR=${JYPPX_TORCH_DIR}"
                "-DJYPPX_TORCH_RUNTIME_DIR=${JYPPX_TORCH_RUNTIME_DIR}"
                "-DJYPPX_CONSUMER_COMPONENT=${component}"
                ${config_args}
        RESULT_VARIABLE configure_result
    )
    if(NOT configure_result EQUAL 0)
        message(FATAL_ERROR
            "Configuring installed ${component} consumer failed")
    endif()

    execute_process(
        COMMAND "${CMAKE_COMMAND}" --build "${consumer_build_dir}"
                ${build_args}
        RESULT_VARIABLE build_result
    )
    if(NOT build_result EQUAL 0)
        message(FATAL_ERROR "Building installed ${component} consumer failed")
    endif()

    execute_process(
        COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${consumer_build_dir}"
                ${test_args} --output-on-failure
        RESULT_VARIABLE test_result
    )
    if(NOT test_result EQUAL 0)
        message(FATAL_ERROR "Running installed ${component} consumer failed")
    endif()
endforeach()

message(STATUS "Installed component matrix passed: ${components}")
