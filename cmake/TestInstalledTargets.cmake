if(NOT DEFINED JYPPX_LIUDING_BUILD_DIR OR NOT DEFINED JYPPX_CONSUMER_SOURCE_DIR OR
   NOT DEFINED JYPPX_LIUDING_TEST_ROOT OR NOT DEFINED JYPPX_TORCH_DIR)
    message(FATAL_ERROR "Installed target test is missing required paths")
endif()

get_filename_component(build_dir "${JYPPX_LIUDING_BUILD_DIR}" REALPATH)
get_filename_component(consumer_source_dir "${JYPPX_CONSUMER_SOURCE_DIR}" REALPATH)
get_filename_component(test_root "${JYPPX_LIUDING_TEST_ROOT}" ABSOLUTE)
file(RELATIVE_PATH test_root_relative "${build_dir}" "${test_root}")
if(test_root_relative MATCHES "^\\.\\." OR IS_ABSOLUTE "${test_root_relative}")
    message(FATAL_ERROR "Installed target test root must stay inside the build tree")
endif()
file(MAKE_DIRECTORY "${test_root}")

set(install_dir "${test_root}/i")
set(consumer_build_dir "${test_root}/b")
# A fresh staging prefix prevents removed package files from surviving between
# runs, while the consumer build tree remains available for incremental builds.
file(REMOVE_RECURSE "${install_dir}")
file(MAKE_DIRECTORY "${install_dir}" "${consumer_build_dir}")
set(config_args)
set(generator_args)
if(JYPPX_CMAKE_GENERATOR)
    list(APPEND generator_args -G "${JYPPX_CMAKE_GENERATOR}")
endif()
if(JYPPX_CMAKE_GENERATOR_PLATFORM)
    list(APPEND generator_args -A "${JYPPX_CMAKE_GENERATOR_PLATFORM}")
endif()
if(JYPPX_CMAKE_GENERATOR_TOOLSET)
    list(APPEND generator_args -T "${JYPPX_CMAKE_GENERATOR_TOOLSET}")
endif()
set(build_args --parallel 2)
set(install_args)
set(test_args)
if(JYPPX_LIUDING_TEST_CONFIG)
    list(APPEND config_args "-DCMAKE_BUILD_TYPE=${JYPPX_LIUDING_TEST_CONFIG}")
    list(APPEND build_args --config "${JYPPX_LIUDING_TEST_CONFIG}")
    list(APPEND install_args --config "${JYPPX_LIUDING_TEST_CONFIG}")
    list(APPEND test_args -C "${JYPPX_LIUDING_TEST_CONFIG}")
endif()
# Propagate the caller's explicitly selected system HTTP SDK to this local
# installed-consumer test. Installed packages themselves never record it.
if(DEFINED JYPPX_LIUDING_TELEMETRY_HTTP_CURL_ROOT)
    set(_jyppx_consumer_http_root "${JYPPX_LIUDING_TELEMETRY_HTTP_CURL_ROOT}")
else()
    load_cache("${build_dir}" READ_WITH_PREFIX _jyppx_consumer_source_
        JYPPX_LIUDING_TELEMETRY_HTTP_CURL_ROOT)
    set(_jyppx_consumer_http_root
        "${_jyppx_consumer_source_JYPPX_LIUDING_TELEMETRY_HTTP_CURL_ROOT}")
endif()
if(_jyppx_consumer_http_root)
    list(APPEND config_args
        "-DJYPPX_LIUDING_TELEMETRY_HTTP_CURL_ROOT=${_jyppx_consumer_http_root}")
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
    list(APPEND config_args
         "-DCUDA_NVCC_EXECUTABLE=${JYPPX_CUDA_COMPILER}")
endif()
if(JYPPX_CUDA_TOOLKIT_ROOT)
    list(APPEND config_args
         "-DCUDAToolkit_ROOT=${JYPPX_CUDA_TOOLKIT_ROOT}")
    list(APPEND config_args
         "-DCUDA_TOOLKIT_ROOT_DIR=${JYPPX_CUDA_TOOLKIT_ROOT}")
    if(NOT JYPPX_CUDA_STANDARD)
        # CMake 3.22 cannot generate CUDA20 try-run projects used by
        # LibTorch's legacy FindCUDA architecture probe.
        list(APPEND config_args "-DCMAKE_CUDA_STANDARD=17")
    endif()
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${build_dir}"
            --prefix "${install_dir}" ${install_args}
    RESULT_VARIABLE install_result
)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "Installing JYPPX for the consumer failed")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "CMAKE_PREFIX_PATH="
            "${CMAKE_COMMAND}"
            -S "${consumer_source_dir}"
            -B "${consumer_build_dir}"
            ${generator_args}
            "-UTorch_*"
            "-UTORCH_*"
            "-UCaffe2_*"
            "-Uc10_*"
            "-Ukineto_*"
            "-DCMAKE_PREFIX_PATH=${install_dir}"
            "-DTorch_DIR=${JYPPX_TORCH_DIR}"
            "-DJYPPX_TORCH_RUNTIME_DIR=${JYPPX_TORCH_RUNTIME_DIR}"
            "-DJYPPX_RUNTIME_STAGER=${CMAKE_CURRENT_LIST_DIR}/StageTorchRuntime.cmake"
            ${config_args}
    RESULT_VARIABLE configure_result
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "Configuring the installed API consumer failed")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${consumer_build_dir}" ${build_args}
    RESULT_VARIABLE build_result
)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "Building the installed API consumer failed")
endif()

if(WIN32 AND DEFINED JYPPX_TORCH_RUNTIME_DIR)
    set(ENV{PATH} "${JYPPX_TORCH_RUNTIME_DIR};$ENV{PATH}")
endif()
execute_process(
    COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${consumer_build_dir}"
            ${test_args} --output-on-failure
    RESULT_VARIABLE test_result
)
if(NOT test_result EQUAL 0)
    message(FATAL_ERROR "Running the installed API consumer failed")
endif()
