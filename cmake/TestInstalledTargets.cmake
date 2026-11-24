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
include("${CMAKE_CURRENT_LIST_DIR}/ComponentMatrixSettings.cmake")
jyppx_component_build_arguments(build_args)

file(MAKE_DIRECTORY "${test_root}")

set(install_dir "${test_root}/i")
set(consumer_build_dir "${test_root}/b")
# A fresh staging prefix prevents removed package files from surviving between
# runs, while the consumer build tree remains available for incremental builds.
file(REMOVE_RECURSE "${install_dir}")
file(MAKE_DIRECTORY "${install_dir}" "${consumer_build_dir}")
set(config_args)
# Reuse the parent's opt-in HIP import closure for installed consumers that
# link the ROCm LibTorch backend. Explicit empty values clear a prior selection.
if(WIN32 AND consumer_source_dir MATCHES "/(migrated|vision_tasks|data_image|sdk)_consumer$")
    load_cache("${build_dir}" READ_WITH_PREFIX _jyppx_consumer_runtime_
        JYPPX_LIUDING_STAGE_ROCM_RUNTIME_DLLS)
    set(_jyppx_rocm_runtime_manifest)
    set(_jyppx_rocm_runtime_source)
    if(_jyppx_consumer_runtime_JYPPX_LIUDING_STAGE_ROCM_RUNTIME_DLLS)
        set(_jyppx_rocm_runtime_manifest
            "${build_dir}/CMakeFiles/jyppx-windows-rocm-runtime-files.txt")
        if(NOT EXISTS "${_jyppx_rocm_runtime_manifest}")
            message(FATAL_ERROR "Parent Windows HIP runtime manifest is missing")
        endif()
        file(STRINGS "${_jyppx_rocm_runtime_manifest}" _jyppx_rocm_runtime_files)
        if(NOT _jyppx_rocm_runtime_files)
            message(FATAL_ERROR "Parent Windows HIP runtime manifest is empty")
        endif()
        list(GET _jyppx_rocm_runtime_files 0 _jyppx_rocm_runtime_first)
        get_filename_component(_jyppx_rocm_runtime_source
            "${_jyppx_rocm_runtime_first}" DIRECTORY)
    endif()
    list(APPEND config_args
        "-DJYPPX_RUNTIME_EXTRA_DLL_LIST=${_jyppx_rocm_runtime_manifest}"
        "-DJYPPX_RUNTIME_EXTRA_DLL_SOURCE_DIR=${_jyppx_rocm_runtime_source}")
endif()
# Keep installed-consumer checks on the selected toolchain, including clang-cl
# for ROCm. A PATH default can silently select a different Windows compiler.
load_cache("${build_dir}" READ_WITH_PREFIX _jyppx_consumer_toolchain_
    CMAKE_C_COMPILER CMAKE_CXX_COMPILER)
foreach(language IN ITEMS C CXX)
    if(_jyppx_consumer_toolchain_CMAKE_${language}_COMPILER)
        list(APPEND config_args
            "-DCMAKE_${language}_COMPILER=${_jyppx_consumer_toolchain_CMAKE_${language}_COMPILER}")
    endif()
endforeach()
set(generator_args)
if(consumer_source_dir MATCHES "/deployment_consumer$")
    load_cache("${build_dir}" READ_WITH_PREFIX _jyppx_consumer_source_
        JYPPX_LIUDING_TEST_DIRECTML)
    list(APPEND config_args
        "-DJYPPX_TEST_DIRECTML=${_jyppx_consumer_source_JYPPX_LIUDING_TEST_DIRECTML}")
endif()
if(JYPPX_CMAKE_GENERATOR)
    list(APPEND generator_args -G "${JYPPX_CMAKE_GENERATOR}")
endif()
if(JYPPX_CMAKE_GENERATOR_PLATFORM)
    list(APPEND generator_args -A "${JYPPX_CMAKE_GENERATOR_PLATFORM}")
endif()
if(JYPPX_CMAKE_GENERATOR_TOOLSET)
    list(APPEND generator_args -T "${JYPPX_CMAKE_GENERATOR_TOOLSET}")
endif()
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
foreach(variable IN ITEMS JYPPX_LIUDING_MIGRAPHX_ROOT JYPPX_LIUDING_MIGRAPHX_HIP_ROOT)
    if(DEFINED ${variable})
        set(_jyppx_consumer_${variable} "${${variable}}")
    else()
        load_cache("${build_dir}" READ_WITH_PREFIX _jyppx_consumer_source_ "${variable}")
        set(_jyppx_consumer_${variable} "${_jyppx_consumer_source_${variable}}")
    endif()
    if(_jyppx_consumer_${variable})
        list(APPEND config_args "-D${variable}=${_jyppx_consumer_${variable}}")
    endif()
endforeach()
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
            "-UCMAKE_RUNTIME_OUTPUT_DIRECTORY*"
            "-UCMAKE_LIBRARY_OUTPUT_DIRECTORY*"
            "-UCMAKE_ARCHIVE_OUTPUT_DIRECTORY*"
            "-DCMAKE_PREFIX_PATH=${install_dir}"
            "-Djyppx_DIR:PATH=${install_dir}/lib/cmake/jyppx"
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

if(WIN32)
    set(_jyppx_consumer_runtime_paths)
    if(JYPPX_TORCH_RUNTIME_DIR)
        list(APPEND _jyppx_consumer_runtime_paths "${JYPPX_TORCH_RUNTIME_DIR}")
    endif()
    if(_jyppx_consumer_JYPPX_LIUDING_MIGRAPHX_ROOT)
        list(APPEND _jyppx_consumer_runtime_paths
            "${_jyppx_consumer_JYPPX_LIUDING_MIGRAPHX_ROOT}/bin")
    endif()
    if(_jyppx_consumer_JYPPX_LIUDING_MIGRAPHX_HIP_ROOT)
        list(APPEND _jyppx_consumer_runtime_paths
            "${_jyppx_consumer_JYPPX_LIUDING_MIGRAPHX_HIP_ROOT}/bin")
    endif()
    list(JOIN _jyppx_consumer_runtime_paths ";" _jyppx_consumer_runtime_path)
    if(_jyppx_consumer_runtime_path)
        set(ENV{PATH} "${_jyppx_consumer_runtime_path};$ENV{PATH}")
    endif()
endif()
execute_process(
    COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${consumer_build_dir}"
            ${test_args} --output-on-failure --no-tests=error
    RESULT_VARIABLE test_result
)
if(NOT test_result EQUAL 0)
    message(FATAL_ERROR "Running the installed API consumer failed")
endif()
