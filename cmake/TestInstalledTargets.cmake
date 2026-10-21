if(NOT DEFINED JYPPX_BUILD_DIR OR NOT DEFINED JYPPX_CONSUMER_SOURCE_DIR OR
   NOT DEFINED JYPPX_TEST_ROOT OR NOT DEFINED JYPPX_TORCH_DIR)
    message(FATAL_ERROR "Installed target test is missing required paths")
endif()

get_filename_component(build_dir "${JYPPX_BUILD_DIR}" REALPATH)
get_filename_component(consumer_source_dir "${JYPPX_CONSUMER_SOURCE_DIR}" REALPATH)
get_filename_component(test_root "${JYPPX_TEST_ROOT}" ABSOLUTE)
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
set(build_args --parallel 2)
set(install_args)
set(test_args)
if(JYPPX_TEST_CONFIG)
    list(APPEND config_args "-DCMAKE_BUILD_TYPE=${JYPPX_TEST_CONFIG}")
    list(APPEND build_args --config "${JYPPX_TEST_CONFIG}")
    list(APPEND install_args --config "${JYPPX_TEST_CONFIG}")
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
