if(NOT DEFINED JYPPX_BUILD_DIR OR NOT DEFINED JYPPX_CONSUMER_SOURCE_DIR OR
   NOT DEFINED JYPPX_TEST_ROOT OR NOT DEFINED JYPPX_TORCH_DIR)
    message(FATAL_ERROR "Installed foundation test is missing required paths")
endif()

get_filename_component(build_dir "${JYPPX_BUILD_DIR}" REALPATH)
get_filename_component(consumer_source_dir "${JYPPX_CONSUMER_SOURCE_DIR}" REALPATH)
get_filename_component(test_root "${JYPPX_TEST_ROOT}" ABSOLUTE)
file(RELATIVE_PATH test_root_relative "${build_dir}" "${test_root}")
if(test_root_relative MATCHES "^\\.\\." OR IS_ABSOLUTE "${test_root_relative}")
    message(FATAL_ERROR "Installed foundation test root must stay inside the build tree")
endif()
file(REMOVE_RECURSE "${test_root}")
file(MAKE_DIRECTORY "${test_root}")

set(install_dir "${test_root}/i")
set(consumer_build_dir "${test_root}/b")
set(config_args)
set(build_args)
set(test_args)
if(JYPPX_TEST_CONFIG)
    list(APPEND config_args "-DCMAKE_BUILD_TYPE=${JYPPX_TEST_CONFIG}")
    list(APPEND build_args --config "${JYPPX_TEST_CONFIG}")
    list(APPEND test_args -C "${JYPPX_TEST_CONFIG}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${build_dir}"
            --prefix "${install_dir}" ${build_args}
    RESULT_VARIABLE install_result
)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "Installing JYPPX for the consumer failed")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}"
            -S "${consumer_source_dir}"
            -B "${consumer_build_dir}"
            "-DCMAKE_PREFIX_PATH=${install_dir}"
            "-DTorch_DIR=${JYPPX_TORCH_DIR}"
            "-DJYPPX_TORCH_RUNTIME_DIR=${JYPPX_TORCH_RUNTIME_DIR}"
            ${config_args}
    RESULT_VARIABLE configure_result
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "Configuring the installed foundation consumer failed")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${consumer_build_dir}" ${build_args}
    RESULT_VARIABLE build_result
)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "Building the installed foundation consumer failed")
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
    message(FATAL_ERROR "Running the installed foundation consumer failed")
endif()
