if(NOT DEFINED JYPPX_SOURCE_DIR OR NOT DEFINED JYPPX_LIUDING_BUILD_DIR OR
   NOT DEFINED JYPPX_CONSUMER_SOURCE_DIR OR NOT DEFINED JYPPX_LIUDING_TEST_ROOT OR
   NOT DEFINED JYPPX_TORCH_DIR)
    message(FATAL_ERROR "Installed component matrix is missing required paths")
endif()

get_filename_component(build_dir "${JYPPX_LIUDING_BUILD_DIR}" REALPATH)
get_filename_component(source_dir "${JYPPX_SOURCE_DIR}" REALPATH)
get_filename_component(consumer_source_dir
                       "${JYPPX_CONSUMER_SOURCE_DIR}" REALPATH)
get_filename_component(test_root "${JYPPX_LIUDING_TEST_ROOT}" ABSOLUTE)
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
set(build_args)
set(test_args)
if(JYPPX_LIUDING_TEST_CONFIG)
    list(APPEND config_args "-DCMAKE_BUILD_TYPE=${JYPPX_LIUDING_TEST_CONFIG}")
    list(APPEND build_args --config "${JYPPX_LIUDING_TEST_CONFIG}")
    list(APPEND test_args -C "${JYPPX_LIUDING_TEST_CONFIG}")
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
        list(APPEND config_args "-DCMAKE_CUDA_STANDARD=17")
    endif()
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${build_dir}"
            --prefix "${install_dir}" ${build_args}
    RESULT_VARIABLE install_result
)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "Installing JYPPX for the component matrix failed")
endif()

# The complete SDK embeds JSON and SQLite implementation code. Check the
# original legal text in the staged binary package, not only in the source tree.
foreach(notice IN ITEMS nlohmann-json-3.12.0-LICENSE.MIT
        sqlite-3.54.1-copyright-blessing.txt sqlite-copyright-page.txt)
    set(installed_notice "${install_dir}/share/doc/jyppx/licenses/${notice}")
    if(NOT EXISTS "${installed_notice}")
        message(FATAL_ERROR "Installed SDK is missing its dependency notice: ${notice}")
    endif()
    file(SHA256 "${source_dir}/third_party/licenses/${notice}" expected_notice_hash)
    file(SHA256 "${installed_notice}" installed_notice_hash)
    if(NOT installed_notice_hash STREQUAL expected_notice_hash)
        message(FATAL_ERROR "Installed SDK dependency notice was altered: ${notice}")
    endif()
endforeach()

execute_process(
    COMMAND "${CMAKE_COMMAND}"
            "-DJYPPX_SOURCE_DIR=${source_dir}"
            "-DJYPPX_LIUDING_INSTALL_DIR=${install_dir}"
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
file(READ "${install_dir}/lib/cmake/jyppx/jyppxConfig.cmake" installed_config)
if(installed_config MATCHES "jyppx_AVAILABLE_COMPONENTS[^\n]*;data_image[;\"]")
    list(APPEND components data_image)
    foreach(notice IN ITEMS tinyxml2-11.0.0-LICENSE.txt yaml-cpp-0.9.0-LICENSE.txt
            stb_image-2.30-LICENSE.txt)
        file(SHA256 "${source_dir}/third_party/licenses/${notice}" expected_hash)
        set(installed_notice "${install_dir}/share/doc/jyppx/licenses/${notice}")
        if(NOT EXISTS "${installed_notice}")
            message(FATAL_ERROR "Installed image profile notice is missing: ${notice}")
        endif()
        file(SHA256 "${installed_notice}" actual_hash)
        if(NOT actual_hash STREQUAL expected_hash)
            message(FATAL_ERROR "Installed image profile notice was altered: ${notice}")
        endif()
    endforeach()
endif()
foreach(component IN LISTS components)
    set(consumer_build_dir "${consumer_build_root}/${component}")
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
