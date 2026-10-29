cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx_liuding_deployment_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(jyppx CONFIG REQUIRED COMPONENTS onnx_export onnx_runtime)
foreach(component IN ITEMS onnx_export onnx_runtime)
    if(NOT TARGET jyppx::liuding::${component} OR NOT jyppx_${component}_FOUND)
        message(FATAL_ERROR "Installed package does not satisfy component ${component}")
    endif()
endforeach()

add_executable(jyppx_liuding_deployment_consumer main.cpp)
target_link_libraries(jyppx_liuding_deployment_consumer PRIVATE
    jyppx::liuding::onnx_export jyppx::liuding::onnx_runtime)

if(WIN32)
    if(NOT JYPPX_TORCH_RUNTIME_DIR)
        message(FATAL_ERROR "Windows deployment consumer requires JYPPX_TORCH_RUNTIME_DIR")
    endif()
    if(JYPPX_RUNTIME_STAGER AND EXISTS "${JYPPX_RUNTIME_STAGER}")
        add_custom_command(TARGET jyppx_liuding_deployment_consumer POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                "-DJYPPX_RUNTIME_SOURCE_DIR=${JYPPX_TORCH_RUNTIME_DIR}"
                "-DJYPPX_RUNTIME_OUTPUT_DIR=$<TARGET_FILE_DIR:jyppx_liuding_deployment_consumer>"
                -P "${JYPPX_RUNTIME_STAGER}" VERBATIM)
    else()
        file(GLOB JYPPX_CONSUMER_RUNTIME_DLLS "${JYPPX_TORCH_RUNTIME_DIR}/*.dll")
        add_custom_command(TARGET jyppx_liuding_deployment_consumer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${JYPPX_CONSUMER_RUNTIME_DLLS}
                "$<TARGET_FILE_DIR:jyppx_liuding_deployment_consumer>" VERBATIM)
    endif()
    get_filename_component(JYPPX_DEPLOYMENT_INSTALL_PREFIX "${jyppx_DIR}/../../.." ABSOLUTE)
    set(JYPPX_DEPLOYMENT_RUNTIME_DIR "${JYPPX_DEPLOYMENT_INSTALL_PREFIX}/bin")
    if(NOT EXISTS "${JYPPX_DEPLOYMENT_RUNTIME_DIR}/onnxruntime.dll")
        message(FATAL_ERROR "Installed deployment package is missing bin/onnxruntime.dll")
    endif()
    # The executable directory precedes Windows system directories and PATH.
    # Stage the package DLL explicitly so Windows' own older ORT cannot win.
    file(GLOB JYPPX_DEPLOYMENT_RUNTIME_DLLS
        "${JYPPX_DEPLOYMENT_RUNTIME_DIR}/onnxruntime*.dll")
    foreach(runtime_dll IN LISTS JYPPX_DEPLOYMENT_RUNTIME_DLLS)
        get_filename_component(runtime_name "${runtime_dll}" NAME)
        add_custom_command(TARGET jyppx_liuding_deployment_consumer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rm -f
                "$<TARGET_FILE_DIR:jyppx_liuding_deployment_consumer>/${runtime_name}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${runtime_dll}"
                "$<TARGET_FILE_DIR:jyppx_liuding_deployment_consumer>" VERBATIM)
    endforeach()
endif()

enable_testing()
if(WIN32)
    # Resolve ORT from the installed package, independently of the SDK or Python.
    add_test(NAME Runs COMMAND ${CMAKE_COMMAND} -E env
        "PATH=${JYPPX_DEPLOYMENT_RUNTIME_DIR};$ENV{PATH}"
        "$<TARGET_FILE:jyppx_liuding_deployment_consumer>")
else()
    add_test(NAME Runs COMMAND jyppx_liuding_deployment_consumer)
endif()
set_tests_properties(Runs PROPERTIES TIMEOUT 60)
