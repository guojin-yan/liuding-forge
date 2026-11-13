cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx_liuding_vision_tasks_consumer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(jyppx CONFIG REQUIRED COMPONENTS vision_tasks
    OPTIONAL_COMPONENTS models models_yolo models_rcnn models_vision onnx_export onnx_runtime)
add_executable(jyppx_liuding_vision_tasks_consumer main.cpp)
target_link_libraries(jyppx_liuding_vision_tasks_consumer PRIVATE jyppx::liuding::vision_tasks)
foreach(component IN ITEMS models models_yolo models_rcnn models_vision onnx_export onnx_runtime)
    if(TARGET jyppx::liuding::${component})
        target_link_libraries(jyppx_liuding_vision_tasks_consumer PRIVATE jyppx::liuding::${component})
        string(TOUPPER "${component}" component_upper)
        target_compile_definitions(jyppx_liuding_vision_tasks_consumer PRIVATE "VISION_CONSUMER_${component_upper}=1")
    endif()
endforeach()
get_filename_component(vision_prefix "${jyppx_DIR}/../../.." ABSOLUTE)
target_compile_definitions(jyppx_liuding_vision_tasks_consumer PRIVATE
    "VISION_INSTALLED_RECIPES=\"${vision_prefix}/share/jyppx/recipes/vision\"")
if(WIN32)
    if(NOT JYPPX_TORCH_RUNTIME_DIR OR NOT EXISTS "${JYPPX_RUNTIME_STAGER}")
        message(FATAL_ERROR "Windows vision consumer requires Torch runtime and stager")
    endif()
    add_custom_command(TARGET jyppx_liuding_vision_tasks_consumer POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            "-DJYPPX_RUNTIME_SOURCE_DIR=${JYPPX_TORCH_RUNTIME_DIR}"
            "-DJYPPX_RUNTIME_OUTPUT_DIR=$<TARGET_FILE_DIR:jyppx_liuding_vision_tasks_consumer>"
            -P "${JYPPX_RUNTIME_STAGER}" VERBATIM)
    if(TARGET jyppx::liuding::onnx_runtime)
        file(GLOB vision_runtime_dlls "${vision_prefix}/bin/onnxruntime*.dll")
        foreach(runtime IN LISTS vision_runtime_dlls)
            add_custom_command(TARGET jyppx_liuding_vision_tasks_consumer POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${runtime}"
                    "$<TARGET_FILE_DIR:jyppx_liuding_vision_tasks_consumer>" VERBATIM)
        endforeach()
    endif()
endif()
enable_testing()
add_test(NAME Runs COMMAND jyppx_liuding_vision_tasks_consumer)
set_tests_properties(Runs PROPERTIES TIMEOUT 60)
