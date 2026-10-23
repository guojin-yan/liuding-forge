cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx_package_component_consumer LANGUAGES CXX)

find_package(jyppx CONFIG QUIET COMPONENTS missing_component)

if(jyppx_FOUND)
    message(FATAL_ERROR "Installed package accepted an unknown component")
endif()
if(NOT jyppx_NOT_FOUND_MESSAGE STREQUAL
   "Unsupported jyppx component: missing_component")
    message(FATAL_ERROR
        "Installed package did not diagnose the unknown component")
endif()

get_filename_component(_jyppx_cmake_root "${jyppx_DIR}" DIRECTORY)
if(EXISTS "${_jyppx_cmake_root}/jyppx_liuding_training_framework")
    message(FATAL_ERROR "Installed package retained the legacy package entry")
endif()

add_custom_target(jyppx_package_component_consumer ALL
    COMMAND ${CMAKE_COMMAND} -E true
)

enable_testing()
add_test(NAME Runs COMMAND ${CMAKE_COMMAND} -E true)
