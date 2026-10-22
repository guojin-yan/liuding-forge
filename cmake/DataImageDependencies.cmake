# Copyright 2026 JYPPX Training Framework Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

include(FetchContent)
foreach(parser IN ITEMS tinyxml2 yaml-cpp)
    if(parser STREQUAL "tinyxml2")
        set(parser_name jyppx_image_tinyxml2)
        set(parser_version 11.0.0)
        set(parser_sha 5556deb5081fb246ee92afae73efd943c889cef0cafea92b0b82422d6a18f289)
    else()
        set(parser_name jyppx_image_yaml)
        set(parser_version 0.9.0)
        set(parser_sha 25cb043240f828a8c51beb830569634bc7ac603978e0f69d6b63558dadefd49a)
    endif()
    set(parser_archive "${PROJECT_SOURCE_DIR}/third_party/downloads/${parser}-${parser_version}.tar.gz")
    if(NOT EXISTS "${parser_archive}")
        message(FATAL_ERROR
            "DATA_IMAGE requires the pinned local ${parser} ${parser_version} archive; "
            "stage third_party/downloads/${parser}-${parser_version}.tar.gz from dependency-manifest.json")
    endif()
    file(SHA256 "${parser_archive}" parser_actual_sha)
    if(NOT parser_actual_sha STREQUAL parser_sha)
        message(FATAL_ERROR "DATA_IMAGE ${parser} archive digest mismatch")
    endif()
    FetchContent_Declare(${parser_name}
        URL "${parser_archive}" URL_HASH "SHA256=${parser_sha}"
        SOURCE_SUBDIR jyppx-no-upstream-project)
    FetchContent_MakeAvailable(${parser_name})
endforeach()

# Embed parser objects in the image archive. No private backend targets or
# parser headers are required by the installed CMake/link interface.
add_library(data_image_xml_backend OBJECT
    "${jyppx_image_tinyxml2_SOURCE_DIR}/tinyxml2.cpp")
target_include_directories(data_image_xml_backend SYSTEM PRIVATE
    "${jyppx_image_tinyxml2_SOURCE_DIR}")
set_target_properties(data_image_xml_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)

file(GLOB yaml_sources CONFIGURE_DEPENDS "${jyppx_image_yaml_SOURCE_DIR}/src/*.cpp")
add_library(data_image_yaml_backend OBJECT ${yaml_sources})
target_include_directories(data_image_yaml_backend SYSTEM PRIVATE
    "${jyppx_image_yaml_SOURCE_DIR}/include"
    "${jyppx_image_yaml_SOURCE_DIR}/src")
target_compile_definitions(data_image_yaml_backend PRIVATE YAML_CPP_STATIC_DEFINE)
set_target_properties(data_image_yaml_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
