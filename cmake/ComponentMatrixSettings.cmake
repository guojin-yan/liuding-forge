# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0.

include_guard(GLOBAL)

# Reuse immutable dependency sources and download policy, never a parent's
# mutable FetchContent build directories.
function(jyppx_write_matrix_dependency_cache path)
    get_cmake_property(cache_variables CACHE_VARIABLES)
    list(SORT cache_variables)
    set(content "# Generated from the parent build's dependency settings.\n")
    foreach(variable IN LISTS cache_variables)
        if(NOT variable MATCHES "^FETCHCONTENT_(FULLY_DISCONNECTED|UPDATES_DISCONNECTED(_[A-Za-z0-9_]+)?|SOURCE_DIR_[A-Za-z0-9_]+)$")
            continue()
        endif()
        get_property(type CACHE "${variable}" PROPERTY TYPE)
        set(delimiter "=")
        while("${${variable}}" MATCHES "]${delimiter}]")
            string(APPEND delimiter "=")
        endwhile()
        string(APPEND content
            "set(${variable} [${delimiter}[${${variable}}]${delimiter}] CACHE ${type} \"\" FORCE)\n")
    endforeach()
    file(WRITE "${path}" "${content}")
endfunction()

function(jyppx_component_build_arguments build_output)
    if(DEFINED JYPPX_MATRIX_BUILD_JOBS AND NOT JYPPX_MATRIX_BUILD_JOBS STREQUAL "")
        set(jobs "${JYPPX_MATRIX_BUILD_JOBS}")
    elseif(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL} AND
           NOT "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}" STREQUAL "")
        set(jobs "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}")
    else()
        set(jobs 2)
    endif()
    if(NOT jobs MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "Component matrix build jobs must be a positive integer")
    endif()
    message(STATUS "Component matrix compile jobs: ${jobs}")
    set(${build_output} --parallel "${jobs}" PARENT_SCOPE)
endfunction()

function(jyppx_component_matrix_settings config_output build_output)
    set(config_args)
    if(JYPPX_MATRIX_DEPENDENCY_CACHE)
        if(NOT EXISTS "${JYPPX_MATRIX_DEPENDENCY_CACHE}")
            message(FATAL_ERROR "Component matrix dependency cache is missing")
        endif()
        list(APPEND config_args -C "${JYPPX_MATRIX_DEPENDENCY_CACHE}")
    endif()
    jyppx_component_build_arguments(build_args)
    set(${config_output} "${config_args}" PARENT_SCOPE)
    set(${build_output} "${build_args}" PARENT_SCOPE)
endfunction()
