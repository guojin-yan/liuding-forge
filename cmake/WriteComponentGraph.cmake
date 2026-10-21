# Copyright 2026 JYPPX Training Framework Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

# Installation and dependency validation must use the same public surface.
function(jyppx_public_headers header_root output_variable)
    file(GLOB_RECURSE public_headers CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/include/jyppx/${header_root}/*.h"
        "${PROJECT_SOURCE_DIR}/include/jyppx/${header_root}/*.hh"
        "${PROJECT_SOURCE_DIR}/include/jyppx/${header_root}/*.hpp"
        "${PROJECT_SOURCE_DIR}/include/jyppx/${header_root}/*.hxx"
        "${PROJECT_SOURCE_DIR}/include/jyppx/${header_root}/*.inc")
    if(header_root STREQUAL "interop" AND NOT JYPPX_BUILD_NN)
        list(REMOVE_ITEM public_headers
            "${PROJECT_SOURCE_DIR}/include/jyppx/interop/model_weight_import.hpp"
            "${PROJECT_SOURCE_DIR}/include/jyppx/interop/torchscript_compatibility.hpp")
    endif()
    if(header_root STREQUAL "training" AND NOT JYPPX_BUILD_INTEROP)
        list(REMOVE_ITEM public_headers
            "${PROJECT_SOURCE_DIR}/include/jyppx/training/file_checkpoint_store.hpp")
    endif()
    set(${output_variable} "${public_headers}" PARENT_SCOPE)
endfunction()

function(jyppx_write_component_graph output_path)
    set(components
        core
        data
        experiment
        foundation
        interop
        nn
        optim
        training
    )
    set(aggregate_targets all sdk)

    file(WRITE "${output_path}" "component\tkind\tdependency\n")
    foreach(component IN LISTS components)
        set(target_name "jyppx_${component}")
        if(TARGET "${target_name}")
            file(APPEND "${output_path}" "${component}\tpresent\t\n")
        else()
            file(APPEND "${output_path}" "${component}\tabsent\t\n")
            continue()
        endif()

        set(header_roots "${component}")
        if(component STREQUAL "core")
            list(APPEND header_roots utils)
        endif()
        foreach(header_root IN LISTS header_roots)
            jyppx_public_headers("${header_root}" component_headers)
            foreach(component_header IN LISTS component_headers)
                file(RELATIVE_PATH header_relative "${PROJECT_SOURCE_DIR}"
                    "${component_header}")
                file(TO_CMAKE_PATH "${header_relative}" header_relative)
                file(APPEND "${output_path}"
                    "${component}\theader\t${header_relative}\n")
            endforeach()
        endforeach()

        get_target_property(component_sources "${target_name}" SOURCES)
        foreach(component_source IN LISTS component_sources)
            if(NOT component_source MATCHES "^\\$<")
                get_filename_component(source_absolute "${component_source}"
                    ABSOLUTE BASE_DIR "${PROJECT_SOURCE_DIR}")
                file(RELATIVE_PATH source_relative "${PROJECT_SOURCE_DIR}"
                    "${source_absolute}")
                file(TO_CMAKE_PATH "${source_relative}" source_relative)
                if(source_relative MATCHES "^src/jyppx/")
                    file(APPEND "${output_path}"
                        "${component}\tsource\t${source_relative}\n")
                endif()
            endif()
        endforeach()

        foreach(graph_property IN ITEMS LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
            if(graph_property STREQUAL "LINK_LIBRARIES")
                set(graph_kind target)
            else()
                set(graph_kind interface)
            endif()
            get_target_property(graph_entries "${target_name}" "${graph_property}")
            if(NOT graph_entries OR graph_entries MATCHES "-NOTFOUND$")
                continue()
            endif()
            foreach(graph_entry IN LISTS graph_entries)
                foreach(provider IN LISTS components aggregate_targets)
                    set(has_provider FALSE)
                    foreach(provider_spelling IN ITEMS
                            "jyppx_${provider}" "jyppx::${provider}")
                        string(FIND "${graph_entry}" "${provider_spelling}"
                               provider_position)
                        if(NOT provider_position EQUAL -1)
                            set(has_provider TRUE)
                        endif()
                    endforeach()
                    if(has_provider)
                        file(APPEND "${output_path}"
                             "${component}\t${graph_kind}\t${provider}\n")
                    endif()
                endforeach()
            endforeach()
        endforeach()
    endforeach()

    get_property(directory_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(directory_target IN LISTS directory_targets)
        if(directory_target MATCHES "^jyppx_(.+)$")
            set(component_name "${CMAKE_MATCH_1}")
            if(NOT component_name IN_LIST components AND
               NOT component_name IN_LIST aggregate_targets)
                file(APPEND "${output_path}"
                     "${component_name}\tunknown-target\t\n")
            endif()
        endif()
    endforeach()
endfunction()
