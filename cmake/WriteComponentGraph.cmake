# Copyright 2026 JYPPX Training Framework Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

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
