# Build-tree runtime dependencies are immutable. Prefer same-volume hard links
# so each test/benchmark directory does not consume another LibTorch copy.
# A copy remains the fallback on filesystems that cannot create hard links.
if(NOT IS_ABSOLUTE "${JYPPX_RUNTIME_SOURCE_DIR}" OR
   NOT IS_ABSOLUTE "${JYPPX_RUNTIME_OUTPUT_DIR}")
    message(FATAL_ERROR "Runtime staging requires absolute directories")
endif()
file(MAKE_DIRECTORY "${JYPPX_RUNTIME_OUTPUT_DIR}")
file(GLOB runtime_files "${JYPPX_RUNTIME_SOURCE_DIR}/*.dll")
foreach(runtime_source IN LISTS runtime_files)
    get_filename_component(runtime_name "${runtime_source}" NAME)
    set(runtime_destination "${JYPPX_RUNTIME_OUTPUT_DIR}/${runtime_name}")
    if(EXISTS "${runtime_destination}")
        execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files
            "${runtime_source}" "${runtime_destination}"
            RESULT_VARIABLE runtime_difference)
        if(runtime_difference EQUAL 0)
            continue()
        endif()
        file(REMOVE "${runtime_destination}")
    endif()
    file(CREATE_LINK "${runtime_source}" "${runtime_destination}"
        RESULT runtime_result COPY_ON_ERROR)
    if(NOT runtime_result STREQUAL "0")
        message(FATAL_ERROR "Could not stage ${runtime_name}: ${runtime_result}")
    endif()
endforeach()
