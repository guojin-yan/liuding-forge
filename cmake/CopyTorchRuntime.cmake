# Copy LibTorch runtime DLLs once per binary directory before dependent
# executables build. Per-target POST_BUILD copies race under parallel MSBuild.
function(jyppx_add_torch_runtime_dependency target_name)
    if(NOT WIN32)
        return()
    endif()

    get_filename_component(jyppx_torch_runtime_dir "${TORCH_LIBRARY}" DIRECTORY)
    file(GLOB jyppx_torch_runtime_dlls "${jyppx_torch_runtime_dir}/*.dll")
    if(NOT jyppx_torch_runtime_dlls)
        return()
    endif()

    string(MD5 jyppx_binary_dir_hash "${CMAKE_CURRENT_BINARY_DIR}")
    string(SUBSTRING "${jyppx_binary_dir_hash}" 0 8 jyppx_binary_dir_id)
    set(jyppx_copy_target "jrt_${jyppx_binary_dir_id}")
    if(NOT TARGET "${jyppx_copy_target}")
        set(jyppx_runtime_output_dir "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
        add_custom_target("${jyppx_copy_target}"
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "${jyppx_runtime_output_dir}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${jyppx_torch_runtime_dlls}
                "${jyppx_runtime_output_dir}"
            COMMENT "Copying LibTorch runtime DLLs for ${CMAKE_CURRENT_BINARY_DIR}"
            VERBATIM
        )
    endif()
    add_dependencies("${target_name}" "${jyppx_copy_target}")
endfunction()
