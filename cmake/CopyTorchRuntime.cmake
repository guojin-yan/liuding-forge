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
        # CMAKE_CFG_INTDIR expands to the selected configuration for
        # multi-config generators and to "." for single-config generators.
        set(jyppx_runtime_output_dir
            "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}")
        set(jyppx_runtime_stamp
            "${jyppx_runtime_output_dir}/jyppx-runtime-${jyppx_binary_dir_id}.stamp")
        add_custom_command(
            OUTPUT "${jyppx_runtime_stamp}"
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "${jyppx_runtime_output_dir}"
            COMMAND ${CMAKE_COMMAND}
                "-DJYPPX_RUNTIME_SOURCE_DIR=${jyppx_torch_runtime_dir}"
                "-DJYPPX_RUNTIME_OUTPUT_DIR=${jyppx_runtime_output_dir}"
                -P "${PROJECT_SOURCE_DIR}/cmake/StageTorchRuntime.cmake"
            COMMAND ${CMAKE_COMMAND} -E touch "${jyppx_runtime_stamp}"
            DEPENDS ${jyppx_torch_runtime_dlls}
                "${PROJECT_SOURCE_DIR}/cmake/StageTorchRuntime.cmake"
            COMMENT "Copying LibTorch runtime DLLs for ${CMAKE_CURRENT_BINARY_DIR}"
            VERBATIM
        )
        add_custom_target("${jyppx_copy_target}"
            DEPENDS "${jyppx_runtime_stamp}"
        )
    endif()
    add_dependencies("${target_name}" "${jyppx_copy_target}")
endfunction()
