# Resolve the selected LibTorch runtime for framework executables.
# Windows copies run once per binary directory to avoid parallel MSBuild races.
function(jyppx_add_torch_runtime_dependency target_name)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND TARGET torch_hip)
        get_property(jyppx_runtime_scanned GLOBAL PROPERTY JYPPX_ROCM_RUNTIME_SCANNED)
        if(NOT jyppx_runtime_scanned)
            get_filename_component(jyppx_torch_runtime_dir "${TORCH_LIBRARY}" DIRECTORY)
            get_filename_component(jyppx_torch_runtime_dir "${jyppx_torch_runtime_dir}" REALPATH)
            get_target_property(jyppx_hip_runtime hip::amdhip64 IMPORTED_LOCATION)
            if(NOT jyppx_hip_runtime)
                get_target_property(jyppx_hip_configs hip::amdhip64 IMPORTED_CONFIGURATIONS)
                string(TOUPPER "${CMAKE_BUILD_TYPE}" jyppx_hip_config)
                foreach(jyppx_config IN ITEMS "${jyppx_hip_config}" ${jyppx_hip_configs})
                    get_target_property(jyppx_hip_runtime hip::amdhip64
                        IMPORTED_LOCATION_${jyppx_config})
                    if(jyppx_hip_runtime)
                        break()
                    endif()
                endforeach()
            endif()
            if(NOT jyppx_hip_runtime)
                message(FATAL_ERROR "Selected ROCm SDK has no imported HIP runtime location")
            endif()
            get_filename_component(jyppx_hip_runtime_dir "${jyppx_hip_runtime}" DIRECTORY)
            get_filename_component(jyppx_hip_runtime_dir "${jyppx_hip_runtime_dir}" REALPATH)
            # Inspect the selected ELF files without loading them. Some SDK DSOs
            # have their own RUNPATH, which also blocks the executable's RPATH
            # for their children. Retain that existing SDK closure as direct
            # dependencies so an empty-environment worker can resolve every DSO.
            set(jyppx_runtime_scan "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/CollectTorchRuntime.cmake")
            set(jyppx_runtime_index "${CMAKE_BINARY_DIR}/CMakeFiles/jyppx-rocm-runtime.files")
            # GET_RUNTIME_DEPENDENCIES belongs in script mode. Its expected
            # search-directory warnings describe the SDK defect handled here;
            # keep all diagnostics if the scan cannot resolve the full closure.
            execute_process(
                COMMAND "${CMAKE_COMMAND}"
                    "-DJYPPX_RUNTIME_TORCH_LIBRARY=${TORCH_LIBRARY}"
                    "-DJYPPX_RUNTIME_HIP_LIBRARY=${jyppx_hip_runtime}"
                    "-DJYPPX_RUNTIME_OUTPUT=${jyppx_runtime_index}"
                    "-DCMAKE_GET_RUNTIME_DEPENDENCIES_COMMAND=${CMAKE_OBJDUMP}"
                    -P "${jyppx_runtime_scan}"
                RESULT_VARIABLE jyppx_runtime_scan_result
                OUTPUT_VARIABLE jyppx_runtime_scan_output
                ERROR_VARIABLE jyppx_runtime_scan_error
                TIMEOUT 120)
            if(NOT jyppx_runtime_scan_result STREQUAL "0")
                message(FATAL_ERROR "Selected ROCm SDK runtime scan failed: ${jyppx_runtime_scan_output}\n${jyppx_runtime_scan_error}")
            endif()
            file(READ "${jyppx_runtime_index}" jyppx_runtime_dependencies)
            set(jyppx_rocm_runtime_dirs "${jyppx_torch_runtime_dir}" "${jyppx_hip_runtime_dir}")
            set(jyppx_rocm_runtime_libraries "${TORCH_LIBRARY}" "${jyppx_hip_runtime}")
            foreach(jyppx_dependency IN LISTS jyppx_runtime_dependencies)
                get_filename_component(jyppx_real_dependency "${jyppx_dependency}" REALPATH)
                string(FIND "${jyppx_real_dependency}" "${jyppx_torch_runtime_dir}/" jyppx_torch_prefix)
                string(FIND "${jyppx_real_dependency}" "${jyppx_hip_runtime_dir}/" jyppx_hip_prefix)
                # System libraries keep their normal loader policy.
                if(jyppx_torch_prefix EQUAL 0 OR jyppx_hip_prefix EQUAL 0)
                    list(APPEND jyppx_rocm_runtime_libraries "${jyppx_dependency}")
                    get_filename_component(jyppx_dependency_dir "${jyppx_dependency}" DIRECTORY)
                    list(APPEND jyppx_rocm_runtime_dirs "${jyppx_dependency_dir}")
                endif()
            endforeach()
            list(REMOVE_DUPLICATES jyppx_rocm_runtime_libraries)
            list(REMOVE_DUPLICATES jyppx_rocm_runtime_dirs)
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                "${jyppx_runtime_scan}" ${jyppx_rocm_runtime_libraries})
            set_property(GLOBAL PROPERTY JYPPX_ROCM_RUNTIME_LIBRARIES ${jyppx_rocm_runtime_libraries})
            set_property(GLOBAL PROPERTY JYPPX_ROCM_RUNTIME_DIRS ${jyppx_rocm_runtime_dirs})
            set_property(GLOBAL PROPERTY JYPPX_ROCM_RUNTIME_SCANNED TRUE)
        endif()
        get_property(jyppx_rocm_runtime_libraries GLOBAL PROPERTY JYPPX_ROCM_RUNTIME_LIBRARIES)
        get_property(jyppx_rocm_runtime_dirs GLOBAL PROPERTY JYPPX_ROCM_RUNTIME_DIRS)
        target_link_libraries(${target_name} PRIVATE
            "-Wl,--no-as-needed" ${jyppx_rocm_runtime_libraries} "-Wl,--as-needed")
        target_link_options(${target_name} PRIVATE "LINKER:--disable-new-dtags")
        set_property(TARGET ${target_name} APPEND PROPERTY
            BUILD_RPATH ${jyppx_rocm_runtime_dirs})
        # LibTorch/HIP remain external SDK dependencies after installation.
        # Preserve any caller-provided paths, including the bundled ORT location.
        set_property(TARGET ${target_name} APPEND PROPERTY
            INSTALL_RPATH ${jyppx_rocm_runtime_dirs})
        return()
    endif()
    if(NOT WIN32)
        return()
    endif()

    get_filename_component(jyppx_torch_runtime_dir "${TORCH_LIBRARY}" DIRECTORY)
    file(GLOB jyppx_torch_runtime_dlls CONFIGURE_DEPENDS
        "${jyppx_torch_runtime_dir}/*.dll")
    if(NOT jyppx_torch_runtime_dlls)
        return()
    endif()
    file(GLOB_RECURSE jyppx_torch_runtime_images CONFIGURE_DEPENDS LIST_DIRECTORIES FALSE
        "${jyppx_torch_runtime_dir}/aotriton.images/*")
    list(APPEND jyppx_torch_runtime_dlls ${jyppx_torch_runtime_images})
    if(EXISTS "${jyppx_torch_runtime_dir}/torch_hip.dll")
        file(GLOB jyppx_torch_runtime_kpacks CONFIGURE_DEPENDS
            "${jyppx_torch_runtime_dir}/../.kpack/*.kpack")
        list(APPEND jyppx_torch_runtime_dlls ${jyppx_torch_runtime_kpacks})
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
        # A removed DLL no longer appears in DEPENDS. Track the selected set
        # separately so switching SDKs also restages when the set shrinks.
        set(jyppx_runtime_inputs
            "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/jyppx-runtime-${jyppx_binary_dir_id}.inputs")
        file(GENERATE OUTPUT "${jyppx_runtime_inputs}"
            CONTENT "${jyppx_torch_runtime_dir}\n${jyppx_torch_runtime_dlls}\n")
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
                "${jyppx_runtime_inputs}"
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
