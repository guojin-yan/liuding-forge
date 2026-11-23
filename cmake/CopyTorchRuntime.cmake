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

    set(jyppx_rocm_extra_runtime_dlls)
    set(jyppx_rocm_extra_runtime_manifest)
    if(JYPPX_LIUDING_STAGE_ROCM_RUNTIME_DLLS)
        if(NOT TARGET hip::amdhip64 OR NOT EXISTS "${jyppx_torch_runtime_dir}/torch_hip.dll")
            message(FATAL_ERROR
                "STAGE_ROCM_RUNTIME_DLLS requires the selected LibTorch HIP backend and hip::amdhip64")
        endif()
        get_property(jyppx_runtime_scanned GLOBAL PROPERTY JYPPX_WINDOWS_ROCM_RUNTIME_SCANNED)
        if(NOT jyppx_runtime_scanned)
            get_target_property(jyppx_hip_runtime hip::amdhip64 IMPORTED_LOCATION)
            if(NOT jyppx_hip_runtime)
                get_target_property(jyppx_hip_configs hip::amdhip64 IMPORTED_CONFIGURATIONS)
                set(jyppx_hip_config_candidates ${jyppx_hip_configs} RELEASE RELWITHDEBINFO DEBUG MINSIZEREL)
                if(CMAKE_BUILD_TYPE)
                    string(TOUPPER "${CMAKE_BUILD_TYPE}" jyppx_hip_build_config)
                    list(PREPEND jyppx_hip_config_candidates "${jyppx_hip_build_config}")
                endif()
                list(REMOVE_DUPLICATES jyppx_hip_config_candidates)
                foreach(jyppx_hip_config IN LISTS jyppx_hip_config_candidates)
                    get_target_property(jyppx_hip_runtime hip::amdhip64
                        "IMPORTED_LOCATION_${jyppx_hip_config}")
                    if(jyppx_hip_runtime)
                        break()
                    endif()
                endforeach()
            endif()
            if(NOT jyppx_hip_runtime OR NOT IS_ABSOLUTE "${jyppx_hip_runtime}")
                message(FATAL_ERROR "Selected Windows ROCm SDK has no absolute HIP runtime DLL location")
            endif()
            if(NOT CMAKE_OBJDUMP OR NOT EXISTS "${CMAKE_OBJDUMP}")
                message(FATAL_ERROR "STAGE_ROCM_RUNTIME_DLLS requires the selected toolchain's CMAKE_OBJDUMP")
            endif()
            get_filename_component(jyppx_hip_runtime_dir "${jyppx_hip_runtime}" DIRECTORY)
            set(jyppx_rocm_runtime_scan
                "${PROJECT_SOURCE_DIR}/cmake/CollectWindowsHipRuntime.cmake")
            set(jyppx_rocm_extra_runtime_manifest
                "${CMAKE_BINARY_DIR}/CMakeFiles/jyppx-windows-rocm-runtime-files.txt")
            execute_process(
                COMMAND "${CMAKE_COMMAND}"
                    "-DJYPPX_ROCM_TORCH_HIP_DLL=${jyppx_torch_runtime_dir}/torch_hip.dll"
                    "-DJYPPX_ROCM_HIP_RUNTIME_DLL=${jyppx_hip_runtime}"
                    "-DJYPPX_ROCM_TORCH_RUNTIME_DIR=${jyppx_torch_runtime_dir}"
                    "-DJYPPX_ROCM_HIP_RUNTIME_DIR=${jyppx_hip_runtime_dir}"
                    "-DJYPPX_ROCM_WINDOWS_RUNTIME_MANIFEST=${jyppx_rocm_extra_runtime_manifest}"
                    "-DCMAKE_GET_RUNTIME_DEPENDENCIES_COMMAND=${CMAKE_OBJDUMP}"
                    -P "${jyppx_rocm_runtime_scan}"
                RESULT_VARIABLE jyppx_runtime_scan_result
                OUTPUT_VARIABLE jyppx_runtime_scan_output
                ERROR_VARIABLE jyppx_runtime_scan_error
                TIMEOUT 120)
            if(NOT jyppx_runtime_scan_result STREQUAL "0")
                message(FATAL_ERROR
                    "Selected Windows ROCm runtime scan failed: ${jyppx_runtime_scan_output}\n${jyppx_runtime_scan_error}")
            endif()
            file(STRINGS "${jyppx_rocm_extra_runtime_manifest}" jyppx_rocm_extra_runtime_dlls)
            if(NOT jyppx_rocm_extra_runtime_dlls)
                message(FATAL_ERROR "Selected Windows ROCm runtime import closure is empty")
            endif()
            set_property(GLOBAL PROPERTY JYPPX_WINDOWS_ROCM_RUNTIME_DLLS
                "${jyppx_rocm_extra_runtime_dlls}")
            set_property(GLOBAL PROPERTY JYPPX_WINDOWS_ROCM_RUNTIME_MANIFEST
                "${jyppx_rocm_extra_runtime_manifest}")
            set_property(GLOBAL PROPERTY JYPPX_WINDOWS_ROCM_RUNTIME_SOURCE_DIR
                "${jyppx_hip_runtime_dir}")
            set_property(GLOBAL PROPERTY JYPPX_WINDOWS_ROCM_RUNTIME_SCANNED TRUE)
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                "${jyppx_rocm_runtime_scan}" "${jyppx_torch_runtime_dir}/torch_hip.dll"
                "${jyppx_hip_runtime}" ${jyppx_rocm_extra_runtime_dlls})
        endif()
        get_property(jyppx_rocm_extra_runtime_dlls GLOBAL PROPERTY JYPPX_WINDOWS_ROCM_RUNTIME_DLLS)
        get_property(jyppx_rocm_extra_runtime_manifest GLOBAL PROPERTY
            JYPPX_WINDOWS_ROCM_RUNTIME_MANIFEST)
        get_property(jyppx_rocm_extra_runtime_source_dir GLOBAL PROPERTY
            JYPPX_WINDOWS_ROCM_RUNTIME_SOURCE_DIR)
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
        set(jyppx_runtime_input_content
            "${jyppx_torch_runtime_dir}\n${jyppx_torch_runtime_dlls}\n")
        if(jyppx_rocm_extra_runtime_manifest)
            file(READ "${jyppx_rocm_extra_runtime_manifest}" jyppx_rocm_runtime_manifest_content)
            string(APPEND jyppx_runtime_input_content
                "${jyppx_rocm_extra_runtime_manifest}\n${jyppx_rocm_runtime_manifest_content}")
        endif()
        file(GENERATE OUTPUT "${jyppx_runtime_inputs}"
            CONTENT "${jyppx_runtime_input_content}")
        add_custom_command(
            OUTPUT "${jyppx_runtime_stamp}"
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "${jyppx_runtime_output_dir}"
            COMMAND ${CMAKE_COMMAND}
                "-DJYPPX_RUNTIME_SOURCE_DIR=${jyppx_torch_runtime_dir}"
                "-DJYPPX_RUNTIME_OUTPUT_DIR=${jyppx_runtime_output_dir}"
                "-DJYPPX_RUNTIME_EXTRA_DLL_LIST=${jyppx_rocm_extra_runtime_manifest}"
                "-DJYPPX_RUNTIME_EXTRA_DLL_SOURCE_DIR=${jyppx_rocm_extra_runtime_source_dir}"
                -P "${PROJECT_SOURCE_DIR}/cmake/StageTorchRuntime.cmake"
            COMMAND ${CMAKE_COMMAND} -E touch "${jyppx_runtime_stamp}"
            DEPENDS ${jyppx_torch_runtime_dlls} ${jyppx_rocm_extra_runtime_dlls}
                "${jyppx_rocm_extra_runtime_manifest}"
                "${jyppx_runtime_inputs}"
                "${PROJECT_SOURCE_DIR}/cmake/StageTorchRuntime.cmake"
            COMMENT "Staging LibTorch runtime for ${CMAKE_CURRENT_BINARY_DIR}"
            VERBATIM
        )
        add_custom_target("${jyppx_copy_target}"
            DEPENDS "${jyppx_runtime_stamp}"
        )
    endif()
    add_dependencies("${target_name}" "${jyppx_copy_target}")
endfunction()
