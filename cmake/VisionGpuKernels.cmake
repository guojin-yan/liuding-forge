# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.

# Only the SDK-only kernel needs a device compiler. Keep LibTorch/custom
# autograd in the existing C++ toolchain, and embed the object in vision_ops.
# This also works with the CMake 3.18 baseline and Windows HIP clang-cl builds.
function(jyppx_add_vision_gpu_kernels target)
    if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8 OR
       NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64)$" OR
       NOT (WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Linux"))
        message(FATAL_ERROR "VISION_GPU_KERNELS requires Windows/Linux x64")
    endif()
    if(TARGET torch_hip)
        set(backend HIP)
        get_filename_component(host_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
        find_program(JYPPX_LIUDING_VISION_GPU_COMPILER NAMES amdclang++ clang++
            HINTS "${host_bin}" "$ENV{HIP_PATH}/lib/llvm/bin"
                  "$ENV{ROCM_PATH}/lib/llvm/bin" NO_DEFAULT_PATH)
        set(default_archs "$ENV{PYTORCH_ROCM_ARCH}")
    elseif(TARGET torch_cuda OR TORCH_CUDA_VERSION)
        set(backend CUDA)
        if(NOT JYPPX_LIUDING_VISION_GPU_COMPILER AND CUDA_NVCC_EXECUTABLE)
            set(JYPPX_LIUDING_VISION_GPU_COMPILER "${CUDA_NVCC_EXECUTABLE}"
                CACHE FILEPATH "Selected SDK device compiler for native vision kernels")
        endif()
        set(default_archs "${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(FATAL_ERROR "VISION_GPU_KERNELS requires a GPU LibTorch package")
    endif()
    set(JYPPX_LIUDING_VISION_GPU_ARCHITECTURES "${default_archs}" CACHE STRING
        "Explicit native vision GPU architectures (HIP gfx names or CUDA SM numbers)")
    if(NOT IS_ABSOLUTE "${JYPPX_LIUDING_VISION_GPU_COMPILER}" OR
       NOT EXISTS "${JYPPX_LIUDING_VISION_GPU_COMPILER}" OR
       NOT JYPPX_LIUDING_VISION_GPU_ARCHITECTURES)
        message(FATAL_ERROR "VISION_GPU_KERNELS needs an existing absolute VISION_GPU_COMPILER and explicit VISION_GPU_ARCHITECTURES")
    endif()
    set(flags)
    set(arch_literals)
    foreach(arch IN LISTS JYPPX_LIUDING_VISION_GPU_ARCHITECTURES)
        if(backend STREQUAL HIP)
            if(NOT arch MATCHES "^gfx[0-9a-f]+$")
                message(FATAL_ERROR "Invalid native vision HIP architecture: ${arch}")
            endif()
            list(APPEND flags "--offload-arch=${arch}")
        else()
            if(NOT arch MATCHES "^[0-9][0-9][0-9]?$" OR arch LESS 60)
                message(FATAL_ERROR "Native vision CUDA architectures require SM >= 60: ${arch}")
            endif()
            list(APPEND flags "--generate-code=arch=compute_${arch},code=[sm_${arch},compute_${arch}]")
        endif()
        string(APPEND arch_literals "\"${arch}\",")
    endforeach()
    set(kernel_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/vision-gpu")
    file(MAKE_DIRECTORY "${kernel_dir}")
    set(config "${kernel_dir}/roi_gpu_build.hpp")
    file(WRITE "${config}.in"
        "#pragma once\n#define JYPPX_VISION_GPU_${backend} 1\n#define JYPPX_VISION_GPU_ARCHS ${arch_literals}\n")
    configure_file("${config}.in" "${config}" COPYONLY)
    set(source "${PROJECT_SOURCE_DIR}/src/jyppx/liuding/vision/ops/roi_gpu_kernel.cpp")
    set(header "${PROJECT_SOURCE_DIR}/src/jyppx/liuding/vision/ops/roi_gpu_launch.hpp")
    # SDK-only code always uses optimized Release CRT, independent of the host
    # configuration. A shared output also keeps OUTPUT compatible with 3.18.
    set(object "${kernel_dir}/roi_gpu${CMAKE_CXX_OUTPUT_EXTENSION}")
    if(backend STREQUAL HIP)
        list(APPEND flags -O3 -std=c++17 -ffp-contract=off -x hip)
        if(WIN32)
            list(APPEND flags -fms-runtime-lib=dll)
        else()
            list(APPEND flags -fPIC)
        endif()
        target_compile_definitions(${target} PRIVATE JYPPX_VISION_GPU_HIP=1)
    else()
        list(APPEND flags -O3 -std=c++17 --fmad=false --cudart=shared -x cu
            -ccbin "${CMAKE_CXX_COMPILER}")
        if(WIN32)
            list(APPEND flags -Xcompiler /MD)
        else()
            list(APPEND flags -Xcompiler -fPIC)
        endif()
    endif()
    add_custom_command(OUTPUT "${object}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${kernel_dir}"
        COMMAND "${JYPPX_LIUDING_VISION_GPU_COMPILER}" ${flags}
            "-I${kernel_dir}" -c "${source}" -o "${object}"
        DEPENDS "${source}" "${header}" "${config}" VERBATIM)
    set_source_files_properties("${object}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    set_source_files_properties("${source}" PROPERTIES HEADER_FILE_ONLY TRUE)
    target_sources(${target} PRIVATE
        src/jyppx/liuding/vision/ops/roi_gpu.cpp
        src/jyppx/liuding/vision/ops/roi_gpu.hpp
        src/jyppx/liuding/vision/ops/roi_gpu_launch.hpp
        src/jyppx/liuding/vision/ops/roi_gpu_kernel.cpp
        "${object}")
    target_compile_definitions(${target} PRIVATE JYPPX_VISION_GPU_KERNELS=1)
    message(STATUS "Native vision kernels: ${backend}, ${JYPPX_LIUDING_VISION_GPU_ARCHITECTURES}, ${JYPPX_LIUDING_VISION_GPU_COMPILER}")
endfunction()
