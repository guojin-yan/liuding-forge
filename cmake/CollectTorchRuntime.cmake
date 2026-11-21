# Inspect the selected Linux ELF dependency closure without executing SDK code.
cmake_minimum_required(VERSION 3.18)
foreach(required IN ITEMS JYPPX_RUNTIME_TORCH_LIBRARY JYPPX_RUNTIME_HIP_LIBRARY JYPPX_RUNTIME_OUTPUT)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "Missing runtime scan argument: ${required}")
    endif()
endforeach()
set(CMAKE_GET_RUNTIME_DEPENDENCIES_PLATFORM "linux+elf")
set(CMAKE_GET_RUNTIME_DEPENDENCIES_TOOL "objdump")
get_filename_component(torch_dir "${JYPPX_RUNTIME_TORCH_LIBRARY}" DIRECTORY)
get_filename_component(hip_dir "${JYPPX_RUNTIME_HIP_LIBRARY}" DIRECTORY)
file(GET_RUNTIME_DEPENDENCIES
    LIBRARIES "${JYPPX_RUNTIME_TORCH_LIBRARY}" "${JYPPX_RUNTIME_HIP_LIBRARY}"
    DIRECTORIES "${torch_dir}" "${hip_dir}"
        "${hip_dir}/host-math/lib" "${hip_dir}/rocm_sysdeps/lib" "${hip_dir}/llvm/lib"
    RESOLVED_DEPENDENCIES_VAR dependencies
    UNRESOLVED_DEPENDENCIES_VAR unresolved)
if(unresolved)
    message(FATAL_ERROR "Selected ROCm SDK has unresolved runtime dependencies: ${unresolved}")
endif()
file(WRITE "${JYPPX_RUNTIME_OUTPUT}" "${dependencies}")
