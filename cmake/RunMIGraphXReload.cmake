# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.

# Development test supervisor only; each native invocation is a fresh process.
foreach(variable IN ITEMS TEST_EXECUTABLE FIXTURE_ROOT MODEL CASE EVIDENCE_PARENT)
    if(NOT DEFINED ${variable} OR "${${variable}}" STREQUAL "")
        message(FATAL_ERROR "MIGraphX reload test requires ${variable}")
    endif()
endforeach()
if(NOT MODEL MATCHES "^(linear|resnet18)$" OR NOT CASE MATCHES "^[01]$")
    message(FATAL_ERROR "Invalid fixed MIGraphX reload model/case")
endif()
string(RANDOM LENGTH 24 ALPHABET 0123456789abcdef token)
set(run_dir "${EVIDENCE_PARENT}/reload-${MODEL}-${CASE}-${token}")
if(EXISTS "${run_dir}")
    message(FATAL_ERROR "Refusing existing reload evidence directory")
endif()
file(MAKE_DIRECTORY "${run_dir}")
message(STATUS "reload_evidence=${run_dir}")

function(run_native stage timeout)
    execute_process(COMMAND "${TEST_EXECUTABLE}" ${ARGN}
        RESULT_VARIABLE result TIMEOUT "${timeout}"
        OUTPUT_FILE "${run_dir}/${stage}.log"
        ERROR_FILE "${run_dir}/${stage}.log")
    file(READ "${run_dir}/${stage}.log" log)
    if(NOT "${result}" STREQUAL "0")
        message(FATAL_ERROR "MIGraphX ${stage} failed (${result}); ${run_dir}/${stage}.log\n${log}")
    endif()
    string(REGEX MATCH "evidence=([^\r\n]+)" matched "${log}")
    if(NOT matched)
        message(FATAL_ERROR "Missing native ${stage} evidence path")
    endif()
    file(TO_CMAKE_PATH "${CMAKE_MATCH_1}" native_dir)
    set("${stage}_dir" "${native_dir}" PARENT_SCOPE)
endfunction()

run_native(compile 600 "${FIXTURE_ROOT}" "${MODEL}" "${CASE}" "${run_dir}/compile")
file(SHA256 "${compile_dir}/compiled.mxr" program_sha)
file(SHA256 "${compile_dir}/actual.f32" output_sha)
run_native(reload 60 "${FIXTURE_ROOT}" "${MODEL}" "${CASE}" "${run_dir}/reload"
    --reload "${compile_dir}/compiled.mxr" "${program_sha}"
    "${compile_dir}/actual.f32" "${output_sha}")
file(SHA256 "${reload_dir}/compiled.mxr" loaded_program_sha)
file(SHA256 "${reload_dir}/baseline.f32" loaded_baseline_sha)
file(SHA256 "${reload_dir}/actual.f32" loaded_output_sha)
file(SHA256 "${compile_dir}/compiled.mxr" retained_program_sha)
file(SHA256 "${compile_dir}/actual.f32" retained_output_sha)
if(NOT program_sha STREQUAL loaded_program_sha OR
   NOT program_sha STREQUAL retained_program_sha OR
   NOT output_sha STREQUAL loaded_baseline_sha OR
   NOT output_sha STREQUAL loaded_output_sha OR
   NOT output_sha STREQUAL retained_output_sha)
    message(FATAL_ERROR "Serialized program or full output changed across processes")
endif()
file(WRITE "${run_dir}/receipt.txt"
    "compile=${compile_dir}\nreload=${reload_dir}\nprogram_sha256=${program_sha}\noutput_sha256=${output_sha}\n")
message(STATUS "Independent GPU reload passed; full output byte-identical: ${output_sha}")
