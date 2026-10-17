# Security fuzz targets are deliberately opt-in because libFuzzer is only
# available with Clang. Each target owns an explicit corpus and bounded runner.
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "JYPPX_BUILD_FUZZERS=ON requires Clang with libFuzzer")
endif()

function(jyppx_add_fuzz_target target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE jyppx_training_framework)
    target_compile_options(${target} PRIVATE -fsanitize=fuzzer)
    target_link_options(${target} PRIVATE -fsanitize=fuzzer)
    jyppx_apply_instrumentation(${target})
    jyppx_add_torch_runtime_dependency(${target})
endfunction()

jyppx_add_fuzz_target(jyppx_fuzz_device_parse device_parse_fuzz.cpp)
jyppx_add_fuzz_target(jyppx_fuzz_data_pipeline_state
                      data_pipeline_state_fuzz.cpp)
jyppx_add_fuzz_target(jyppx_fuzz_iterable_pipeline_state
                      iterable_pipeline_state_fuzz.cpp)
