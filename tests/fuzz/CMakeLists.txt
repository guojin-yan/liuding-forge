# Security fuzz targets are deliberately opt-in because libFuzzer is only
# available with Clang. Each target owns an explicit corpus and bounded runner.
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "JYPPX_BUILD_FUZZERS=ON requires Clang with libFuzzer")
endif()

add_executable(jyppx_fuzz_device_parse device_parse_fuzz.cpp)
target_link_libraries(jyppx_fuzz_device_parse PRIVATE jyppx_training_framework)
target_compile_options(jyppx_fuzz_device_parse PRIVATE -fsanitize=fuzzer)
target_link_options(jyppx_fuzz_device_parse PRIVATE -fsanitize=fuzzer)
jyppx_apply_instrumentation(jyppx_fuzz_device_parse)
jyppx_add_torch_runtime_dependency(jyppx_fuzz_device_parse)
