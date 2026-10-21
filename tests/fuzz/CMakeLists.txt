# Security fuzz targets are deliberately opt-in because libFuzzer is only
# available with Clang. Each target owns an explicit corpus and bounded runner.
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "JYPPX_BUILD_FUZZERS=ON requires Clang with libFuzzer")
endif()

# Instrument the implementation archives, not just LLVMFuzzerTestOneInput.
# This directory is entered only for the explicit fuzz build. Consumers that
# provide their own main use the no-main coverage runtime; each fuzz target
# below additionally links libFuzzer's main. Third-party dependencies stay out.
foreach(implementation IN LISTS JYPPX_ENABLED_TARGETS)
    get_target_property(implementation_type ${implementation} TYPE)
    if(implementation_type STREQUAL "STATIC_LIBRARY" OR
       implementation_type STREQUAL "SHARED_LIBRARY" OR
       implementation_type STREQUAL "OBJECT_LIBRARY")
        target_compile_options(${implementation} PRIVATE -fsanitize=fuzzer-no-link)
    endif()
endforeach()

function(jyppx_add_fuzz_target target source)
    add_executable(${target} ${source})
    set_property(TARGET ${target} PROPERTY JYPPX_LIBFUZZER_MAIN TRUE)
    target_link_libraries(${target} PRIVATE jyppx::sdk)
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

jyppx_add_fuzz_target(jyppx_fuzz_telemetry_codec telemetry_codec_fuzz.cpp)

if(TARGET jyppx::interop)
    jyppx_add_fuzz_target(jyppx_fuzz_checkpoint_manifest
                          checkpoint_manifest_fuzz.cpp)
    jyppx_add_fuzz_target(jyppx_fuzz_model_bundle_manifest
                          model_bundle_manifest_fuzz.cpp)
    jyppx_add_fuzz_target(jyppx_fuzz_safetensors safetensors_fuzz.cpp)
endif()

if(TARGET jyppx::experiment)
    jyppx_add_fuzz_target(jyppx_fuzz_evaluation_json evaluation_json_fuzz.cpp)
    jyppx_add_fuzz_target(jyppx_fuzz_backend_protocol backend_protocol_fuzz.cpp)
    target_link_libraries(jyppx_fuzz_backend_protocol
        PRIVATE nlohmann_json::nlohmann_json)
endif()
