# Benchmarks. Built when JYPPX_BUILD_BENCHMARKS=ON (off by default).

jyppx_add_bench(jyppx_bench_data_pipeline bench_data_pipeline.cpp)
jyppx_add_bench(jyppx_bench_trainer_throughput bench_trainer_throughput.cpp)
jyppx_add_bench(jyppx_bench_optimizer_convergence bench_optimizer_convergence.cpp)
jyppx_add_bench(jyppx_bench_loss_vertical_slice
    LINK_TARGET jyppx::nn
    bench_loss_vertical_slice.cpp
)
jyppx_add_bench(jyppx_bench_adadelta_native_throughput bench_adadelta_native_throughput.cpp)
jyppx_add_bench(jyppx_bench_lamb_native_throughput bench_lamb_native_throughput.cpp)
jyppx_add_bench(jyppx_bench_adagrad_adapter_overhead bench_adagrad_adapter_overhead.cpp)
jyppx_add_bench(jyppx_bench_sgd_adapter_overhead bench_sgd_adapter_overhead.cpp)
jyppx_add_bench(jyppx_bench_adam_adapter_overhead bench_adam_adapter_overhead.cpp)
jyppx_add_bench(jyppx_bench_adamw_adapter_overhead bench_adamw_adapter_overhead.cpp)
jyppx_add_bench(jyppx_bench_nadam_native_throughput bench_nadam_native_throughput.cpp)
jyppx_add_bench(jyppx_bench_rmsprop_adapter_overhead bench_rmsprop_adapter_overhead.cpp)
jyppx_add_bench(jyppx_bench_memory_usage bench_memory_usage.cpp)
if(WIN32)
    target_link_libraries(jyppx_bench_data_pipeline PRIVATE Psapi)
    target_link_libraries(jyppx_bench_memory_usage PRIVATE Psapi)
endif()

# The target includes CUDA allocator APIs and is intentionally absent from
# CPU-only LibTorch builds rather than compiling a benchmark that can only skip.
if(TARGET torch_cuda OR TORCH_CUDA_VERSION)
    jyppx_add_bench(jyppx_bench_cuda_training bench_cuda_training.cpp)
    jyppx_add_bench(jyppx_bench_cuda_optimizer bench_cuda_optimizer.cpp)
endif()

# Resource evidence is opt-in with JYPPX_BUILD_BENCHMARKS. Preview executables
# are present only when their independently optional owner is really enabled.
function(jyppx_add_resource_bench name owner source)
    jyppx_add_bench(${name} LINK_TARGET ${owner} ${source})
    target_link_libraries(${name} PRIVATE jyppx::core nlohmann_json::nlohmann_json)
    if(WIN32)
        target_link_libraries(${name} PRIVATE Psapi)
    endif()
endfunction()

jyppx_add_resource_bench(jyppx_bench_resource_stable jyppx::sdk
    bench_resource_stable.cpp)
if(TARGET jyppx::models)
    jyppx_add_resource_bench(jyppx_bench_resource_models jyppx::models
        bench_resource_models.cpp)
endif()
if(TARGET jyppx::quantization)
    jyppx_add_resource_bench(jyppx_bench_resource_quantization jyppx::quantization
        bench_resource_quantization.cpp)
endif()
if(TARGET jyppx::serving)
    jyppx_add_resource_bench(jyppx_bench_resource_serving jyppx::serving
        bench_resource_serving.cpp)
endif()
if(TARGET torch_cuda OR TORCH_CUDA_VERSION)
    foreach(resource_target jyppx_bench_resource_models jyppx_bench_resource_serving)
        if(TARGET ${resource_target})
            target_compile_definitions(${resource_target} PRIVATE JYPPX_RESOURCE_CUDA=1)
            if(TARGET torch::cudart)
                target_link_libraries(${resource_target} PRIVATE torch::cudart)
            elseif(TARGET CUDA::cudart)
                target_link_libraries(${resource_target} PRIVATE CUDA::cudart)
            endif()
        endif()
    endforeach()
endif()
