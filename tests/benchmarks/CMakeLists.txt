# Benchmarks. Built when JYPPX_LIUDING_BUILD_BENCHMARKS=ON (off by default).

jyppx_add_bench(jyppx_bench_data_pipeline bench_data_pipeline.cpp)
jyppx_add_bench(jyppx_bench_trainer_throughput bench_trainer_throughput.cpp)
jyppx_add_bench(jyppx_bench_optimizer_convergence bench_optimizer_convergence.cpp)
jyppx_add_bench(jyppx_bench_loss_vertical_slice
    LINK_TARGET jyppx::liuding::nn
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
if(TARGET jyppx::liuding::data_image)
    jyppx_add_bench(jyppx_bench_image_decode
        LINK_TARGET jyppx::liuding::data_image
        bench_image_decode.cpp)
    target_link_libraries(jyppx_bench_image_decode PRIVATE nlohmann_json::nlohmann_json)
    jyppx_add_bench(jyppx_bench_image_training
        LINK_TARGET jyppx::liuding::data_image
        bench_image_training.cpp)
    target_link_libraries(jyppx_bench_image_training PRIVATE
        jyppx::liuding::sdk nlohmann_json::nlohmann_json)
    if(TARGET jyppx::liuding::models)
        target_link_libraries(jyppx_bench_image_training PRIVATE
            jyppx::liuding::models)
        target_compile_definitions(jyppx_bench_image_training PRIVATE
            JYPPX_BENCH_MODELS=1)
    endif()
    if(TARGET torch_cuda OR TORCH_CUDA_VERSION)
        target_compile_definitions(jyppx_bench_image_training PRIVATE
            JYPPX_BENCH_CUDA=1)
    endif()
endif()
if(WIN32)
    target_link_libraries(jyppx_bench_data_pipeline PRIVATE Psapi)
    target_link_libraries(jyppx_bench_memory_usage PRIVATE Psapi)
    if(TARGET jyppx_bench_image_decode)
        target_link_libraries(jyppx_bench_image_decode PRIVATE Psapi)
    endif()
    if(TARGET jyppx_bench_image_training)
        target_link_libraries(jyppx_bench_image_training PRIVATE Psapi)
    endif()
endif()

# Compile against the allocator and device APIs of the selected LibTorch.
# CPU-only builds have no accelerator targets.
if(TARGET torch_hip)
    foreach(suite training optimizer)
        jyppx_add_bench(jyppx_bench_rocm_${suite} bench_cuda_${suite}.cpp)
        target_compile_definitions(jyppx_bench_rocm_${suite} PRIVATE JYPPX_BENCH_ROCM=1)
        target_link_libraries(jyppx_bench_rocm_${suite} PRIVATE
            hip::host nlohmann_json::nlohmann_json)
    endforeach()
elseif(TARGET torch_cuda OR TORCH_CUDA_VERSION)
    foreach(suite training optimizer)
        jyppx_add_bench(jyppx_bench_cuda_${suite} bench_cuda_${suite}.cpp)
        target_link_libraries(jyppx_bench_cuda_${suite} PRIVATE nlohmann_json::nlohmann_json)
    endforeach()
endif()

# Stability mixes pressure, recovery and checkpoint I/O; it is separate from
# the performance sampling targets and is never registered as a default test.
if(TARGET torch_hip)
    jyppx_add_bench(jyppx_bench_rocm_stability bench_accelerator_stability.cpp)
    target_compile_definitions(jyppx_bench_rocm_stability PRIVATE JYPPX_BENCH_ROCM=1)
    target_link_libraries(jyppx_bench_rocm_stability PRIVATE
        hip::host nlohmann_json::nlohmann_json)
    if(WIN32)
        target_link_libraries(jyppx_bench_rocm_stability PRIVATE Psapi)
    endif()
elseif(TARGET torch_cuda OR TORCH_CUDA_VERSION)
    jyppx_add_bench(jyppx_bench_cuda_stability bench_accelerator_stability.cpp)
    target_link_libraries(jyppx_bench_cuda_stability PRIVATE nlohmann_json::nlohmann_json)
    if(WIN32)
        target_link_libraries(jyppx_bench_cuda_stability PRIVATE Psapi)
    endif()
endif()

# Resource evidence is opt-in with JYPPX_LIUDING_BUILD_BENCHMARKS. Preview executables
# are present only when their independently optional owner is really enabled.
function(jyppx_add_resource_bench name owner source)
    jyppx_add_bench(${name} LINK_TARGET ${owner} ${source})
    target_link_libraries(${name} PRIVATE jyppx::liuding::core nlohmann_json::nlohmann_json)
    if(WIN32)
        target_link_libraries(${name} PRIVATE Psapi)
    endif()
endfunction()

jyppx_add_resource_bench(jyppx_bench_resource_stable jyppx::liuding::sdk
    bench_resource_stable.cpp)
if(TARGET jyppx::liuding::models)
    jyppx_add_resource_bench(jyppx_bench_resource_models jyppx::liuding::models
        bench_resource_models.cpp)
endif()
if(TARGET jyppx::liuding::serving)
    jyppx_add_resource_bench(jyppx_bench_resource_serving jyppx::liuding::serving
        bench_resource_serving.cpp)
endif()
if(TARGET torch_hip OR TARGET torch_cuda OR TORCH_CUDA_VERSION)
    foreach(resource_target jyppx_bench_resource_models jyppx_bench_resource_serving)
        if(TARGET ${resource_target})
            if(TARGET torch_hip)
                target_compile_definitions(${resource_target} PRIVATE
                    JYPPX_RESOURCE_ROCM=1 JYPPX_BENCH_ROCM=1)
                target_link_libraries(${resource_target} PRIVATE hip::host)
            else()
                target_compile_definitions(${resource_target} PRIVATE JYPPX_RESOURCE_CUDA=1)
                if(TARGET torch::cudart)
                    target_link_libraries(${resource_target} PRIVATE torch::cudart)
                elseif(TARGET CUDA::cudart)
                    target_link_libraries(${resource_target} PRIVATE CUDA::cudart)
                endif()
            endif()
        endif()
    endforeach()
endif()
