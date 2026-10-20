add_executable(jyppx_basic_data_pipeline basic_data_pipeline.cpp)
target_link_libraries(jyppx_basic_data_pipeline PRIVATE jyppx::sdk)

add_executable(jyppx_basic_trainer basic_trainer.cpp)
target_link_libraries(jyppx_basic_trainer PRIVATE jyppx::sdk)

add_executable(jyppx_mnist mnist/mnist.cpp)
target_link_libraries(jyppx_mnist PRIVATE jyppx::sdk)

add_executable(jyppx_distillation distillation.cpp)
target_link_libraries(jyppx_distillation PRIVATE jyppx::sdk)

# LibTorch's Windows distribution keeps runtime DLLs beside its import
# libraries. Copy the DLLs (not the whole directory, which also holds large
# import libraries) once before the examples build.
foreach(example_target IN ITEMS
        jyppx_basic_data_pipeline jyppx_basic_trainer
        jyppx_mnist jyppx_distillation)
    jyppx_add_torch_runtime_dependency(${example_target})
endforeach()
