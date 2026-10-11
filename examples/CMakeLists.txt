add_executable(jyppx_basic_dataloader basic_dataloader.cpp)
target_link_libraries(jyppx_basic_dataloader PRIVATE jyppx_training_framework)

add_executable(jyppx_basic_trainer basic_trainer.cpp)
target_link_libraries(jyppx_basic_trainer PRIVATE jyppx_training_framework)

add_executable(jyppx_complete_training complete_training.cpp)
target_link_libraries(jyppx_complete_training PRIVATE jyppx_training_framework)

add_executable(jyppx_mnist mnist/mnist.cpp)
target_link_libraries(jyppx_mnist PRIVATE jyppx_training_framework)

# LibTorch's Windows distribution keeps runtime DLLs beside its import
# libraries. Copy the DLLs (not the whole directory, which also holds large
# import libraries) next to the examples so a freshly built example can be
# launched without manually editing PATH.
if(WIN32)
    get_filename_component(JYPPX_TORCH_RUNTIME_DIR "${TORCH_LIBRARY}" DIRECTORY)
    file(GLOB JYPPX_TORCH_RUNTIME_DLLS "${JYPPX_TORCH_RUNTIME_DIR}/*.dll")
    foreach(example_target IN ITEMS jyppx_basic_dataloader jyppx_basic_trainer)
        add_custom_command(TARGET ${example_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${JYPPX_TORCH_RUNTIME_DLLS}
                "$<TARGET_FILE_DIR:${example_target}>")
    endforeach()
endif()

