# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0.
# Optional vision components share a physical root and retain separate targets.

if(JYPPX_LIUDING_BUILD_VISION_OPS)
    add_library(jyppx_liuding_vision_ops STATIC
        src/jyppx/liuding/vision/ops/boxes.cpp
        src/jyppx/liuding/vision/ops/assignment.cpp
        src/jyppx/liuding/vision/ops/deformable.cpp
        src/jyppx/liuding/vision/ops/roi.cpp
        src/jyppx/liuding/vision/ops/rotated.cpp)
    add_library(jyppx::liuding::vision_ops ALIAS jyppx_liuding_vision_ops)
    set_target_properties(jyppx_liuding_vision_ops PROPERTIES
        OUTPUT_NAME liuding-forge-vision-ops EXPORT_NAME vision_ops)
    target_include_directories(jyppx_liuding_vision_ops PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_link_libraries(jyppx_liuding_vision_ops
        PUBLIC jyppx_liuding_core PRIVATE jyppx_liuding_foundation)
    list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_vision_ops)
    list(APPEND JYPPX_ENABLED_COMPONENTS vision_ops)
    list(APPEND JYPPX_ENABLED_HEADER_DIRS vision/ops)
endif()

if(JYPPX_LIUDING_BUILD_MODELS_YOLO)
    add_library(jyppx_liuding_models_yolo STATIC
        src/jyppx/liuding/vision/models/yolo/yolov8.cpp
        src/jyppx/liuding/vision/models/yolo/yolov5.cpp
        src/jyppx/liuding/vision/models/yolo/fusion.cpp
        src/jyppx/liuding/vision/models/yolo/model.cpp)
    add_library(jyppx::liuding::models_yolo ALIAS jyppx_liuding_models_yolo)
    set_target_properties(jyppx_liuding_models_yolo PROPERTIES
        OUTPUT_NAME liuding-forge-models-yolo EXPORT_NAME models_yolo)
    target_include_directories(jyppx_liuding_models_yolo PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_link_libraries(jyppx_liuding_models_yolo
        PUBLIC jyppx_liuding_core jyppx_liuding_nn
        PRIVATE jyppx_liuding_foundation jyppx_liuding_interop)
    list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_models_yolo)
    list(APPEND JYPPX_ENABLED_COMPONENTS models_yolo)
    list(APPEND JYPPX_ENABLED_HEADER_DIRS vision/models/yolo)
endif()

if(JYPPX_LIUDING_BUILD_MODELS_RCNN)
    add_library(jyppx_liuding_models_rcnn STATIC
        src/jyppx/liuding/vision/models/rcnn/faster_rcnn.cpp)
    add_library(jyppx::liuding::models_rcnn ALIAS jyppx_liuding_models_rcnn)
    set_target_properties(jyppx_liuding_models_rcnn PROPERTIES
        OUTPUT_NAME liuding-forge-models-rcnn EXPORT_NAME models_rcnn)
    target_include_directories(jyppx_liuding_models_rcnn PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_link_libraries(jyppx_liuding_models_rcnn
        PUBLIC jyppx_liuding_models jyppx_liuding_nn
        PRIVATE jyppx_liuding_core jyppx_liuding_foundation jyppx_liuding_vision_ops)
    list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_models_rcnn)
    list(APPEND JYPPX_ENABLED_COMPONENTS models_rcnn)
    list(APPEND JYPPX_ENABLED_HEADER_DIRS vision/models/rcnn)
endif()

if(JYPPX_LIUDING_BUILD_MODELS_VISION)
    add_library(jyppx_liuding_models_vision STATIC
        src/jyppx/liuding/vision/models/extended/pyramid.cpp
        src/jyppx/liuding/vision/models/extended/detectors.cpp
        src/jyppx/liuding/vision/models/extended/dense.cpp
        src/jyppx/liuding/vision/models/extended/rtdetr.cpp
        src/jyppx/liuding/vision/models/extended/vit_adapter.cpp
        src/jyppx/liuding/vision/models/extended/text.cpp
        src/jyppx/liuding/vision/models/extended/open_vocabulary.cpp
        src/jyppx/liuding/vision/models/extended/region.cpp)
    add_library(jyppx::liuding::models_vision ALIAS jyppx_liuding_models_vision)
    set_target_properties(jyppx_liuding_models_vision PROPERTIES
        OUTPUT_NAME liuding-forge-models-vision EXPORT_NAME models_vision)
    target_include_directories(jyppx_liuding_models_vision PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_link_libraries(jyppx_liuding_models_vision PUBLIC jyppx_liuding_models
        jyppx_liuding_nn PRIVATE jyppx_liuding_core jyppx_liuding_foundation jyppx_liuding_vision_ops)
    list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_models_vision)
    list(APPEND JYPPX_ENABLED_COMPONENTS models_vision)
    list(APPEND JYPPX_ENABLED_HEADER_DIRS vision/models/extended)
endif()

if(JYPPX_LIUDING_BUILD_VISION_TASKS)
    add_library(jyppx_liuding_vision_tasks STATIC
        src/jyppx/liuding/vision/tasks/task.cpp
        src/jyppx/liuding/vision/tasks/task_identity.cpp
        src/jyppx/liuding/vision/tasks/extended.cpp
        src/jyppx/liuding/vision/tasks/advanced_metrics.cpp
        src/jyppx/liuding/vision/tasks/workflow/evaluate_extended.cpp
        src/jyppx/liuding/vision/tasks/data/context.cpp
        src/jyppx/liuding/vision/tasks/data/extended.cpp
        src/jyppx/liuding/vision/tasks/data/sampling.cpp
        src/jyppx/liuding/vision/tasks/tracking/workflow.cpp
        src/jyppx/liuding/vision/tasks/tracking/tracker.cpp
        src/jyppx/liuding/vision/tasks/tracking/metrics.cpp
        src/jyppx/liuding/vision/tasks/original_v5.cpp
        src/jyppx/liuding/vision/tasks/faster_rcnn.cpp
        src/jyppx/liuding/vision/tasks/dense_detection.cpp
        src/jyppx/liuding/vision/tasks/rtdetr.cpp
        src/jyppx/liuding/vision/tasks/open_vocabulary.cpp
        src/jyppx/liuding/vision/tasks/results.cpp
        src/jyppx/liuding/vision/tasks/metrics.cpp
        src/jyppx/liuding/vision/tasks/workflow/recipe.cpp
        src/jyppx/liuding/vision/tasks/workflow/binding.cpp
        src/jyppx/liuding/vision/tasks/workflow/binding_disabled.cpp
        src/jyppx/liuding/vision/tasks/workflow/suite.cpp
        src/jyppx/liuding/vision/tasks/workflow/search.cpp
        src/jyppx/liuding/vision/tasks/workflow/output.cpp
        src/jyppx/liuding/vision/tasks/workflow/preflight.cpp
        src/jyppx/liuding/vision/tasks/yolo_detection.cpp
        src/jyppx/liuding/vision/tasks/prediction.cpp
        src/jyppx/liuding/vision/tasks/evaluation.cpp
        src/jyppx/liuding/vision/tasks/training/strategy.cpp
        src/jyppx/liuding/vision/tasks/training/advanced.cpp
        src/jyppx/liuding/vision/tasks/training/distillation.cpp
        src/jyppx/liuding/vision/tasks/training/ema.cpp
        src/jyppx/liuding/vision/tasks/training/musgd.cpp
        src/jyppx/liuding/vision/tasks/training/model_selection.cpp
        src/jyppx/liuding/vision/tasks/training/transfer.cpp
        src/jyppx/liuding/vision/tasks/training/weight_conversion.cpp
        src/jyppx/liuding/vision/tasks/data/orientation.cpp
        src/jyppx/liuding/vision/tasks/data/inspection.cpp
        src/jyppx/liuding/vision/tasks/data/augmentation.cpp
        src/jyppx/liuding/vision/tasks/data/composition.cpp
        src/jyppx/liuding/vision/tasks/data/classification.cpp
        src/jyppx/liuding/vision/tasks/data/targets.cpp
        src/jyppx/liuding/vision/tasks/data/preparation.cpp
        src/jyppx/liuding/vision/tasks/data/dataset.cpp)
    add_library(jyppx::liuding::vision_tasks ALIAS jyppx_liuding_vision_tasks)
    if(MSVC)
        target_compile_options(jyppx_liuding_vision_tasks PRIVATE /MP2)
    endif()
    set_target_properties(jyppx_liuding_vision_tasks PROPERTIES
        OUTPUT_NAME liuding-forge-vision-tasks EXPORT_NAME vision_tasks)
    target_include_directories(jyppx_liuding_vision_tasks PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_link_libraries(jyppx_liuding_vision_tasks
        PUBLIC jyppx_liuding_core jyppx_liuding_nn jyppx_liuding_optim jyppx_liuding_training
               jyppx_liuding_data_image jyppx_liuding_data
        PRIVATE jyppx_liuding_foundation
                jyppx_liuding_experiment jyppx_liuding_interop jyppx_liuding_vision_ops
                $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)
    list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_vision_tasks)
    foreach(_pack IN ITEMS models models_yolo models_rcnn models_vision)
        if(TARGET jyppx_liuding_${_pack})
            target_link_libraries(jyppx_liuding_vision_tasks PRIVATE jyppx_liuding_${_pack})
        endif()
    endforeach()
    if(TARGET jyppx_liuding_onnx_export)
        target_sources(jyppx_liuding_vision_tasks PRIVATE src/jyppx/liuding/vision/tasks/workflow/export.cpp src/jyppx/liuding/vision/tasks/workflow/onnx_lowering.cpp src/jyppx/liuding/vision/tasks/workflow/extended_onnx.cpp)
        target_link_libraries(jyppx_liuding_vision_tasks PRIVATE jyppx_liuding_onnx_export)
        if(TARGET jyppx_liuding_models_rcnn)
            target_sources(jyppx_liuding_vision_tasks PRIVATE src/jyppx/liuding/vision/tasks/workflow/rcnn_onnx.cpp)
        endif()
        target_compile_definitions(jyppx_liuding_vision_tasks PRIVATE JYPPX_VISION_WITH_ONNX_EXPORT=1)
    else()
        target_sources(jyppx_liuding_vision_tasks PRIVATE src/jyppx/liuding/vision/tasks/workflow/export_disabled.cpp)
    endif()
    if(TARGET torch_cuda)
        target_compile_definitions(jyppx_liuding_vision_tasks PRIVATE JYPPX_VISION_WITH_CUDA=1)
    endif()
    if(TARGET jyppx_liuding_telemetry_native)
        target_sources(jyppx_liuding_vision_tasks PRIVATE src/jyppx/liuding/vision/tasks/workflow/telemetry.cpp)
        target_link_libraries(jyppx_liuding_vision_tasks PRIVATE jyppx_liuding_telemetry_native)
    else()
        target_sources(jyppx_liuding_vision_tasks PRIVATE src/jyppx/liuding/vision/tasks/workflow/telemetry_disabled.cpp)
    endif()
    if(TARGET jyppx_liuding_vision_render)
        target_sources(jyppx_liuding_vision_tasks PRIVATE src/jyppx/liuding/vision/tasks/workflow/render.cpp)
        target_link_libraries(jyppx_liuding_vision_tasks PRIVATE jyppx_liuding_vision_render)
    else()
        target_sources(jyppx_liuding_vision_tasks PRIVATE src/jyppx/liuding/vision/tasks/workflow/render_disabled.cpp)
    endif()
    if(TARGET jyppx_liuding_onnx_runtime)
        target_sources(jyppx_liuding_vision_tasks PRIVATE src/jyppx/liuding/vision/tasks/workflow/bundle.cpp src/jyppx/liuding/vision/tasks/workflow/bundle_extended.cpp)
        target_link_libraries(jyppx_liuding_vision_tasks PRIVATE jyppx_liuding_onnx_runtime)
        target_compile_definitions(jyppx_liuding_vision_tasks PRIVATE JYPPX_VISION_WITH_ONNX_RUNTIME=1)
    else()
        target_sources(jyppx_liuding_vision_tasks PRIVATE src/jyppx/liuding/vision/tasks/workflow/bundle_disabled.cpp)
    endif()
    if(TARGET jyppx_liuding_models)
        target_sources(jyppx_liuding_vision_tasks PRIVATE src/jyppx/liuding/vision/tasks/workflow/binding_classification.cpp)
        target_compile_definitions(jyppx_liuding_vision_tasks PRIVATE JYPPX_VISION_WITH_MODELS=1)
    endif()
    if(TARGET jyppx_liuding_models_yolo)
        target_sources(jyppx_liuding_vision_tasks PRIVATE src/jyppx/liuding/vision/tasks/workflow/binding_yolo.cpp)
        target_compile_definitions(jyppx_liuding_vision_tasks PRIVATE JYPPX_VISION_WITH_YOLO=1)
    endif()
    if(TARGET jyppx_liuding_models_vision)
        target_sources(jyppx_liuding_vision_tasks PRIVATE src/jyppx/liuding/vision/tasks/workflow/binding_extended.cpp)
        target_compile_definitions(jyppx_liuding_vision_tasks PRIVATE JYPPX_VISION_WITH_EXTENDED=1)
    endif()
    if(TARGET jyppx_liuding_models_rcnn)
        target_sources(jyppx_liuding_vision_tasks PRIVATE src/jyppx/liuding/vision/tasks/workflow/binding_rcnn.cpp)
        target_compile_definitions(jyppx_liuding_vision_tasks PRIVATE JYPPX_VISION_WITH_RCNN=1)
    endif()
    list(APPEND JYPPX_ENABLED_COMPONENTS vision_tasks)
    list(APPEND JYPPX_ENABLED_HEADER_DIRS vision/tasks)
endif()
