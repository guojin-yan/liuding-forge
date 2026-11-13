// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#pragma once
#include <jyppx/liuding/vision/tasks/workflow/export.hpp>
#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
namespace jyppx::liuding::vision::tasks::detail {
struct ExtendedExport {
  std::vector<Tensor> examples;
  std::vector<std::string> input_names,output_names;
  onnx_export::ForwardFunction forward;
  std::string decoder;
};
bool extended_bundle(const RecipeSpec&);
ExtendedExport extended_export(const TaskBinding&,const RecipeSpec&,
    const VisionExportOptions&,const Tensor&);
void add_extended_onnx_lowerings(std::map<std::string,onnx_export::OperatorLowering>&);
std::string predict_extended_bundle(const std::filesystem::path&,
    std::span<const data::image::VisionSample>,std::stop_token);
}
