// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DEPLOYMENT_ONNX_EXPORT_GENERIC_LOWERING_HPP
#define JYPPX_LIUDING_DEPLOYMENT_ONNX_EXPORT_GENERIC_LOWERING_HPP

#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>

#include <torch/csrc/jit/ir/ir.h>

#include <map>
#include <memory>
#include <string>

namespace jyppx::liuding::onnx_export::detail {

void lower_to_onnx(const std::shared_ptr<torch::jit::Graph>& graph,
                   const ExportOptions& options,
                   std::map<std::string, at::Tensor>& initializers);

}  // namespace jyppx::liuding::onnx_export::detail

#endif  // JYPPX_LIUDING_DEPLOYMENT_ONNX_EXPORT_GENERIC_LOWERING_HPP
