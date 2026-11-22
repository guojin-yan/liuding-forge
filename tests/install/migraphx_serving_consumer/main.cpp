// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#include <jyppx/liuding/core/runtime.hpp>
#include <jyppx/liuding/serving/runtime.hpp>
#ifndef JYPPX_MIGRAPHX_SERVING_ONLY
#include "../../integration/deployment/migraphx_serving_fixture.hpp"
#include "../../integration/deployment/native_modules.hpp"
#endif
#include <iostream>

int main(int argc, char** argv) {
  try {
    jyppx::liuding::initialize_runtime();
#ifdef JYPPX_MIGRAPHX_SERVING_ONLY
    jyppx::liuding::serving::validate_request_resources(
        {.positional = {torch::ones({2, 2})}, .keyword = {}}, {}, {});
    return 0;
#else
    if (argc != 2 && argc != 4)
      throw std::runtime_error("fixture root and optional model/case required");
    const std::string_view model = argc == 4 ? argv[2] : "linear";
    const std::string_view sample = argc == 4 ? argv[3] : "0";
    if ((model != "linear" && model != "resnet18") ||
        (sample != "0" && sample != "1"))
      throw std::runtime_error("explicit fixed model/case required");
    using namespace jyppx_test::migraphx_serving;
    Fixture fixture(argv[1], sample == "1" ? 1 : 0, model == "resnet18");
    Bundle bundle("owner", fixture);
    const auto directory = bundle.Publish();
    const std::array known{fixture.Known()};
    auto predictor = bundle.Backend(fixture).load(directory, {}, known);
    auto response = predictor->predict(fixture.Request("migraphx-v1"));
    Compare(response.output.value, fixture);
    WriteOutput("actual.f32", response.output.value);
    if (!predictor->close()) throw std::runtime_error("close incomplete");
    predictor.reset();
    Compare(response.output.value, fixture);
    std::ofstream lock("environment-lock.json", std::ios::binary);
    lock << mgx::serialize_environment_lock(fixture.environment);
    lock.close();
    if (!lock) throw std::runtime_error("environment evidence write");
    jyppx_test::Modules("modules.txt");
    return 0;
#endif
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
