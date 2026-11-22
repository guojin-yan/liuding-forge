// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#include <jyppx/liuding/core/runtime.hpp>
#include <jyppx/liuding/serving/runtime.hpp>
#ifndef JYPPX_MIGRAPHX_SERVING_ONLY
#include "../../integration/deployment/migraphx_serving_fixture.hpp"
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
    if (argc != 2) throw std::runtime_error("immutable fixture root required");
    using namespace jyppx_test::migraphx_serving;
    Fixture fixture(argv[1]);
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
    return 0;
#endif
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
