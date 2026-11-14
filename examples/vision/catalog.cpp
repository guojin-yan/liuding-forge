// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
/** @file catalog.cpp @brief Query installed model/task metadata without
 * allocating models. */
#include <jyppx/liuding/core/runtime.hpp>
#include <jyppx/liuding/vision/tasks/task.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>

#include <exception>
#include <iostream>

int main() {
  try {
    jyppx::liuding::initialize_runtime();
    namespace tasks = jyppx::liuding::vision::tasks;
    const auto catalog = tasks::vision_catalog();
    std::cout << "installed selections=" << catalog.size() << '\n';
    for (const auto& entry : catalog) {
      std::cout << entry.architecture
                << " task=" << tasks::task_name(entry.task)
                << " variant=" << entry.variant
                << " component=" << entry.component
                << " divisor=" << entry.input_divisor
                << " minimum=" << entry.minimum_input_size
                << " fixed=" << entry.fixed_input_size[0] << 'x'
                << entry.fixed_input_size[1]
                << " pretrained_weights=" << std::boolalpha
                << entry.pretrained_weights
                << " reference_qualified=" << entry.reference_qualified
                << " quality_qualified=" << entry.quality_qualified << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
