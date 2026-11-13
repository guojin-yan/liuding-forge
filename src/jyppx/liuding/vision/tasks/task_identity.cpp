// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/task.hpp>
#include <array>
namespace jyppx::liuding::vision::tasks {
namespace {
constexpr std::array<std::pair<TaskKind, std::string_view>, 10> names{{
  {TaskKind::detect,"detect"},{TaskKind::classify,"classify"},
  {TaskKind::instance_segment,"segment"},{TaskKind::pose,"pose"},
  {TaskKind::obb,"obb"},{TaskKind::classify_multilabel,"classify_multilabel"},
  {TaskKind::semantic_segment,"semantic_segment"},{TaskKind::depth,"depth"},
  {TaskKind::pretrain_simmim,"pretrain_simmim"},{TaskKind::reid,"reid"}}};
}
std::string_view task_name(TaskKind task) {
  for (auto [value,name]:names) if(value==task)return name;
  throw InvalidArgument("Unknown vision task identity");
}
TaskKind parse_task(std::string_view name) {
  for (auto [value,text]:names) if(text==name)return value;
  throw InvalidArgument("Unknown vision task name: "+std::string(name));
}
bool is_dense_task(TaskKind task) noexcept {
  return task==TaskKind::semantic_segment || task==TaskKind::depth ||
         task==TaskKind::pretrain_simmim;
}
}
