// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/optim/adamw_adapter.hpp>

#include <torch/optim/adamw.h>

#define JYPPX_ADAM_FAMILY_ADAPTER AdamWOptimizerAdapter
#define JYPPX_ADAM_FAMILY_OPTIONS AdamWAdapterOptions
#define JYPPX_ADAM_FAMILY_GROUP AdamWParameterGroup
#define JYPPX_ADAM_FAMILY_CHECKPOINT_OPTIONS AdamWOptimizerCheckpointOptions
#define JYPPX_ADAM_FAMILY_PARTICIPANT AdamWOptimizerCheckpointParticipant
#define JYPPX_ADAM_FAMILY_NATIVE_OPTIMIZER torch::optim::AdamW
#define JYPPX_ADAM_FAMILY_NATIVE_OPTIONS torch::optim::AdamWOptions
#define JYPPX_ADAM_FAMILY_NATIVE_STATE torch::optim::AdamWParamState
#define JYPPX_ADAM_FAMILY_NAME "AdamW"
#define JYPPX_ADAM_FAMILY_COMPONENT "optim.adamw_adapter"
#define JYPPX_ADAM_FAMILY_MAGIC_TAG 'W'
#define JYPPX_ADAM_FAMILY_TOPOLOGY_MAGIC "JYPPXAWM"
#define JYPPX_ADAM_FAMILY_CONTROL_MAGIC "JYPPXAWC"
#define JYPPX_ADAM_FAMILY_MEDIA_STEM "adamw"
#define JYPPX_ADAM_FAMILY_NATIVE_OPTIONS_NAME "torch::optim::AdamWOptions"
#define JYPPX_ADAM_FAMILY_NATIVE_STATE_NAME "torch::optim::AdamWParamState"
#define JYPPX_ADAM_FAMILY_ADAPTER_DESCRIPTION \
  "a non-null shared AdamWOptimizerAdapter"

#include "detail/adam_family_adapter.inc"

#undef JYPPX_ADAM_FAMILY_ADAPTER_DESCRIPTION
#undef JYPPX_ADAM_FAMILY_NATIVE_STATE_NAME
#undef JYPPX_ADAM_FAMILY_NATIVE_OPTIONS_NAME
#undef JYPPX_ADAM_FAMILY_MEDIA_STEM
#undef JYPPX_ADAM_FAMILY_CONTROL_MAGIC
#undef JYPPX_ADAM_FAMILY_TOPOLOGY_MAGIC
#undef JYPPX_ADAM_FAMILY_MAGIC_TAG
#undef JYPPX_ADAM_FAMILY_COMPONENT
#undef JYPPX_ADAM_FAMILY_NAME
#undef JYPPX_ADAM_FAMILY_NATIVE_STATE
#undef JYPPX_ADAM_FAMILY_NATIVE_OPTIONS
#undef JYPPX_ADAM_FAMILY_NATIVE_OPTIMIZER
#undef JYPPX_ADAM_FAMILY_PARTICIPANT
#undef JYPPX_ADAM_FAMILY_CHECKPOINT_OPTIONS
#undef JYPPX_ADAM_FAMILY_GROUP
#undef JYPPX_ADAM_FAMILY_OPTIONS
#undef JYPPX_ADAM_FAMILY_ADAPTER
