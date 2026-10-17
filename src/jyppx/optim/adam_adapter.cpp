// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/optim/adam_adapter.hpp>

#include <torch/optim/adam.h>

#define JYPPX_ADAM_FAMILY_ADAPTER AdamOptimizerAdapter
#define JYPPX_ADAM_FAMILY_OPTIONS AdamAdapterOptions
#define JYPPX_ADAM_FAMILY_GROUP AdamParameterGroup
#define JYPPX_ADAM_FAMILY_CHECKPOINT_OPTIONS AdamOptimizerCheckpointOptions
#define JYPPX_ADAM_FAMILY_PARTICIPANT AdamOptimizerCheckpointParticipant
#define JYPPX_ADAM_FAMILY_NATIVE_OPTIMIZER torch::optim::Adam
#define JYPPX_ADAM_FAMILY_NATIVE_OPTIONS torch::optim::AdamOptions
#define JYPPX_ADAM_FAMILY_NATIVE_STATE torch::optim::AdamParamState
#define JYPPX_ADAM_FAMILY_NAME "Adam"
#define JYPPX_ADAM_FAMILY_COMPONENT "optim.adam_adapter"
#define JYPPX_ADAM_FAMILY_MAGIC_TAG 'D'
#define JYPPX_ADAM_FAMILY_TOPOLOGY_MAGIC "JYPPXADM"
#define JYPPX_ADAM_FAMILY_CONTROL_MAGIC "JYPPXADC"
#define JYPPX_ADAM_FAMILY_MEDIA_STEM "adam"
#define JYPPX_ADAM_FAMILY_NATIVE_OPTIONS_NAME "torch::optim::AdamOptions"
#define JYPPX_ADAM_FAMILY_NATIVE_STATE_NAME "torch::optim::AdamParamState"
#define JYPPX_ADAM_FAMILY_ADAPTER_DESCRIPTION \
  "a non-null shared AdamOptimizerAdapter"

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
