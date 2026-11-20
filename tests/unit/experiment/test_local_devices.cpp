// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/experiment/local_execution/devices.hpp>

#include <gtest/gtest.h>
#include <string>

namespace {
namespace e = jyppx::liuding::experiment;

TEST(LocalDeviceIdentity, DistinguishesCanonicalPhysicalCudaAndRocmIds) {
  const std::string amd = "AMD-0000-0a-00-0-0123456789abcdef0123456789abcdef";
  const std::string cuda = "GPU-01234567-89ab-cdef-0123-456789abcdef";
  EXPECT_TRUE(e::local_rocm_id_is_valid(amd));
  EXPECT_FALSE(e::local_cuda_uuid_is_valid(amd));
  EXPECT_TRUE(e::local_cuda_uuid_is_valid(cuda));
  EXPECT_FALSE(e::local_rocm_id_is_valid(cuda));
  for (std::size_t index = 0; index < amd.size(); ++index) {
    auto changed = amd;
    changed[index] = '!';
    EXPECT_FALSE(e::local_rocm_id_is_valid(changed)) << index;
    EXPECT_FALSE(e::local_rocm_id_is_valid(amd.substr(0, index))) << index;
  }
  EXPECT_FALSE(e::local_rocm_id_is_valid(amd + "0"));
  EXPECT_FALSE(e::local_rocm_id_is_valid(
      "AMD-0000-0a-00-0-00000000000000000000000000000000"));
  EXPECT_FALSE(e::local_rocm_id_is_valid(
      "AMD-0000-0a-20-0-0123456789abcdef0123456789abcdef"));
  EXPECT_FALSE(e::local_rocm_id_is_valid(
      "AMD-0000-0a-00-8-0123456789abcdef0123456789abcdef"));
}
}  // namespace
