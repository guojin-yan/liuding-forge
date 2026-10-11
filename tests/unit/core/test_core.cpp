// Unit tests for jyppx::core: device helpers and tensor utilities.

#include <jyppx/core/device.hpp>
#include <jyppx/core/tensor_utils.hpp>
#include <jyppx/core/types.hpp>

#include <gtest/gtest.h>

#include <limits>

namespace {

using namespace jyppx;

TEST(DeviceTest, ParseDeviceStrings) {
    const auto cpu = parse_device("cpu");
    EXPECT_EQ(cpu.type(), torch::kCPU);

    const auto cuda = parse_device("cuda");
    EXPECT_EQ(cuda.type(), torch::kCUDA);

    const auto cuda_indexed = parse_device("cuda:1");
    EXPECT_EQ(cuda_indexed.type(), torch::kCUDA);
    EXPECT_EQ(cuda_indexed.index(), 1);
}

TEST(DeviceTest, ParseInvalidDeviceThrows) {
    EXPECT_THROW(parse_device("not-a-device"), std::exception);
}

TEST(DeviceTest, DefaultDeviceIsValid) {
    const auto device = get_default_device();
    if (cuda_is_available()) {
        EXPECT_EQ(device.type(), torch::kCUDA);
        EXPECT_GT(cuda_device_count(), 0);
    } else {
        EXPECT_EQ(device.type(), torch::kCPU);
    }
}

TEST(TensorUtilsTest, HasNanDetectsNaN) {
    EXPECT_FALSE(has_nan(torch::ones({3})));
    EXPECT_TRUE(has_nan(torch::full({3}, std::numeric_limits<double>::quiet_NaN())));
}

TEST(TensorUtilsTest, HasInfDetectsInfinity) {
    EXPECT_FALSE(has_inf(torch::ones({3})));
    EXPECT_TRUE(has_inf(torch::full({3}, std::numeric_limits<double>::infinity())));
}

TEST(TensorUtilsTest, IsFiniteRejectsNaNAndInf) {
    EXPECT_TRUE(is_finite(torch::tensor({0.5, -1.25})));
    EXPECT_FALSE(is_finite(torch::full({2}, std::numeric_limits<double>::quiet_NaN())));
    EXPECT_FALSE(is_finite(torch::full({2}, std::numeric_limits<double>::infinity())));
}

TEST(TensorUtilsTest, CloneTensorsProducesIndependentCopies) {
    auto first = torch::ones({2, 2});
    auto second = torch::full({3}, 2.0);
    TensorList originals{first, second};

    const auto copies = clone_tensors(originals);

    ASSERT_EQ(copies.size(), originals.size());
    torch::NoGradGuard no_grad;
    copies[0].mul_(5.0);
    EXPECT_TRUE(torch::equal(first, torch::ones({2, 2})));
}

TEST(TensorUtilsTest, ToDeviceMovesTensorsToCpu) {
    TensorList tensors{torch::ones({2}), torch::zeros({3})};
    const auto moved = to_device(tensors, torch::kCPU);
    ASSERT_EQ(moved.size(), 2);
    for (const auto& tensor : moved) {
        EXPECT_EQ(tensor.device().type(), torch::kCPU);
    }
}

} // namespace
