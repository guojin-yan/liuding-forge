// Unit tests for jyppx::core: device helpers and tensor utilities.

#include <jyppx/core/device.hpp>
#include <jyppx/core/tensor_utils.hpp>
#include <jyppx/core/types.hpp>

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <string>

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

TEST(DeviceTest, ParseInvalidDevicePreservesStableErrorContext) {
  const auto expect_invalid = [](const std::string& value,
                                 const std::string& actual) {
    try {
      (void)parse_device(value);
      FAIL() << "Invalid device input was accepted";
    } catch (const InvalidArgument& error) {
      EXPECT_EQ(error.code(), ErrorCode::invalid_argument);
      EXPECT_EQ(error.details().component, "core.device");
      EXPECT_EQ(error.details().operation, "parse_device");
      EXPECT_EQ(error.details().actual, actual);
      EXPECT_FALSE(error.details().expected.empty());
      EXPECT_FALSE(error.details().hint.empty());
    } catch (...) {
      FAIL() << "Invalid device input raised the wrong exception type";
    }
  };

  expect_invalid("", "empty");
  expect_invalid(std::string(65, 'c'), "more than 64 bytes");
  expect_invalid(std::string("cpu\0cuda", 8), "embedded NUL byte");
  expect_invalid("not-a-device", "rejected by the LibTorch device parser");
}

TEST(DeviceTest, DefaultDeviceIsValid) {
  const auto device = get_default_device();
  EXPECT_GE(cuda_device_count(), 0);
  if (cuda_is_available()) {
    EXPECT_EQ(device.type(), torch::kCUDA);
    EXPECT_GT(cuda_device_count(), 0);
  } else {
    EXPECT_EQ(device.type(), torch::kCPU);
  }
}

TEST(TensorUtilsTest, HasNanDetectsNaN) {
  EXPECT_FALSE(has_nan(torch::ones({3})));
  EXPECT_TRUE(
      has_nan(torch::full({3}, std::numeric_limits<double>::quiet_NaN())));
}

TEST(TensorUtilsTest, HasInfDetectsInfinity) {
  EXPECT_FALSE(has_inf(torch::ones({3})));
  EXPECT_TRUE(
      has_inf(torch::full({3}, std::numeric_limits<double>::infinity())));
}

TEST(TensorUtilsTest, IsFiniteRejectsNaNAndInf) {
  EXPECT_TRUE(is_finite(torch::tensor({0.5, -1.25})));
  EXPECT_FALSE(
      is_finite(torch::full({2}, std::numeric_limits<double>::quiet_NaN())));
  EXPECT_FALSE(
      is_finite(torch::full({2}, std::numeric_limits<double>::infinity())));
}

TEST(TensorUtilsTest, CloneTensorsProducesIndependentCopies) {
  auto first = torch::ones({2, 2});
  auto second = torch::full({3}, 2.0);
  TensorList originals{first, Tensor{}, second};

  const auto copies = clone_tensors(originals);

  ASSERT_EQ(copies.size(), originals.size());
  torch::NoGradGuard no_grad;
  copies[0].mul_(5.0);
  EXPECT_TRUE(torch::equal(first, torch::ones({2, 2})));
  EXPECT_FALSE(copies[1].defined());
}

TEST(TensorUtilsTest, ToDeviceMovesTensorsToCpu) {
  TensorList tensors{torch::ones({2}), Tensor{}, torch::zeros({3})};
  const auto moved = to_device(tensors, torch::kCPU);
  ASSERT_EQ(moved.size(), 3);
  for (const auto& tensor : moved) {
    if (tensor.defined()) {
      EXPECT_EQ(tensor.device().type(), torch::kCPU);
    }
  }
  EXPECT_FALSE(moved[1].defined());
}

}  // namespace
