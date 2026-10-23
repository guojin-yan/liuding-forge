// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_MODELS_VISION_HPP
#define JYPPX_LIUDING_MODELS_VISION_HPP
/** @file vision.hpp
 * @brief Independently optional residual and mobile image classifiers.
 */
#include <jyppx/liuding/nn/model_adapter.hpp>

#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/pimpl.h>

#include <cstdint>
#include <vector>
namespace jyppx::liuding::models {
/** @brief The two supported ImageNet-style residual block schedules. */
enum class ResNetVariant : std::uint8_t {
  resnet18 = 1,  ///< Basic blocks in a 2/2/2/2 schedule.
  resnet50 = 2,  ///< Bottlenecks in a 3/4/6/3 schedule with v1.5 strides.
};
/** @brief Architecture options; inputs are already normalized NCHW images. */
struct ResNetOptions {
  std::int64_t num_classes = 1000;  ///< Positive classifier width.
  std::int64_t input_channels = 3;  ///< Positive stem input width.
  bool zero_init_residual = false;  ///< Zero final residual BN scale.
};
/** @brief Actual trainable torchvision-style ResNet18 or ResNet50. */
class ResNetImpl final : public torch::nn::Module {
 public:
  /** @brief Constructs a randomly initialized classifier, without downloading.
   * @param variant Fixed basic/bottleneck schedule.
   * @param options Input/classifier and initialization settings.
   */
  explicit ResNetImpl(ResNetVariant variant, ResNetOptions options = {});
  /** @brief Maps positive NCHW floating images to [N,num_classes] logits.
   * @param input Dense images on the module's exact device and dtype.
   * @return Unnormalized trainable logits; no implicit preprocessing.
   */
  Tensor forward(const Tensor& input);
  /** @brief Returns four residual-stage maps in increasing stride order.
   * @param input Same image contract as forward.
   * @return Stage outputs at nominal strides 4,8,16,32.
   */
  std::vector<Tensor> forward_features(const Tensor& input);
  /** @brief Returns immutable architecture settings.
   * @return Constructor options retained for the module lifetime.
   */
  const ResNetOptions& options() const noexcept;

 private:
  ResNetOptions options_;  ///< Validated immutable constructor configuration.
  torch::nn::Conv2d conv1_{nullptr};  ///< Registered ImageNet stem convolution.
  torch::nn::BatchNorm2d bn1_{nullptr};    ///< Registered stem normalization.
  torch::nn::Sequential layer1_{nullptr};  ///< Residual stage at stride four.
  torch::nn::Sequential layer2_{nullptr};  ///< Residual stage at stride eight.
  torch::nn::Sequential layer3_{
      nullptr};  ///< Residual stage at stride sixteen.
  torch::nn::Sequential layer4_{
      nullptr};                    ///< Residual stage at stride thirty-two.
  torch::nn::Linear fc_{nullptr};  ///< Registered unnormalized classifier.
};
/** @brief Shared owning LibTorch holder for the residual classifier. */
TORCH_MODULE(ResNet);
/** @brief Constructs the 2/2/2/2 basic-block classifier.
 * @param options Explicit image channels, class count and initialization.
 * @return Owning randomly initialized ResNet18 holder.
 */
ResNet resnet18(ResNetOptions options = {});
/** @brief Constructs the 3/4/6/3 bottleneck classifier.
 * @param options Explicit image channels, class count and initialization.
 * @return Owning randomly initialized ResNet50 holder.
 */
ResNet resnet50(ResNetOptions options = {});
/** @brief Creates an owning NCHW/logit schema-checked adapter.
 * @param model Nonempty shared holder whose mode and identity are preserved.
 * @return Adapter retaining the same registered module and state.
 */
nn::ModelAdapter adapt(ResNet model);
/** @brief Supported mobile inverted-residual schedules. */
enum class MobileNetV3Variant : std::uint8_t {
  small_variant = 1,  ///< Eleven inverted residual blocks and a 1024-wide head.
  large_variant =
      2,  ///< Fifteen inverted residual blocks and a 1280-wide head.
};
/** @brief MobileNetV3 classification and width settings. */
struct MobileNetV3Options {
  std::int64_t num_classes = 1000;  ///< Positive classifier width.
  std::int64_t input_channels = 3;  ///< Positive image channel count.
  double width_multiplier = 1.0;  ///< Finite value in [0.25,2], divisor eight.
  double dropout = 0.2;  ///< Training-only classifier dropout in [0,1).
};
/** @brief Trainable MobileNetV3 with depthwise blocks and squeeze excitation.
 */
class MobileNetV3Impl final : public torch::nn::Module {
 public:
  /** @brief Constructs the exact supported small/large block schedule.
   * @param variant Fixed block schedule before width scaling.
   * @param options Image channels, class count, width and dropout settings.
   */
  explicit MobileNetV3Impl(MobileNetV3Variant variant,
                           MobileNetV3Options options = {});
  /** @brief Maps normalized positive NCHW images to class logits.
   * @param input Dense floating image tensor matching module device and dtype.
   * @return Unnormalized [batch,num_classes] logits with normal autograd.
   */
  Tensor forward(const Tensor& input);
  /** @brief Returns the last convolutional NCHW feature map.
   * @param input Same image contract as forward.
   * @return Last registered convolutional feature map before pooling.
   */
  Tensor forward_features(const Tensor& input);
  /** @brief Returns immutable architecture settings.
   * @return Constructor options retained for the module lifetime.
   */
  const MobileNetV3Options& options() const noexcept;

 private:
  MobileNetV3Options options_;  ///< Validated immutable constructor options.
  torch::nn::Sequential features_{
      nullptr};  ///< Registered convolutional stack.
  torch::nn::Sequential classifier_{
      nullptr};  ///< Registered classification head.
};
/** @brief Shared owning LibTorch holder for the mobile classifier. */
TORCH_MODULE(MobileNetV3);
/** @brief Constructs the small 11-block mobile classifier.
 * @param options Image channels, class count, width and dropout settings.
 * @return Owning randomly initialized small MobileNetV3 holder.
 */
MobileNetV3 mobilenet_v3_small(MobileNetV3Options options = {});
/** @brief Constructs the large 15-block mobile classifier.
 * @param options Image channels, class count, width and dropout settings.
 * @return Owning randomly initialized large MobileNetV3 holder.
 */
MobileNetV3 mobilenet_v3_large(MobileNetV3Options options = {});
/** @brief Creates an owning mobile NCHW/logit schema-checked adapter.
 * @param model Nonempty shared holder whose mode and identity are preserved.
 * @return Adapter retaining the same registered module and state.
 */
nn::ModelAdapter adapt(MobileNetV3 model);
}  // namespace jyppx::liuding::models
#endif  // JYPPX_LIUDING_MODELS_VISION_HPP