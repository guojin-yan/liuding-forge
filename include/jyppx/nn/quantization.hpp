#ifndef JYPPX_NN_QUANTIZATION_HPP
#define JYPPX_NN_QUANTIZATION_HPP

#include <jyppx/core/types.hpp>

#include <stdexcept>

namespace jyppx {
namespace nn {
namespace quantization {

/**
 * @brief An int8-quantized tensor with its dequantization parameters
 */
struct QuantizedTensor {
    Tensor data;        ///< int8 quantized values
    double scale = 1.0; ///< dequantization multiplier
    int32_t zero_point = 0; ///< dequantization offset
};

/**
 * @brief Symmetric per-tensor int8 quantization
 *
 * scale = max(|values|) / 127, zero point 0, so a zero maps exactly to zero
 * and dequantization is a single multiply.
 */
inline QuantizedTensor quantize_symmetric(const Tensor& values) {
    auto floats = values.to(torch::kFloat32);
    const double amplitude =
        floats.abs().max().item<double>();
    if (amplitude == 0.0) {
        return {torch::zeros_like(floats, torch::TensorOptions().dtype(torch::kInt8)),
                1.0, 0};
    }
    const double scale = amplitude / 127.0;
    auto quantized =
        torch::round(floats / scale).clamp(-128.0, 127.0).to(torch::kInt8);
    return {std::move(quantized), scale, 0};
}

/**
 * @brief Dequantizes back to float32: scale * (q - zero_point)
 */
inline Tensor dequantize(const QuantizedTensor& quantized) {
    return quantized.scale *
           (quantized.data.to(torch::kFloat32) -
            static_cast<float>(quantized.zero_point));
}

/**
 * @brief Weight-only int8 quantized linear layer computation
 *
 * y = x @ dequantize(W).T + b. Weights are quantized once at construction
 * (weights are inference-static); activations stay float, which keeps the
 * memory benefit without requiring quantized-kernel kernels.
 */
class QuantizedLinear {
public:
    QuantizedLinear(const Tensor& weight, const Tensor& bias) {
        if (!weight.defined() || weight.dim() != 2) {
            throw std::invalid_argument("QuantizedLinear weight must be {out, in}");
        }
        quantized_weight_ = quantize_symmetric(weight);
        if (bias.defined()) {
            bias_ = bias.to(torch::kFloat32);
        }
    }

    static QuantizedLinear from_linear(const torch::nn::Module& linear) {
        const auto params = linear.parameters();
        if (params.empty()) {
            throw std::invalid_argument("from_linear requires a module with parameters");
        }
        return QuantizedLinear(params[0].detach(),
                               params.size() > 1 ? params[1].detach() : Tensor{});
    }

    Tensor forward(const Tensor& input) const {
        auto weight = dequantize(quantized_weight_);
        Tensor output = torch::matmul(input.to(torch::kFloat32), weight.t());
        if (bias_.defined()) {
            output = output + bias_;
        }
        return output;
    }

    const QuantizedTensor& quantized_weight() const { return quantized_weight_; }

private:
    QuantizedTensor quantized_weight_;
    Tensor bias_;
};

} // namespace quantization
} // namespace nn
} // namespace jyppx

#endif // JYPPX_NN_QUANTIZATION_HPP
