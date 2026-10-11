#ifndef JYPPX_NN_QUANTIZATION_HPP
#define JYPPX_NN_QUANTIZATION_HPP

#include <jyppx/core/types.hpp>
#include <torch/nn/module.h>
#include <torch/serialize.h>

#include <stdexcept>
#include <string>
#include <vector>

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
    const Tensor& bias() const { return bias_; }

private:
    QuantizedTensor quantized_weight_;
    Tensor bias_;
};

/**
 * @brief One quantized linear entry in a whole-model snapshot
 *
 * `path` is the registered module path (e.g. "encoder.fc1") so the snapshot
 * can be matched back to the original model.
 */
struct NamedQuantizedLinear {
    std::string path;
    QuantizedTensor weight;
    Tensor bias;
};

/**
 * @brief Quantizes the weights of every 2D-weight (Linear-like) submodule
 *
 * Walks `named_modules()` recursively and quantizes each submodule whose
 * first parameter is a 2D tensor, keeping the registered path. The result is
 * a deployment snapshot: int8 weights with per-tensor scales, saveable with
 * save_snapshot(). This is weight-only quantization — activations stay float.
 */
inline std::vector<NamedQuantizedLinear> quantize_linear_weights(
    const torch::nn::Module& module) {
    // include_self=false: the root module is not a Linear-like child, and
    // include_self=true would require the module to live in a shared_ptr.
    const auto modules =
        module.named_modules(/*name_prefix=*/"", /*include_self=*/false);
    std::vector<NamedQuantizedLinear> snapshot;
    for (const auto& item : modules.items()) {
        const auto& path = item.key();
        const auto& child = item.value();
        const auto params = child->parameters(/*recurse=*/false);
        if (params.size() < 1 || !params[0].defined() || params[0].dim() != 2) {
            continue; // only Linear-like modules (2D weight)
        }
        NamedQuantizedLinear entry;
        entry.path = path; // child paths from named_modules(include_self=false)
        entry.weight = quantize_symmetric(params[0].detach());
        if (params.size() > 1 && params[1].defined()) {
            entry.bias = params[1].detach().to(torch::kFloat32);
        }
        snapshot.push_back(std::move(entry));
    }
    if (snapshot.empty()) {
        throw std::invalid_argument(
            "quantize_linear_weights found no 2D-weight modules to quantize");
    }
    return snapshot;
}

namespace detail {

// Tensor-only serialization helpers (paths stored as uint8 byte tensors).
inline Tensor string_to_tensor(const std::string& text) {
    std::vector<uint8_t> bytes(text.begin(), text.end());
    bytes.push_back(0);
    return torch::from_blob(bytes.data(),
                            {static_cast<int64_t>(bytes.size())},
                            torch::TensorOptions().dtype(torch::kUInt8))
        .clone();
}

inline std::string tensor_to_string(const Tensor& tensor) {
    const auto bytes = tensor.to(torch::kUInt8).contiguous();
    const auto* data = bytes.data_ptr<uint8_t>();
    return std::string(reinterpret_cast<const char*>(data),
                       bytes.numel() > 0 ? bytes.numel() - 1 : 0);
}

inline Tensor scalar_tensor(double value) {
    return torch::full({}, value, torch::TensorOptions().dtype(torch::kFloat64));
}

} // namespace detail

/**
 * @brief Saves a quantized snapshot to a LibTorch archive (tensor-only fields)
 */
inline void save_snapshot(const std::vector<NamedQuantizedLinear>& snapshot,
                          const std::string& path) {
    torch::serialize::OutputArchive archive;
    archive.write("count",
                  torch::full({}, static_cast<double>(snapshot.size())));
    for (size_t i = 0; i < snapshot.size(); ++i) {
        const std::string prefix = "entry/" + std::to_string(i) + "/";
        archive.write(prefix + "path", detail::string_to_tensor(snapshot[i].path));
        archive.write(prefix + "scale", detail::scalar_tensor(snapshot[i].weight.scale));
        archive.write(prefix + "zero_point",
                      detail::scalar_tensor(snapshot[i].weight.zero_point));
        archive.write(prefix + "data", snapshot[i].weight.data);
        if (snapshot[i].bias.defined()) {
            archive.write(prefix + "bias", snapshot[i].bias);
        }
    }
    archive.save_to(path);
}

/**
 * @brief Loads a quantized snapshot previously written by save_snapshot()
 */
inline std::vector<NamedQuantizedLinear> load_snapshot(const std::string& path) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    torch::Tensor count_tensor;
    archive.read("count", count_tensor);
    const auto count = static_cast<size_t>(count_tensor.item<double>());

    std::vector<NamedQuantizedLinear> snapshot;
    snapshot.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const std::string prefix = "entry/" + std::to_string(i) + "/";
        NamedQuantizedLinear entry;
        torch::Tensor path_tensor, scale_tensor, zero_tensor;
        archive.read(prefix + "path", path_tensor);
        entry.path = detail::tensor_to_string(path_tensor);
        archive.read(prefix + "scale", scale_tensor);
        archive.read(prefix + "zero_point", zero_tensor);
        entry.weight.scale = scale_tensor.item<double>();
        entry.weight.zero_point = static_cast<int32_t>(zero_tensor.item<double>());
        archive.read(prefix + "data", entry.weight.data);
        torch::Tensor bias;
        if (archive.try_read(prefix + "bias", bias)) {
            entry.bias = bias;
        }
        snapshot.push_back(std::move(entry));
    }
    if (snapshot.size() != count) {
        throw std::runtime_error("quantized snapshot is truncated");
    }
    return snapshot;
}

} // namespace quantization
} // namespace nn
} // namespace jyppx

#endif // JYPPX_NN_QUANTIZATION_HPP
