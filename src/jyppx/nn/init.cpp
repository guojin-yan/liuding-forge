#include <jyppx/nn/init.hpp>

#include <torch/nn/init.h>

#include <stdexcept>
#include <string>

namespace jyppx {
namespace nn {
namespace init {

namespace {

// torch::nn::init functions take the tensor by value and modify the shared
// storage in place (guarded by NoGradGuard internally), so every wrapper below
// returns the caller's tensor with new contents and no autograd graph changes.

torch::nn::init::FanModeType parse_fan_mode(const std::string& mode) {
    if (mode == "fan_in") {
        return torch::kFanIn;
    }
    if (mode == "fan_out") {
        return torch::kFanOut;
    }
    throw std::invalid_argument("kaiming mode must be 'fan_in' or 'fan_out'");
}

} // namespace

void uniform_(Tensor& tensor, double a, double b) {
    torch::nn::init::uniform_(tensor, a, b);
}

void normal_(Tensor& tensor, double mean, double std) {
    torch::nn::init::normal_(tensor, mean, std);
}

void constant_(Tensor& tensor, double val) {
    torch::nn::init::constant_(tensor, val);
}

void zeros_(Tensor& tensor) {
    torch::nn::init::zeros_(tensor);
}

void ones_(Tensor& tensor) {
    torch::nn::init::ones_(tensor);
}

void xavier_uniform_(Tensor& tensor, double gain) {
    torch::nn::init::xavier_uniform_(tensor, gain);
}

void xavier_normal_(Tensor& tensor, double gain) {
    torch::nn::init::xavier_normal_(tensor, gain);
}

void kaiming_uniform_(Tensor& tensor, double a, const std::string& mode) {
    torch::nn::init::kaiming_uniform_(tensor, a, parse_fan_mode(mode), torch::kLeakyReLU);
}

void kaiming_normal_(Tensor& tensor, double a, const std::string& mode) {
    torch::nn::init::kaiming_normal_(tensor, a, parse_fan_mode(mode), torch::kLeakyReLU);
}

} // namespace init
} // namespace nn
} // namespace jyppx
