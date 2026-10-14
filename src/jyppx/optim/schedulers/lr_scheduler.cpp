#include <jyppx/optim/schedulers/lr_scheduler.hpp>

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace jyppx {
namespace optim {
namespace schedulers {

namespace {
constexpr double kMaxExactInteger = 9007199254740991.0;

const char* step_mode_name(SchedulerStepMode mode) {
    switch (mode) {
        case SchedulerStepMode::batch: return "batch";
        case SchedulerStepMode::epoch: return "epoch";
        case SchedulerStepMode::metric: return "metric";
        case SchedulerStepMode::manual: return "manual";
    }
    return "unknown";
}

void append_descriptor(std::ostringstream& stream, const SchedulerDescriptor& descriptor) {
    stream << descriptor.schema_version << '|' << descriptor.type_id << '|'
           << step_mode_name(descriptor.step_mode) << '|'
           << (descriptor.checkpoint_safe ? '1' : '0') << '|';
    for (const auto& [key, value] : descriptor.config) {
        stream << key.size() << ':' << key << '=' << value.size() << ':' << value << ';';
    }
    stream << '[';
    for (const auto& child : descriptor.children) {
        append_descriptor(stream, child);
    }
    stream << ']';
}

uint64_t descriptor_hash(const SchedulerDescriptor& descriptor) {
    std::ostringstream canonical;
    append_descriptor(canonical, descriptor);
    uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : canonical.str()) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}
}

LRScheduler::LRScheduler(std::shared_ptr<Optimizer> optimizer)
    : optimizer_(std::move(optimizer)), step_count_(0), base_lr_(0.0), last_lr_(0.0) {
    if (!optimizer_) {
        throw std::invalid_argument("Scheduler optimizer cannot be null");
    }
    const auto lrs = optimizer_->get_lrs();
    if (lrs.empty()) {
        throw std::invalid_argument("Scheduler optimizer must contain a parameter group");
    }
    base_lrs_ = lrs;
    last_lrs_ = lrs;
    base_lr_ = lrs.front();
    last_lr_ = lrs.front();
    configure_descriptor(
        "jyppx.scheduler.unconfigured/v1", SchedulerStepMode::manual, {}, {}, false);
}

void LRScheduler::configure_descriptor(
    std::string type_id,
    SchedulerStepMode step_mode,
    std::map<std::string, std::string> config,
    std::vector<SchedulerDescriptor> children,
    bool checkpoint_safe) {
    if (type_id.empty()) {
        throw std::invalid_argument("Scheduler descriptor type_id cannot be empty");
    }
    descriptor_ = SchedulerDescriptor{
        1, std::move(type_id), step_mode, std::move(config),
        std::move(children), checkpoint_safe};
}

std::string LRScheduler::descriptor_number(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("Scheduler descriptor numbers must be finite");
    }
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return stream.str();
}

std::string LRScheduler::descriptor_integer(int64_t value) {
    return std::to_string(value);
}

std::string LRScheduler::descriptor_boolean(bool value) {
    return value ? "true" : "false";
}

std::string LRScheduler::descriptor_fingerprint() const {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16)
           << descriptor_hash(descriptor_);
    return stream.str();
}

void LRScheduler::validate_descriptor(
    const std::map<std::string, double>& state) const {
    const auto version = state.find("format_version");
    if (version == state.end() || version->second < 2.0) {
        return;
    }
    const auto schema = state.find("descriptor_schema_version");
    const auto low = state.find("descriptor_hash_low");
    const auto high = state.find("descriptor_hash_high");
    const auto safe = state.find("descriptor_checkpoint_safe");
    if (schema == state.end() || low == state.end() || high == state.end() ||
        safe == state.end()) {
        throw std::invalid_argument("Scheduler checkpoint descriptor is incomplete");
    }
    const uint64_t hash = descriptor_hash(descriptor_);
    const double expected_low = static_cast<double>(hash & 0xffffffffULL);
    const double expected_high = static_cast<double>((hash >> 32) & 0xffffffffULL);
    if (schema->second != static_cast<double>(descriptor_.schema_version) ||
        low->second != expected_low || high->second != expected_high ||
        safe->second != (descriptor_.checkpoint_safe ? 1.0 : 0.0)) {
        throw std::invalid_argument(
            "Scheduler checkpoint descriptor does not match the configured scheduler");
    }
}

double LRScheduler::get_last_lr() const {
    return last_lr_;
}

void LRScheduler::set_last_lrs(const std::vector<double>& lrs) {
    if (lrs.size() != base_lrs_.size()) {
        throw std::invalid_argument("Scheduler learning-rate count must match parameter groups");
    }
    optimizer_->set_lrs(lrs);
    last_lrs_ = lrs;
    last_lr_ = lrs.front();
}

std::map<std::string, double> LRScheduler::state_dict() const {
    std::map<std::string, double> state{
        {"format_version", 2.0},
        {"descriptor_schema_version", static_cast<double>(descriptor_.schema_version)},
        {"descriptor_hash_low", static_cast<double>(descriptor_hash(descriptor_) & 0xffffffffULL)},
        {"descriptor_hash_high", static_cast<double>((descriptor_hash(descriptor_) >> 32) & 0xffffffffULL)},
        {"descriptor_checkpoint_safe", descriptor_.checkpoint_safe ? 1.0 : 0.0},
        {"param_group_count", static_cast<double>(base_lrs_.size())},
        {"step_count", static_cast<double>(step_count_)}};
    for (size_t i = 0; i < last_lrs_.size(); ++i) {
        state["base_lr_" + std::to_string(i)] = base_lrs_[i];
        state["last_lr_" + std::to_string(i)] = last_lrs_[i];
    }
    return state;
}

void LRScheduler::decode_state_dict(
    const std::map<std::string, double>& state,
    bool allow_negative_lrs,
    int64_t& restored_step,
    std::vector<double>& restored_lrs) const {
    const bool has_format = state.count("format_version") != 0;
    if (has_format && state.at("format_version") != 1.0 &&
        state.at("format_version") != 2.0) {
        throw std::invalid_argument("Unsupported scheduler state format version");
    }
    validate_descriptor(state);
    if (optimizer_->param_groups().size() != base_lrs_.size()) {
        throw std::invalid_argument(
            "Scheduler parameter-group topology changed after construction");
    }
    if (has_format) {
        const auto count = state.find("param_group_count");
        if (count == state.end() || !std::isfinite(count->second) ||
            std::floor(count->second) != count->second || count->second < 0.0 ||
            count->second != static_cast<double>(base_lrs_.size())) {
            throw std::invalid_argument(
                "Scheduler checkpoint parameter-group topology does not match");
        }
        for (size_t i = 0; i < base_lrs_.size(); ++i) {
            const auto base = state.find("base_lr_" + std::to_string(i));
            if (base == state.end() || !std::isfinite(base->second) ||
                base->second != base_lrs_[i]) {
                throw std::invalid_argument(
                    "Scheduler checkpoint base learning rates do not match");
            }
        }
    }

    restored_step = step_count_;
    const auto step_it = state.find("step_count");
    if (step_it != state.end()) {
        if (!std::isfinite(step_it->second) || step_it->second < 0.0 ||
            std::floor(step_it->second) != step_it->second ||
            step_it->second > kMaxExactInteger) {
            throw std::invalid_argument(
                "Scheduler step_count must be an exactly representable non-negative integer");
        }
        restored_step = static_cast<int64_t>(step_it->second);
    } else if (has_format) {
        throw std::invalid_argument("Scheduler checkpoint is missing step_count");
    }

    restored_lrs = optimizer_->get_lrs();
    for (size_t i = 0; i < restored_lrs.size(); ++i) {
        const auto lr_it = state.find("last_lr_" + std::to_string(i));
        if (lr_it == state.end()) {
            if (has_format) {
                throw std::invalid_argument(
                    "Scheduler checkpoint is missing a parameter-group learning rate");
            }
            continue;
        }
        if (!std::isfinite(lr_it->second) ||
            (!allow_negative_lrs && lr_it->second < 0.0)) {
            throw std::invalid_argument(
                "Scheduler learning rates must be finite and non-negative");
        }
        restored_lrs[i] = lr_it->second;
    }
}

void LRScheduler::load_state_dict(const std::map<std::string, double>& state) {
    int64_t restored_step = step_count_;
    std::vector<double> restored_lrs;
    decode_state_dict(state, false, restored_step, restored_lrs);
    set_last_lrs(restored_lrs);
    step_count_ = restored_step;
}

} // namespace schedulers
} // namespace optim
} // namespace jyppx
