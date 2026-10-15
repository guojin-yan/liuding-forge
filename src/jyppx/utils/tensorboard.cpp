#include <jyppx/utils/tensorboard.hpp>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace jyppx {
namespace utils {

namespace {

// ---- protobuf wire-format primitives (hand-rolled; TensorBoard's Event and
// Summary messages are simple enough to encode directly). --------------------

// Appends a varint-encoded field key: (field_number << 3) | wire_type.
void append_key(std::string& out, uint32_t field, uint32_t wire_type) {
    const uint32_t key = (field << 3) | wire_type;
    uint32_t value = key;
    while (value >= 0x80) {
        out.push_back(static_cast<char>((value & 0x7F) | 0x80));
        value >>= 7;
    }
    out.push_back(static_cast<char>(value));
}

void append_varint(std::string& out, uint64_t value) {
    while (value >= 0x80) {
        out.push_back(static_cast<char>((value & 0x7F) | 0x80));
        value >>= 7;
    }
    out.push_back(static_cast<char>(value));
}

void append_string_field(std::string& out, uint32_t field, const std::string& value) {
    append_key(out, field, /*length-delimited=*/2);
    append_varint(out, value.size());
    out.append(value);
}

void append_double_field(std::string& out, uint32_t field, double value) {
    append_key(out, field, /*64-bit=*/1);
    char bytes[8];
    std::memcpy(bytes, &value, 8); // x86/ARM are little-endian, matching protobuf
    out.append(bytes, 8);
}

void append_float_field(std::string& out, uint32_t field, float value) {
    append_key(out, field, /*32-bit=*/5);
    char bytes[4];
    std::memcpy(bytes, &value, 4);
    out.append(bytes, 4);
}

// Event fields: 1 = wall_time (double), 2 = step (varint), 3 = summary,
// 5 = file_version (string). Summary: 1 = repeated Value.
// Value: 1 = node_name (string), 2 = simple_value (float), 8 = tensor.
std::string encode_scalar_event(const std::string& tag, float value, int64_t step,
                                double wall_time) {
    std::string value_msg;
    append_string_field(value_msg, 1, tag);
    append_float_field(value_msg, 2, value);

    std::string summary;
    append_string_field(summary, 1, value_msg);

    std::string event;
    append_double_field(event, 1, wall_time);
    append_key(event, 2, 0);
    append_varint(event, static_cast<uint64_t>(step));
    append_string_field(event, 3, summary);
    return event;
}

std::string encode_text_event(const std::string& tag, const std::string& text,
                              int64_t step, double wall_time) {
    // TensorProto { dtype: DT_STRING(8), tensor_shape: {dim: [size 1]},
    //               string_val: [text] }
    std::string tensor;
    append_key(tensor, 1, 0);
    append_varint(tensor, 8); // DT_STRING
    std::string shape_dim;
    append_key(shape_dim, 1, 0);
    append_varint(shape_dim, 1);
    std::string shape;
    append_string_field(shape, 2, shape_dim);
    append_string_field(tensor, 2, shape);
    append_string_field(tensor, 8, text);

    std::string value_msg;
    append_string_field(value_msg, 1, tag);
    append_string_field(value_msg, 8, tensor);

    std::string summary;
    append_string_field(summary, 1, value_msg);

    std::string event;
    append_double_field(event, 1, wall_time);
    append_key(event, 2, 0);
    append_varint(event, static_cast<uint64_t>(step));
    append_string_field(event, 3, summary);
    return event;
}

// ---- TFRecord framing --------------------------------------------------------

uint32_t crc32c(const char* data, size_t length) {
    static uint32_t table[256];
    static const bool table_ready = [] {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t crc = i;
            for (int bit = 0; bit < 8; ++bit) {
                crc = (crc & 1) != 0 ? (0x82F63B78u ^ (crc >> 1)) : (crc >> 1);
            }
            table[i] = crc;
        }
        return true;
    }();
    (void)table_ready;

    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < length; ++i) {
        crc = table[(crc ^ static_cast<unsigned char>(data[i])) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

uint32_t masked_crc(const char* data, size_t length) {
    const uint32_t crc = crc32c(data, length);
    return ((crc >> 15) | (crc << 17)) + 0xA282EAD8u;
}

void append_u32(std::string& out, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
    }
}

void append_u64(std::string& out, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
    }
}

std::string host_name() {
#ifdef _WIN32
    char* name = nullptr;
    std::size_t name_size = 0;
    if (_dupenv_s(&name, &name_size, "COMPUTERNAME") == 0 && name != nullptr) {
        std::string result(name);
        std::free(name);
        return result;
    }
    std::free(name);
    return "localhost";
#else
    const char* name = std::getenv("HOSTNAME");
    return name != nullptr ? name : "localhost";
#endif
}

// TensorBoard event filenames include the process id to avoid collisions when
// multiple writers share a log directory. The platform APIs use different
// spellings, so keep the portability boundary in this one helper.
std::uint64_t process_id() noexcept {
#ifdef _WIN32
    return static_cast<std::uint64_t>(_getpid());
#else
    return static_cast<std::uint64_t>(getpid());
#endif
}

} // namespace

uint32_t tfrecord_crc(const char* data, size_t length) {
    return masked_crc(data, length);
}

// ---- TensorBoardWriter ---------------------------------------------------------

TensorBoardWriter::TensorBoardWriter(const std::string& log_dir)
    : log_dir_(log_dir) {
    const auto now = std::chrono::system_clock::now();
    wall_time_base_ =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            now.time_since_epoch())
            .count();

    file_path_ = log_dir_ + "/events.out.tfevents." +
                 std::to_string(static_cast<long long>(wall_time_base_)) + "." +
                 host_name() + "." + std::to_string(process_id());

    if (!std::filesystem::create_directories(log_dir_) &&
        !std::filesystem::is_directory(log_dir_)) {
        throw std::runtime_error("TensorBoardWriter cannot create log dir: " +
                                 log_dir_);
    }
    file_.open(file_path_, std::ios::binary | std::ios::app);
    if (!file_.is_open()) {
        throw std::runtime_error("TensorBoardWriter cannot open " + file_path_);
    }

    // Mandatory file-version event.
    std::string event;
    append_double_field(event, 1, wall_time_base_);
    append_string_field(event, 5, "brain.Event:2");
    write_event(event);
}

TensorBoardWriter::~TensorBoardWriter() {
    if (file_.is_open()) {
        file_.flush();
    }
}

void TensorBoardWriter::write_event(const std::string& encoded_event) {
    if (!file_.is_open()) {
        throw std::runtime_error("TensorBoardWriter event file is closed");
    }
    std::string length_bytes;
    append_u64(length_bytes, encoded_event.size());

    std::string record;
    record.append(length_bytes);
    append_u32(record, masked_crc(length_bytes.data(), length_bytes.size()));
    record.append(encoded_event);
    append_u32(record, masked_crc(encoded_event.data(), encoded_event.size()));
    file_.write(record.data(), static_cast<std::streamsize>(record.size()));
}

void TensorBoardWriter::add_scalar(const std::string& tag, double value,
                                   int64_t step) {
    const double wall_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    write_event(encode_scalar_event(tag, static_cast<float>(value), step, wall_time));
}

void TensorBoardWriter::add_text(const std::string& tag, const std::string& text,
                                 int64_t step) {
    const double wall_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    write_event(encode_text_event(tag, text, step, wall_time));
}

void TensorBoardWriter::flush() {
    if (file_.is_open()) {
        file_.flush();
    }
}

} // namespace utils
} // namespace jyppx
