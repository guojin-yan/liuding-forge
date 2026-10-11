#ifndef JYPPX_UTILS_TENSORBOARD_HPP
#define JYPPX_UTILS_TENSORBOARD_HPP

#include <jyppx/core/types.hpp>

#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace jyppx {
namespace utils {

/**
 * @brief Minimal TensorBoard event-file writer
 *
 * Writes `events.out.tfevents.*` files consumable by TensorBoard: TFRecord
 * framing (length + masked CRC32C checksums) around hand-encoded Event
 * protobuf messages. Supports scalar summaries — the feature the framework's
 * training loop needs; histograms and images are future work.
 *
 * @example
 * @code
 * utils::TensorBoardWriter writer("runs/linear");
 * writer.add_scalar("train/loss", 0.5, /*step=*\/1);
 * writer.flush();
 * @endcode
 */
class TensorBoardWriter {
public:
    explicit TensorBoardWriter(const std::string& log_dir);
    ~TensorBoardWriter();

    TensorBoardWriter(const TensorBoardWriter&) = delete;
    TensorBoardWriter& operator=(const TensorBoardWriter&) = delete;

    /** @brief Adds a scalar summary point. */
    void add_scalar(const std::string& tag, double value, int64_t step);

    /** @brief Adds a plain text entry. */
    void add_text(const std::string& tag, const std::string& text, int64_t step);

    /** @brief Flushes buffered bytes to disk. */
    void flush();

    /** @brief Directory this writer logs into. */
    const std::string& log_dir() const { return log_dir_; }

private:
    void write_event(const std::string& encoded_event);

    std::string log_dir_;
    std::string file_path_;
    std::ofstream file_;
    double wall_time_base_ = 0.0;
};

/**
 * @brief CRC32C (Castagnoli) checksum with the TFRecord mask applied
 */
uint32_t tfrecord_crc(const char* data, size_t length);

} // namespace utils
} // namespace jyppx

#endif // JYPPX_UTILS_TENSORBOARD_HPP
