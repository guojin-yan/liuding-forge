#ifndef JYPPX_UTILS_TENSORBOARD_HPP
#define JYPPX_UTILS_TENSORBOARD_HPP

#include <jyppx/core/types.hpp>
#include <jyppx/foundation/error.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>

namespace jyppx {
namespace utils {

/**
 * @brief Minimal TensorBoard event-file writer
 *
 * Writes `events.out.tfevents.*` files consumable by TensorBoard: TFRecord
 * framing (length + masked CRC32C checksums) around hand-encoded Event
 * protobuf messages. Supports scalar summaries, the feature the framework's
 * training loop needs; histograms and images are future work.
 *
 * @par Example
 * @code
 * utils::TensorBoardWriter writer("runs/linear");
 * writer.add_scalar("train/loss", 0.5, 1);
 * writer.flush();
 * @endcode
 */
class TensorBoardWriter {
 public:
  /**
   * @brief Creates an append-only TensorBoard event writer.
   * @param log_dir Directory that will contain the event file.
   * @throws jyppx::IoError If the directory or event file cannot be created.
   */
  explicit TensorBoardWriter(const std::string& log_dir);
  /** @brief Flushes the event stream before destruction. */
  ~TensorBoardWriter();

  TensorBoardWriter(const TensorBoardWriter&) = delete;  ///< Non-copyable.
  TensorBoardWriter& operator=(const TensorBoardWriter&) = delete;

  /**
   * @brief Adds a scalar summary point.
   * @param tag TensorBoard summary tag.
   * @param value Scalar value to record.
   * @param step Training step associated with the value.
   * @throws jyppx::NotReady If the event stream is closed.
   * @throws jyppx::IoError If the event cannot be written.
   */
  void add_scalar(const std::string& tag, double value, int64_t step);

  /**
   * @brief Adds a plain text summary entry.
   * @param tag TensorBoard summary tag.
   * @param text Text value to record.
   * @param step Training step associated with the value.
   * @throws jyppx::NotReady If the event stream is closed.
   * @throws jyppx::IoError If the event cannot be written.
   */
  void add_text(const std::string& tag, const std::string& text, int64_t step);

  /**
   * @brief Closes the event stream and releases its file handle.
   *
   * Closing is idempotent. A closed writer cannot accept new events or flush;
   * those operations raise `jyppx::NotReady` so lifecycle mistakes are
   * visible to the caller.
   */
  void close();

  /**
   * @brief Flushes buffered bytes to disk.
   * @throws jyppx::NotReady If the event stream is closed.
   * @throws jyppx::IoError If the stream reports a write failure.
   */
  void flush();

  /**
   * @brief Returns the directory this writer logs into.
   * @return Configured directory path, valid for the writer's lifetime.
   */
  const std::string& log_dir() const { return log_dir_; }

 private:
  /**
   * @brief Writes one already-serialized Event message using TFRecord framing.
   * @param encoded_event Serialized TensorBoard Event protobuf bytes.
   * @throws jyppx::NotReady If the event stream is closed.
   * @throws jyppx::IoError If the framed event cannot be written.
   */
  void write_event(const std::string& encoded_event);

  std::string log_dir_;          ///< Directory containing the event file.
  std::string file_path_;        ///< Fully qualified event file path.
  std::ofstream file_;           ///< Binary TFRecord output stream.
  double wall_time_base_ = 0.0;  ///< Creation timestamp used for file header.
};

/**
 * @brief Computes a masked CRC32C (Castagnoli) checksum for TFRecord data.
 * @param data Bytes to checksum.
 * @param length Number of bytes at @p data.
 * @return TensorFlow's masked CRC32C value.
 */
uint32_t tfrecord_crc(const char* data, size_t length);

}  // namespace utils
}  // namespace jyppx

#endif  // JYPPX_UTILS_TENSORBOARD_HPP
