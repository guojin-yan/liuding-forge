#ifndef JYPPX_DATA_THREAD_SAFE_QUEUE_HPP
#define JYPPX_DATA_THREAD_SAFE_QUEUE_HPP

/**
 * @file thread_safe_queue.hpp
 * @brief Defines a closeable blocking queue for DataLoader worker pipelines.
 */

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <utility>

namespace jyppx {
namespace data {

/**
 * @brief Thread-safe blocking queue for producer/consumer pipelines
 *
 * @tparam T Element type. `put` blocks while the queue is full (unless
 *          unbounded), `get` blocks until an element is available or the
 *          queue is closed and drained. Closing wakes all waiters; items
 *          already queued remain retrievable, which lets consumers drain
 *          in-flight work during shutdown.
 */
template <typename T>
class ThreadSafeQueue {
 public:
  /**
   * @brief Creates an open queue.
   * @param max_size Maximum queued items, or zero for no capacity limit.
   */
  explicit ThreadSafeQueue(std::size_t max_size = 0) : max_size_(max_size) {}

  /** @brief Copy construction is disabled because synchronization is owned. */
  ThreadSafeQueue(const ThreadSafeQueue&) = delete;

  /** @brief Copy assignment is disabled because synchronization is owned. */
  ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

  /**
   * @brief Enqueues an item, blocking while a bounded queue is full.
   * @param item Value moved into the queue after capacity becomes available.
   * @return False when the queue was closed before insertion.
   */
  bool put(T item) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock, [this] {
      return closed_ || max_size_ == 0 || queue_.size() < max_size_;
    });
    if (closed_) {
      return false;
    }
    queue_.push(std::move(item));
    lock.unlock();
    not_empty_.notify_one();
    return true;
  }

  /**
   * @brief Removes an item, blocking while an open queue is empty.
   * @param item Destination assigned from the oldest queued value.
   * @return False when the queue is closed and drained.
   */
  bool get(T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [this] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) {
      return false;
    }
    item = std::move(queue_.front());
    queue_.pop();
    lock.unlock();
    not_full_.notify_one();
    return true;
  }

  /**
   * @brief Attempts to remove one item without blocking.
   * @param item Destination assigned from the oldest queued value on success.
   * @return False when no item is currently available.
   */
  bool try_get(T& item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    item = std::move(queue_.front());
    queue_.pop();
    not_full_.notify_one();
    return true;
  }

  /**
   * @brief Close the queue and wake every waiting thread
   *
   * Items already queued stay retrievable; `put` fails from now on.
   */
  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  /**
   * @brief Returns the current queued item count under the queue lock.
   * @return Number of values available to consumers.
   */
  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  /**
   * @brief Returns whether no item is currently queued.
   * @return True when the FIFO storage is empty.
   */
  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  /**
   * @brief Returns whether insertion has been permanently disabled.
   * @return True after `close()` has been called.
   */
  bool is_closed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
  }

 private:
  mutable std::mutex mutex_;  ///< Protects all queue state and predicates.
  std::condition_variable not_empty_;  ///< Wakes consumers after insertion.
  std::condition_variable not_full_;   ///< Wakes producers after removal.
  std::queue<T> queue_;                ///< FIFO storage guarded by mutex_.
  std::size_t max_size_;  ///< Maximum queued items, or zero if unbounded.
  bool closed_ = false;   ///< Permanent insertion shutdown state.
};

}  // namespace data
}  // namespace jyppx

#endif  // JYPPX_DATA_THREAD_SAFE_QUEUE_HPP
