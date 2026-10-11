#ifndef JYPPX_DATA_THREAD_SAFE_QUEUE_HPP
#define JYPPX_DATA_THREAD_SAFE_QUEUE_HPP

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
     * @brief Construct a queue
     * @param max_size Maximum capacity (0 = unbounded)
     */
    explicit ThreadSafeQueue(std::size_t max_size = 0) : max_size_(max_size) {}

    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    /**
     * @brief Push an item, blocking while the queue is full
     * @return false if the queue was closed (the item was not enqueued)
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
     * @brief Pop an item, blocking until one is available
     * @return false when the queue is closed and empty
     */
    bool get(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) {
            // Only possible when closed.
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        not_full_.notify_one();
        return true;
    }

    /**
     * @brief Pop an item without blocking
     * @return false when the queue is empty
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

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    bool is_closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<T> queue_;
    std::size_t max_size_;
    bool closed_ = false;
};

} // namespace data
} // namespace jyppx

#endif // JYPPX_DATA_THREAD_SAFE_QUEUE_HPP
