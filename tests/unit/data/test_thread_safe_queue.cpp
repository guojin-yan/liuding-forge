// Unit tests for jyppx::data::ThreadSafeQueue.

#include <jyppx/data/thread_safe_queue.hpp>

#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <numeric>
#include <thread>
#include <vector>

namespace {

using jyppx::data::ThreadSafeQueue;

TEST(ThreadSafeQueueTest, BasicPutGet) {
  ThreadSafeQueue<int> queue(10);
  queue.put(42);

  int value = 0;
  EXPECT_TRUE(queue.get(value));
  EXPECT_EQ(value, 42);
  EXPECT_TRUE(queue.empty());
}

TEST(ThreadSafeQueueTest, FifoOrder) {
  ThreadSafeQueue<int> queue;
  for (int i = 0; i < 5; ++i) {
    queue.put(i);
  }

  int value = 0;
  for (int expected = 0; expected < 5; ++expected) {
    EXPECT_TRUE(queue.get(value));
    EXPECT_EQ(value, expected);
  }
}

TEST(ThreadSafeQueueTest, MultiThreadedProducerConsumer) {
  ThreadSafeQueue<int> queue(64);

  constexpr int kCount = 10000;
  std::atomic<long long> sum{0};

  std::thread producer([&] {
    for (int i = 0; i < kCount; ++i) {
      queue.put(i);
    }
    queue.close();
  });

  std::thread consumer([&] {
    int value = 0;
    while (queue.get(value)) {
      sum += value;
    }
  });

  producer.join();
  consumer.join();

  EXPECT_EQ(sum.load(), static_cast<long long>(kCount) * (kCount - 1) / 2);
}

TEST(ThreadSafeQueueTest, PutBlocksWhenFull) {
  ThreadSafeQueue<int> queue(2);
  queue.put(1);
  queue.put(2);

  std::atomic<bool> put_returned{false};
  std::thread blocked_producer([&] {
    queue.put(3);
    put_returned = true;
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_FALSE(put_returned.load());  // still blocked on the bounded queue

  int value = 0;
  EXPECT_TRUE(queue.get(value));
  blocked_producer.join();
  EXPECT_TRUE(put_returned.load());
  EXPECT_EQ(value, 1);
}

TEST(ThreadSafeQueueTest, CloseWakesWaitingGet) {
  ThreadSafeQueue<int> queue;

  std::thread waiter([&] {
    int value = 0;
    // Returns false once the queue is closed and drained.
    while (queue.get(value)) {
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  queue.close();
  waiter.join();  // must not hang
}

TEST(ThreadSafeQueueTest, CloseRejectsPutButKeepsQueuedItems) {
  ThreadSafeQueue<int> queue;
  queue.put(7);
  queue.close();

  EXPECT_FALSE(queue.put(8));
  EXPECT_TRUE(queue.is_closed());

  int value = 0;
  EXPECT_TRUE(queue.get(value));  // queued item survives close
  EXPECT_EQ(value, 7);
  EXPECT_FALSE(queue.get(value));
}

TEST(ThreadSafeQueueTest, TryGetNeverBlocks) {
  ThreadSafeQueue<int> queue;
  int value = 0;
  EXPECT_FALSE(queue.try_get(value));

  queue.put(3);
  EXPECT_TRUE(queue.try_get(value));
  EXPECT_EQ(value, 3);
}

TEST(ThreadSafeQueueTest, MoveOnlyElements) {
  ThreadSafeQueue<std::unique_ptr<int>> queue;
  queue.put(std::make_unique<int>(11));

  std::unique_ptr<int> value;
  ASSERT_TRUE(queue.get(value));
  EXPECT_EQ(*value, 11);
}

}  // namespace
