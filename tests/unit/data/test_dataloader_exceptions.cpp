// Exception propagation contract for multi-worker DataLoader.

#include <jyppx/data/dataloader.hpp>

#include <c10/util/Exception.h>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

class ThrowingDataset final : public jyppx::data::Dataset<jyppx::Tensor> {
 public:
  enum class Kind { standard, runtime, torch };

  explicit ThrowingDataset(Kind kind) : kind_(kind) {}

  jyppx::Tensor get(std::size_t) const override {
    switch (kind_) {
      case Kind::standard:
        throw std::logic_error("worker logic error");
      case Kind::runtime:
        throw std::runtime_error("worker runtime error");
      case Kind::torch:
        TORCH_CHECK(false, "worker torch error");
    }
    return {};
  }

  std::size_t size() const override { return 4; }

 private:
  Kind kind_;
};

template <typename Exception>
void consume_and_expect_message(const std::shared_ptr<ThrowingDataset>& dataset,
                                const std::string& message) {
  jyppx::data::DataLoader loader(dataset, {.batch_size = 2, .num_workers = 2});
  try {
    for (const auto& batch : loader) {
      (void)batch;
    }
    FAIL() << "worker exception was not propagated";
  } catch (const Exception& error) {
    EXPECT_NE(std::string(error.what()).find(message), std::string::npos);
  } catch (...) {
    FAIL() << "exception type was not preserved";
  }
}

TEST(DataLoaderExceptionTest, StandardExceptionKeepsTypeAndMessage) {
  consume_and_expect_message<std::logic_error>(
      std::make_shared<ThrowingDataset>(ThrowingDataset::Kind::standard),
      "worker logic error");
}

TEST(DataLoaderExceptionTest, RuntimeExceptionKeepsTypeAndMessage) {
  consume_and_expect_message<std::runtime_error>(
      std::make_shared<ThrowingDataset>(ThrowingDataset::Kind::runtime),
      "worker runtime error");
}

TEST(DataLoaderExceptionTest, TorchExceptionKeepsTypeAndMessage) {
  consume_and_expect_message<c10::Error>(
      std::make_shared<ThrowingDataset>(ThrowingDataset::Kind::torch),
      "worker torch error");
}

TEST(DataLoaderExceptionTest, SingleWorkerAndCallingThreadHaveSameContract) {
  for (const auto workers : {std::size_t{0}, std::size_t{1}, std::size_t{4}}) {
    auto dataset =
        std::make_shared<ThrowingDataset>(ThrowingDataset::Kind::runtime);
    jyppx::data::DataLoader loader(dataset,
                                   {.batch_size = 1, .num_workers = workers});
    EXPECT_THROW(
        [&] {
          for (const auto& batch : loader) {
            (void)batch;
          }
        }(),
        std::runtime_error);
  }
}

}  // namespace
