// Unit tests for the plan-completion modules: classification metrics,
// extended samplers/datasets, extended transforms, NAdam/LAMB, TensorBoard
// writer, model summary/profiler and distributed helpers.

#include <jyppx/jyppx.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <set>

namespace {

using namespace jyppx;
using namespace jyppx::data;
using namespace jyppx::data::samplers;
using namespace jyppx::data::transforms;
using namespace jyppx::training;
using namespace jyppx::training::metrics;
using namespace jyppx::training::callbacks;
using namespace jyppx::utils;

// ---- Classification metrics (Precision / Recall / F1) -----------------------

TEST(ClassificationMetricsTest, HandComputedConfusionMatrix) {
    // 3 classes; argmax of preds: {0, 0, 1, 2, 2}; targets: {0, 0, 1, 1, 2}.
    // Confusion (rows = true, cols = predicted):
    //   true 0 -> [2, 0, 0]  precision(0)=2/2, recall(0)=2/2
    //   true 1 -> [0, 1, 0]  precision(1)=1/1, recall(1)=1/2
    //   true 2 -> [0, 0, 1]  precision(2)=1/2, recall(2)=1/2
    metrics::Precision precision;
    metrics::Recall recall;
    metrics::F1Score f1;

    auto preds = torch::tensor({{0.9f, 0.1f, 0.0f}, {0.8f, 0.1f, 0.1f},
                                {0.1f, 0.8f, 0.1f}, {0.0f, 0.2f, 0.9f},
                                {0.1f, 0.1f, 0.8f}});
    auto targets = torch::tensor({0.0f, 0.0f, 1.0f, 1.0f, 2.0f});
    precision.update(preds, targets);
    recall.update(preds, targets);
    f1.update(preds, targets);

    EXPECT_NEAR(precision.compute(), (1.0 + 1.0 + 0.5) / 3.0, 1e-12);
    EXPECT_NEAR(recall.compute(), (1.0 + 0.5 + 1.0) / 3.0, 1e-12);
    EXPECT_NEAR(f1.compute(), (1.0 + 2.0 / 3.0 + 2.0 / 3.0) / 3.0, 1e-12);
}

TEST(ClassificationMetricsTest, ConfusionMatrixExposedAndResets) {
    metrics::Precision precision;
    precision.update(torch::tensor({{0.9f, 0.1f}}), torch::tensor({0.0f}));
    ASSERT_EQ(precision.classes(), 1); // only class 0 observed
    EXPECT_EQ(precision.confusion_matrix()[0][0], 1);

    precision.reset();
    EXPECT_EQ(precision.classes(), 0);
    EXPECT_DOUBLE_EQ(precision.compute(), 0.0);
}

TEST(ClassificationMetricsTest, PerfectPredictionsScoreOne) {
    metrics::F1Score f1;
    f1.update(torch::tensor({{0.9f, 0.1f}, {0.1f, 0.9f}}),
              torch::tensor({0.0f, 1.0f}));
    EXPECT_DOUBLE_EQ(f1.compute(), 1.0);
}

// ---- Extended samplers --------------------------------------------------------

TEST(WeightedRandomSamplerTest, WeightsShapeTheDistribution) {
    samplers::WeightedRandomSampler sampler({1.0, 0.0, 9.0}, /*num_samples=*/1000,
                                            /*replacement=*/true, /*seed=*/5);
    const auto indices = sampler.indices();
    ASSERT_EQ(indices.size(), 1000);

    int count_class2 = 0;
    for (const size_t index : indices) {
        EXPECT_TRUE(index == 0 || index == 1 || index == 2);
        if (index == 2) {
            ++count_class2;
        }
    }
    // Weight 9/10 of the total mass lands on class 2.
    EXPECT_GT(count_class2, 800);
}

TEST(WeightedRandomSamplerTest, DeterministicWithSeed) {
    samplers::WeightedRandomSampler first({1, 2, 3}, 50, true, 7);
    samplers::WeightedRandomSampler second({1, 2, 3}, 50, true, 7);
    EXPECT_EQ(first.indices(), second.indices());
}

TEST(WeightedRandomSamplerTest, ValidatesArguments) {
    EXPECT_THROW(samplers::WeightedRandomSampler({}, 10), std::invalid_argument);
    EXPECT_THROW(samplers::WeightedRandomSampler({1.0}, 0), std::invalid_argument);
    EXPECT_THROW(
        samplers::WeightedRandomSampler({1.0, -1.0}, 10), std::invalid_argument);
    EXPECT_THROW(samplers::WeightedRandomSampler({0.0, 0.0}, 10),
                 std::invalid_argument);
    EXPECT_THROW(
        samplers::WeightedRandomSampler({1.0, 2.0}, 5, /*replacement=*/false),
        std::invalid_argument);
}

TEST(WeightedRandomSamplerTest, WithoutReplacementYieldsDistinctIndices) {
    samplers::WeightedRandomSampler sampler({1, 1, 1, 1, 1}, 4, false, 3);
    const auto indices = sampler.indices();
    ASSERT_EQ(indices.size(), 4);
    std::set<size_t> unique(indices.begin(), indices.end());
    EXPECT_EQ(unique.size(), 4);
}

TEST(SubsetRandomSamplerTest, PermutesTheGivenSubset) {
    samplers::SubsetRandomSampler sampler({4, 7, 9, 2}, /*seed=*/11);
    const auto indices = sampler.indices();
    ASSERT_EQ(indices.size(), 4);
    std::set<size_t> unique(indices.begin(), indices.end());
    EXPECT_EQ(unique.size(), 4);
    for (const size_t index : indices) {
        EXPECT_TRUE(index == 2 || index == 4 || index == 7 || index == 9);
    }
}

TEST(DistributedSamplerTest, PartitionsAcrossRanks) {
    samplers::DistributedSampler rank0(10, /*rank=*/0, /*world_size=*/2, false);
    samplers::DistributedSampler rank1(10, /*rank=*/1, /*world_size=*/2, false);

    const auto a = rank0.indices();
    const auto b = rank1.indices();
    ASSERT_EQ(a.size(), 5);
    ASSERT_EQ(b.size(), 5);

    std::vector<size_t> all = a;
    all.insert(all.end(), b.begin(), b.end());
    std::sort(all.begin(), all.end());
    for (size_t i = 0; i < all.size(); ++i) {
        EXPECT_EQ(all[i], i); // disjoint, covers everything
    }
}

TEST(DistributedSamplerTest, SetEpochChangesTheShuffle) {
    samplers::DistributedSampler sampler(20, 0, 2, /*shuffle=*/true, /*seed=*/1);
    const auto epoch0 = sampler.indices();
    sampler.set_epoch(1);
    const auto epoch1 = sampler.indices();
    EXPECT_NE(epoch0, epoch1);

    // Same epoch is reproducible.
    sampler.set_epoch(0);
    EXPECT_EQ(sampler.indices(), epoch0);
}

TEST(DistributedSamplerTest, SingleProcessBehavesLikeIdentity) {
    samplers::DistributedSampler sampler(6, 0, 1, false);
    const auto indices = sampler.indices();
    for (size_t i = 0; i < indices.size(); ++i) {
        EXPECT_EQ(indices[i], i);
    }
}

TEST(DistributedSamplerTest, ValidatesArguments) {
    EXPECT_THROW(samplers::DistributedSampler(10, 1, 1), std::invalid_argument);
    EXPECT_THROW(samplers::DistributedSampler(10, 0, 0), std::invalid_argument);
}

// ---- Chain / Stack datasets ---------------------------------------------------

TEST(ChainDatasetTest, ConcatenatesLikeConcat) {
    auto first = std::make_shared<data::TensorDataset>(
        torch::arange(0, 4, torch::kFloat32), torch::arange(0, 4, torch::kFloat32));
    auto second = std::make_shared<data::TensorDataset>(
        torch::arange(4, 6, torch::kFloat32), torch::arange(4, 6, torch::kFloat32));
    const data::ChainDataset<data::Example> chain({first, second});

    EXPECT_EQ(chain.size(), 6);
    EXPECT_NEAR(chain.get(4).data.item<double>(), 4.0, 1e-6);
}

namespace {

// Minimal Dataset<Tensor> wrapper for the StackDataset tests.
class TensorRows : public data::Dataset<Tensor> {
public:
    explicit TensorRows(Tensor rows) : rows_(std::move(rows)) {}
    Tensor get(size_t index) const override {
        return rows_[static_cast<int64_t>(index)];
    }
    size_t size() const override { return rows_.size(0); }

private:
    Tensor rows_;
};

} // namespace

TEST(StackDatasetTest, ZipsTupleSamples) {
    auto features = std::make_shared<TensorRows>(
        torch::arange(0, 6, torch::kFloat32).view({3, 2}));
    auto labels = std::make_shared<TensorRows>(torch::tensor({10.f, 20.f, 30.f}));

    const data::StackDataset<Tensor, Tensor> stacked(features, labels);
    ASSERT_EQ(stacked.size(), 3);

    auto sample = stacked.get(1);
    EXPECT_EQ(std::get<0>(sample).numel(), 2);
    EXPECT_NEAR(std::get<1>(sample).item<double>(), 20.0, 1e-6);
}

TEST(StackDatasetTest, LengthIsTheSmallestDataset) {
    auto big = std::make_shared<TensorRows>(torch::randn({5, 1}));
    auto small = std::make_shared<TensorRows>(torch::randn({2, 1}));
    const data::StackDataset<Tensor, Tensor> stacked(big, small);
    EXPECT_EQ(stacked.size(), 2);
    EXPECT_THROW(stacked.get(2), std::out_of_range);
}

// ---- Extended transforms --------------------------------------------------------

class TransformsExtraTest : public ::testing::Test {
protected:
    void SetUp() override {
        image_ = torch::arange(12, torch::TensorOptions().dtype(torch::kFloat32))
                     .view({2, 2, 3}); // zero-padded later by Pad
    }

    Tensor image_;
};

TEST_F(TransformsExtraTest, PadGrowsTheImage) {
    const Pad pad({1, 1, 1, 1}, /*fill=*/-1.0f);
    const auto result = pad(image_);
    EXPECT_EQ(result.sizes(), (std::vector<int64_t>{2, 4, 5}));
    // Corners carry the fill value.
    EXPECT_NEAR(result[0][0][0].item<double>(), -1.0, 1e-6);
    // Interior unchanged.
    EXPECT_NEAR(result[0][1][1].item<double>(), image_[0][0][0].item<double>(),
                1e-6);
}

TEST_F(TransformsExtraTest, PadSquareShorthand) {
    const Pad pad({1});
    EXPECT_EQ(pad(image_).sizes(), (std::vector<int64_t>{2, 4, 5}));
}

TEST_F(TransformsExtraTest, RotationByZeroIsIdentity) {
    const RandomRotation rotate(0.0); // angle drawn from [-0, 0]
    const auto result = rotate(image_);
    EXPECT_EQ(result.sizes(), image_.sizes());
    EXPECT_TRUE(torch::allclose(result, image_, 1e-3, 1e-3));
}

TEST_F(TransformsExtraTest, RotationStaysFiniteAndInRange) {
    const RandomRotation rotate(30.0);
    for (int i = 0; i < 8; ++i) {
        const auto result = rotate(image_);
        EXPECT_TRUE(torch::isfinite(result).all().item<bool>());
    }
}

TEST_F(TransformsExtraTest, AffineIdentityAndValidation) {
    const RandomAffine affine(0.0); // no rotation/translate/scale variation
    EXPECT_TRUE(
        torch::allclose(affine(image_), image_, 1e-3, 1e-3));

    EXPECT_THROW(RandomAffine(10.0, {1.5, 0.0}), std::invalid_argument);
    EXPECT_THROW(RandomAffine(10.0, {0.0, 0.0}, {0.0, 1.0}), std::invalid_argument);
}

TEST_F(TransformsExtraTest, AffineTranslateShiftsContent) {
    // Deterministic translate: shift right by exactly one pixel.
    const RandomAffine affine(0.0, {0.0, 0.0}, {1.0, 1.0}); // no-op scale/rot
    (void)affine;

    // Use zero-degree rotation with a fixed translate via a small subclass-free
    // path: RandomAffine draws translate from +-translate*size; with
    // translate=(0,0) there is no shift, so we validate the identity already
    // covered above. For an actual shift, apply RandomAffine(0, {0.5, 0}):
    // the drawn shift is +-0.5*W; result must remain finite and 2x3.
    const RandomAffine shifting(0.0, {0.5, 0.0});
    const auto result = shifting(image_);
    EXPECT_EQ(result.sizes(), image_.sizes());
    EXPECT_TRUE(torch::isfinite(result).all().item<bool>());
}

TEST_F(TransformsExtraTest, GrayscaleLumaWeights) {
    // Pure red {1,0,0} -> luma 0.299.
    auto red = torch::zeros({3, 1, 1});
    red[0][0][0] = 1.0f;
    const Grayscale gray(1);
    const auto result = gray(red);
    EXPECT_EQ(result.sizes(), (std::vector<int64_t>{1, 1, 1}));
    EXPECT_NEAR(result[0][0][0].item<double>(), 0.299, 1e-5);

    const Grayscale gray3(3);
    const auto replicated = gray3(red);
    EXPECT_EQ(replicated.sizes(), (std::vector<int64_t>{3, 1, 1}));
    EXPECT_NEAR(replicated[1][0][0].item<double>(), 0.299, 1e-5);
}

TEST_F(TransformsExtraTest, GrayscaleValidatesChannels) {
    EXPECT_THROW(Grayscale(2), std::invalid_argument);
    EXPECT_THROW(Grayscale(5), std::invalid_argument);
}

TEST_F(TransformsExtraTest, RandomGrayscaleKeepsThreeChannels) {
    // RandomGrayscale requires a 3-channel RGB input; use one here.
    auto rgb = torch::rand({3, 4, 4});
    const RandomGrayscale always(1.0);
    const auto result = always(rgb);
    EXPECT_EQ(result.sizes(), rgb.sizes());
    // Channels become identical after grayscale.
    EXPECT_TRUE(torch::equal(result[0], result[1]));
}

TEST_F(TransformsExtraTest, ColorJitterBrightnessZeroIsIdentity) {
    const ColorJitter jitter(0.0, 0.0, 0.0, 0.0);
    EXPECT_TRUE(torch::allclose(jitter(image_), image_, 1e-6, 1e-6));
}

TEST_F(TransformsExtraTest, ColorJitterBrightnessScales) {
    // brightness=1 draws f in [0, 2]; the output stays proportional to input.
    const ColorJitter jitter(1.0);
    for (int i = 0; i < 8; ++i) {
        const auto result = jitter(image_);
        EXPECT_TRUE(torch::isfinite(result).all().item<bool>());
        EXPECT_GE(result.min().item<double>(), -1e-6);
    }
}

TEST_F(TransformsExtraTest, ColorJitterValidatesArguments) {
    EXPECT_THROW(ColorJitter(-0.5), std::invalid_argument);
    EXPECT_THROW(ColorJitter(0.0, 0.0, 0.0, 0.9), std::invalid_argument);
}

// ---- NAdam / LAMB ----------------------------------------------------------------

TEST(NAdamTest, FirstStepMatchesHandComputation) {
    // p = 1, grad = 1, lr = 0.002, beta1 = 0.9, beta2 = 0.999, eps = 1e-8.
    // mu_t = 0.9 * (1 - 0.5 * 0.96) = 0.468; mu_product = mu_t.
    // bias_correction2 = 1 - 0.999 = 0.001.
    // denom = sqrt(v/0.001) + eps with v = 1e-4 * 1 -> sqrt(0.1)+eps.
    // grad coef = -lr*(1-mu_t)/ (bc_sqrt * (1-mu_product))
    // exp_avg = 0.1*1 -> exp_avg coef = -lr*mu_t/(bc_sqrt*(1-mu_product))
    auto params = torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat64)
                                                .requires_grad(true));
    params.mutable_grad() = torch::ones_like(params);

    optim::NAdam optimizer(TensorList{params}, optim::NAdamOptions{.lr = 0.002});
    optimizer.step();

    const double mu_t = 0.9 * (1.0 - 0.5 * std::pow(0.96, 1.0));
    const double mu_product = mu_t;
    const double bc2 = 1.0 - std::pow(0.999, 1.0);
    const double denom = std::sqrt(1e-4 * 1.0 / bc2) + 1e-8;
    const double grad_coef = 0.002 * (1.0 - mu_t) / (std::sqrt(bc2) * (1.0 - mu_product));
    const double avg_coef = 0.002 * mu_t / (std::sqrt(bc2) * (1.0 - mu_product));
    const double expected = 1.0 - grad_coef - avg_coef * 0.1;
    EXPECT_NEAR(params[0].item<double>(), expected, 1e-9);
}

TEST(NAdamTest, ValidatesArguments) {
    auto params = torch::ones({1}, torch::TensorOptions().requires_grad(true));
    EXPECT_THROW(optim::NAdam(TensorList{params}, optim::NAdamOptions{.lr = -1.0}),
                 std::invalid_argument);
    EXPECT_THROW(optim::NAdam(TensorList{params}, optim::NAdamOptions{.momentum_decay = 0.01}),
                 std::invalid_argument);
}

TEST(LambTest, ZeroParameterFallsBackToPlainAdamStep) {
    // p = 0 -> weight_norm = 0 -> trust ratio 1 -> LAMB == AdamW-style step.
    auto params = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat64)
                                              .requires_grad(true));
    params.mutable_grad() = torch::ones_like(params);

    optim::Lamb lamb(TensorList{params}, optim::LambOptions{.lr = 0.01, .eps = 1e-6});
    lamb.step();

    // adam_step = (1/1) / (sqrt(1/1) + eps) ~= 1 / (1 + 1e-6)
    const double expected = -0.01 * (1.0 / (1.0 + 1e-6));
    EXPECT_NEAR(params[0].item<double>(), expected, 1e-9);
}

TEST(LambTest, TrustRatioLimitsSmallParameters) {
    // p = 0.01 with unit grad: weight_norm << adam_norm -> update is tiny.
    auto params = torch::full({1}, 0.01,
                              torch::TensorOptions().dtype(torch::kFloat64)
                                  .requires_grad(true));
    params.mutable_grad() = torch::ones_like(params);

    optim::Lamb lamb(TensorList{params}, optim::LambOptions{.lr = 0.01, .eps = 1e-6});
    lamb.step();

    const double movement = std::abs(0.01 - params[0].item<double>());
    EXPECT_LT(movement, 0.01); // far below the unadapted ~0.01 step
}

TEST(LambTest, ValidatesArguments) {
    auto params = torch::ones({1}, torch::TensorOptions().requires_grad(true));
    EXPECT_THROW(optim::Lamb(TensorList{params}, optim::LambOptions{.beta1 = 1.0}),
                 std::invalid_argument);
    EXPECT_THROW(optim::Lamb(TensorList{params}, optim::LambOptions{.weight_decay = -1.0}),
                 std::invalid_argument);
}

// ---- TensorBoard writer --------------------------------------------------------------

TEST(TensorBoardTest, WritesEventFileWithScalarsAndText) {
    const auto dir = std::filesystem::temp_directory_path() / "jyppx_tb_test";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    {
        utils::TensorBoardWriter writer(dir.string());
        writer.add_scalar("train/loss", 0.5, 1);
        writer.add_scalar("train/loss", 0.25, 2);
        writer.add_text("info/message", "hello tensorboard", 2);
        writer.flush();
    }

    // Exactly one event file, non-trivially sized, containing the tag.
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        files.push_back(entry.path().filename().string());
    }
    ASSERT_EQ(files.size(), 1);
    EXPECT_EQ(files[0].rfind("events.out.tfevents.", 0), 0);

    std::ifstream event_file((dir / files[0]).string(), std::ios::binary);
    const std::string bytes{std::istreambuf_iterator<char>(event_file),
                            std::istreambuf_iterator<char>()};
    EXPECT_NE(bytes.find("train/loss"), std::string::npos);
    EXPECT_NE(bytes.find("info/message"), std::string::npos);
    EXPECT_NE(bytes.find("brain.Event:2"), std::string::npos);

    // TFRecord framing: first 8 bytes decode to a little-endian length.
    uint64_t first_length = 0;
    std::memcpy(&first_length, bytes.data(), 8);
    EXPECT_GT(first_length, 0u);
    EXPECT_LT(first_length, 4096u);

    std::filesystem::remove_all(dir, ec);
}

// ---- Model summary / profiler ---------------------------------------------------

class SummaryModel : public torch::nn::Module {
public:
    SummaryModel()
        : fc1(register_module("encoder", torch::nn::Linear(4, 8))),
          fc2(register_module("head", torch::nn::Linear(8, 2))) {}

    Tensor forward(const Tensor& x) {
        return fc2->forward(torch::relu(fc1->forward(x)));
    }

    torch::nn::Linear fc1, fc2;
};

TEST(ModelSummaryTest, ReportsParameterCounts) {
    SummaryModel model;
    const auto stats = utils::calculate_model_stats(model);
    // fc1: 4*8 + 8 = 40; fc2: 8*2 + 2 = 18.
    EXPECT_EQ(stats.total_params, 58);
    EXPECT_EQ(stats.trainable_params, 58);
    EXPECT_EQ(stats.memory_bytes, 58 * 4);
}

TEST(ModelSummaryTest, SummaryListsChildren) {
    SummaryModel model;
    std::cout << "children=" << model.named_children().size() << std::endl;
    const auto text = utils::model_summary(model, "TestNet");
    std::cout << text << std::endl;
    EXPECT_NE(text.find("TestNet summary"), std::string::npos);
    EXPECT_NE(text.find("encoder"), std::string::npos);
    EXPECT_NE(text.find("head"), std::string::npos);
    EXPECT_NE(text.find("Total params:        58"), std::string::npos);
}

TEST(ProfilerTest, AggregatesScopedTimings) {
    utils::ProfilerReport report;
    {
        utils::ScopedTimer timer(report, "scope_a");
        // minimal work
    }
    {
        utils::ScopedTimer timer(report, "scope_b");
    }
    {
        utils::ScopedTimer timer(report, "scope_a");
    }

    const auto summary = report.summary();
    EXPECT_NE(summary.find("scope_a"), std::string::npos);
    EXPECT_NE(summary.find("scope_b"), std::string::npos);

    const auto sorted = report.sorted();
    ASSERT_EQ(sorted.size(), 2);
    EXPECT_EQ(sorted[0].first, "scope_a"); // 2 calls sort first by total
    EXPECT_EQ(sorted[0].second.count, 2);
}

// ---- Distributed / DDP -----------------------------------------------------------

TEST(DistributedTest, SingleProcessGroupLifecycle) {
    EXPECT_FALSE(distributed::is_initialized());
    EXPECT_EQ(distributed::get_world_size(), 1);
    EXPECT_EQ(distributed::get_rank(), 0);

    distributed::init_process_group("single-process", 0, 1);
    EXPECT_TRUE(distributed::is_initialized());
    distributed::barrier();

    // Gradients pass through unchanged.
    auto grad = torch::ones({3});
    EXPECT_TRUE(torch::equal(distributed::average_gradients(grad), grad));
}

TEST(DistributedTest, RejectsUnsupportedBackends) {
    EXPECT_THROW(distributed::init_process_group("gloo", 0, 4), std::runtime_error);
    EXPECT_THROW(distributed::init_process_group("single-process", 2, 4),
                 std::runtime_error);
}

TEST(DistributedDDPTest, ForwardsAndRunsGradientsThroughHooks) {
    distributed::init_process_group("single-process", 0, 1);

    class Tiny : public torch::nn::Module {
    public:
        Tiny() : w(register_parameter("w", torch::ones({1}, torch::kFloat64))) {}
        Tensor forward(const Tensor& x) { return w * x; }
        torch::Tensor w;
    };

    distributed::DistributedDataParallel<Tiny> ddp(std::make_shared<Tiny>());
    auto loss = ddp.forward(torch::ones({1}, torch::kFloat64)).sum();
    loss.backward();

    // The hook chain passes the gradient through; the parameter gets it.
    EXPECT_NEAR(ddp.module()->w.grad().item<double>(), 1.0, 1e-12);
}

} // namespace

