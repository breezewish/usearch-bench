#include <benchmark/benchmark.h>

#include <filesystem>
#include <highfive/highfive.hpp>

#include "SimSIMD.h"
#include "USearch.h"

std::vector<std::vector<float>> loadDataset() {
    auto dataset_directory =
        std::filesystem::path(__FILE__).parent_path().string() +
        "/bench_dataset";
    auto dataset_path = dataset_directory + "/fashion-mnist-784-euclidean.hdf5";

    if (!std::filesystem::exists(dataset_path)) {
        throw "Benchmark cannot run because dataset file not found. ";
    }

    auto file = HighFive::File(dataset_path, HighFive::File::ReadOnly);
    auto train_dataset = file.getDataSet("train");

    std::vector<std::vector<float>> data;
    train_dataset.read(data);  // Read all data into memory

    return data;
}

const std::vector<std::vector<float>> &getDataset() {
    static auto data = loadDataset();
    return data;
}

class Index {
   public:
    void build() {
        auto data = getDataset();
        auto metric = unum::usearch::metric_punned_t(
            /* dimensions */ data[0].size(),
            /* metric_kind */ unum::usearch::metric_kind_t::cos_k,
            unum::usearch::scalar_kind_t::f32_k);

        std::cout << "ISA=" << metric.isa_name() << std::endl;

        index_ =
            ImplType::make(metric, unum::usearch::index_dense_config_t(
                                       unum::usearch::default_connectivity(),
                                       unum::usearch::default_expansion_add(),
                                       64 /* default is 64 */));

        index_.reserve(data.size());

        for (uint64_t i = 0; i < data.size(); ++i) {
            index_.add(/* key */ i, data[i].data());
        }
    }

    void search(uint64_t idx) {
        auto data = getDataset();
        auto result = index_.search(data[idx].data(), 100);
    }

   private:
    using ImplType =
        unum::usearch::index_dense_gt</* key_at */ uint64_t,
                                      /* compressed_slot_at */ uint32_t>;

    ImplType index_;
};

static void BM_BuildIndex(benchmark::State &state) {
    getDataset();

    for (auto _ : state) {
        auto index = std::make_unique<Index>();
        index->build();
    }
}

static void BM_SearchIndex(benchmark::State &state) {
    auto data = getDataset();

    auto index = std::make_unique<Index>();
    index->build();

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, data.size() - 1);

    for (auto _ : state) {
        auto idx = dist(rng);
        index->search(idx);
    }
}

// BENCHMARK(BM_BuildIndex);
// BENCHMARK(BM_SearchIndex);

namespace simsimd_details {

simsimd_capability_t simd_capabilities() {
    static simsimd_capability_t static_capabilities = simsimd_cap_any_k;
    if (static_capabilities == simsimd_cap_any_k)
        static_capabilities = simsimd_capabilities_implementation();
    return static_capabilities;
}

}  // namespace simsimd_details

static void BM_Distance_Cosine_SIMD(benchmark::State &state) {
    auto data = getDataset();

    static simsimd_metric_punned_t metric = nullptr;
    if (metric == nullptr) {
        simsimd_capability_t used_capability;
        simsimd_find_metric_punned(simsimd_metric_cos_k, simsimd_datatype_f32_k,
                                   simsimd_details::simd_capabilities(),
                                   simsimd_cap_any_k, &metric,
                                   &used_capability);
        if (!metric) throw std::runtime_error("No suitable metric found");
    }

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, data.size() - 1);

    for (auto _ : state) {
        for (size_t i = 0; i < 60000; i++) {
            simsimd_distance_t distance;
            metric(data[dist(rng)].data(), data[dist(rng)].data(),
                   data[0].size(), &distance);
            benchmark::DoNotOptimize(distance);
        }
    }
}

static void BM_Distance_Cosine_Serial(benchmark::State &state) {
    auto data = getDataset();

    static simsimd_metric_punned_t metric = nullptr;
    if (metric == nullptr) {
        simsimd_capability_t used_capability;
        simsimd_find_metric_punned(simsimd_metric_cos_k, simsimd_datatype_f32_k,
                                   simsimd_cap_serial_k, simsimd_cap_any_k,
                                   &metric, &used_capability);
        if (!metric) throw std::runtime_error("No suitable metric found");
    }

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, data.size() - 1);

    for (auto _ : state) {
        for (size_t i = 0; i < 60000; i++) {
            simsimd_distance_t distance;
            metric(data[dist(rng)].data(), data[dist(rng)].data(),
                   data[0].size(), &distance);
            benchmark::DoNotOptimize(distance);
        }
    }
}

template <typename at>
at square(at value) noexcept {
    return value * value;
}

/**
 *  @brief  Cosine (Angular) distance.
 *          Identical to the Inner Product of normalized vectors.
 *          Unless you are running on an tiny embedded platform, this metric
 *          is recommended over `::metric_ip_gt` for low-precision scalars.
 */
template <typename scalar_at = float, typename result_at = scalar_at>
struct metric_cos_gt {
    using scalar_t = scalar_at;
    using result_t = result_at;

    inline result_t operator()(scalar_t const *a, scalar_t const *b,
                               std::size_t dim) const noexcept {
        result_t ab{}, a2{}, b2{};
#pragma clang loop vectorize(enable)
        for (std::size_t i = 0; i != dim; ++i) {
            result_t ai = static_cast<result_t>(a[i]);
            result_t bi = static_cast<result_t>(b[i]);
            ab += ai * bi, a2 += square(ai), b2 += square(bi);
        }

        result_t result_if_zero[2][2];
        result_if_zero[0][0] = 1 - ab / (std::sqrt(a2) * std::sqrt(b2));
        result_if_zero[0][1] = result_if_zero[1][0] = 1;
        result_if_zero[1][1] = 0;
        return result_if_zero[a2 == 0][b2 == 0];
    }
};

static void BM_Distance_Cosine_Naive(benchmark::State &state) {
    auto data = getDataset();

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, data.size() - 1);

    for (auto _ : state) {
        for (size_t i = 0; i < 60000; i++) {
            metric_cos_gt<float, float> metric;
            auto distance = metric(data[dist(rng)].data(),
                                   data[dist(rng)].data(), data[0].size());
            benchmark::DoNotOptimize(distance);
        }
    }
}

static void BM_Distance_L2_SIMD(benchmark::State &state) {
    auto data = getDataset();

    static simsimd_metric_punned_t metric = nullptr;
    if (metric == nullptr) {
        simsimd_capability_t used_capability;
        simsimd_find_metric_punned(
            simsimd_metric_l2sq_k, simsimd_datatype_f32_k,
            simsimd_details::simd_capabilities(), simsimd_cap_any_k, &metric,
            &used_capability);
        if (!metric) throw std::runtime_error("No suitable metric found");
    }

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, data.size() - 1);

    for (auto _ : state) {
        for (size_t i = 0; i < 60000; i++) {
            simsimd_distance_t distance;
            metric(data[dist(rng)].data(), data[dist(rng)].data(),
                   data[0].size(), &distance);
            benchmark::DoNotOptimize(distance);
        }
    }
}

static void BM_Distance_L2_Serial(benchmark::State &state) {
    auto data = getDataset();

    static simsimd_metric_punned_t metric = nullptr;
    if (metric == nullptr) {
        simsimd_capability_t used_capability;
        simsimd_find_metric_punned(
            simsimd_metric_l2sq_k, simsimd_datatype_f32_k, simsimd_cap_serial_k,
            simsimd_cap_any_k, &metric, &used_capability);
        if (!metric) throw std::runtime_error("No suitable metric found");
    }

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, data.size() - 1);

    for (auto _ : state) {
        for (size_t i = 0; i < 60000; i++) {
            simsimd_distance_t distance;
            metric(data[dist(rng)].data(), data[dist(rng)].data(),
                   data[0].size(), &distance);
            benchmark::DoNotOptimize(distance);
        }
    }
}

/**
 *  @brief  Squared Euclidean (L2) distance.
 *          Square root is avoided at the end, as it won't affect the ordering.
 */
template <typename scalar_at = float, typename result_at = scalar_at>
struct metric_l2sq_gt {
    using scalar_t = scalar_at;
    using result_t = result_at;

    inline result_t operator()(scalar_t const *a, scalar_t const *b,
                               std::size_t dim) const noexcept {
        result_t ab_deltas_sq{};
#pragma clang loop vectorize(enable)
        for (std::size_t i = 0; i != dim; ++i) {
            result_t ai = static_cast<result_t>(a[i]);
            result_t bi = static_cast<result_t>(b[i]);
            ab_deltas_sq += square(ai - bi);
        }
        return ab_deltas_sq;
    }
};

static void BM_Distance_L2_Native(benchmark::State &state) {
    auto data = getDataset();

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, data.size() - 1);

    for (auto _ : state) {
        for (size_t i = 0; i < 60000; i++) {
            metric_l2sq_gt<float, float> metric;
            auto distance = metric(data[dist(rng)].data(),
                                   data[dist(rng)].data(), data[0].size());
            benchmark::DoNotOptimize(distance);
        }
    }
}

BENCHMARK(BM_Distance_Cosine_SIMD);
BENCHMARK(BM_Distance_Cosine_Serial);
BENCHMARK(BM_Distance_Cosine_Naive);
BENCHMARK(BM_Distance_L2_SIMD);
BENCHMARK(BM_Distance_L2_Serial);
BENCHMARK(BM_Distance_L2_Native);

BENCHMARK_MAIN();
