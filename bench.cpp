#include <benchmark/benchmark.h>

#include <filesystem>
#include <highfive/highfive.hpp>

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
BENCHMARK(BM_SearchIndex);

BENCHMARK_MAIN();
