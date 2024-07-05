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

        using ImplType =
            unum::usearch::index_dense_gt</* key_at */ uint64_t,
                                          /* compressed_slot_at */ uint32_t>;
        auto metric = unum::usearch::metric_punned_t(
            /* dimensions */ data[0].size(),
            /* metric_kind */ unum::usearch::metric_kind_t::cos_k,
            unum::usearch::scalar_kind_t::f32_k);

        index_ = ImplType::make(metric);

        index_.reserve(data.size());

        for (uint64_t i = 0; i < data.size(); ++i) {
            index_.add(/* key */ i, data[i].data());
        }
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

BENCHMARK(BM_BuildIndex);

BENCHMARK_MAIN();
