#include <chrono>
#include <numeric>
#include <random>

#include <cuda_runtime.h>

#include <cuda/std/numeric>

static void __global__ csum(_Inout_ double* const slice, _In_ const unsigned size) {
    double sum {};
    for (unsigned i {}; i < size; ++i) sum += slice[i];
    slice[0] = sum;
}

template<typename _IteratorType> static // NOLINTNEXTLINE(modernize-use-constraints)
    typename std::enable_if<!std::is_same<typename _IteratorType::value_type, void>::value, void>::type __global__
    cppsum(_Inout_ _IteratorType _begin, _IteratorType _end) {
    *_begin = cuda::std::accumulate(_begin, _end, _IteratorType::value_type {}, cuda::std::sum {});
}

int main() {
    std::vector<double>                    randoms(1'000'000'000);
    std::knuth_b                           rengine { std::chrono::high_resolution_clock::now().time_since_epoch().count() };
    std::uniform_real_distribution<double> urdist { -100.0, 1000.0 };

    std::generate(randoms.begin(), randoms.end(), [&rengine, &urdist]() noexcept -> auto { return urdist(rengine); });
    const auto hsum { std::reduce(randoms.cbegin(), randoms.cend(), 0.0L) };
    double *   drandoms {}, dsum {};
    ::cudaMalloc(&drandoms, randoms.size() * sizeof(double));

    ::cudaFree(drandoms);

    return EXIT_SUCCESS;
}
