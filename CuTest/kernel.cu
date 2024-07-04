#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

static constexpr unsigned threads { 700 };
static constexpr unsigned nums_per_threads { 12'000 };

template<typename T>
__global__ void __cdecl kernel(
    _In_ const T* const        devmem,
    _Inout_ long double* const sums,
    _In_opt_ typename std::enable_if<std::is_arithmetic<T>::value, bool>::type = false
) {
    long double _sum {};
    const auto  index { threadIdx.x + threadIdx.y + threadIdx.z };
    for (unsigned i = index * nums_per_threads; i < (index * nums_per_threads + nums_per_threads); ++i) _sum += devmem[i];
    sums[index] = _sum;
}

template<typename T>
__global__ void __cdecl reduce(
    _In_ T* const collection, _In_ const unsigned length, _In_opt_ typename std::enable_if<std::is_arithmetic<T>::value, bool>::type = false
) {
    for (unsigned i = 1; i < length; ++i) collection[0] += collection[i];
}

template<typename T, unsigned nthreads = threads, unsigned ntasks = nums_per_threads>
[[nodiscard]] static inline long double dsum(
    _In_ const T* const hmem, _In_ const unsigned size, _In_opt_ typename std::enable_if<std::is_arithmetic<T>::value, bool>::type = false
) noexcept {
    T*           devmem {};
    long double* devsums {};
    long double  final {};

    cudaMalloc(&devmem, size * sizeof(T));
    cudaMalloc(&devsums, nthreads * sizeof(long double));
    cudaMemcpy(devmem, hmem, size * sizeof(T), cudaMemcpyHostToDevice);

    kernel<<<1, nthreads>>>(devmem, devsums);
    reduce<<<1, 1>>>(devsums, nthreads);

    cudaMemcpy(&final, devsums, sizeof(long double), cudaMemcpyDeviceToHost);
    cudaFree(devmem);
    cudaFree(devsums);

    return final;
}

extern "C" auto wmain() -> int {
    std::vector<int>   rintegers(threads * nums_per_threads);
    std::vector<float> rfloats(threads * nums_per_threads);

    std::random_device       seeder {};
    std::mt19937_64          reng { seeder() };
    std::normal_distribution rnorm { 0.0, 10.0 }; // mean, and std

    std::generate(rintegers.begin(), rintegers.end(), reng);
    for (std::vector<float>::iterator it = rfloats.begin(), end = rfloats.end(); it != end; ++it) *it = rnorm(reng);

    const long double isum { std::accumulate(rintegers.cbegin(), rintegers.cend(), 0.0L) };
    const long double fsum { std::accumulate(rfloats.cbegin(), rfloats.cend(), 0.0L) };

    const auto gisum { ::dsum(rintegers.data(), rintegers.size()) };
    const auto gfsum { ::dsum(rfloats.data(), rfloats.size()) };

    wprintf_s(L"integers sum :: host %Lf, device %Lf\n", isum, gisum);
    wprintf_s(L"floats sum :: host %Lf, device %Lf\n", gfsum, gfsum);

    return EXIT_SUCCESS;
}