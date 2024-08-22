#include <algorithm>
#include <cassert>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

static constexpr size_t NTHREADS { 1'000 }, NTASKS { 2'000 };
static constexpr size_t COUNT { NTHREADS * NTASKS };

static_assert(COUNT * sizeof(double) < 2LLU * 1024 * 1024 * 1024);

template<std::integral T> __device__ __host__ constexpr double factorial(_In_ const T n) {
    double res { 1.0000 };
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-narrowing-conversions)
    for (size_t i = 1; i <= n; ++i) res *= i;
    return res;
}

template<std::integral T> __device__ __host__ constexpr double unrolled_factorial(_In_ const T n) {
    double res { 1.0000 };
#pragma unroll
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-narrowing-conversions)
    for (size_t i = 1; i <= n; ++i) res *= i;
    return res;
}

static_assert(::factorial(0) == 1.000);
static_assert(::factorial(1) == 1.000);
static_assert(::factorial(4) == 24.000);
static_assert(::factorial(5) == 120.000);
static_assert(::factorial(10) == 3628800.000);

static_assert(::unrolled_factorial(0) == 1.000);
static_assert(::unrolled_factorial(1) == 1.000);
static_assert(::unrolled_factorial(4) == 24.000);
static_assert(::unrolled_factorial(5) == 120.000);
static_assert(::unrolled_factorial(10) == 3628800.000);

template<class T, bool unroll> requires std::integral<T>
static __global__ void kernel(_In_ const T* const numbers, _Inout_ double* const results, _In_ const size_t njobs) {
    const auto id { threadIdx.x + threadIdx.y + threadIdx.z };

    if constexpr (unroll)
        for (size_t i = id; i < id + njobs; ++i)
            results[i] = unrolled_factorial(numbers[i]); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    else
        for (size_t i = id; i < id + njobs; ++i)
            results[i] = factorial(numbers[i]); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

template<class T> __device__ __host__ void reduce(_Inout_ double* const numbers, _In_ const size_t count) { }

auto wmain() -> int {
    srand(time(nullptr));
    std::vector<uint8_t> randoms(COUNT);
    std::vector<double>  results(COUNT);

    std::vector<double> host_results(COUNT);

    std::generate(randoms.begin(), randoms.end(), []() noexcept -> uint8_t { return rand() % 25; });

    const auto host_start = std::chrono::high_resolution_clock::now();
    std::transform(randoms.cbegin(), randoms.cend(), host_results.begin(), [](const uint8_t& n) noexcept -> double {
        double res { 1.0000 };
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-narrowing-conversions)
        for (size_t i = 1; i <= n; ++i) res *= i;
        return res;
    });
    const auto host_duration = (std::chrono::high_resolution_clock::now() - host_start).count();

    const auto host_sum      = std::accumulate(host_results.cbegin(), host_results.cend(), 0.00L);

    uint8_t* device_randoms {};
    double*  device_results {};
    ::cudaMalloc(&device_randoms, COUNT * sizeof(decltype(randoms)::value_type));
    ::cudaMalloc(&device_results, COUNT * sizeof(decltype(results)::value_type));

    ::cudaMemcpy(device_randoms, randoms.data(), COUNT * sizeof(decltype(randoms)::value_type), cudaMemcpyKind::cudaMemcpyHostToDevice);

    const auto start_non_unrolled = std::chrono::high_resolution_clock::now();
    kernel<uint8_t, false><<<1, NTHREADS>>>(device_randoms, device_results, NTASKS);
    const auto non_unrolled_duration = (std::chrono::high_resolution_clock::now() - start_non_unrolled).count();
    ::cudaMemcpy(results.data(), device_results, COUNT * sizeof(decltype(results)::value_type), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    ::cudaDeviceSynchronize();
    const auto non_unrolled_device_sum = std::accumulate(results.cbegin(), results.cend(), 0.00L);

    const auto start_unrolled          = std::chrono::high_resolution_clock::now();
    kernel<uint8_t, true><<<1, NTHREADS>>>(device_randoms, device_results, NTASKS);
    const auto unrolled_duration = (std::chrono::high_resolution_clock::now() - start_unrolled).count();
    ::cudaMemcpy(results.data(), device_results, COUNT * sizeof(decltype(results)::value_type), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    ::cudaDeviceSynchronize();
    const auto unrolled_device_sum = std::accumulate(results.cbegin(), results.cend(), 0.00L);

    std::wcout << L"host duration " << host_duration << L'\n';
    std::wcout << L"non-unrolled duration " << non_unrolled_duration << L'\n';
    std::wcout << L"unrolled duration " << unrolled_duration << L'\n';

    std::wcout << L"sums :: host - " << host_sum << L'\n';
    std::wcout << L"unrolled device - " << unrolled_device_sum << L'\n';
    std::wcout << L"non-unrolled device - " << non_unrolled_device_sum << L'\n';

    ::cudaFree(device_randoms);
    ::cudaFree(device_results);

    return EXIT_SUCCESS;
}
