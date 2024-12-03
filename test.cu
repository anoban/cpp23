#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>

static constexpr auto NRANDS { 100'000'000LLU };
static constexpr auto NKERNELS { 1000LLU };
static constexpr auto NELEMENTS_PER_KERNEL { NRANDS / NKERNELS };

// launch a 1,000 kernels to sum this vector in parallel, each kernel will have to sum 100,000 doubles

void __global__       sum(_Inout_ double* const vector) {
    const auto index { threadIdx.x + threadIdx.y + threadIdx.z }; // will range from 0 to 999
    const auto start_offset { index * NELEMENTS_PER_KERNEL };
    double     sum {};
#pragma unroll
    for (unsigned long long i = start_offset; i < start_offset + NELEMENTS_PER_KERNEL; ++i)
        sum += vector[i]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    // store the sum as the first element in their partition of the vector
    vector[start_offset] = sum; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

void __global__ fold(_Inout_ double* const vector) {
    double sum {};
#pragma unroll
    for (unsigned long long i = 0; i < NRANDS; i += NELEMENTS_PER_KERNEL)
        sum += vector[i]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) fold all the partition sums into the first element of the vector
    vector[0] = sum;
}

auto wmain() -> int {
    auto         randoms { std::make_unique_for_overwrite<double[]>(NRANDS) };
    std::knuth_b randeng { std::random_device {}() };
    auto         rddist {
        std::uniform_real_distribution<double> { -100.00, 1000.000 }
    };
    std::generate(randoms.get(), randoms.get() + NRANDS, [&randeng, &rddist]() noexcept -> double { return rddist(randeng); });

    const auto  host_sum { std::reduce(randoms.get(), randoms.get() + NRANDS, 0.000L) };
    long double device_sum {};

    double*     device_vector {};
    ::cudaMalloc(&device_vector, NRANDS * sizeof(decltype(randoms)::element_type));
    ::cudaMemcpy(device_vector, randoms.get(), NRANDS * sizeof(decltype(randoms)::element_type), ::cudaMemcpyHostToDevice);

    ::sum<<<1, 1000>>>(device_vector);
    ::fold<<<1, 1>>>(device_vector);
    ::cudaDeviceSynchronize();

    ::cudaMemcpy(&device_sum, device_vector, sizeof(decltype(randoms)::element_type), ::cudaMemcpyDeviceToHost);

    ::cudaFree(device_vector);

    std::wcout << std::fixed << std::setprecision(10) << L"host => " << host_sum << L" device => " << device_sum << L'\n';

    return EXIT_SUCCESS;
}
