// nvcc .\reduce.cu -std=c++20 -O3 -o .\reduce.exe

#include <numeric>
#include <random>
#include <vector>

static constexpr size_t NTHREADS { 1024 };
static constexpr size_t NRANDOMS { NTHREADS * 1024 };

template<typename T> static constexpr typename std::enable_if<std::is_scalar<T>::value, size_t>::type memsize = sizeof(T) * NRANDOMS;

__global__ void sum(_In_reads_(NRANDOMS) const int32_t* const numbers, _Inout_count_(NTHREADS) double* const sums) {
    const auto index { threadIdx.x + threadIdx.y + threadIdx.z };
    double     sum {};
    for (unsigned i = index * NTHREADS; i < (index * NTHREADS) + NTHREADS; ++i) sum += numbers[i];
    sums[index] = sum;
    return;
}

// a fold kernel to sum the NTHREADS sums produced by each kernel
__global__ void fold(_In_ const double* const sums, _Inout_ double* const result) {
    double sum {};
    for (size_t i {}; i < NTHREADS; ++i) sum += sums[i];
    *result = sum;
}

auto main() -> int {
    // std::random_device rdevice {};
    // std::mt19937_64    rengine { rdevice() };

    srand(time(nullptr));
    std::vector<int32_t> randoms(NRANDOMS);
    std::generate(randoms.begin(), randoms.end(), rand);

    const auto host_sum { std::accumulate(randoms.cbegin(), randoms.cend(), 0.0L) };

    int32_t*    d_randoms {};
    double*     d_sums {};
    double*     d_total {};
    cudaError_t cuStatus {};

    // TODO: Error handling
    cuStatus = ::cudaMalloc(&d_randoms, memsize<int32_t>);
    cuStatus = ::cudaMalloc(&d_sums, sizeof(double) * NTHREADS);
    cuStatus = ::cudaMalloc(&d_total, sizeof(double));

    cuStatus = ::cudaMemcpy(d_randoms, randoms.data(), memsize<int32_t>, cudaMemcpyHostToDevice);
    sum<<<1, NTHREADS>>>(d_randoms, d_sums);
    cuStatus = ::cudaDeviceSynchronize();

    fold<<<1, 1>>>(d_sums, d_total);
    cuStatus = ::cudaDeviceSynchronize();

    double device_sum {};
    cuStatus = ::cudaMemcpy(&device_sum, d_total, sizeof(double), cudaMemcpyDeviceToHost);

    ::cudaFree(d_randoms);
    ::cudaFree(d_sums);
    ::cudaFree(d_total);

    ::wprintf_s(L"device sum :: %.5lf, host sum :: %.5Lf\n", device_sum, host_sum);

    return EXIT_SUCCESS;
}
