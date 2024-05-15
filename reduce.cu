// nvcc .\reduce.cu -std=c++20 -O3 -o .\reduce.exe

#include <algorithm>
#include <ctime>
#include <numeric>
#include <random>
#include <ranges>
#include <type_traits>
#include <vector>

static constexpr size_t NTHREADS { 1024 };
static constexpr size_t NTASKSPERTHREAD { 200'000 };
static constexpr size_t NRANDOMS { NTHREADS * NTASKSPERTHREAD };

template<typename T> static constexpr typename std::enable_if<std::is_scalar<T>::value, size_t>::type memsize = sizeof(T) * NRANDOMS;

template<typename scalar_in_t, typename scalar_out_t> requires std::is_scalar<scalar_in_t>::value && std::is_scalar<scalar_out_t>::value
__global__ void sum(_In_count_(NRANDOMS) const scalar_in_t* const numbers, _Inout_count_(NTHREADS) scalar_out_t* const sums) {
    const unsigned index { threadIdx.x + threadIdx.y + threadIdx.z };
    scalar_out_t   sum {};
    // cannot use STL in device code
    // for (const auto& i : std::ranges::views::iota(index * NTASKSPERTHREAD, (index * NTASKSPERTHREAD) + NTASKSPERTHREAD)) sum += numbers[i];
    const auto     start = index * NTASKSPERTHREAD;
    for (auto i = start; i < start + NTASKSPERTHREAD; ++i) sum += numbers[i];
    sums[index] = sum;
    return;
}

// a fold kernel to sum the NTHREADS sums produced by each kernel
template<typename scalar_in_t, typename scalar_out_t>
__global__ void fold(
    _In_count_(NTHREADS) const scalar_in_t* const sums,
    _Inout_count_(1) scalar_out_t* const          result,
    typename std::enable_if<std::is_scalar_v<scalar_in_t>, scalar_in_t>::type   = static_cast<scalar_in_t>(0),
    typename std::enable_if<std::is_scalar_v<scalar_out_t>, scalar_out_t>::type = static_cast<scalar_out_t>(0)
) {
    scalar_out_t sum {};
    for (size_t i {}; i < NTHREADS; ++i) sum += sums[i];
    *result = sum;
}

auto wmain() -> int {
    std::random_device rdevice {};
    std::mt19937_64    rengine { rdevice() };

    srand(time(nullptr));
    std::vector<int32_t> randoms(NRANDOMS);
    std::generate(randoms.begin(), randoms.end(), rand);

    int32_t* d_randoms {};
    double*  d_sums {};
    double*  d_total {};

    double     device_sum {};
    const auto host_sum { std::accumulate(randoms.cbegin(), randoms.cend(), 0.0L) };

    cudaError_t cuStatus {};
    cuStatus = ::cudaMalloc(&d_randoms, memsize<int32_t>);
    if (cuStatus != cudaSuccess) {
        ::fwprintf_s(stderr, L"Error @ line %d in %s:: cudaMalloc failed!", __LINE__, __FILEW__);
        return EXIT_FAILURE;
    }

    cuStatus = ::cudaMalloc(&d_sums, sizeof(double) * NTHREADS);
    if (cuStatus != cudaSuccess) {
        ::fwprintf_s(stderr, L"Error @ line %d in %s:: cudaMalloc failed!", __LINE__, __FILEW__);
        goto PREMATURE_EXIT;
    }

    cuStatus = ::cudaMalloc(&d_total, sizeof(double));
    if (cuStatus != cudaSuccess) {
        ::fwprintf_s(stderr, L"Error @ line %d in %s:: cudaMalloc failed!", __LINE__, __FILEW__);
        goto PREMATURE_EXIT;
    }

    cuStatus = ::cudaMemcpy(d_randoms, randoms.data(), memsize<int32_t>, cudaMemcpyHostToDevice);
    if (cuStatus != cudaSuccess) {
        ::fwprintf_s(stderr, L"Error @ line %d in %s:: cudaMemcpy failed!", __LINE__, __FILEW__);
        goto PREMATURE_EXIT;
    }

    sum<<<1, NTHREADS>>>(d_randoms, d_sums);
    cuStatus = cudaGetLastError();
    if (cuStatus != cudaSuccess) {
        ::fwprintf_s(stderr, L"Error @ line %d in %s:: sum kernel launch failed! (%S)", __LINE__, __FILEW__, cudaGetErrorString(cuStatus));
        goto PREMATURE_EXIT;
    }

    cuStatus = ::cudaDeviceSynchronize();
    if (cuStatus != cudaSuccess) {
        ::fwprintf_s(stderr, L"cudaDeviceSynchronize returned error code %d after launching sum kernel!\n", cuStatus);
        goto PREMATURE_EXIT;
    }

    fold<<<1, 1>>>(d_sums, d_total);
    cuStatus = cudaGetLastError();
    if (cuStatus != cudaSuccess) {
        ::fwprintf_s(stderr, L"Error @ line %d in %s:: fold kernel launch failed! (%S)", __LINE__, __FILEW__, cudaGetErrorString(cuStatus));
        goto PREMATURE_EXIT;
    }

    cuStatus = ::cudaDeviceSynchronize();
    if (cuStatus != cudaSuccess) {
        ::fwprintf_s(stderr, L"cudaDeviceSynchronize returned error code %d after launching fold kernel!\n", cuStatus);
        goto PREMATURE_EXIT;
    }

    cuStatus = ::cudaMemcpy(&device_sum, d_total, sizeof(double), cudaMemcpyDeviceToHost);
    if (cuStatus != cudaSuccess) {
        ::fwprintf_s(stderr, L"Error @ line %d in %s:: cudaMemcpy failed!", __LINE__, __FILEW__);
        goto PREMATURE_EXIT;
    }

    ::wprintf_s(L"device sum :: %.5lf, host sum :: %.5Lf\n", device_sum, host_sum);

PREMATURE_EXIT:
    ::cudaFree(d_randoms);
    ::cudaFree(d_sums);
    ::cudaFree(d_total);
    return EXIT_SUCCESS;
}
