#include <algorithm>
#include <array>
#include <cstdio>
#include <random>
#include <ranges>
#include <type_traits>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// NOLINTBEGIN(cppcoreguidelines-pro-type-vararg)

template<typename _TyNumeric> requires std::is_arithmetic_v<_TyNumeric> __global__ static void addKernel(
    _Inout_ _TyNumeric* const results, _In_ const _TyNumeric* const input_01, _In_ const _TyNumeric* const input_02
) {
    results[threadIdx.x] = input_01[threadIdx.x] + input_02[threadIdx.x]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

// NOLINTNEXTLINE(readability-redundant-inline-specifier)
template<typename _TyNumeric, unsigned long long _ArraySize> static inline // NOLINTNEXTLINE(modernize-use-constraints)
    typename std::enable_if<std::is_integral<_TyNumeric>::value || std::is_floating_point<_TyNumeric>::value, cudaError_t>::
        type __stdcall addWithCuda(
            _Inout_ std::array<_TyNumeric, _ArraySize>& results,
            _In_ const std::array<_TyNumeric, _ArraySize>& input_01,
            _In_ const std::array<_TyNumeric, _ArraySize>& input_02
        ) noexcept {
    _TyNumeric* dev_inputs_01 {};
    _TyNumeric* dev_inputs_02 {};
    _TyNumeric* dev_results {};

    cudaError_t cudaStatus { ::cudaSetDevice(0) }; // choose which GPU to run on, change this on a multi-GPU system.
    if (cudaStatus != cudaSuccess) {
        ::fputws(L"cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", stderr);
        goto ERROR;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = ::cudaMalloc(&dev_results, _ArraySize * sizeof(_TyNumeric));
    if (cudaStatus != cudaSuccess) {
        ::fputws(L"cudaMalloc failed!", stderr);
        goto ERROR;
    }

    cudaStatus = ::cudaMalloc(&dev_inputs_01, _ArraySize * sizeof(_TyNumeric));
    if (cudaStatus != cudaSuccess) {
        ::fputws(L"cudaMalloc failed!", stderr);
        goto ERROR;
    }

    cudaStatus = ::cudaMalloc(&dev_inputs_02, _ArraySize * sizeof(_TyNumeric));
    if (cudaStatus != cudaSuccess) {
        ::fputws(L"cudaMalloc failed!", stderr);
        goto ERROR;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = ::cudaMemcpy(dev_inputs_01, input_01.data(), _ArraySize * sizeof(_TyNumeric), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        ::fputws(L"cudaMemcpy failed!", stderr);
        goto ERROR;
    }

    cudaStatus = ::cudaMemcpy(dev_inputs_02, input_02.data(), _ArraySize * sizeof(_TyNumeric), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        ::fputws(L"cudaMemcpy failed!", stderr);
        goto ERROR;
    }

    // Launch a kernel on the GPU with one thread for each element.
    ::addKernel<<<1, _ArraySize>>>(dev_results, dev_inputs_01, dev_inputs_02);

    // Check for any errors launching the kernel
    cudaStatus = ::cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        ::fwprintf_s(stderr, L"addKernel launch failed: %S\n", ::cudaGetErrorString(cudaStatus));
        goto ERROR;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = ::cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        ::fwprintf_s(stderr, L"cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto ERROR;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = ::cudaMemcpy(results.data(), dev_results, _ArraySize * sizeof(_TyNumeric), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        ::fputws(L"cudaMemcpy failed!", stderr);
        goto ERROR;
    }

ERROR:
    ::cudaFree(dev_results);
    ::cudaFree(dev_inputs_01);
    ::cudaFree(dev_inputs_02);

    return cudaStatus;
}

int wmain() {
    constexpr unsigned long long        ARRAY_LENGTH { 250 };
    std::array<long long, ARRAY_LENGTH> left {}, right {}, sums {}; // NOLINT(readability-isolate-declaration)

    std::mt19937_64 randeng { std::random_device {}() };

    std::generate(left.begin(), left.end(), [&randeng]() noexcept -> long long { return static_cast<long long>(randeng() % 100LL); });
    std::generate(right.begin(), right.end(), [&randeng]() noexcept -> long long { return static_cast<long long>(randeng() % 100LL); });

    // Add vectors in parallel.
    cudaError_t cudaStatus = ::addWithCuda(sums, left, right);
    if (cudaStatus != cudaSuccess) {
        ::fputws(L"addWithCuda failed!", stderr);
        return EXIT_FAILURE;
    }

    for (const auto& i : std::ranges::views::iota(0LLU, ARRAY_LENGTH))
        ::wprintf_s(L"%3lld + %3lld = %4lld\n", left.at(i), right.at(i), sums.at(i));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = ::cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        ::fwprintf_s(stderr, L"cudaDeviceReset failed!");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// NOLINTEND(cppcoreguidelines-pro-type-vararg)
