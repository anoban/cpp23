#include <algorithm>
#include <array>
#include <cstdio>
#include <random>
#include <ranges>
#include <type_traits>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template<typename _TyNumeric> requires std::is_arithmetic_v<_TyNumeric>
__global__ static void addKernel(_Inout_ _TyNumeric* res, _In_ const _TyNumeric* inp_01, _In_ const _TyNumeric* inp_02) {
    const unsigned i = threadIdx.x;
    res[i]           = inp_01[i] + inp_02[i];
}

// Helper function for using CUDA to add vectors in parallel.
template<typename _TyNumeric, unsigned long long _Size> static inline
    typename std::enable_if<std::is_integral<_TyNumeric>::value || std::is_floating_point<_TyNumeric>::value, cudaError_t>::type __stdcall
    addWithCuda(
        _Inout_ std::array<_TyNumeric, _Size>& results,
        _In_ const std::array<_TyNumeric, _Size>& input_01,
        _In_ const std::array<_TyNumeric, _Size>& input_02
    ) noexcept {
    typename _TyNumeric* dev_inputs_01 {};
    typename _TyNumeric* dev_inputs_02 {};
    typename _TyNumeric* dev_results {};
    cudaError_t          cudaStatus {};

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = ::cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        ::fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = ::cudaMalloc((void**) &dev_results, _Size * sizeof(_TyNumeric));
    if (cudaStatus != cudaSuccess) {
        ::fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = ::cudaMalloc((void**) &dev_inputs_01, _Size * sizeof(_TyNumeric));
    if (cudaStatus != cudaSuccess) {
        ::fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = ::cudaMalloc((void**) &dev_inputs_02, _Size * sizeof(_TyNumeric));
    if (cudaStatus != cudaSuccess) {
        ::fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = ::cudaMemcpy(dev_inputs_01, input_01.data(), _Size * sizeof(_TyNumeric), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        ::fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = ::cudaMemcpy(dev_inputs_02, input_02.data(), _Size * sizeof(_TyNumeric), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        ::fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    ::addKernel<<<1, size>>>(dev_results, dev_inputs_01, dev_inputs_02);

    // Check for any errors launching the kernel
    cudaStatus = ::cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        ::fprintf(stderr, "addKernel launch failed: %s\n", ::cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = ::cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        ::fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = ::cudaMemcpy(results.data(), dev_results, _Size * sizeof(_TyNumeric), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        ::fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    ::cudaFree(dev_results);
    ::cudaFree(dev_inputs_01);
    ::cudaFree(dev_inputs_02);

    return cudaStatus;
}

int main() {
    constexpr unsigned long long        ARRAY_LENGTH { 25'000 };
    std::array<long long, ARRAY_LENGTH> left {}, right {}, sums {};

    std::mt19937_64 randeng { std::random_device {}() };

    std::generate(left.begin(), left.end(), randeng);
    std::generate(right.begin(), right.end(), randeng);

    // Add vectors in parallel.
    cudaError_t cudaStatus = ::addWithCuda(sums, left, right);
    if (cudaStatus != cudaSuccess) {
        ::fprintf(stderr, "addWithCuda failed!");
        return EXIT_FAILURE;
    }

    for (const auto& i : std::ranges::views::iota(0LLU, ARRAY_LENGTH))
        ::printf_s("%LLU + %LLU = %LLU\n", left.at(i), right.at(i), sums.at(i));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = ::cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        ::fprintf(stderr, "cudaDeviceReset failed!");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
