// nvcc .\kernel.cu -std=c++20 -O3 -o .\kernel.exe

#include <algorithm>
#include <array>
#include <cstdio>
#include <numeric>
#include <random>
#include <ranges>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template<typename scalar_t> requires std::is_scalar_v<scalar_t>
__global__ void addKernel(_Inout_ scalar_t* out, _In_ const scalar_t* const in_0, _In_ const scalar_t* const in_1) {
    const auto i { threadIdx.x + threadIdx.y + threadIdx.z };
    out[i] = in_0[i] + in_1[i];
    return;
}

static constexpr size_t nthreads { 450 };

template<typename T, typename = std::enable_if<std::is_scalar<T>::value, T>::type> static constexpr size_t memsize = sizeof(T) * nthreads;

// Helper function for using CUDA to add vectors in parallel.
template<typename scalar_t>
cudaError_t addWithCuda(
    _Inout_ std::array<scalar_t, nthreads>& out,
    _In_ const std::array<scalar_t, nthreads>& in_0,
    _In_ const std::array<scalar_t, nthreads>& in_1,
    _In_opt_ typename std::enable_if<std::is_scalar<scalar_t>::value, scalar_t>::type = static_cast<scalar_t>(0)
) noexcept {
    scalar_t*   dev_in0 {};
    scalar_t*   dev_in1 {};
    scalar_t*   dev_out {};
    cudaError_t cudaStatus {};

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = ::cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        ::fputws(L"cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", stderr);
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = ::cudaMalloc(reinterpret_cast<void**>(&dev_out), memsize<scalar_t>);
    if (cudaStatus != cudaSuccess) {
        ::fputws(L"cudaMalloc failed!", stderr);
        goto Error;
    }

    cudaStatus = ::cudaMalloc(reinterpret_cast<void**>(&dev_in0), memsize<scalar_t>);
    if (cudaStatus != cudaSuccess) {
        ::fputws(L"cudaMalloc failed!", stderr);
        goto Error;
    }

    cudaStatus = ::cudaMalloc(reinterpret_cast<void**>(&dev_in1), memsize<scalar_t>);
    if (cudaStatus != cudaSuccess) {
        ::fputws(L"cudaMalloc failed!", stderr);
        goto Error;
    }

    // copy input arrays from host memory to GPU buffers.
    cudaStatus = ::cudaMemcpy(dev_in0, in_0.data(), memsize<scalar_t>, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        ::fputws(L"cudaMemcpy failed!", stderr);
        goto Error;
    }

    cudaStatus = ::cudaMemcpy(dev_in1, in_1.data(), memsize<scalar_t>, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        ::fputws(L"cudaMemcpy failed!", stderr);
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<scalar_t><<<1, nthreads>>>(dev_out, dev_in0, dev_in1);

    // Check for any errors launching the kernel
    cudaStatus = ::cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        ::fwprintf_s(stderr, L"addKernel launch failed: %S\n", ::cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = ::cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        ::fwprintf_s(stderr, L"cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = ::cudaMemcpy(out.data(), dev_out, memsize<scalar_t>, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        ::fputws(L"cudaMemcpy failed!", stderr);
        goto Error;
    }

Error:
    ::cudaFree(dev_out);
    ::cudaFree(dev_in0);
    ::cudaFree(dev_in1);

    return cudaStatus;
}

int wmain() {
    std::array<float, nthreads> a {};
    std::array<float, nthreads> b {};
    std::array<float, nthreads> c {};

    std::random_device rdevice {};
    std::mt19937_64    rand_engine { rdevice() };

    // fill arrays a and b with random floats
    std::generate(a.begin(), a.end(), [&rand_engine]() noexcept {
        return static_cast<float>(rand_engine() / static_cast<double>(RAND_MAX));
    });
    std::generate(b.begin(), b.end(), [&rand_engine]() noexcept {
        return static_cast<float>(rand_engine() / static_cast<double>(RAND_MAX));
    });

    const auto host_sum { std::accumulate(a.cbegin(), a.cend(), 0.0F, std::plus<float> {}) +
                          std::accumulate(b.cbegin(), b.cend(), 0.0F, std::plus<float> {}) };

    ::_putws(L"so far so good :)");

    cudaError_t cudaStatus { ::addWithCuda<float>(c, a, b) };
    if (cudaStatus != cudaSuccess) {
        ::fputws(L"addWithCuda failed!", stderr);
        return EXIT_FAILURE;
    }

    ::_putws(L"kernel launch is over :)");

    for (const auto& i : std::ranges::views::iota(0LLU, nthreads)) ::wprintf_s(L"%.4f + %.4f = %.4f\n", a.at(i), b.at(i), c.at(i));
    // for (size_t i {}; i < nthreads; ++i) ::wprintf_s(L"%.4f + %.4f = %.4f\n", a.at(i), b.at(i), c.at(i));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = ::cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        ::fputws(L"cudaDeviceReset failed!", stderr);
        return EXIT_FAILURE;
    }

    const auto device_sum { std::reduce(c.cbegin(), c.cend(), 0.0F, std::plus<float> {}) };

    ::_putws(L"all's good :)");
    ::wprintf_s(L"host :: %.5f, device :: %.5f\n", host_sum, device_sum);

    return EXIT_SUCCESS;
}
