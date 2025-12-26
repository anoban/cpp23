/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// This sample queries the properties of the CUDA devices present in the system.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <cuda.h>

#pragma comment(lib, "cuda.lib") // required for the driver API calls

#define checkCudaErrors(error) _checkCudaErrors(error, __FILEW__, __LINE__) // NOLINT(cppcoreguidelines-macro-usage)

// NOLINTBEGIN(cppcoreguidelines-pro-type-vararg)

static __forceinline void __stdcall _checkCudaErrors(
    _In_ const CUresult& error, _In_ const wchar_t* const file, _In_ const long& line
) noexcept {
    if (cudaError_enum::CUDA_SUCCESS != error) {
        const char* error_string {};
        ::cuGetErrorString(error, &error_string);
        ::fwprintf(stderr, L"checkCudaErrors() Driver API error = %04d \"%S\" from file <%s>, line %d.\n", error, error_string, file, line);
        ::exit(EXIT_FAILURE);
    }
}

static inline int __stdcall _ConvertSMVer2CoresDRV(_In_ const int& major, _In_ const int& minor) noexcept {
    // Defines for GPU Architecture types (using the SM version to determine the of cores per SM
    struct sSMtoCores {
            int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
            int Cores;
    };

    static constexpr sSMtoCores nGpuArchCoresPerSM[25] = {
        { .SM = 0x30, .Cores = 192 },
        { .SM = 0x32, .Cores = 192 },
        { .SM = 0x35, .Cores = 192 },
        { .SM = 0x37, .Cores = 192 },
        { .SM = 0x50, .Cores = 128 },
        { .SM = 0x52, .Cores = 128 },
        { .SM = 0x53, .Cores = 128 },
        { .SM = 0x60,  .Cores = 64 },
        { .SM = 0x61, .Cores = 128 },
        { .SM = 0x62, .Cores = 128 },
        { .SM = 0x70,  .Cores = 64 },
        { .SM = 0x72,  .Cores = 64 },
        { .SM = 0x75,  .Cores = 64 },
        { .SM = 0x80,  .Cores = 64 },
        { .SM = 0x86, .Cores = 128 },
        { .SM = 0x87, .Cores = 128 },
        { .SM = 0x89, .Cores = 128 },
        { .SM = 0x90, .Cores = 128 },
        { .SM = 0xa0, .Cores = 128 },
        { .SM = 0xa1, .Cores = 128 },
        { .SM = 0xa3, .Cores = 128 },
        { .SM = 0xb0, .Cores = 128 },
        { .SM = 0xc0, .Cores = 128 },
        { .SM = 0xc1, .Cores = 128 },
        {   .SM = -1,  .Cores = -1 }
    };

    int index {};

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) return nGpuArchCoresPerSM[index].Cores;
        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    ::wprintf_s(
        L"MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index - 1].Cores
    );
    return nGpuArchCoresPerSM[index - 1].Cores;
}

int wmain() {
    int         major {}, minor {}, device_count {}; // NOLINT(readability-isolate-declaration)
    std::string device_name(256, ' ');

    // note your project will need to link with cuda.lib files on windows
    ::_putws(L"CUDA Device Query (Driver API) statically linked version");

    checkCudaErrors(::cuInit(0));
    checkCudaErrors(::cuDeviceGetCount(&device_count));

    // This function call returns 0 if there are no CUDA capable devices.
    if (!device_count)
        ::_putws(L"There are no available device(s) that support CUDA");
    else
        ::wprintf_s(L"Detected %d CUDA Capable device(s)\n", device_count);

    for (int n_device {}; n_device < device_count; ++n_device) {
        checkCudaErrors(::cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, n_device));
        checkCudaErrors(::cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, n_device));

        checkCudaErrors(::cuDeviceGetName(device_name.data(), 256, n_device));

        ::wprintf_s(L"\nDevice %d: \"%S\"\n", n_device, device_name.c_str());

        // NOLINTBEGIN(readability-isolate-declaration,modernize-avoid-c-arrays)

        size_t total_global_mem {};
        int    multi_processor_count {}, clock_rate {}, memory_clock {}, mem_bus_width {}, l2_cache_size {}, max_tex_1d {};
        int    totalConstantMemory {}, sharedMemPerBlock {}, regsPerBlock {}, warpSize {}, maxThreadsPerMultiProcessor {};
        int    maxThreadsPerBlock {}, textureAlign {}, asyncEngineCount {}, gpuOverlap {}, memPitch {};
        int    unified_addressing {}, managed_memory {}, compute_preemption {}, cooperative_launch {}, cooperative_multi_dev_launch {};
        int    pci_domain_id {}, pci_bus_id {}, pci_device_id {}, compute_mode {}, driver_version {};

        int gridDim[3] {}, blockDim[3] {}, max_tex_2d[2] {}, max_tex_3d[3] {}, max_tex_1d_layered[2] {}, max_tex_2d_layered[3] {};

        // NOLINTEND(readability-isolate-declaration,modernize-avoid-c-arrays)

        checkCudaErrors(::cuDriverGetVersion(&driver_version));
        ::wprintf_s(L"  CUDA Driver Version:                           %d.%d\n", driver_version / 1000, (driver_version % 100) / 10);
        ::wprintf_s(L"  CUDA Capability Major/Minor version number:    %d.%d\n", major, minor);

        checkCudaErrors(cuDeviceTotalMem(&total_global_mem, n_device));

        ::wprintf_s(
            L"  Total amount of global memory:                 %.0Lf MBytes (%llu bytes)\n", total_global_mem / 1048576.0L, total_global_mem
        );

        ::cuDeviceGetAttribute(&multi_processor_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, n_device);

        ::wprintf_s(
            L"  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
            multi_processor_count,
            ::_ConvertSMVer2CoresDRV(major, minor),
            ::_ConvertSMVer2CoresDRV(major, minor) * multi_processor_count
        );

        ::cuDeviceGetAttribute(&clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, n_device);
        ::wprintf_s(L"  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", clock_rate * 1e-3f, clock_rate * 1e-6f);

        ::cuDeviceGetAttribute(&memory_clock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, n_device);
        ::wprintf_s(L"  Memory Clock rate:                             %.0f Mhz\n", memory_clock * 1e-3f);

        ::cuDeviceGetAttribute(&mem_bus_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, n_device);
        ::wprintf_s(L"  Memory Bus Width:                              %d-bit\n", mem_bus_width);

        ::cuDeviceGetAttribute(&l2_cache_size, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, n_device);
        if (l2_cache_size) ::wprintf_s(L"  L2 Cache Size:                                 %d bytes\n", l2_cache_size);

        ::cuDeviceGetAttribute(&max_tex_1d, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, n_device);
        ::cuDeviceGetAttribute(&max_tex_2d[0], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, n_device);
        ::cuDeviceGetAttribute(&max_tex_2d[1], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, n_device);
        ::cuDeviceGetAttribute(&max_tex_3d[0], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, n_device);
        ::cuDeviceGetAttribute(&max_tex_3d[1], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, n_device);
        ::cuDeviceGetAttribute(&max_tex_3d[2], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, n_device);
        ::wprintf_s(
            L"  Max Texture Dimension Sizes                    1D=(%d) 2D=(%d, %d) 3D=(%d, %d, %d)\n",
            max_tex_1d,
            max_tex_2d[0],
            max_tex_2d[1],
            max_tex_3d[0],
            max_tex_3d[1],
            max_tex_3d[2]
        );

        ::cuDeviceGetAttribute(&max_tex_1d_layered[0], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH, n_device);
        ::cuDeviceGetAttribute(&max_tex_1d_layered[1], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS, n_device);
        ::wprintf_s(L"  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n", max_tex_1d_layered[0], max_tex_1d_layered[1]);

        ::cuDeviceGetAttribute(&max_tex_2d_layered[0], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH, n_device);
        ::cuDeviceGetAttribute(&max_tex_2d_layered[1], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, n_device);
        ::cuDeviceGetAttribute(&max_tex_2d_layered[2], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS, n_device);
        ::wprintf_s(
            L"  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
            max_tex_2d_layered[0],
            max_tex_2d_layered[1],
            max_tex_2d_layered[2]
        );

        ::cuDeviceGetAttribute(&totalConstantMemory, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, n_device);
        ::wprintf_s(L"  Total amount of constant memory:               %u bytes\n", totalConstantMemory);

        ::cuDeviceGetAttribute(&sharedMemPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, n_device);
        ::wprintf_s(L"  Total amount of shared memory per block:       %u bytes\n", sharedMemPerBlock);

        ::cuDeviceGetAttribute(&regsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, n_device);
        ::wprintf_s(L"  Total number of registers available per block: %d\n", regsPerBlock);

        ::cuDeviceGetAttribute(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, n_device);
        ::wprintf_s(L"  Warp size:                                     %d\n", warpSize);

        ::cuDeviceGetAttribute(&maxThreadsPerMultiProcessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, n_device);
        ::wprintf_s(L"  Maximum number of threads per multiprocessor:  %d\n", maxThreadsPerMultiProcessor);

        ::cuDeviceGetAttribute(&maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, n_device);
        ::wprintf_s(L"  Maximum number of threads per block:           %d\n", maxThreadsPerBlock);

        ::cuDeviceGetAttribute(&blockDim[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, n_device);
        ::cuDeviceGetAttribute(&blockDim[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, n_device);
        ::cuDeviceGetAttribute(&blockDim[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, n_device);
        ::wprintf_s(L"  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n", blockDim[0], blockDim[1], blockDim[2]);

        ::cuDeviceGetAttribute(&gridDim[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, n_device);
        ::cuDeviceGetAttribute(&gridDim[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, n_device);
        ::cuDeviceGetAttribute(&gridDim[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, n_device);
        ::wprintf_s(L"  Max dimension size of a grid size (x,y,z):    (%d, %d, %d)\n", gridDim[0], gridDim[1], gridDim[2]);

        ::cuDeviceGetAttribute(&textureAlign, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, n_device);
        ::wprintf_s(L"  Texture alignment:                             %u bytes\n", textureAlign);

        ::cuDeviceGetAttribute(&memPitch, CU_DEVICE_ATTRIBUTE_MAX_PITCH, n_device);
        ::wprintf_s(L"  Maximum memory pitch:                          %u bytes\n", memPitch);

        ::cuDeviceGetAttribute(&gpuOverlap, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, n_device);

        ::cuDeviceGetAttribute(&asyncEngineCount, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, n_device);
        ::wprintf_s(
            L"  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", gpuOverlap ? "Yes" : "No", asyncEngineCount
        );

        int kernelExecTimeoutEnabled;
        ::cuDeviceGetAttribute(&kernelExecTimeoutEnabled, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, n_device);
        ::wprintf_s(L"  Run time limit on kernels:                     %s\n", kernelExecTimeoutEnabled ? "Yes" : "No");
        int integrated;
        ::cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, n_device);
        ::wprintf_s(L"  Integrated GPU sharing Host Memory:            %s\n", integrated ? "Yes" : "No");
        int canMapHostMemory;
        ::cuDeviceGetAttribute(&canMapHostMemory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, n_device);
        ::wprintf_s(L"  Support host page-locked memory mapping:       %s\n", canMapHostMemory ? "Yes" : "No");

        int concurrentKernels;
        ::cuDeviceGetAttribute(&concurrentKernels, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, n_device);
        ::wprintf_s(L"  Concurrent kernel execution:                   %s\n", concurrentKernels ? "Yes" : "No");

        int surfaceAlignment;
        ::cuDeviceGetAttribute(&surfaceAlignment, CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, n_device);
        ::wprintf_s(L"  Alignment requirement for Surfaces:            %s\n", surfaceAlignment ? "Yes" : "No");

        int eccEnabled;
        ::cuDeviceGetAttribute(&eccEnabled, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, n_device);
        ::wprintf_s(L"  Device has ECC support:                        %s\n", eccEnabled ? "Enabled" : "Disabled");

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        int tccDriver {};
        ::cuDeviceGetAttribute(&tccDriver, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, n_device);
        ::wprintf_s(
            L"  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
            tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)"
        );
#endif

        ::cuDeviceGetAttribute(&unified_addressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, n_device);
        ::wprintf_s(L"  Device supports Unified Addressing (UVA):      %s\n", unified_addressing ? "Yes" : "No");

        ::cuDeviceGetAttribute(&managed_memory, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, n_device);
        ::wprintf_s(L"  Device supports Managed Memory:                %s\n", managed_memory ? "Yes" : "No");

        ::cuDeviceGetAttribute(&compute_preemption, CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, n_device);
        ::wprintf_s(L"  Device supports Compute Preemption:            %s\n", compute_preemption ? "Yes" : "No");

        ::cuDeviceGetAttribute(&cooperative_launch, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, n_device);
        ::wprintf_s(L"  Supports Cooperative Kernel Launch:            %s\n", cooperative_launch ? "Yes" : "No");

        ::cuDeviceGetAttribute(&cooperative_multi_dev_launch, CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH, n_device);
        ::wprintf_s(L"  Supports MultiDevice Co-op Kernel Launch:      %s\n", cooperative_multi_dev_launch ? "Yes" : "No");

        ::cuDeviceGetAttribute(&pci_domain_id, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, n_device);
        ::cuDeviceGetAttribute(&pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, n_device);
        ::cuDeviceGetAttribute(&pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, n_device);
        ::wprintf_s(L"  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", pci_domain_id, pci_bus_id, pci_device_id);

        static const wchar_t* const sComputeMode[] = {
            L"Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
            L"Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
            L"Prohibited (no host thread can use ::cudaSetDevice() with this device)",
            L"Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
            L"Unknown",
            nullptr
        };

        ::cuDeviceGetAttribute(&compute_mode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, n_device);
        ::_putws(L"  Compute Mode:\n");
        ::wprintf_s(L"     < %s >\n", sComputeMode[compute_mode]);
    }

    // If there are 2 or more GPUs, query to determine whether RDMA is supported
    if (device_count >= 2) {
        int gpuid[64]; // we want to find the first two GPUs that can support P2P
        int gpu_p2p_count = 0;
        int tccDriver     = 0;

        for (int i = 0; i < device_count; i++) {
            checkCudaErrors(::cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, i));
            checkCudaErrors(::cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, i));
            ::cuDeviceGetAttribute(&tccDriver, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, i);

            // Only boards based on Fermi or later can support P2P
            if (
                (major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
                // on Windows (64-bit), the Tesla Compute Cluster driver for windows
                // must be enabled to support this
                && tccDriver
#endif
            ) {
                // This is an array of P2P capable GPUs
                gpuid[gpu_p2p_count++] = i;
            }
        }

        // Show all the combinations of support P2P GPUs
        int  can_access_peer;
        char deviceName0[256], deviceName1[256];

        if (gpu_p2p_count >= 2) {
            for (int i = 0; i < gpu_p2p_count; i++) {
                for (int j = 0; j < gpu_p2p_count; j++) {
                    if (gpuid[i] == gpuid[j]) continue;
                    checkCudaErrors(::cuDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
                    checkCudaErrors(::cuDeviceGetName(deviceName0, 256, gpuid[i]));
                    checkCudaErrors(::cuDeviceGetName(deviceName1, 256, gpuid[j]));
                    ::wprintf_s(
                        L"> Peer-to-Peer (P2P) access from %s (GPU%d) -> %s (GPU%d) : %s\n",
                        deviceName0,
                        gpuid[i],
                        deviceName1,
                        gpuid[j],
                        can_access_peer ? L"Yes" : L"No"
                    );
                }
            }
        }
    }

    ::_putws(L"Result = PASS\n");

    return EXIT_SUCCESS;
}

// NOLINTEND(cppcoreguidelines-pro-type-vararg)
