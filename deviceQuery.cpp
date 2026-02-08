// refactored version of https://raw.githubusercontent.com/NVIDIA/cuda-samples/master/Samples/1_Utilities/deviceQuery/deviceQuery.cpp
// tailored for CUDA runtimes 12.3 and later on Windows

// clang .\deviceQuery.cpp -Wall -Wextra -pedantic -O3 -std=c++20 -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include" -L "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64\" -o .\deviceQuery.exe
// or simply use nvcc
// nvcc .\deviceQuery.cpp -O3 -std=c++20  --run

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

// this queries the properties of the CUDA devices present in the system via the CUDA Runtime API.

#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// NOLINTBEGIN(cppcoreguidelines-pro-type-vararg)
// we don't want clang-tidy moaning about C's stdio functions

#pragma comment(lib, "cudart.lib") // CUDA runtime API
#pragma comment(lib, "cuda.lib")   // CUDA driver API

constexpr auto    BYTES_PER_MB { 1024.0L * 1024.0L };
constexpr int64_t MAX_ANTICIPATED_DEVICES { 4 }; // define the maximum number of devices you expect a system to have
// defaulting to 64 seems a little far fetched

static_assert(sizeof(cudaDeviceProp) == 1032); // because cudaDeviceProp is a really huge struct

// imagine the size 64 such structs would take up on the stack

// an error handler for CUDA runtime API calls
// NOTE THE CHECK TAKES PLACE ON A PRVALUE!
// THE RETURN VALUE OF API CALLS IS NOT BEING CAPTURED IN A VARIABLE
static void check(cudaError_t status, const char* const function, const char* const file, const int line) noexcept {
    // handle if the status is not cudaSuccess
    if (status != cudaSuccess) {
        ::fprintf(
            stderr,
            "CUDA error in %s @ line %d :: code = %d (%s) in call to \"%s\" \n",
            file,
            line,
            static_cast<unsigned int>(status),
            ::cudaGetErrorName(status),
            function
        );
        ::exit(EXIT_FAILURE); // avoiding heap allocations as this premature exit could potentially lead to memory leaks
    }
}

// this has to be a macro, with a constexpr alternative L#expression, __FILEW__, __LINE__ will expand in-line, will be of no
// practical use to us
#define checkCudaErrors(expression) ::check((expression), #expression, __FILE__, __LINE__)

// function copied and refactored from https://raw.githubusercontent.com/NVIDIA/cuda-samples/master/Common/helper_cuda.h
static inline int CoreCountFromSMLookup(int& major, int& minor) noexcept {
    // defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    struct sSMtoCores {
            int32_t SM; // 0xMm (hexidecimal notation), M = SM Major version and m = SM minor version
            int32_t Cores;
    };

    static constexpr std::array<sSMtoCores, 19> nGpuArchCoresPerSM {
        {
         { 0x30, 192 },
         { 0x32, 192 },
         { 0x35, 192 },
         { 0x37, 192 },
         { 0x50, 128 },
         { 0x52, 128 },
         { 0x53, 128 },
         { 0x60, 64 },
         { 0x61, 128 },
         { 0x62, 128 },
         { 0x70, 64 },
         { 0x72, 64 },
         { 0x75, 64 },
         { 0x80, 64 },
         { 0x86, 128 },
         { 0x87, 128 },
         { 0x89, 128 },
         { 0x90, 128 },
         { -1, -1 } // indicator - SM version not in the records
        }
    };

    int32_t index {};

    while (nGpuArchCoresPerSM[index].SM != -1) { // until we reach to the last sSMtoCores struct
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) return nGpuArchCoresPerSM[index].Cores;
        index++;
    }

    // if we don't find the values, we default use the previous one to run properly
    ::printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}

int main() {
    ::puts(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int32_t     deviceCount {};
    // cudaError_t is a typedef to cudaError, which is a plain C style enum
    cudaError_t cudaStatus = ::cudaGetDeviceCount(&deviceCount);

    if (cudaStatus != cudaSuccess) {
        // looks like NVIDIA isn't as keen as MS in providing wchar_t variants for all their APIs
        ::printf("cudaGetDeviceCount returned %d (%s)\n", static_cast<int32_t>(cudaStatus), ::cudaGetErrorName(cudaStatus));
        ::puts("Result = FAIL\n");
        ::exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (!deviceCount)
        ::puts("There are no available device(s) that support CUDA\n");
    else
        ::printf("Detected %d CUDA Capable device(s)\n", deviceCount);

    int32_t        driverVersion {}, runtimeVersion {};
    cudaDeviceProp deviceProp {};

    for (int32_t i {}; i < deviceCount; ++i) {
        ::cudaSetDevice(i);                           // select device i for consideration
        ::cudaGetDeviceProperties_v2(&deviceProp, i); // collect device i's properties

        ::printf("\nDevice %d: \"%s\"\n", i, deviceProp.name);

        ::cudaDriverGetVersion(&driverVersion);
        ::cudaRuntimeGetVersion(&runtimeVersion);
        ::printf(
            "  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
            driverVersion / 1000,
            (driverVersion % 100) / 10,
            runtimeVersion / 1000,
            (runtimeVersion % 100) / 10
        );
        ::printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        ::printf(
            "  Total amount of global memory:                 %.0Lf MBytes (%zu bytes)\n",
            deviceProp.totalGlobalMem / BYTES_PER_MB,
            deviceProp.totalGlobalMem
        );

        ::printf(
            "  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
            deviceProp.multiProcessorCount,
            ::CoreCountFromSMLookup(deviceProp.major, deviceProp.minor),
            ::CoreCountFromSMLookup(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount
        );
        ::printf(
            "  GPU Max Clock rate:                            %.0Lf MHz (%0.2Lf GHz)\n",
            deviceProp.clockRate * 1e-3L,
            deviceProp.clockRate * 1e-6L
        );

        // This is supported in CUDA 5.0 (runtime API device properties)
        ::printf("  Memory Clock rate:                             %.0Lf Mhz\n", deviceProp.memoryClockRate * 1e-3L);
        ::printf("  Memory Bus Width:                              %d-bit\n", deviceProp.memoryBusWidth);

        // if the device does have a L2 cache,
        if (deviceProp.l2CacheSize) ::printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);

        ::printf(
            "  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
            deviceProp.maxTexture1D,
            deviceProp.maxTexture2D[0],
            deviceProp.maxTexture2D[1],
            deviceProp.maxTexture3D[0],
            deviceProp.maxTexture3D[1],
            deviceProp.maxTexture3D[2]
        );
        ::printf(
            "  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n"
            "  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
            deviceProp.maxTexture1DLayered[0],
            deviceProp.maxTexture1DLayered[1],
            deviceProp.maxTexture2DLayered[0],
            deviceProp.maxTexture2DLayered[1],
            deviceProp.maxTexture2DLayered[2]
        );

        ::printf(
            "  Total amount of constant memory:               %zu bytes\n"
            "  Total amount of shared memory per block:       %zu bytes\n"
            "  Total shared memory per multiprocessor:        %zu bytes\n"
            "  Total number of registers available per block: %d\n"
            "  Warp size:                                     %d\n"
            "  Maximum number of threads per multiprocessor:  %d\n"
            "  Maximum number of threads per block:           %d\n",
            deviceProp.totalConstMem,
            deviceProp.sharedMemPerBlock,
            deviceProp.sharedMemPerMultiprocessor,
            deviceProp.regsPerBlock,
            deviceProp.warpSize,
            deviceProp.maxThreadsPerMultiProcessor,
            deviceProp.maxThreadsPerBlock
        );

        ::printf(
            "  Max dimension size of a thread block  (x,y,z): (%d, %d, %d)\n"
            "  Max dimension size of a grid size     (x,y,z): (%d, %d, %d)\n",
            deviceProp.maxThreadsDim[0],
            deviceProp.maxThreadsDim[1],
            deviceProp.maxThreadsDim[2],
            deviceProp.maxGridSize[0],
            deviceProp.maxGridSize[1],
            deviceProp.maxGridSize[2]
        );

        ::printf(
            "  Maximum memory pitch:                          %zu bytes\n"
            "  Texture alignment:                             %zu bytes\n",
            deviceProp.memPitch,
            deviceProp.textureAlignment
        );

        ::printf(
            "  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n",
            deviceProp.deviceOverlap ? "Yes" : "No",
            deviceProp.asyncEngineCount
        );

        ::printf(
            "  Run time limit on kernels:                     %s\n"
            "  Integrated GPU sharing Host Memory:            %s\n"
            "  Support host page-locked memory mapping:       %s\n"
            "  Alignment requirement for Surfaces:            %s\n"
            "  Device has ECC support:                        %s\n",
            deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No",
            deviceProp.integrated ? "Yes" : "No",
            deviceProp.canMapHostMemory ? "Yes" : "No",
            deviceProp.surfaceAlignment ? "Yes" : "No",
            deviceProp.ECCEnabled ? "Enabled" : "Disabled"
        );

        ::printf(
            "  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
            deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)"
        );

        ::printf(
            "  Device supports Unified Addressing (UVA):      %s\n"
            "  Device supports Managed Memory:                %s\n"
            "  Device supports Compute Preemption:            %s\n"
            "  Supports Cooperative Kernel Launch:            %s\n"
            "  Supports MultiDevice Co-op Kernel Launch:      %s\n",
            deviceProp.unifiedAddressing ? "Yes" : "No",
            deviceProp.managedMemory ? "Yes" : "No",
            deviceProp.computePreemptionSupported ? "Yes" : "No",
            deviceProp.cooperativeLaunch ? "Yes" : "No",
            deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No"
        );

        ::printf(
            "  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
            deviceProp.pciDomainID,
            deviceProp.pciBusID,
            deviceProp.pciDeviceID
        );

        static constexpr std::array<const char*, 6> sComputeMode {
            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
            "Unknown",
            nullptr
        };
        ::puts("  Compute Mode:\n");
        ::printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
    }

    // if there are 2 or more GPUs, query to determine whether RDMA is supported
    if (deviceCount >= 2) {
        // dynamic allocation will probably be better here, but PERFORMANCE!!!
        static std::array<cudaDeviceProp, MAX_ANTICIPATED_DEVICES> devProps {};
        static std::array<int32_t, MAX_ANTICIPATED_DEVICES>        P2PGpuIds {}; // this is an array of GPU ids capable of P2P
        int32_t                                                    p2pCount {};

        for (int32_t i {}; i < deviceCount; i++) {
            checkCudaErrors(::cudaGetDeviceProperties_v2(&devProps[i], i));

            // only boards based on Fermi or later can support P2P
            // on Windows (64-bit), the Tesla Compute Cluster driver for windows must be enabled to support this
            if ((devProps[i].major >= 2) && devProps[i].tccDriver) P2PGpuIds[p2pCount++] = i;
        }

        // show all the combinations of support P2P GPUs
        int32_t can_access_peer {};

        if (p2pCount >= 2) {
            for (int32_t i = 0; i < p2pCount; i++) {
                for (int32_t j = 0; j < p2pCount; j++) {
                    if (P2PGpuIds[i] == P2PGpuIds[j]) continue;
                    checkCudaErrors(::cudaDeviceCanAccessPeer(&can_access_peer, P2PGpuIds[i], P2PGpuIds[j]));
                    ::printf(
                        "> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
                        devProps[P2PGpuIds[i]].name,
                        P2PGpuIds[i],
                        devProps[P2PGpuIds[j]].name,
                        P2PGpuIds[j],
                        can_access_peer ? "Yes" : "No"
                    );
                }
            }
        }
    }
    ::puts("");
    ::printf(
        "deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = %d.%d, CUDA Runtime Version = %d.%d, Number of Devices = %d\nResult = PASS\n\n",
        driverVersion / 1000,
        (driverVersion % 1000) / 10,
        runtimeVersion / 1000,
        (runtimeVersion % 100) / 10,
        deviceCount
    );

    return EXIT_SUCCESS;
}

// NOLINTEND(cppcoreguidelines-pro-type-vararg)
