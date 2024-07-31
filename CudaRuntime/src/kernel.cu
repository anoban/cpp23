#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <curand.h>
#include <device_launch_parameters.h>

static constexpr size_t NELEMENTS { 1024 * 1024 * 1024 }; // 1 GiB

template<typename scalar_t, typename generator_t> __global__ static void fill_randoms(scalar_t* const device_array, const unsigned size) {
    //
}

auto wmain() -> int {
    //

    ::cudaMalloc();

    return EXIT_SUCCESS;
}