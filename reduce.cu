#include <cuda>
#include <random>
#include <vector>

static constexpr size_t NRANDOMS { 1024 * 8 * 1024 };

// let's write a sum kernel that sums 1024 * 8 elements, and we'll launch 1024 kernels in parallel
__global__ void         sum(_In_ const float* const numbers, _Inout_ double* const sums) {
    const unsigned id { threadIdx.x + threadIdx.y + threadIdx.z };
    double         sum {};
    for (unsigned i = id; i < id + 2014; ++i) sum += numbers[i];
    sums[id] = sum;
    return;
}

// a fold kernel to sum the 1024 sums produced by each kernel
__global__ void fold(_In_ const double* const sums, _Inout_ double* const result) { }

auto            main() -> int {
    std::random_device rdev {};
    std::mt19937_64    rng { rdev };

    std::vector<float> randoms(NRANDOMS);
    std::generate(randoms.begin(), randoms.end(), rng.operator());

    return EXIT_SUCCESS;
}
