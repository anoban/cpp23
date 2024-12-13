#include <chrono>
#include <numeric>
#include <random>

#include <cuda_runtime.h>

#include <cuda/std/numeric>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

static void __global__ csum(_Inout_ double* const array, _In_ const unsigned size) {
    double sum {};
#pragma unroll
    for (unsigned i {}; i < size; ++i) sum += array[i];
    array[0] = sum;
}

static void __global__ custdsum(_In_ const double* const array, _In_ const unsigned size, _Inout_ double* const res) {
    const auto sum { cuda::std::reduce(array, array + size, 0.0L) };
    *res = sum;
}

template<typename _TyDeviceIterator> static
    typename std::enable_if<std::is_arithmetic<typename _TyDeviceIterator::value_type>::value, void>::type __global__
    cudastdsum(_In_ _TyDeviceIterator _begin, _In_ _TyDeviceIterator _end, _Inout_ typename _TyDeviceIterator::value_type* const result) {
    *result = cuda::std::reduce(
        _begin,
        _end,
        static_cast<typename _TyDeviceIterator::value_type>(0) /* _TyDeviceIterator::value_type {} seems problematic here WTF?? */,
        cuda::std::plus<typename _TyDeviceIterator::value_type> {}
    );
}

int main() {
    std::vector<double> randoms(1'000'000);
    std::knuth_b        rengine { static_cast<unsigned>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) };
    std::uniform_real_distribution<double> urdist { -10.0, 100.0 };

    std::generate(randoms.begin(), randoms.end(), [&rengine, &urdist]() noexcept -> auto { return urdist(rengine); });
    const auto                    hsum { std::reduce(randoms.cbegin(), randoms.cend(), 0.0L) };

    thrust::device_vector<double> drandoms { randoms.cbegin(), randoms.cend() };
    const auto                    dsum { thrust::reduce(drandoms.cbegin(), drandoms.cend(), 0.000L) };

    double *                      sum {}, custdsum {}, ksum {};

    cudaMalloc(&sum, sizeof(double));
    // ::custdsum<<<1, 1>>>(drandoms.data().get(), drandoms.size(), sum);
    ::cudastdsum<<<1, 1>>>(drandoms.cbegin(), drandoms.cend(), sum);
    ::cudaMemcpy(&custdsum, sum, sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    ::csum<<<1, 1>>>(drandoms.data().get(), drandoms.size()); // destructive
    ::cudaMemcpy(&ksum, drandoms.data().get(), sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    ::cudaThreadSynchronize();
    ::cudaFree(sum);

    std::cout << std::setw(30) << std::setprecision(20) << "std::reduce " << hsum << '\n';
    std::cout << std::setw(30) << "thrust::reduce " << dsum << '\n';
    std::cout << std::setw(30) << "kernel " << ksum << '\n';
    std::cout << std::setw(30) << "cuda::std::reduce " << custdsum << '\n';

    auto x { 10.00 + long double {} };

    return EXIT_SUCCESS;
}
