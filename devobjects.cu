// nvcc .\devobjects.cu -O3 -std=c++20 --expt-relaxed-constexpr -o .\devobjects.exe     :)

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <vector>

template<typename T, typename = std::enable_if<std::is_arithmetic<T>::value, T>::type> class scalar_wrapper {
    public:
        using type = T;

    private:
        T _value {};

    public:
        __host__ __device__ scalar_wrapper() noexcept : _value {} { }

        __host__ __device__ explicit scalar_wrapper(const T& _init) noexcept : _value { _init } { }

        __host__ __device__ scalar_wrapper(const scalar_wrapper& other) noexcept : _value { other._value } { }

        __host__ __device__ scalar_wrapper(scalar_wrapper&& other) noexcept : _value { other._value } { other._value = 0; }

        __host__ __device__ scalar_wrapper& operator=(const scalar_wrapper& other) noexcept {
            if (this == &other) return *this;
            _value = other._value;
            return *this;
        }

        __host__ __device__ scalar_wrapper& operator=(scalar_wrapper&& other) noexcept {
            if (this == &other) return *this;
            _value       = other._value;
            other._value = 0;
            return *this;
        }

        __host__ __device__ ~scalar_wrapper() noexcept { _value = 0; }

        __host__ __device__ scalar_wrapper operator+(const scalar_wrapper& other) const noexcept {
            return scalar_wrapper { _value + other._value };
        }

        __host__ __device__ scalar_wrapper& operator+=(const scalar_wrapper& other) noexcept {
            _value += other._value;
            return *this;
        }

        __host__ __device__ type unwrapped() const noexcept { return _value; }
};

static constexpr auto size { 1024 * 1024 };

// DO NOT USE C++ REFERENCES IN KERNEL FUNCTIONS
template<typename T> requires std::is_arithmetic_v<T>
__global__ void kernel(_In_ const scalar_wrapper<T>* const array, _In_ const unsigned length, _Inout_ scalar_wrapper<T>* const out) {
    scalar_wrapper<T> temp {};
    // for (const auto& i : std::ranges::views::iota(0u, length)) temp = temp + array[i];   // error: identifier "std::ranges::views::iota" is undefined in device code
    for (auto i = 0; i < length; ++i) temp += array[i];
    printf("sum is %Lf\n", temp.unwrapped());
    *out = temp;
}

auto wmain() -> int {
    srand(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    std::vector<scalar_wrapper<double>> randoms;
    randoms.reserve(size);

    for (const auto& _ : std::ranges::views::iota(0, size)) randoms.emplace_back(rand());
    std::wcout << L"size = " << size << L" randoms.size() = " << randoms.size() << L'\n';

    const auto sum { std::reduce(randoms.cbegin(), randoms.cend()).unwrapped() };

    long double loopsum {};
    for (const auto& i : std::ranges::views::iota(0, size)) loopsum += randoms[i].unwrapped();

    decltype(randoms)::value_type* device_vector {};
    decltype(randoms)::value_type *device_sum {}, copy {};

    cudaMalloc(&device_vector, sizeof(decltype(randoms)::value_type) * randoms.size());
    cudaMalloc(&device_sum, sizeof(decltype(randoms)::value_type));
    cudaMemcpy(device_vector, randoms.data(), sizeof(decltype(randoms)::value_type) * randoms.size(), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(device_vector, randoms.size(), device_sum);
    cudaDeviceSynchronize();

    cudaMemcpy(&copy, device_sum, sizeof copy, cudaMemcpyDeviceToHost);

    cudaFree(device_vector);
    cudaFree(device_sum);

    std::wcout << L"Sum from host :: " << sum << L" :: " << loopsum << L" sum from device :: " << copy.unwrapped() << L'\n';

    return EXIT_SUCCESS;
}
