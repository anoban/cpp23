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
        scalar_wrapper() noexcept : _value {} { }

        scalar_wrapper(const T& _init) noexcept : _value { _init } { }

        scalar_wrapper(const scalar_wrapper& other) noexcept : _value { other._value } { }

        scalar_wrapper(scalar_wrapper&& other) noexcept : _value { other._value } { other._value = 0; }

        scalar_wrapper& operator=(const scalar_wrapper& other) noexcept {
            if (this == std::addressof(other)) return *this;
            _value = other._value;
            return *this;
        }

        scalar_wrapper& operator=(scalar_wrapper&& other) noexcept {
            if (this == std::addressof(other)) return *this;
            _value       = other._value;
            other._value = 0;
            return *this;
        }

        ~scalar_wrapper() = default;

        scalar_wrapper operator+(const scalar_wrapper& other) const noexcept { return { _value + other._value }; }

        scalar_wrapper& operator+=(const scalar_wrapper& other) noexcept {
            _value += other._value;
            return *this;
        }

        type unwrapped() const noexcept { return _value; }
};

static constexpr auto size { 1024 * 1024 };

/*
error: calling a __host__ function("scalar_wrapper<double, double> ::operator +=(const scalar_wrapper<double, double> &)") from a __global__ function("kernel<double> ") is not allowed
error: identifier "scalar_wrapper<double, double> ::operator +=" is undefined in device code
error: calling a __host__ function("scalar_wrapper<double, double> ::operator =(const scalar_wrapper<double, double> &)") from a __global__ function("kernel<double> ") is not allowed
error: identifier "scalar_wrapper<double, double> ::operator =" is undefined in device code
*/

template<typename T> requires std::is_arithmetic_v<T>
__global__ void kernel(_In_ const scalar_wrapper<T>* const array, _In_ const unsigned& length, _Inout_ scalar_wrapper<T>* const out) {
    scalar_wrapper<T> temp {};
    // for (const auto& i : std::ranges::views::iota(0u, length)) temp = temp + array[i];   // error: identifier "std::ranges::views::iota" is undefined in device code
    for (auto i = 0; i < length; ++i) temp += array[i];
    *out = temp;
}

auto wmain() -> int {
    srand(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    std::vector<scalar_wrapper<double>> randoms;
    randoms.reserve(size);

    for (const auto& _ : std::ranges::views::iota(0, size)) randoms.emplace_back(rand());
    const auto sum { std::reduce(randoms.cbegin(), randoms.cend()).unwrapped() };

    long double loopsum {};
    for (const auto& i : std::ranges::views::iota(0, size)) loopsum += randoms[i].unwrapped();

    decltype(randoms)::value_type* device_vector {};
    decltype(randoms)::value_type *device_sum {}, copy {};

    cudaMalloc(&device_vector, sizeof(decltype(randoms)::value_type) * randoms.size());
    cudaMalloc(&device_sum, sizeof(decltype(randoms)::value_type));
    cudaMemcpy(device_vector, randoms.data(), sizeof(decltype(randoms)::value_type) * randoms.size(), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(device_vector, randoms.size(), device_sum);
    cudaMemcpy(&copy, device_sum, sizeof copy, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(device_vector);
    cudaFree(device_sum);

    std::wcout << L"Sum from host :: " << sum << L" :: " << loopsum << L" sum from device :: " << copy.unwrapped() << L'\n';

    return EXIT_SUCCESS;
}
