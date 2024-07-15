
#include <algorithm>
#include <concepts>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

template<typename T> concept arithmetic                = std::integral<T> || std::floating_point<T>;

template<typename T> static constexpr inline T newline = static_cast<T>('\n');

template<typename T> requires ::arithmetic<T> class wrapper {
    private:
        T _value {};

    public:
        constexpr __host__ __device__ wrapper() noexcept : _value {} { }

        constexpr __host__ __device__ wrapper(T&& _init) noexcept : _value { _init } { } // NOLINT(google-explicit-constructor)

        constexpr __host__ __device__ wrapper(const wrapper& other) noexcept : _value { other._value } { }

        constexpr __host__ __device__ wrapper& operator=(const wrapper& other) noexcept { _value = other._value; }

        constexpr __host__ __device__ wrapper(wrapper&& other) noexcept : _value { other._value } { other._value = 0; };

        constexpr __host__ __device__ wrapper& operator=(wrapper&& other) noexcept {
            _value       = other._value;
            other._value = 0;
        };

        constexpr __host__ __device__ ~wrapper() noexcept { _value = 0; }

        constexpr __host__ __device__ const T& unwrapped() const noexcept { return _value; }

        constexpr __host__ __device__ T& unwrapped() noexcept { return _value; }

        constexpr __host__ __device__ wrapper operator+(const wrapper& other) const noexcept { return _value + other._value; }

        constexpr __host__ __device__ wrapper& operator+=(const wrapper& other) noexcept {
            _value += other._value;
            return *this;
        }

        constexpr __host__ __device__ wrapper operator-(const wrapper& other) const noexcept { return _value - other._value; }

        constexpr __host__ __device__ wrapper& operator-=(const wrapper& other) noexcept {
            _value -= other._value;
            return *this;
        }

        constexpr __host__ __device__ wrapper operator*(const wrapper& other) const noexcept { return _value * other._value; }

        constexpr __host__ __device__ wrapper& operator*=(const wrapper& other) noexcept {
            _value *= other._value;
            return *this;
        }

        constexpr __host__ __device__ wrapper operator/(const wrapper& other) const noexcept { return _value / other._value; }

        constexpr __host__ __device__ wrapper& operator/=(const wrapper& other) noexcept {
            _value /= other._value;
            return *this;
        }

        constexpr __host__ __device__ wrapper& operator++() noexcept {
            _value++;
            return *this;
        }

        constexpr __host__ __device__ wrapper operator++(int) noexcept {
            _value++;
            return _value - 1;
        }

        constexpr __host__ __device__ wrapper& operator--() noexcept {
            _value--;
            return *this;
        }

        constexpr __host__ __device__ wrapper operator--(int) noexcept {
            _value--;
            return _value + 1;
        }

        template<typename U> friend std::basic_ostream<U>& operator<<(std::basic_ostream<U>& ostream, const wrapper& object) {
            ostream << object._value << newline<U>;
            return ostream;
        }
};

static constexpr unsigned length { 12'000 };

auto wmain() -> int {
    std::random_device                    seeder {};
    std::mt19937_64                       engine { seeder() };
    std::uniform_real_distribution<float> distr { 0.0, 100.0 }; // min = 0.0, max = 100.0

    wrapper<int> fmax { std::numeric_limits<unsigned char>::max() };
    std::wcout << fmax;
    fmax += fmax;
    std::wcout << fmax;

    std::vector<wrapper<float>> collection {};
    collection.reserve(length);

    // the pains of having a deleted move ctor huh!
    // let's use the copy ctor

    for (unsigned i {}; i < length; ++i) collection.emplace_back(distr(engine));
    for (const auto& elem : collection) std::wcout << elem;

    return EXIT_SUCCESS;
}
