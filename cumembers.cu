#include <algorithm>
#include <concepts>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

template<typename T> concept arithmetic                = std::integral<T> || std::floating_point<T>;

template<typename T> static constexpr inline T newline = static_cast<T>('\n');

template<typename T> requires ::arithmetic<T> class wrapper {
    private:
        T _value {};

    public:
        typedef T value_type;

        constexpr __host__ __device__ wrapper() noexcept : _value {} { }

        constexpr __host__ __device__ wrapper(const T& _init) noexcept : _value { _init } { } // NOLINT(google-explicit-constructor)

        constexpr __host__ __device__ wrapper(const wrapper& other) noexcept : _value { other._value } { }

        constexpr __host__ __device__ wrapper& operator=(const wrapper& other) noexcept {
            if (this == &other) return *this; // identity check is not gonna save us much here but anyways
            _value = other._value;
            return *this;
        }

        constexpr __host__ __device__ wrapper(wrapper&& other) noexcept : _value { other._value } { other._value = 0; };

        constexpr __host__ __device__ wrapper& operator=(wrapper&& other) noexcept {
            if (this == &other) return *this;
            _value       = other._value;
            other._value = 0;
            return *this;
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

        template<typename U> friend __host__ std::basic_ostream<U>& operator<<(std::basic_ostream<U>& ostream, const wrapper& object) {
            ostream << object._value << newline<U>;
            return ostream;
        }
};

static constexpr unsigned N_THREADS { 240 };
static constexpr unsigned N_OPERATIONS { 10'000 };
static constexpr unsigned MAXX_SIZE { N_THREADS * N_OPERATIONS }; // requisite for the semantics of kernel and reduce functions

template<typename T> __global__ void kernel(T* const _rsrc_ptr, const unsigned _rsrc_count) {
    const auto index { threadIdx.x + threadIdx.y + threadIdx.z };
    T          temporary {};
    for (unsigned i = index * _rsrc_count; i < index * _rsrc_count + _rsrc_count; ++i)
        temporary += _rsrc_ptr[i];              // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    _rsrc_ptr[index * _rsrc_count] = temporary; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

template<typename T> __global__ void reduce(T* const _rsrc_ptr, const unsigned _stride, const unsigned _length) {
    T temporary {};
    for (unsigned i = 0; i < _length; i += _stride) temporary += _rsrc_ptr[i]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    _rsrc_ptr[0] = temporary;                                                  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

auto wmain() -> int {
    std::wcout << std::setprecision(std::numeric_limits<float>::digits10);

    std::random_device seeder {};
    std::mt19937_64    engine { seeder() };

    wrapper<double> fmax { std::numbers::pi_v<double> };
    std::wcout << fmax;
    fmax += fmax;
    std::wcout << fmax;

    std::vector<wrapper<float>> collection {};
    collection.reserve(MAXX_SIZE);
    std::uniform_real_distribution<decltype(collection)::value_type::value_type> dist { 10.0, 20.0 }; // min, max

    for (unsigned i {}; i < MAXX_SIZE; ++i) collection.emplace_back(dist(engine));
    const auto sum { std::accumulate(collection.cbegin(), collection.cend(), decltype(collection)::value_type {}) };

    decltype(collection)::value_type* device_array {};
    decltype(collection)::value_type  device_sum {};
    cudaError_t                       cudaStatus {};

    cudaStatus = ::cudaMalloc(&device_array, sizeof(decltype(collection)::value_type) * MAXX_SIZE);
    if (cudaStatus != cudaError_t::cudaSuccess) {
        std::wcerr << L"cudaMalloc failed @ line " << __LINE__ << L'\n';
        return EXIT_FAILURE;
    }

    cudaStatus = ::cudaMemcpy(
        device_array, collection.data(), sizeof(decltype(collection)::value_type) * MAXX_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice
    );
    if (cudaStatus != cudaError_t::cudaSuccess) {
        std::wcerr << L"cudaMemcpy failed @ line " << __LINE__ << L'\n';
        goto ERROR;
    }

    kernel<<<1, N_THREADS>>>(device_array, N_OPERATIONS);
    cudaStatus = ::cudaDeviceSynchronize();
    if (cudaStatus != cudaError_t::cudaSuccess) {
        std::wcerr << L"cudaDeviceSynchronize failed @ line " << __LINE__ << L'\n';
        goto ERROR;
    }

    reduce<<<1, 1>>>(device_array, N_OPERATIONS, MAXX_SIZE);
    cudaStatus = ::cudaDeviceSynchronize();
    if (cudaStatus != cudaError_t::cudaSuccess) {
        std::wcerr << L"cudaDeviceSynchronize failed @ line " << __LINE__ << L'\n';
        goto ERROR;
    }

    cudaStatus = ::cudaMemcpy(&device_sum, device_array, sizeof(decltype(collection)::value_type), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaError_t::cudaSuccess) {
        std::wcerr << L"cudaMemcpy failed @ line " << __LINE__ << L'\n';
        goto ERROR;
    }

    std::wcout << sum.unwrapped() << L" @ line " << __LINE__ << L'\n';
    std::wcout << device_sum.unwrapped() << L" @ line " << __LINE__ << L'\n';

    // there is a tiny variation between the device sum and the host sum

    ::cudaFree(device_array);
    return EXIT_SUCCESS;

ERROR:
    ::cudaFree(device_array);
    return EXIT_FAILURE;
}
