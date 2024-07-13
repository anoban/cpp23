#include <concepts>

template<typename T> concept arithmetic = std::integral<T> || std::floating_point<T>;

template<typename T> requires ::arithmetic<T> class wrapper {
    private:
        T _value {};

    public:
        __host__ __device__ wrapper() noexcept : _value {} { }

        __host__ __device__ wrapper(const T& _init) noexcept : _value { _init } { } // NOLINT(google-explicit-constructor)

        __host__ __device__ wrapper(const wrapper& other) noexcept : _value { other._value } { }

        __host__ __device__ wrapper& operator=(const wrapper& other) noexcept { _value = other._value; }

        __host__ __device__ wrapper(wrapper&& other)            = delete;

        __host__ __device__ wrapper& operator=(wrapper&& other) = delete;

        __host__ __device__ ~wrapper() noexcept { _value = 0; }

        __host__ __device__ const T& unwrapped() const noexcept { return _value; }

        __host__ __device__ T& unwrapped() noexcept { return _value; }

        __host__ __device__ wrapper operator+(const wrapper& other) const noexcept { return _value + other._value; }

        __host__ __device__ wrapper& operator+=(const wrapper& other) noexcept {
            _value += other._value;
            return *this;
        }

        __host__ __device__ wrapper operator-(const wrapper& other) const noexcept { return _value - other._value; }

        __host__ __device__ wrapper& operator-=(const wrapper& other) noexcept {
            _value -= other._value;
            return *this;
        }

        __host__ __device__ wrapper operator*(const wrapper& other) const noexcept { return _value * other._value; }

        __host__ __device__ wrapper& operator*=(const wrapper& other) noexcept {
            _value *= other._value;
            return *this;
        }

        __host__ __device__ wrapper operator/(const wrapper& other) const noexcept { return _value / other._value; }

        __host__ __device__ wrapper& operator/=(const wrapper& other) noexcept {
            _value /= other._value;
            return *this;
        }

        __host__ __device__ wrapper& operator++() noexcept {
            _value++;
            return *this;
        }

        __host__ __device__ wrapper operator++(int) noexcept {
            _value++;
            return _value - 1;
        }

        __host__ __device__ wrapper& operator--() noexcept {
            _value--;
            return *this;
        }

        __host__ __device__ wrapper operator--(int) noexcept {
            _value--;
            return _value + 1;
        }
};

auto wmain() -> int {
    //
    return EXIT_SUCCESS;
}
