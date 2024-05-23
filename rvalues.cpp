#include <cstdlib>
#include <numbers>
#include <type_traits>

template<typename T, typename = std::enable_if<std::is_scalar<T>::value, T>::type> class pair {
    private:
        T first;
        T second;

    public:
        constexpr pair() noexcept : first(), second() { }

        constexpr explicit pair(const T& _val) noexcept : first(_val), second(_val) { }

        constexpr explicit pair(const T& _val0, const T& _val1) noexcept : first(_val0), second(_val1) { }

        constexpr ~pair() = default;

        constexpr pair(const pair& other) noexcept : first(other.first), second(other.second) { }

        // template<typename U> constexpr explicit pair<U, true>(const pair<U>& other) noexcept : first(other.first), second(other.second) { }
};

int wmain() {
    constexpr ::pair<float> a { std::numbers::pi_v<float> };
    constexpr auto          x { ::pair<short> {} };
    constexpr auto          y {
        ::pair<double> { 12.086, 6543.0974 }
    };

    ::pair<float>           z { y };
    constexpr ::pair<float> q { a };

    return EXIT_SUCCESS;
}
