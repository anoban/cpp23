#include <cstddef>
#include <cstdlib>
#include <limits>
#include <ranges>
#include <type_traits>

template<typename T, size_t count, bool is_integer = std::is_integral<T>::value> struct type;

template<typename T, size_t count> struct type<T, count, true> {
        typedef T          value_type;
        static constexpr T value { static_cast<T>(count) };
        consteval T        operator()() const noexcept { return static_cast<T>(count); }
        bool               is_integer { true };
};

template<typename T, size_t count> struct type<T, count, false> {
        typedef T          value_type;
        static constexpr T value { static_cast<T>(count) };
        consteval T        operator()() const noexcept { return static_cast<T>(count); }
        bool               is_integer { false };
};

template<typename scalar_t>
[[nodiscard("Just Don't")]] consteval double pow(_In_ const scalar_t& _base, _In_ const unsigned _exponent) noexcept {
    double result = _base;
    for (const auto& _ : std::ranges::views::iota(1u, _exponent)) result *= _base;
    return result;
}

int wmain() {
    [[maybe_unused]] constexpr auto twohundred = type<unsigned, 100>::value;
    [[maybe_unused]] constexpr auto thirteen { type<unsigned char, 13> {}.operator()() };
    [[maybe_unused]] constexpr auto no { type<decltype(12.65789), 8> {}.is_integer };

    [[maybe_unused]] constexpr auto x { ::pow(4, 3) };

    [[maybe_unused]] constexpr typename ::type<unsigned short, 12>::value_type p { std::numeric_limits<decltype(45ui16)>::max() };
    return EXIT_SUCCESS;
}
