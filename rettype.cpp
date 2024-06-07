// templated function return types

#include <cstdlib>
#include <format>
#include <iostream>
#include <print>
#include <type_traits>

template<typename scalar_t>
[[nodiscard]] static inline constexpr typename std::enable_if<std::is_arithmetic_v<scalar_t>, scalar_t>::type function(
    const scalar_t& _scalar0, const scalar_t& _scalar1
) noexcept {
    // the return type will always be the type of the arguments!
    return _scalar0 * _scalar1;
}

template<typename T, typename U> requires std::is_arithmetic_v<T> && std::is_arithmetic_v<U>
[[nodiscard]] static inline constexpr T function(const T& _scalar0, const U& _scalar1) noexcept {
    // the return type will always be the type of the first argument!
    return _scalar0 * _scalar1;
}

template<typename T, typename U> static inline constexpr auto func(const T& _scalar0, const U& _scalar1) noexcept -> T {
    // trailing return type, but will always be the type of the first argument
    return _scalar0 * _scalar1;
}

template<typename T, typename U>
static inline constexpr auto funcc(const T& _scalar0, const U& _scalar1) noexcept requires requires { _scalar0* _scalar1; } {
    // let the compiler figure out the rrturn type
    return _scalar0 * _scalar1;
}

namespace max {
    template<typename T, typename U> // requires requires {
                                     // std::numeric_limits<T>::max(); // implicitly expects T and U to be numeric types
    // std::numeric_limits<U>::max(); looks like std::numeric_limits does not impose any type constraints on its arg YIKES!
    // }
    constexpr auto max(const T& _arg0, const U& _arg1) noexcept requires requires { _arg0 < _arg1; } {
        return _arg0 > _arg1 ? _arg0 : _arg1;
    }
} // namespace max

auto wmain() -> int {
    constexpr auto constint { function(12, 97) };
    constexpr auto constfloat { function(12.354f, 97) }; // the second overload will be instantiated here!

    constexpr auto what { funcc(1.5456, 54567) };        // compiler has opted for the type with largest range
    constexpr auto whatnow { funcc(7634LLU, 4.21567L) }; // sure, long double has a larger range than unsigned long long

    constexpr auto maxx { max::max(400LLU, 6.8764f) };
    std::wcout << std::format(L"{:.10f}", maxx);

    return EXIT_SUCCESS;
}
