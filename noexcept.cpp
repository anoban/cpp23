#include <type_traits>

template<typename T>
static constexpr typename std::enable_if<std::is_scalar<T>::value, T>::type square(const T x
) noexcept(std::is_nothrow_default_constructible<T>::value) {
    return x * x;
}

template<typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
static constexpr long double power(const T base, unsigned exp) { // no exception specifiers here
    long double result = base;
    for (unsigned i = 1; i < exp; ++i) result *= base;
    return result;
}

template<typename T>
long double static constexpr cube(
    const T x, typename std::enable_if<std::is_scalar<T>::value && !std::is_floating_point<T>::value, T>::type = 0
) noexcept(noexcept(::power(x, 3))) {
    return ::power(x, 3);
}

int main() {
    //
    constexpr auto four { ::square(2.000) };
    constexpr auto thousand { ::cube(10i16) };
    constexpr auto sixtyfour { ::cube(::square(20 / 10)) };
    return 0;
}
