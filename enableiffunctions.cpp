// g++ enableiffunctions.cpp  -Wall -Wextra -Wpedantic -O3 -std=c++20

#include <iostream>
#include <type_traits>

// using std::enable_if in return types
// if std::enable_if<condition>::type evaluates to nothing, function definition will fail as it will lack a valid return type
template<typename T> [[nodiscard]] static consteval std::enable_if<std::is_integral_v<T>, T>::type square(T x) noexcept { return x * x; }

template<typename T> [[nodiscard]] static consteval long long                                      power(
                                         long base,
                                         T    exponent,
                                         typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value, T>::type =
                                             0 // third optional type with a default value
                                         // when std::enable_if fails, function definition will become invalid as the type of the third argument will disappear!
                                     ) noexcept {
    long long result { 1 };
    for (unsigned i = 0; i < exponent; ++i) result *= base;
    return result;
}

int main() {
    constexpr long value { 9 };
    std::wcout << ::square(value) << std::endl;
    ::square(7.76471);
    ::square('A');
    ::square(67Ui16);

    ::power(9, 3U);
    ::power(9, 3.764532);
    ::power(9, 2);
    ::power(9, 3Ui64);
    ::power(9, 3Ui8);

    return 0;
}
