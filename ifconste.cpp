#include <cmath>
#include <iostream>

namespace ifconstexxx {
    template<typename T> [[nodiscard]] static inline constexpr double power(const T& base, const unsigned& exponent) noexcept {
        if (std::is_constant_evaluated()) { // if the function is being evaluated @ compile time
            T temp { 1 };
            for (unsigned i = 0; i < exponent; ++i) temp *= base;
            return temp;
        } else {
            return pow(base, exponent);
        }
    }

    template<typename T> [[nodiscard]] static inline constexpr double _power(const T& base, const unsigned& exponent) noexcept {
        if consteval { // consteval if is a C++23 extension
            T temp { 1 };
            for (unsigned i = 0; i < exponent; ++i) temp *= base;
            return temp;
        } else {
            return pow(base, exponent);
        }
    }

    template<typename T> [[nodiscard]] static inline consteval double __power(const T& base, const unsigned& exponent) noexcept {
        T temp { 1 };
        for (unsigned i = 0; i < exponent; ++i) temp *= base;
        return temp;
    }

} // namespace ifconstexxx

static_assert(::ifconstexxx::power(6.00, 6) == 46656.000); // constexpr

auto wmain() -> int {
    constexpr auto shouldwork { ::ifconstexxx::power(2.451, 7) };       // 531.377521456277
    volatile auto  nonconstexpr { ::ifconstexxx::power(8.65472, 11) };  // 20407191608.2931
    const auto     shouldntwork { ::ifconstexxx::power(3.2056411, 4) }; // 105.598947702201

    std::wcout << shouldwork << L' ' << nonconstexpr << L' ' << shouldntwork << L'\n';

    constexpr auto _shouldwork { ::ifconstexxx::_power(2.451, 7) };       // 531.377521456277
    volatile auto  _nonconstexpr { ::ifconstexxx::_power(8.65472, 11) };  // 20407191608.2931
    const auto     _shouldntwork { ::ifconstexxx::_power(3.2056411, 4) }; // 105.598947702201

    std::wcout << _shouldwork << L' ' << _nonconstexpr << L' ' << _shouldntwork << L'\n';

    constexpr auto __shouldwork { ::ifconstexxx::__power(2.451, 7) };       // 531.377521456277
    volatile auto  __nonconstexpr { ::ifconstexxx::__power(8.65472, 11) };  // 20407191608.2931
    const auto     __shouldntwork { ::ifconstexxx::__power(3.2056411, 4) }; // 105.598947702201

    std::wcout << __shouldwork << L' ' << __nonconstexpr << L' ' << __shouldntwork << L'\n';

    return EXIT_SUCCESS;
}
