// clang .\traits.cpp -Wall -Wextra -pedantic -std=c++20 -O3

#include <iostream>
#include <numbers>
#include <type_traits>

// variable template

template<typename T> inline constexpr std::enable_if<std::is_floating_point<T>::value, T>::type pi_value = std::numbers::pi_v<T>;

using std::numbers::pi_v;

struct cstruct {
        float  first;
        int    second;
        double third;
};

// doesn't work as an overload for bool already exists in namespace std
namespace std {
    template<typename T> static std::basic_ostream<T>& operator<<(std::basic_ostream<T>& ostr, const bool predicate) {
        if (predicate)
            return std::operator<<(ostr, T("Yes"));
        else
            return std::operator<<(ostr, T("No"));
    }
} // namespace std

auto main() -> int {
    constexpr auto    pi { ::pi_v<double> };
    constexpr auto    pif { ::pi_value<float> };
    const auto&       refpi { pi };
    static const auto _pi { ::pi_value<decltype(pi)> };
    std::wcout << std::boolalpha;
    std::wcout << L"is pi a const ? " << std::is_const<decltype(pi)>::value << L'\n';
    std::wcout << L"is pi a real number ? " << std::is_floating_point<decltype(pi)>::value << L'\n';
    std::wcout << L"is pi an integer ? " << std::is_integral<decltype(pi)>::value << L'\n';

    std::wcout << L"is refpi a reference ? " << std::is_reference<decltype(refpi)>::value << L'\n';
    std::wcout << L"is refpi a const ? " << std::is_const<decltype(refpi)>::value << L'\n';

    std::wcout << L"is pi trivially constructible ? " << std::is_trivially_constructible<decltype(pi)>::value << L'\n';
    std::wcout << L"is pi trivially destructible ? " << std::is_trivially_destructible<decltype(pi)>::value << L'\n';
    std::wcout << L"is pi trivially copy constructible ? " << std::is_trivially_constructible<decltype(pi)>::value << L'\n';

    std::wcout << L"is _pi a scalar ? " << std::is_scalar<decltype(_pi)>::value << L'\n';
    std::wcout << L"is _pi signed ? " << std::is_signed<decltype(_pi)>::value << L'\n';

    std::wcout << L"is cstruct a scalar ? " << std::is_scalar<cstruct>::value << L'\n';
    std::wcout << L"is cstruct signed ? " << std::is_signed<cstruct>::value << L'\n';
    std::wcout << L"is cstruct trivially constructible ? " << std::is_trivially_constructible<cstruct>::value << L'\n';
    std::wcout << L"is cstruct trivially destructible ? " << std::is_trivially_destructible<cstruct>::value << L'\n';
    std::wcout << L"is cstruct trivially copy constructible ? " << std::is_trivially_constructible<cstruct>::value << L'\n';
    return 0;
}
