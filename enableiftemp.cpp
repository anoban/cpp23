// clang .\enableiftemp.cpp -Wall -Wextra -pedantic -O3 -std=c++20 -DUNICODE

#include <array>
#include <concepts>
#include <iostream>

template<bool value, typename T> struct enable_if;

template<typename T> struct enable_if<true, T> {
        using type = T;
        static constexpr bool value { true };
};

template<typename T> struct is_printable;

template<> struct is_printable<char> {
        using type = char;
        static constexpr bool value { true };
};

template<> struct is_printable<wchar_t> {
        using type = wchar_t;
        static constexpr bool value { true };
};

template<
    typename scalar_t,
    typename char_t,
    size_t size,
    typename = enable_if<is_printable<char_t>::value, char_t>::type,                                   // type constraint for char_t
    typename = enable_if<std::is_integral_v<scalar_t> || std::is_floating_point_v<scalar_t>, scalar_t> // type constraint for scalar_t
    >
std::basic_ostream<char_t>& operator<<(std::basic_ostream<char_t>& ostr, const std::array<scalar_t, size>& array) {
    if constexpr (std::is_same_v<char_t, char>) {
        ostr << "[ ";
        for (const auto& e : array) ostr << e << ", ";
        ostr << "\b\b ]";
    } else {
        ostr << L"[ ";
        for (const auto& e : array) ostr << e << L", ";
        ostr << L"\b\b ]";
    }
    return ostr;
}

int main() {
    constexpr std::array<float, 10> arr { 0.8819892324752042,  0.8095204458082461, 0.836852912371145,  0.12479819549608095,
                                          0.30229920227151164, 0.7177711128026765, 0.5224925910732104, 0.40065640840088623,
                                          0.28688727750819054, 0.6019107387850805 };

#if defined(UNICODE)
    std::wcout << L"Using wchar_t s\n";
    std::wcout << arr;
#else
    std::cout << "Using regular chars\n";
    std::cout << arr;
#endif
    return EXIT_SUCCESS;
}
