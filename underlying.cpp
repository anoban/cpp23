// g++ underlying.cpp  -Wall -Wextra -Wpedantic -std=c++23 -O3 -o underlying.exe

#include <climits>
#include <cstdlib>
#include <ios>
#include <iostream>
#include <type_traits>
#include <utility>

using cstyle = enum cstyle { O, P, Q };

enum dummy : unsigned short { A, B };

enum class scoped : long long { X, Y, Z };

auto main() -> int {
    constexpr typename std::underlying_type<dummy>::type x { USHRT_MAX };
    constexpr decltype(dummy::A)                         y { 0xAFA };
    [[maybe_unused]] constexpr decltype(B)               z { 0xAFA };

    constexpr auto sc { typename std::underlying_type<scoped>::type { LONG_LONG_MAX } };

    constexpr auto p { static_cast<decltype(P)>(2) };
    constexpr auto q { static_cast<std::underlying_type<cstyle>::type>(2) };

    std::wcout << std::hex << std::uppercase << std::to_underlying(y) << L' ' << x << std::endl;
    return EXIT_SUCCESS;
}
