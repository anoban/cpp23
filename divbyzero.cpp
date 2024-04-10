// clang .\divbyzero.cpp -Wall -Wextra -pedantic -O3 -std=c++20

#include <iostream>

template<unsigned value> struct tricky {
        // chances for a division by zero when value is 0
        // however this will work just fine as long as the function is executed at runtime
        [[nodiscard]] constexpr unsigned func(unsigned i = 0) noexcept { return i / value; }

        // this will always result in compile time errors if value is 0
        [[nodiscard]] consteval unsigned funcv2(unsigned i = 0) noexcept { return i / value; }
};

auto main() -> int {
    tricky<100> hundred {};
    tricky<0>   zero {};

    hundred.func(); // 0 / 100

    // expression evaluation at runtime, only a warning issued at compile time
    zero.func(100); // 100 / 0  warning: division by zero is undefined
    // to force the evaluation at compile time we'd have to explictly initialize a constexpr variable using this function

    std::wcout << hundred.func(500) << std::endl; // 5
    std::wcout << zero.func(100) << std::endl;    // 0

    constexpr auto x { zero.func(12) };           // since our function is constexpr we could evaluate it at
    // compile time to detect the zero division errors

    // since division by zero is undefined, constexpr result is not technically a constant - is undefined
    // we'll get a hard compile time error for this!

    zero.funcv2(); // division by zero :) told ya

    return EXIT_SUCCESS;
}
