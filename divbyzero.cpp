// clang .\divbyzero.cpp -Wall -Wextra -pedantic -O3 -std=c++20

#include <iostream>

template<unsigned V> struct tricky {
        // chances for a division by zero when V is 0
        // however this will work just fine as long as the function is executed at runtime
        [[nodiscard]] constexpr unsigned func(unsigned i = 0) noexcept { return i / V; }

        // this will always result in compile time errors if V is 0
        [[nodiscard]] consteval unsigned funcv2(unsigned i = 0) noexcept { return i / V; }

        // this will result in a compile time error without the need to be made consteval or constexpr
        [[nodiscard]] unsigned funcv3() noexcept { return V / V; }
};

template<size_t N> static size_t func() { return N / N; }

auto main() -> int {
    tricky<100> hundred {};
    tricky<0>   zero {};

    hundred.func(); // 0 / 100

    // expression evaluation at runtime, only a warning issued at compile time
    zero.func(100); // 100 / 0  warning: division by zero is undefined
    // to force the evaluation at compile time we'd have to explictly initialize a constexpr variable using this function

    std::wcout << hundred.func(500) << std::endl; // 5
    std::wcout << zero.func(100) << std::endl;    // 0

    constexpr auto x { zero.func(12) }; // since our function is constexpr we could evaluate it at
    // compile time to detect the zero division errors

    // since division by zero is undefined, constexpr result is not technically a constant - is undefined
    // we'll get a hard compile time error for this!

    // call to consteval function "tricky<V>::funcv2 [with V=0U]" did not produce a valid constant expression
    // integer operation result is out of range
    zero.funcv2(); // division by zero :) told ya

    zero.funcv3(); // clang only gives a warning for this
    // MSVC & g++ did not even give a warning

    ::func<0>();
    ::func<0 / 0>(); // hard compile time error: division by zero

    return EXIT_SUCCESS;
}

template<unsigned eval> static unsigned evaluate() noexcept { return eval; }

struct align {
        bool      first;
        int16_t   second;
        double    third;
        char      fourth;
        long long fifth;
};

// compile time constants can be...
static void examples() noexcept {
    evaluate<sizeof(double)>();        // okay
    evaluate<alignof(align::fifth)>(); // looks like a reordering took place
    evaluate<offsetof(align, fourth)>();
    evaluate<19>();
}
