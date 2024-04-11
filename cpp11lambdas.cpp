//  clang .\cpp11lambdas.cpp -Wall -O3 -std=c++23 -Wextra -pedantic

#include <cstdio>
#include <print>

// C++11 lambda syntax is
// [closure captures](arguments) specifiers -> return type { function body };

// the simplest form of lambda
const auto bare_minimum    = [] {};                        // argument paranthesis is not mandatory

const auto add_10          = [](int x) { return x + 10; }; // remember the trailing return type is optional
// the above is equivalent to
const auto add_10_explicit = [](int x) -> int { return x + 10; };

// noexcept qualified lambda
const auto increment       = [](int& x) noexcept -> void { x++; };

// constexpr lambdas
const auto decrement       = [](int& x) constexpr noexcept -> void { --x; };
const auto decrement2      = [](int& x) consteval noexcept -> void { x -= 2; };

const auto printer         = [](const auto& o) { std::print("{}", o); }; // std::print requires C++23

int        main() {
    constexpr auto x { 0.67573F };

    printer(x);

    return 0;
}
