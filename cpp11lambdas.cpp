//  clang .\cpp11lambdas.cpp -Wall -O3 -std=c++23 -Wextra -pedantic

#include <cstdio>
#include <print>
#include <string>

// C++11 lambda syntax is
// [closure captures](arguments) specifiers -> return type { function body };

// the simplest form of lambda
const auto bare_minimum    = [] {}; // argument paranthesis is not mandatory
// unless there are qualifiers like mutable, constexpr, consteval, noexcept involved the argument parenthesis is deemed optional

const auto add_10          = [](int x) { return x + 10; }; // remember the trailing return type is optional
// the above is equivalent to
const auto add_10_explicit = [](int x) -> int { return x + 10; };

// noexcept qualified lambda
const auto increment       = [](int& x) noexcept -> void { x++; };

// constexpr lambdas
const auto decrement       = [](int& x) constexpr noexcept -> void { --x; };
const auto decrement2      = [](int& x) consteval noexcept -> int { return x - 2; };
const auto decrementconst  = [](const int& x) consteval noexcept -> int { return x - 2; };

const auto printer         = [](const auto& o) { std::print("{}", o); }; // std::print requires C++23

static void captures() noexcept {
    // lambdas can capture variables from their environment

    [[maybe_unused]] float local { 0.12345 }; // a local variable confined to the scope of captures()
    const std::wstring     name { L"Anoban" };
    size_t                 age { 78 };

    // the mutable keyword is essential if a lambda is to mutate its capture!
    // however if variables are passed in as arguments, mutable qualifier is not necessary!
    // cannot use const here as age is not const qualified
    auto grow = [&age]() mutable noexcept -> void { age++; };
    grow(); // takes in age as a closure capture

    const auto degrow = [](size_t& x) constexpr noexcept -> void { x--; };
    degrow(age); // takes in age as an argument, no need for mutable here as we do not have any captures
}

// evaluation of lambdas produce a closure object, which is a non-union class type with the captures from its environment
// i.e a functor object
const auto cube = [](unsigned& x) constexpr noexcept -> void { x = x * x * x; };
// the above lambda will first be translated into a class type by the compiler
struct __cpp20$msvcpp$$ucrt$440414_34jhart_constexprvoid$unsigned { // some horrendously mangled unreadable name
        constexpr void operator()(unsigned& arg$x) noexcept { arg$x = arg$x * arg$x * arg$x * arg$x; }
};

int main() {
    auto x { 0.67573F };
    printer(x);

    auto xx { 123 }; // runtime storage
    printer(xx);

    increment(xx);
    printer(xx);

    decrement(xx);
    printer(xx);

    decrement2(xx); // Error: read of non-const variable 'xx' is not allowed in a constant expression
    // Error: attempt to access run-time storage

    constexpr auto y { 89 };
    decrement2(y);     // Error: candidate function not viable: 1st argument ('const int') would lose const qualifier
    decrementconst(y); // okay :)

    printer(xx);

    return 0;
}
