#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>

extern constexpr int function(const int&) noexcept;
extern constexpr int function(int&) noexcept;

namespace scope {
    extern constexpr int function(const long long&) noexcept;
} // namespace scope

namespace ambiguity {
    constexpr decltype(auto) func(const float&) throw();
    constexpr decltype(auto) func(float) throw();
} // namespace ambiguity

auto wmain() -> int {
    int            nine { 9 };
    constexpr auto ten { 10 };
    ::function(11); // prvalues can bind to const references
    ::function(ten);
    ::function(nine); // binds to the plain reference overload
    // but in the absence of a non const overload, nine will bind to a const reference overload
    scope::function(nine); // okay, a type promotion also happens here

    constexpr auto pi { M_PI };
    ambiguity::func(pi); // more than one fitting overloads

    int& wont { 122 };
    const int* { &6432 };
    int&& will { 7059 }; // rvalue reference

    return EXIT_SUCCESS;
}
