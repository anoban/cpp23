#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifdef __GNUG__
    #define __FUNCSIG__ __PRETTY_FUNCTION__
#endif

namespace overloads {

    static inline float func(float& n) noexcept {
        ::puts(__FUNCSIG__);
        return n;
    }

    static inline float func(const float& n) noexcept {
        ::puts(__FUNCSIG__);
        return n;
    }

    static inline float func(float&& n) noexcept { // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        ::puts(__FUNCSIG__);
        return n;
    }

    static inline float func(const float&& n) noexcept { // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        ::puts(__FUNCSIG__);
        return n;
    }

} // namespace overloads

auto main() -> int {
    constexpr float pi { M_PI };
    float           eps { M_E };
    overloads::func(12.98769458); // calls the rvalue reference overload
    // clang and g++ are okay with this but MSVC errs saying this is an ambiguous call
    overloads::func(pi);  // calls the const lvalue reference overload
    overloads::func(eps); // calls the non-const lvalue reference overload
    return EXIT_SUCCESS;
}
