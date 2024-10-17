#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstring>

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

namespace recursion {
    [[nodiscard]] static ::sstring&& append(_Inout_ ::sstring&& string, _In_ ::sstring&& annexure, _In_ const unsigned times) noexcept {
        string += annexure;
        if (!times) return std::move(string);
        return append(std::move(string), std::move(annexure), times - 1);
    }
} // namespace recursion

auto wmain() -> int {
    constexpr float pi { M_PI };
    float           eps { M_E };
    const float&&   rref { M_SQRT2 }; // temporary materialization, xvalues

    overloads::func(12.98769458); // calls the rvalue reference overload
    // clang and g++ are okay with this but MSVC overload resolution errs saying this is an ambiguous call

    overloads::func(pi);  // calls the const lvalue reference overload
    overloads::func(eps); // calls the non-const lvalue reference overload
    overloads::func(rref);
    overloads::func(std::move(pi)); // calls the const rvalue reference overload

    return EXIT_SUCCESS;
}
