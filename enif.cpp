#include <numbers>
#include <type_traits>

template<bool, typename T> struct enable_if { };

template<typename T> struct enable_if<true, T> { // partial specialization of enable_if
        static const bool value { true };
        using type = T;
};

template<typename T> struct enable_if<false, T> { // partial specialization of enable_if
        static const bool value { false };
};

template<typename T> static consteval ::enable_if<std::is_integral<T>::value, T>::type isum(T a, T b) noexcept { return a + b; }

int main() {
    constexpr double x { 2.0000 }, pi { std::numbers::pi };
    constexpr short  p { 12 }, q { 78 };

    ::isum(p, p);
    ::isum(p, q);
    ::isum(p, x);  // mixed type
    ::isum(pi, x); // non-integral argument types
    return 0;
}
