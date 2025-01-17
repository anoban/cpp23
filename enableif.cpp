// clang .\enableif.cpp -O3 -std=c++20 -Wall -Wextra -pedantic

#include <concepts>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

// templated function without any explicit type constraints
// implicit requirement :: operand scalar_t must support the binary + operator
template<typename scalar_t> [[nodiscard]] static constexpr scalar_t gsum(scalar_t x, scalar_t y) noexcept {
    ::_putws(L"template<typename scalar_t> [[nodiscard]] static constexpr scalar_t isum(scalar_t x, scalar_t y) noexcept");
    return x + y;
}

template<typename _Ty> struct is_integral; // declaration of the type trait is_integral

// type trait is_integral for integrals :: char, unsigned char, short, unsigned short, int, unsigned, long, unsigned long, long long, unsigned long long
template<> struct is_integral<char> {
        static const bool value { true };
        using type = char;
};

template<> struct is_integral<unsigned char> {
        static const bool value { true };
        using type = unsigned char;
};

template<> struct is_integral<short> {
        static const bool value { true };
        using type = short;
};

template<> struct is_integral<unsigned short> {
        static const bool value { true };
        using type = unsigned short;
};

template<> struct is_integral<int> {
        static const bool value { true };
        using type = int;
};

template<> struct is_integral<unsigned> {
        static const bool value { true };
        using type = unsigned int;
};

template<> struct is_integral<long> {
        static const bool value { true };
        using type = long;
};

template<> struct is_integral<unsigned long> {
        static const bool value { true };
        using type = unsigned long;
};

template<> struct is_integral<long long> {
        static const bool value { true };
        using type = long long;
};

template<> struct is_integral<unsigned long long> {
        static const bool value { true };
        using type = unsigned long long;
};

// std::enable_if basically takes advantage of template substitution failure to disable the use of certain types as template arguments!

// composite concept combining our own is_integral and std::floating_point
template<typename _Ty> concept is_arithmetic = is_integral<_Ty>::value || std::floating_point<_Ty>;

// constraint leveraging std::enable_if<T>::type member as a template type parameter
template<typename _Ty, typename = std::enable_if<is_integral<_Ty>::value, _Ty>::type>
[[nodiscard]] constexpr _Ty isum(const _Ty& x, const _Ty& y) noexcept { // will only work with integral argument types
    ::_putws(
        L"template<typename T, typename = std::enable_if<is_integral<T>::value>::type>[[nodiscard]] constexpr T isum(const T x, const T& y) noexcept"
    );
    return x + y;
}

// if std::enable_if failed, the above template will be attempted to be instantiated as,
/* template<typename _Ty, typename = no member type in struct std::enable_if >
[[nodiscard]] constexpr _Ty iisum(const _Ty& x, const _Ty& y) noexcept { // will only work with integral argument types
    ::_putws(
        L"template<typename T, typename = std::enable_if<is_integral<T>::value>::type>[[nodiscard]] constexpr T isum(const T x, const T& y) noexcept"
    );
    return x + y;
}
*/

// using std::enable_if<T>::type as argument types
template<typename _Ty> [[nodiscard]] static constexpr _Ty imul(
    const _Ty& x, const _Ty& y, typename std::enable_if<is_integral<std::remove_const_t<_Ty>>::value, _Ty>::type = 0
    // all this just to make the argument type -> const T&
) noexcept {
    return x * y;
}

// hand rolled enable_if
template<bool is_true, class _Ty> struct enable_if final { };

template<class _Ty> struct enable_if<true, _Ty> final {
        using type = _Ty;
};

template<bool is_true, class _Ty> using enable_if_t = typename ::enable_if<is_true, _Ty>::type;

auto main() -> int {
    constexpr float one { 634.8567623 }, two { 6.046654 };
    constexpr short shirt { 54 }, tshirt { 84 };

    ::isum(one, two); // invoked with float arguments
    ::isum(shirt, shirt);

    ::imul(one, one); // invoked with float arguments
    ::imul(shirt, tshirt);

    constexpr ::enable_if_t<::is_arithmetic<unsigned>, bool> yes { false };
    constexpr ::enable_if_t<!::is_arithmetic<double>, bool>  nope { false }; // error

    return EXIT_SUCCESS;
}

// starting to like C++ :))
