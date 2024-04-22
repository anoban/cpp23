// clang .\enableif.cpp -O3 -std=c++20 -Wall -Wextra -pedantic

#include <concepts>
#include <cstdio>
#include <type_traits>

// templated function without any explicit type constraints
// implicit requirement :: operand scalar_t must support the binary + operator
template<typename scalar_t> [[nodiscard]] static constexpr scalar_t gsum(scalar_t x, scalar_t y) noexcept {
    ::_putws(L"template<typename scalar_t> [[nodiscard]] static constexpr scalar_t isum(scalar_t x, scalar_t y) noexcept");
    return x + y;
}

template<typename T> struct is_integral; // declaration of the type trait is_integral

// type trait is_integral for integrals :: char, unsigned char, short, unsigned short, int, unsigned, long, unsigned long, long long, unsigned long long
template<> struct is_integral<char> {
        static const bool value { true };
        typedef char      type;
};

template<> struct is_integral<unsigned char> {
        static const bool     value { true };
        typedef unsigned char type;
};

template<> struct is_integral<short> {
        static const bool value { true };
        typedef short     type;
};

template<> struct is_integral<unsigned short> {
        static const bool      value { true };
        typedef unsigned short type;
};

template<> struct is_integral<int> {
        static const bool value { true };
        typedef int       type;
};

template<> struct is_integral<unsigned> {
        static const bool value { true };
        typedef unsigned  type;
};

template<> struct is_integral<long> {
        static const bool value { true };
        typedef long      type;
};

template<> struct is_integral<unsigned long> {
        static const bool     value { true };
        typedef unsigned long type;
};

template<> struct is_integral<long long> {
        static const bool value { true };
        typedef long long type;
};

template<> struct is_integral<unsigned long long> {
        static const bool          value { true };
        typedef unsigned long long type;
};

// std::enable_if basically takes advantage of template substitution failure to disable the use of certain types as template arguments!

// composite concept combining our own is_integral and std::floating_point
template<typename T> concept is_arithmetic = is_integral<T>::value || std::floating_point<T>;

// constraint leveraging std::enable_if<T>::type member as a template type parameter
template<typename T, typename = std::enable_if<is_integral<T>::value, T>::type>
[[nodiscard]] constexpr T isum(const T& x, const T& y) noexcept { // will only work with integral argument types
    ::_putws(
        L"template<typename T, typename = std::enable_if<is_integral<T>::value>::type>[[nodiscard]] constexpr T isum(const T x, const T& y) noexcept"
    );
    return x + y;
}

// if std::enable_if failed, the above template will be attempted to be instantiated as,
template<typename T, typename = /* no member type in struct std::enable_if */>
[[nodiscard]] constexpr T iisum(const T& x, const T& y) noexcept { // will only work with integral argument types
    ::_putws(
        L"template<typename T, typename = std::enable_if<is_integral<T>::value>::type>[[nodiscard]] constexpr T isum(const T x, const T& y) noexcept"
    );
    return x + y;
}

// using std::enable_if<T>::type as argument types
template<typename T>
[[nodiscard]] static constexpr T imul(
    const T& x, const T& y, typename std::enable_if<is_integral<std::remove_const_t<T>>::value, T>::type = 0
    // all this just to make the argument type -> const T&
) noexcept {
    return x * y;
}

auto main() -> int {
    constexpr float one { 634.8567623 }, two { 6.046654 };
    constexpr short shirt { 54 }, tshirt { 84 };
    ::isum(one, two);
    ::isum(shirt, shirt);

    ::imul(one, one); // called with float arguments
    ::imul(shirt, tshirt);

    return 0;
}

// starting to like C++ :))
