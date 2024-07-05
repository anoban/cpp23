// Item 01 : Understanding template type deduction

#define _USE_MATH_DEFINES
#include <cmath>
#include <string>

using std::wstring;

template<typename T> [[nodiscard]] constexpr double power_v(T base /* by value */, const unsigned exp) noexcept {
    if (!exp || !base) return 1.00;
    double result = base;
    for (unsigned i = 1; i < exp; ++i) result *= base;
    return result;
}

template<typename T> [[nodiscard]] constexpr double power_r(T& base /* by reference */, const unsigned exp) noexcept {
    if (!exp || !base) return 1.00;
    double result = base;
    for (unsigned i = 1; i < exp; ++i) result *= base;
    return result;
}

template<typename T> [[nodiscard]] constexpr double power_cr(const T& base /* by const reference */, const unsigned exp) noexcept {
    if (!exp || !base) return 1.00;
    double result = base;
    for (unsigned i = 1; i < exp; ++i) result *= base;
    return result;
}

template<typename T> [[nodiscard]] constexpr double power_p(T* base /* by pointer */, const unsigned exp) noexcept {
    if (!exp || !base) return 1.00;
    double result = *base;
    for (unsigned i = 1; i < exp; ++i) result *= *base;
    return result;
}

template<typename T> [[nodiscard]] constexpr double power_cp(const T* base /* by pointer to const */, const unsigned exp) noexcept {
    if (!exp || !base) return 1.00;
    double result = *base;
    for (unsigned i = 1; i < exp; ++i) result *= *base;
    return result;
}

template<typename T> [[nodiscard]] constexpr double power_fr(T&& base /* by a forwarding reference */, const unsigned exp) noexcept {
    if (!exp || !base) return 1.00;
    double result = std::forward(base);
    for (unsigned i = 1; i < exp; ++i) result *= base;
    return result;
}

static_assert(::power_cr(4, 2) == 16);
static_assert(::power_cr(3, 4) == 81);
static_assert(::power_cr(2, 5) == 32);
static_assert(::power_cr(2, 7) == 128);
static_assert(::power_cr(2, 10) == 1024);

// let's implement a type capturer
template<typename T> struct _type_capture_v {
        using deduced_type = T;
        explicit _type_capture_v([[maybe_unused]] T _deduce_from) noexcept { }
};

template<typename T> struct _type_capture_ref {
        using deduced_type = T;
        explicit _type_capture_ref([[maybe_unused]] T& _deduce_from) noexcept { }
};

template<typename T> struct _type_capture_cref {
        using deduced_type = T;
        explicit _type_capture_cref([[maybe_unused]] const T& _deduce_from) noexcept { }
};

template<typename T> struct is_reference {
        static constexpr bool value { false };
        using type = T;
};

template<typename T> struct is_reference<T&> {
        static constexpr bool value { true };
        using type = T;
};

template<typename T> struct is_reference<T&&> {
        static constexpr bool value { true };
        using type = T;
};

template<typename T> constexpr bool is_reference_v = ::is_reference<T>::value;

template<typename T> using is_reference_t          = typename ::is_reference<T>::type;

auto wmain() -> int {
    // in type deduction, when we pass an int for base, T is deduced as int, not const int&!
    constexpr double two { 2.000 };

    ::power_cr(two, 5); // constexpr double power_cr<double>(const double &base, unsigned int exp) noexcept
    // the deduced type is just double, the adornments used in the function argument doesn't impact the type of T

    // when T is a reference or pointer
    float              p { M_PI };
    float* const       ppi { &p };
    const float        pi { M_PI };
    const float* const cppi { &pi };
    const float&       piref { pi };

    ::power_v(p, 3);     // T is float
    ::power_v(pi, 3);    // T is float
    ::power_v(piref, 3); // T is float
        // note the disappearance of const and reference qualifiers in the deduced type
        // this happens because having const or reference qualifiers doesn't matter when the argument is passed by value
        // references will yield their values to the function when the argument type is a value type

    ::power_r(p, 3);     // T is float
    ::power_r(pi, 3);    // T is const float
    ::power_r(piref, 3); // T is const float
    // when the argument is passed as reference, there's a need to preserve the constness in order to prevent mutations of const arguments inside the
    // function body
    // here we cannot afford to ignore the const qualifier, hence when the type is T&, constness is preserved

    ::power_cr(p, 3);     // T is float
    ::power_cr(pi, 3);    // T is float
    ::power_cr(piref, 3); // T is float
    // here the argument type is const T&
    // it makes no sense to deduce the type T to be a const qualified type because the function is explicitly forbidden from mutating its argument
    // because of the argument type const T&
    // so the const qualifer is ignored during the type deduction
    // referenceness is ignored as usual.

    ::power_p(ppi, 5);
    ::power_p(cppi, 5);

    ::power_cp(ppi, 3);
    ::power_cp(cppi, 3);

    [[maybe_unused]] decltype(_type_capture_v { piref })::deduced_type    deduct { M_PI_2 };    // just float, NOT const float&
    [[maybe_unused]] decltype(_type_capture_ref { piref })::deduced_type  deduct_ref { M_PI };  // const float NOT const float&
    [[maybe_unused]] decltype(_type_capture_cref { piref })::deduced_type deduct_cref { M_PI }; // float NOT const float&

    wstring name { L"BARRACUDAAAAAA" };
    auto&   nameref { name };

    static_assert(!::is_reference_v<decltype(name)>);
    static_assert(::is_reference_v<decltype(nameref)>);
    static_assert(::is_reference_v<decltype(std::move(name))>);

    ::power_fr(p, 4);
    ::power_fr(pi, 4);
    ::power_fr(piref, 4);
    ::power_fr(4.7657, 4);

    return EXIT_SUCCESS;
}
