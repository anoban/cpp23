#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <type_traits>

template<class T, class = std::enable_if<std::is_integral_v<T>, T>::type> class Integral final {
    private:
        T _value {};

    public:
        using value_type                               = T;

        constexpr Integral()                           = delete; // no default ctor
        constexpr Integral(const Integral&)            = delete; // no copy ctor
        constexpr Integral(Integral&&)                 = delete; // no move ctor
        constexpr Integral& operator=(const Integral&) = delete; // no copy assignment operator
        constexpr Integral& operator=(Integral&&)      = delete; // no move assignment operator

        constexpr explicit Integral(const T& init) noexcept : _value(init) { }
        constexpr ~Integral() = default;

        // conversion operator, to convert to arithmetic types
        template<class U, bool is_unsigned = std::is_unsigned<U>::value> requires std::is_arithmetic_v<U>
        [[nodiscard]] constexpr operator U() const noexcept {
            return static_cast<U>(_value);
        }

        // templated copy ctor
        template<class U> requires std::is_arithmetic_v<U>
        constexpr Integral(const Integral<U>& other) noexcept : _value(static_cast<T>(other._value)) { }
};

// delete the conversion operator of Integral template when the required type is unsigned

template<class T, class = std::enable_if<std::is_floating_point_v<T>, T>::type> class Floating final {
    private:
        T _value {};

    public:
        using value_type                               = T;

        constexpr Floating()                           = delete; // no default ctor
        constexpr Floating(const Floating&)            = delete; // no copy ctor
        constexpr Floating(Floating&&)                 = delete; // no move ctor
        constexpr Floating& operator=(const Floating&) = delete; // no copy assignment operator
        constexpr Floating& operator=(Floating&&)      = delete; // no move assignment operator

        constexpr explicit Floating(const T& init) noexcept : _value(init) { }
        constexpr ~Floating() = default;

        // conversion operator
        template<class U, bool is_unsigned = std::is_unsigned_v<U>> requires std::is_arithmetic_v<U>
        [[nodiscard]] constexpr operator U() const noexcept {
            return static_cast<U>(_value);
        }
};

auto wmain() -> int {
    constexpr ::Integral<short>          five46 { 546 };
    [[maybe_unused]] constexpr long long fivefour6 { five46 }; // implicit invocation of the conversion operator

    constexpr double pi { ::Floating { ::std::numbers::pi_v<float> } };
    return EXIT_SUCCESS;
}

// function templates does not allow partial specializations

template<template<typename> class unary_predicate, typename... TList> struct any_of final {
        static constexpr bool value = (... || unary_predicate<TList>::value);
};

template<typename... TList>
[[nodiscard]] long double consteval sum(
    const bool has_unsigned_types = ::any_of<std::is_unsigned, TList...>::value, const TList&... args
) noexcept {
    return (... + args);
}

static_assert(::sum(10, 11, 12, 13, 14, 15, 16) == 91); // :) cool

// can't do it! function template partial specialization is not allowed
// template<typename... TList> [[nodiscard]] long double consteval sum<TList..., true>(const TList&... args) noexcept { return (... + args); }
