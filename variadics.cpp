#include <concepts>
#include <type_traits>

// NOLINTBEGIN(cppcoreguidelines-narrowing-conversions)

// make sure the passed predicate template has a public static const bool memeber named value
template<template<class> class predicate, class candidate> concept validate_predicate =
    std::is_same<class predicate<candidate>::value, bool>::value;

namespace recursive {
    template<template<class> class predicate, class T, class... TList> struct any_of final {
            static constexpr bool value = predicate<T>::value || any_of<predicate, TList...>::value;
    };

    template<template<class> class predicate, class T> struct any_of<predicate, T> final {
            static constexpr bool value = predicate<T>::value;
    };
} // namespace recursive

static_assert(recursive::any_of<std::is_floating_point, float, double, short, unsigned, int>::value);
static_assert(!recursive::any_of<std::is_floating_point, char, long long, short, unsigned, int>::value);

namespace fold_expressions {

    template<template<typename> typename predicate, typename... TList> struct any_of final {
            static constexpr bool value = (predicate<TList>::value || ...);
    };

    template<template<typename> typename predicate, typename... TList> struct all_of final {
            static constexpr bool value = (predicate<TList>::value && ...);
    };

} // namespace fold_expressions

static_assert(fold_expressions::any_of<std::is_floating_point, float, double, short, unsigned, int>::value);
static_assert(!fold_expressions::any_of<std::is_floating_point, char, long long, short, unsigned, int, long>::value);
static_assert(fold_expressions::all_of<std::is_floating_point, float, double, long double>::value);
static_assert(!fold_expressions::all_of<std::is_floating_point, char, float, double, long double>::value);

namespace constexprif { // still recursive though

    template<template<typename> typename predicate, typename T, typename... TList> struct any_of final {
            [[nodiscard]] static constexpr bool operator()() noexcept {
                if constexpr (!sizeof...(TList)) // when TList is empty
                    return predicate<T>::value;
                else
                    return predicate<T>::value || any_of<predicate, TList...>::operator()();
            }
    };

    template<template<typename> typename predicate, typename T, typename... TList> struct all_of final {
            [[nodiscard]] static constexpr bool operator()() noexcept {
                if constexpr (!sizeof...(TList)) // when TList is empty
                    return predicate<T>::value;
                else
                    return predicate<T>::value && all_of<predicate, TList...>::operator()();
            }
    };

} // namespace constexprif

static_assert(constexprif::any_of<std::is_floating_point, float, double, short, unsigned, int>::operator()());
static_assert(!constexprif::any_of<std::is_floating_point, char, long long, short, unsigned, int, long>::operator()());
static_assert(constexprif::all_of<std::is_floating_point, float, double, long double>::operator()());
static_assert(!constexprif::all_of<std::is_floating_point, char, float, double, long double>::operator()());

// using recursion and specializations
template<class T> [[nodiscard]] static constexpr long double sum(const T& tail) noexcept { return tail; }

template<class T, class... TList> [[nodiscard]] static constexpr long double sum(const T& head, const TList&... rest) noexcept {
    return head + ::sum(rest...);
}

template<class T> [[nodiscard]] static constexpr long double mul(const T& tail) noexcept { return tail; }

template<class T, class... TList> [[nodiscard]] static constexpr long double mul(const T& head, const TList&... rest) noexcept {
    return head * ::mul(rest...);
}

template<std::floating_point T>
[[nodiscard]] static consteval bool is_close(const T& arg_0, const T& arg_1, const long double epsilon = 1.0E5) noexcept {
    // incomplete
    return arg_0 - arg_1;
}

static_assert(
    ::sum(5U, 92.0F, 57, 28L, 73, 62.0L, 'D', 17, 52, 68LL, 12, 51.0, 40, 9, 93Ui8, 88, 22, 'S', 13, 62Ui16, 3, 42, 53LLU, 21, 69) ==
    1183.0L
);
static_assert(
    ::mul(5U, 92.0F, 57, 28L, 73, 62.0L, 'D', 17, 52, 68LL, 12, 51.0, 40, 9, 93Ui8, 88, 22, 'S', 13, 62Ui16, 3, 42, 53LLU, 21, 69) ==
    3.4877513665571724E+38
);

// using sizeof... to switch branches
template<class T, class... TList> [[nodiscard]] static constexpr long double ssum(const T& head, const TList&... tail) noexcept {
    if constexpr (!sizeof...(TList))
        return head;
    else
        return head + ::ssum(tail...);
}

template<class T, class... TList> [[nodiscard]] static constexpr long double smul(const T& head, const TList&... tail) noexcept {
    if constexpr (!sizeof...(TList))
        return head;
    else
        return head * ::smul(tail...);
}

static_assert(
    ::ssum(5U, 92.0F, 57, 28L, 73, 62.0L, 'D', 17, 52, 68LL, 12, 51.0, 40, 9, 93Ui8, 88, 22, 'S', 13, 62Ui16, 3, 42, 53LLU, 21, 69) ==
    1183.0L
);
static_assert(
    ::smul(5U, 92.0F, 57, 28L, 73, 62.0L, 'D', 17, 52, 68LL, 12, 51.0, 40, 9, 93Ui8, 88, 22, 'S', 13, 62Ui16, 3, 42, 53LLU, 21, 69) ==
    3.4877513665571724E+38
);

// 2, 3, 7, 5, 4, 2, 5, 4, 4, 8, 2, 0, 9, 2, 4, 5, 6, 6, 5, 7, 7, 4, 9, 2, 8 sum of factorials is 823577

template<class T>
[[nodiscard]] static consteval typename std::enable_if<std::is_arithmetic<T>::value, long double>::type factorial(const T& val) noexcept {
    if (!val || val == 1) return 1;
    long double result = val; // braced initializer { } results in narrowing errors
    for (unsigned long long i = 1; i < val; ++i) result *= i;
    return result;
}

static_assert(::factorial(10) == 3628800);
static_assert(::factorial(7) == 5040);
static_assert(::factorial(0) == 1);
static_assert(::factorial(1) == 1);
static_assert(::factorial(4) == 24);

// sum the factorials
template<class... TList> static consteval long double fsum(const TList&... values) noexcept { return (... + ::factorial(values)); }

static_assert(::fsum(2.0, 3, 7.0F, 5U, 4.0L, 2, 5LLU, 4L, 4, 8LU, 2, 0, 9, 2, 4Ui8, 5, 6I16, 6, 5, 7LL, 7, 4Ui16, 9, 2, 8) == 823577);

// NOLINTEND(cppcoreguidelines-narrowing-conversions)
