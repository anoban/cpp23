#include <cstdlib>
#include <type_traits>

template<bool predicate, class T> struct enablde_if final { };

template<class T> struct enablde_if<true, T> final {
        static constexpr bool value { true };
        using type = T;
};

// alias template
template<bool value, class T> using enable_if_t                 = typename ::enablde_if<value, T>::type;

// alias templates DO NOT allow partial or explicit specializations!
// template<template<class> class conditional, class T> using enable_if_t = typename ::enablde_if<conditional<T>::value, T>::type;

// variable template
template<bool value, class T> static constexpr bool enable_if_v = ::enablde_if<value, T>::value;

auto wmain() -> int {
    [[maybe_unused]] constexpr ::enable_if_t<true, float> valid {};
    return EXIT_SUCCESS;
}
