#include <cstdlib>
#include <type_traits>

template<bool predicate, class T> struct enablde_if final { };

template<class T> struct enablde_if<true, T> final {
        static constexpr bool value { true };
        using type = T;
};

// alias template
template<bool value, class T> using enable_if_t                 = typename /* cannot use class here */ ::enablde_if<value, T>::type;

// alias templates DO NOT allow partial or explicit specializations!
// template<template<class> class conditional, class T> using enable_if_t = typename ::enablde_if<conditional<T>::value, T>::type;

// variable template
template<bool value, class T> static constexpr bool enable_if_v = ::enablde_if<value, T>::value;

template<template<typename> typename predicate, typename T, bool _is_true = predicate<T>::value> struct predicate_if final { };

template<template<typename> typename predicate, typename T> struct predicate_if<predicate, T, true> final {
        using type = T;
};

auto wmain() -> int {
    [[maybe_unused]] constexpr ::enable_if_t<true, float>                      valid {};
    [[maybe_unused]] constexpr ::predicate_if<std::is_arithmetic, float>::type okay {};
    return EXIT_SUCCESS;
}
