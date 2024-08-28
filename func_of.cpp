// any_of and all_of but as function templates

#include <type_traits>

namespace sizeof_ellipsis {
    // template<template<class> class predicate, class T> static constexpr bool any_of() noexcept { return predicate<T>::value; }

    template<template<class> class predicate, class T, class... TList> static constexpr bool any_of() noexcept {
        if constexpr (!sizeof...(TList)) // when TList is empty
            return predicate<T>::value;
        else
            return predicate<T>::value || sizeof_ellipsis::any_of<predicate, TList...>();
    }

    // template<template<class> class predicate, class T> static constexpr bool all_of() noexcept { return predicate<T>::value; }

    template<template<class> class predicate, class T, class... TList> static constexpr bool all_of() noexcept {
        if constexpr (!sizeof...(TList)) // when TList is empty
            return predicate<T>::value;
        else
            return predicate<T>::value && sizeof_ellipsis::any_of<predicate, TList...>();
    }
} // namespace sizeof_ellipsis

static_assert(sizeof_ellipsis::any_of<std::is_floating_point, float, double, short, unsigned, int>());
static_assert(!sizeof_ellipsis::any_of<std::is_floating_point, char, long long, short, unsigned, int, long>());
static_assert(sizeof_ellipsis::all_of<std::is_floating_point, float, double, long double>());
static_assert(!sizeof_ellipsis::all_of<std::is_floating_point, char, float, double, long double>());
