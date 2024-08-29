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

namespace overloads {

    template<class T, class... TList> struct __cxx_typelist_counter final {
            static const size_t count = 1 + __cxx_typelist_counter<TList...>::count;
    };

    template<class T> struct __cxx_typelist_counter<T> final {
            static const size_t count = 1;
    };

    static_assert(__cxx_typelist_counter<wchar_t, long long, volatile short&&, unsigned, const int, long&>::count == 6);
    static_assert(__cxx_typelist_counter<float, double, volatile long double&&>::count == 3);

    template<template<class> class predicate, class T> static constexpr bool all_of() noexcept { return predicate<T>::value; }

    template<template<class> class predicate, class T, class... TList>
    // int to bool narrowing conversion is invalid in templates
    static constexpr std::enable_if_t<
        __cxx_typelist_counter<TList...>::count != 0
        /* when __cxx_typelist_counter<TList...>::count is illformed, i.e when TList is empty, this overload will be SFINAEd away */,
        bool>
    all_of() noexcept {
        return predicate<T>::value && all_of<predicate, TList...>();
    }
} // namespace overloads

static_assert(overloads::all_of<std::is_floating_point, float, const double, long double>());
static_assert(!overloads::all_of<std::is_floating_point, char, float, double, long double>());
static_assert(overloads::all_of<std::is_arithmetic, char, unsigned short, int, float, long, double, long long, long double>());
