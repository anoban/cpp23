#include <iostream>
#include <type_traits>

template<class T> struct is_std_ostream_compatible final {
        using type = void;
        static constexpr bool value { false };
};

template<> struct is_std_ostream_compatible<char> final {
        using type = char;
        static constexpr bool value { true };
};

template<> struct is_std_ostream_compatible<wchar_t> final {
        using type = wchar_t;
        static constexpr bool value { true };
};

namespace __cxx_trait_helpers { // not using sizeof...()

    template<class... TList> struct __cxx_paramater_pack_counter;

    template<> struct __cxx_paramater_pack_counter<> final { // for empty TLists
            static constexpr size_t value { 0 };
    };

    template<class T, class... TList> struct __cxx_paramater_pack_counter<T, TList...> final {
            static constexpr size_t value { 1 + __cxx_paramater_pack_counter<TList...>::value };
    };

    static_assert(__cxx_paramater_pack_counter<char, float, unsigned short, int, long, const float&, double&&>::value == 7);

} // namespace __cxx_trait_helpers

template<template<class> class _cxx_predicate, class T>
[[nodiscard]] static consteval bool all_of_v() noexcept requires requires { _cxx_predicate<T>::value; } {
    return _cxx_predicate<T>::value;
}

template<
    template<class>
    class _cxx_predicate,
    class T,
    class... TList,
    typename = std::enable_if<__cxx_trait_helpers::__cxx_paramater_pack_counter<TList...>::value != 0, T>::type>
[[nodiscard]] static consteval bool all_of_v() noexcept {
    return _cxx_predicate<T>::value && ::all_of_v<_cxx_predicate, TList...>();
}

static_assert(::all_of_v<::is_std_ostream_compatible, char, wchar_t>());
static_assert(!::all_of_v<::is_std_ostream_compatible, char, wchar_t, unsigned>());

template<class __arg0_type, class __arg1_type, class __rtrn_type> struct multiply final {
        using first_argument_type  = __arg0_type;
        using second_argument_type = __arg1_type;
        using result_type          = __rtrn_type;

        constexpr result_type operator()(const __arg0_type& arg_0, const __arg1_type& arg_1) const noexcept { return arg_0 * arg_1; }
};
