// any_of and all_of but as function templates

#include <type_traits>

template<class T, class... TList> struct __cxx_typelist_counter final {
        static constexpr size_t count = 1 + __cxx_typelist_counter<TList...>::count;
};

template<class T> struct __cxx_typelist_counter<T> final {
        static constexpr size_t count = 1;
};

static_assert(__cxx_typelist_counter<wchar_t, long long, volatile short&&, unsigned, const int, long&>::count == 6);
static_assert(__cxx_typelist_counter<float, double, volatile long double&&>::count == 3);

template<class T, class... TList> struct get_first final {
        using type = T;
};

static_assert(std::is_same_v<get_first<wchar_t, long long, volatile short&&, unsigned, const int, long&>::type, wchar_t>);
static_assert(std::is_same_v<get_first<const float&, double, volatile long double&&>::type, const float&>);

namespace sizeof_ellipsis {
    // template<template<class> class predicate, class T> static constexpr bool any_of() noexcept { return predicate<T>::value; }

    template<template<class> class predicate, class T, class... TList> static consteval bool any_of() noexcept {
        if constexpr (!sizeof...(TList)) // when TList is empty
            return predicate<T>::value;
        else
            return predicate<T>::value || sizeof_ellipsis::any_of<predicate, TList...>();
    }

    // template<template<class> class predicate, class T> static constexpr bool all_of() noexcept { return predicate<T>::value; }

    template<template<class> class predicate, class T, class... TList> static consteval bool all_of() noexcept {
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

    template<template<class> class predicate, class T> static consteval bool all_of() noexcept { return predicate<T>::value; }

    template<
        template<class> class predicate,
        class T,
        class... TList,
        // when __cxx_typelist_counter<TList...>::count is illformed, i.e when TList is empty, this overload will be SFINAEd away
        // int to bool is a narrowing conversion which is invalid in templates
        unsigned _sfinae_tlist_size = ::__cxx_typelist_counter<TList...>::count> // see, we did not need std::enable_if
    static consteval bool all_of() noexcept {
        return predicate<T>::value && all_of<predicate, TList...>();
    }

    template<template<class> class predicate, class T> static consteval bool any_of() noexcept { return predicate<T>::value; }

    template<
        template<class> class predicate,
        class T,
        class... TList,
        // when __cxx_typelist_counter<TList...>::count is illformed, i.e when TList is empty, this overload will be SFINAEd away
        // int to bool is a narrowing conversion which is invalid in templates
        unsigned _sfinae_tlist_size = ::__cxx_typelist_counter<TList...>::count> // see, we did not need std::enable_if
    static consteval bool any_of() noexcept {
        return predicate<T>::value || any_of<predicate, TList...>();
    }

} // namespace overloads

static_assert(overloads::all_of<std::is_floating_point, float, const double, long double>());
static_assert(!overloads::all_of<std::is_floating_point, char, float, double, long double>());
static_assert(overloads::all_of<std::is_arithmetic, char, unsigned short, int, float, long, double, long long, long double>());
static_assert(overloads::any_of<std::is_floating_point, float, double, short, unsigned, int>());
static_assert(!overloads::any_of<std::is_floating_point, char, long long, short, unsigned, int, long>());

namespace foldexpressions {

    template<template<class> class predicate, class... TList> requires requires { predicate<typename ::get_first<TList...>::type>::value; }
    static consteval bool all_of() noexcept {
        return (... && predicate<TList>::value);
    }

    template<template<class> class predicate, class... TList> requires requires { predicate<typename ::get_first<TList...>::type>::value; }
    static consteval bool any_of() noexcept {
        return (... || predicate<TList>::value);
    }

} // namespace foldexpressions

static_assert(foldexpressions::all_of<std::is_floating_point, float, const double, long double>());
static_assert(!foldexpressions::all_of<std::is_floating_point, char, float, double, long double>());
static_assert(foldexpressions::all_of<std::is_arithmetic, char, unsigned short, int, float, long, double, long long, long double>());
static_assert(foldexpressions::any_of<std::is_floating_point, float, double, short, unsigned, int>());
static_assert(!foldexpressions::any_of<std::is_floating_point, char, long long, short, unsigned, int, long>());

// yeehawww :)))

// test whether the requires clause is actually working using a hand rolled type trait

namespace type_traits {

    template<class T> struct is_usable_on_x86 final {
            static constexpr bool value { false };
    };

    template<> struct is_usable_on_x86<char> final {
            static constexpr bool value { true };
            using type = char;
    };

    template<> struct is_usable_on_x86<unsigned char> final {
            static constexpr bool value { true };
            using type = unsigned char;
    };

    template<> struct is_usable_on_x86<short> final {
            static constexpr bool value { true };
            using type = short;
    };

    template<> struct is_usable_on_x86<unsigned short> final {
            static constexpr bool value { true };
            using type = unsigned short;
    };

    template<> struct is_usable_on_x86<int> final {
            static constexpr bool value { true };
            using type = int;
    };

    template<> struct is_usable_on_x86<unsigned int> final {
            static constexpr bool value { true };
            using type = unsigned int;
    };

    template<> struct is_usable_on_x86<float> final {
            static constexpr bool value { true };
            using type = float;
    };

} // namespace type_traits

static_assert(!foldexpressions::all_of<type_traits::is_usable_on_x86, float, const double, long double>());
static_assert(!foldexpressions::all_of<type_traits::is_usable_on_x86, char, float, double, long double>());
static_assert(
    !foldexpressions::all_of<type_traits::is_usable_on_x86, char, unsigned short, int, float, long, double, long long, long double>()
);
static_assert(foldexpressions::any_of<type_traits::is_usable_on_x86, float, short, unsigned, int>());
static_assert(foldexpressions::any_of<type_traits::is_usable_on_x86, char, unsigned char, short, unsigned, int>());

template<class T> struct is_leq_4bytes final {
        // instead of naming the predicate `value`, we'll use `qualified`, this should fail the requires clause
        static constexpr bool qualified { (sizeof(T) <= 4LLU) };
};

static_assert(::is_leq_4bytes<float>::qualified);
static_assert(::is_leq_4bytes<const char&>::qualified);
static_assert(!::is_leq_4bytes<volatile double&&>::qualified);
static_assert(::is_leq_4bytes<wchar_t>::qualified);
static_assert(!::is_leq_4bytes<unsigned long long*>::qualified);
static_assert(::is_leq_4bytes<long&&>::qualified);
static_assert(::is_leq_4bytes<char>::qualified);

template<template<typename> class unary_predicate, class T, class... TList> requires requires { unary_predicate<T>::qualified; }
struct all_of_v2 final {
        // static_assert(!unary_predicate<T>::qualified); // providing default arguments is a terrible idea here!
        // becaue the compiler always uses the default `false` instead of using unary_predicate::qualified
        static constexpr bool value { unary_predicate<T>::qualified && all_of_v2<unary_predicate, TList...>::value };
};

template<template<typename> class unary_predicate, class T> requires requires { unary_predicate<T>::qualified; }
struct all_of_v2<unary_predicate, T> final {
        static constexpr bool value { unary_predicate<T>::qualified };
};

template<template<typename> class unary_predicate, class T, class... TList> requires requires { unary_predicate<T>::qualified; }
struct any_of_v2 final {
        static constexpr bool value { unary_predicate<T>::qualified || any_of_v2<unary_predicate, TList...>::value };
};

template<template<typename> class unary_predicate, class T> requires requires { unary_predicate<T>::qualified; }
struct any_of_v2<unary_predicate, T> final {
        static constexpr bool value { unary_predicate<T>::qualified };
};

static_assert(!::all_of_v2<::is_leq_4bytes, float, const double, long double>::value);
static_assert(!::all_of_v2<::is_leq_4bytes, char, float, double, long double>::value);
static_assert(::all_of_v2<::is_leq_4bytes, char, volatile unsigned short, int, float, long, const char, long&&, const unsigned&>::value);
static_assert(::any_of_v2<::is_leq_4bytes, float, short, unsigned, int>::value);
static_assert(::any_of_v2<::is_leq_4bytes, char, unsigned char, short, unsigned, int>::value);
static_assert(!::any_of_v2<::is_leq_4bytes, const char*, unsigned long long&&, const double&, long long* const, volatile float*>::value);

namespace foldexpr {

    template<template<class> class predicate, class... TList>
    requires requires { predicate<typename ::get_first<TList...>::type>::qualified; } static consteval bool all_of() noexcept {
        return (... && predicate<TList>::qualified);
    }

    template<template<class> class predicate, class... TList>
    requires requires { predicate<typename ::get_first<TList...>::type>::qualified; } static consteval bool any_of() noexcept {
        return (... || predicate<TList>::qualified);
    }

} // namespace foldexpr

static_assert(!foldexpr::all_of<::is_leq_4bytes, float, const double, long double>());
static_assert(foldexpr::all_of<::is_leq_4bytes, char, float, short, unsigned char, wchar_t, bool>());
static_assert(!foldexpr::all_of<::is_leq_4bytes, char, unsigned short, int, float, long, double, long long, long double>());
static_assert(foldexpr::any_of<::is_leq_4bytes, float, short, unsigned, int>());
static_assert(foldexpr::any_of<::is_leq_4bytes, char, unsigned char, short, unsigned, int>());
