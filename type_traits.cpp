#include <cstdlib>
#include <numbers>
#include <string>
#include <type_traits>

// C++ entities can be envisioned as belonging to one of the following spaces
// type space - consists of types int, const float&, volatile std::string and what not
// value space - 65, 8.3846, L"Anoban"
// SFINAE space ???

// SFINAE space is like a high dimensional type space where we leverage the formation quality to develop type abstractions

template<typename _Ty, _Ty _value> requires std::integral<_Ty> struct integral_constant {
        using value_type = _Ty;
        static inline constexpr value_type value { _value };

        [[nodiscard]] constexpr inline explicit operator value_type() const noexcept { return _value; }

        [[nodiscard]] constexpr inline value_type operator()() const noexcept { return value; }
};

template<typename _Ty, _Ty _value> using integral_constant_t                = typename ::integral_constant<_Ty, _value>::value_type;
template<typename _Ty, _Ty _value> constexpr inline _Ty integral_constant_v = ::integral_constant<_Ty, _value>::value;

// template<bool _value> struct bool_constant : ::integral_constant<bool, _value> { };

template<bool _value> using bool_constant                                   = ::integral_constant<bool, _value>;
using true_type                                                             = ::bool_constant<true>;
using false_type                                                            = ::bool_constant<false>;
template<bool _value> inline constexpr bool bool_constant_v                 = ::bool_constant<_value>::value;

static_assert(::bool_constant<std::is_integral_v<decltype(12)>>::value);
static_assert(::bool_constant_v<std::is_floating_point_v<decltype(12.0)>>);

template<typename T> struct is_reference : false_type { // base template
        using type = T;
};

template<typename T> struct is_reference<T&> : true_type { // lvalue references
        using type = T&;
};

template<typename T> struct is_reference<T&&> : true_type { // rvalue references
        using type = T&&;
};

template<typename T> using is_reference_t                 = typename ::is_reference<T>::type;
template<typename T> inline constexpr bool is_reference_v = ::is_reference<T>::value;

template<typename T> struct remove_reference : false_type {
        using type = T;
};

template<typename T> struct remove_reference<T&> : true_type {
        using type = T;
};

template<typename T> struct remove_reference<T&&> : true_type {
        using type = T;
};

template<typename T> struct add_lvalue_reference {
        using type = T&;
};

template<typename T> using add_lvalue_reference_t = typename ::add_lvalue_reference<T>::type;

// circumnavigating the void& problem with partial specializations
template<> struct add_lvalue_reference<void> {
        using type = void;
};

template<> struct add_lvalue_reference<const void> {
        using type = const void;
};

template<> struct add_lvalue_reference<volatile void> {
        using type = volatile void;
};

template<> struct add_lvalue_reference<const volatile void> {
        using type = const volatile void;
};

// the above approach works but is too wordy and inelegant

// an alternative approach to the partial specializations on void
template<typename T, bool is_void = std::is_same_v<void, std::remove_cv_t<std::remove_reference_t<T>>>> struct is_referrable {
        using type           = T;
        using reference_type = T&;
        static constexpr bool value { true };
};

template<typename T> struct is_referrable<T, true> { // partial specialzation for void variants
        using type           = T;
        using reference_type = T;
        static constexpr bool value { false };
};

static_assert(::is_referrable<bool>::value);
static_assert(::is_referrable<const float&>::value);
static_assert(::is_referrable<const volatile int>::value);
static_assert(::is_referrable<const volatile short&&>::value);
static_assert(!::is_referrable<void>::value);
static_assert(::is_referrable<void*>::value);

// an alternate implementation of add_lvalue_reference, but naming it as add_reference because we don't want the oldname_v2 sort of suffixes
template<typename T> struct add_reference {
        using type = typename ::is_referrable<T>::reference_type; // this will return void when a reference to void is required
        // instead of a compile time hard error
        static constexpr bool is_referrable = ::is_referrable<T>::value;
};

// without the presence of partial specializations for add_lvalue_reference or without the is_referrable mediated add_reference
// we will end up with illformed types with void variants as inputs

static_assert(std::is_same_v<add_reference<bool>::type, bool&>);
static_assert(std::is_same_v<add_reference<const float>::type, const float&>);
static_assert(std::is_same_v<add_reference<volatile short&&>::type, volatile short&>);
static_assert(std::is_same_v<add_reference<void>::type, void>);                               // :)
static_assert(std::is_same_v<add_reference<const void>::type, const void>);                   // :)
static_assert(std::is_same_v<add_reference<const volatile void>::type, const volatile void>); // :)

static_assert(std::is_same_v<::std::remove_reference_t<void>, void>);
static_assert(!::remove_reference<void>::value);

extern "C" int wmain() {
    constexpr auto pi { std::numbers::pi_v<float> };
    auto* const    pip { &pi };
    auto&          piref { pi };

    static_assert(!::is_reference_v<decltype(pi)>);
    static_assert(!::is_reference_v<decltype(pip)>);
    static_assert(::is_reference_v<decltype(piref)>);

    static_assert(::is_reference_v<decltype(std::move(pi))>);

    static_assert(::is_reference_v<const float&&>);
    static_assert(::is_reference_v<volatile std::string&>);

    static_assert(std::is_same_v<float, ::remove_reference<float&&>::type>);
    static_assert(std::is_same_v<const float, ::remove_reference<const float&>::type>);
    static_assert(!std::is_same_v<float, ::remove_reference<const float&>::type>);
    static_assert(std::is_same_v<volatile long, ::remove_reference<volatile long>::type>);

    static_assert(std::is_same_v<float&, ::add_lvalue_reference_t<float>>);
    static_assert(std::is_same_v<float&, ::add_lvalue_reference_t<float&>>);
    static_assert(std::is_same_v<float&, ::add_lvalue_reference_t<float&&>>);

    static_assert(std::is_same_v<void, ::add_lvalue_reference_t<void>>); // w/o p spl - error C7683: you cannot create a reference to 'void'
    // okay with p spl

    // typename ::add_lvalue_reference<void>::type cannot_be {};
    //  w/o the partial specialization - error C7683: you cannot create a reference to 'void'
    // with partial specialization -  error C2182: 'cannot_be': this use of 'void' is not valid

    // adding an lvalue reference to void should yield void&
    // but void& is not a valid type in C++, hence the type transformation is ill formed!
    // hence we end up with a hard error at compile time

    // what if we fancy a gentler treatment
    // instead of a hard compile time error, fancy a false_type
    // one solution is to provide partial specializations for void type and its qualifier adorned variants

    return EXIT_SUCCESS;
}
