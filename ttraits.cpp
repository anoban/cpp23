#include <type_traits>

template<typename _Ty, _Ty _value> struct integral_constant {
        using type = _Ty;
        static inline constexpr type value { _value };

        [[nodiscard]] constexpr inline type operator()() const noexcept { return value; }
};

template<typename _Ty, _Ty _value> using integral_constant_t                = typename ::integral_constant<_Ty, _value>::type;

template<typename _Ty, _Ty _value> constexpr inline _Ty integral_constant_v = ::integral_constant<_Ty, _value>::value;

template<bool _value> struct bool_constant : public ::integral_constant<bool, _value> { };

// or template<bool _value> using bool_constant = ::integral_constant<bool, _value>;

template<bool _value> inline constexpr bool bool_constant_v = bool_constant<_value>::value;

static_assert(::bool_constant<std::is_integral_v<decltype(12)>>::value);
static_assert(::bool_constant_v<std::is_floating_point_v<decltype(12.0)>>);

template<typename T> struct is_reference : public ::bool_constant<false> { // base template
        using type = T;
};

template<typename T> struct is_reference<T&> : public ::bool_constant<true> { // lvalue references
        using type = T&;
};

template<typename T> struct is_reference<T&&> : public ::bool_constant<true> { // rvalue references
        using type = T&&;
};

template<typename T> using is_reference_t                 = typename ::is_reference<T>::type;

template<typename T> inline constexpr bool is_reference_v = ::is_reference<T>::value;
