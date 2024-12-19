#include <concepts>
#include <type_traits>

template<typename _TyIntegral, _TyIntegral _value> requires std::integral<_TyIntegral> struct integral_constant {
        static constexpr _TyIntegral value { _value };
};

static_assert(std::is_integral<bool>::value); // okay, that's good

template<bool _value> struct bool_constant : public integral_constant<bool, _value> { // deriving from the base class integral_constant
        static constexpr bool value { false };
};

template<> struct bool_constant<true> { // class template complete specialization
        static constexpr bool value { true };
};

static_assert(bool_constant<std::is_signed<long>::value>::value);

// base template
template<typename _TyCandidate> struct is_reference final : public bool_constant<false> { };

// specialization
template<typename _TyCandidate> struct is_reference<_TyCandidate&> final : public bool_constant<true> {
        using type = _TyCandidate;
};

// rvalue references cannot be considered as lvalue references, we'll need a separate specialization for rvalue references
template<typename _TyCandidate> struct is_reference<_TyCandidate&&> final : public bool_constant<true> {
        using type = _TyCandidate;
};

// variable template
template<typename _TyCandidate> static inline constexpr bool is_reference_v = ::is_reference<_TyCandidate>::value;

// alias template
template<typename _TyCandidate> using is_reference_t                        = typename ::is_reference<_TyCandidate>::type;

// this is okay, but with void passed as the candidate type, this will err because void& is not a valid type
template<typename _TyCandidate> struct add_lvalue_reference {
        using type = _TyCandidate&;
};

// to prevent that we could provide a specialization for void, like so
template<> struct add_lvalue_reference<void> {
        using type = void;
};

// this will require separate specializations for cv qualified variants of void :(
template<typename _TyCandidate> struct add_lvalue_reference<typename std::remove_cv<_TyCandidate>::type> {
        using type = void;
};
