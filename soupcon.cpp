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
// this will require separate specializations for cv qualified variants of void :(
template<> struct add_lvalue_reference<void> {
        using type = void;
};

// we need to implement a specialization that applies to a cv qualifiers stripped version of void
template<typename _Ty> static constexpr bool is_stripped_void_v = std::is_same_v<std::remove_cv_t<_Ty>, void>;

template<typename _TyCandidate, bool = is_stripped_void_v<_TyCandidate>> struct add_lvalue_reference_refined {
        using type = _TyCandidate&;
        // using type = void;
};

template<typename _TyCandidate> struct add_lvalue_reference_refined<_TyCandidate, true> { // partial class template specialization
        using type = void;
};

static_assert(std::is_same_v<void, ::add_lvalue_reference_refined<void>::type>);
static_assert(std::is_same_v<long&, ::add_lvalue_reference_refined<long>::type>);
static_assert(std::is_same_v<volatile double&, ::add_lvalue_reference_refined<volatile double&&>::type>);
static_assert(std::is_same_v<const float&, ::add_lvalue_reference_refined<const float>::type>);

template<typename... _TyList> using any_to_void_type                                                                      = void;

// in c++20 we could accomplish this simply using a requires clause
template<typename _TyCandidate> requires(!std::is_void_v<std::remove_cv_t<_TyCandidate>>) using add_lvalue_reference_t_v2 = _TyCandidate&;

// but this will give a compile time error stating that the requirements are not satisifed

// static_assert(std::is_same_v<void, ::add_lvalue_reference_t_v2<void>>);

// using fold expressions
static consteval double square(_In_ const double& _value) noexcept { return _value * _value; }

template<typename... _TyList> static consteval double sum(_In_ const _TyList&... args) noexcept { return (... + args); }

template<typename... _TyList> static consteval double squaresum(_In_ const _TyList&... args) noexcept { return (... + ::square(args)); }

static_assert(::sum(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10) == 55.00);
static_assert(::squaresum(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10) == 385.00);

template<typename _TyCandidate, typename... _TyList> static consteval bool is_in() noexcept {
    return (std::is_same_v<_TyCandidate, _TyList> || ...);
}

static_assert(!::is_in<double&&, float&, const int, volatile long, double&, const double&&>());
static_assert(::is_in<double&&, float&, const int, volatile long, double&, const double&&, double&&>()); // yeehaw

static_assert(std::is_assignable_v<double&, float>);

template<typename _Ty> static typename std::add_rvalue_reference<_Ty>::type declval() noexcept;

// static_assert(::declval<double>() == 0.00); // won't work because our implementation lacks the function definition
// declval<T>() isn't meant to be used in such situations

// the problem with this implementation is when the _TyLeft is a const reference, the decltype() expression cannot be evaluated at compile time
// we'll get a hard compile time error
template<typename _TyLeft, typename _TyRight, typename _TyResult = decltype(declval<_TyLeft>() = declval<_TyRight>())>
struct is_assignable {
        static constexpr bool value { false };
};

// this leverages thefact that any valid assignment will yield the same type as the left operand
template<typename _TyLeft, typename _TyRight> struct is_assignable<_TyLeft, _TyRight, _TyLeft> {
        static constexpr bool value { true };
};

template<typename _TyLeft, typename _TyRight> static constexpr bool is_assignable_v = ::is_assignable<_TyLeft, _TyRight>::value;

static_assert(::is_assignable_v<double&, float>);
static_assert(::is_assignable_v<double&, const float&>);
static_assert(::is_assignable_v<double&, volatile long long&&>);
static_assert(!::is_assignable_v<
              const unsigned&,
              unsigned>); // ERROR becaue decltype(declval<const unsigned&>() = declval<unsigned>()) assignment is illegal

template<typename _TyLeft, typename _TyRight> struct is_assignable_refined {
        static constexpr bool value {
            (!std::is_const_v<_TyLeft>) &&
            // if _TyLeft is const, short circuiting WILL NOT prevent the second subexpression from being evaluated,
            // WE STILL GET COMPILE TIME ERRORS
            std::is_same_v<decltype(declval<_TyLeft>() = declval<_TyRight>()), _TyLeft>
        };
};

static_assert(::is_assignable_refined<double&, float>::value);
static_assert(::is_assignable_refined<double&, const float&>::value);
static_assert(::is_assignable_refined<double&, volatile long long&&>::value);
static_assert(!::is_assignable_refined<const unsigned&, unsigned>::value);

template<typename _TyLeft, typename _TyRight> static consteval bool is_assignable_func() noexcept {
    return !std::is_const_v<_TyLeft> &&
           // if _TyLeft is const, short circuiting will prevent the second subexpression from being evaluated,
           // so hopefully we won't get compile time errors
           std::is_same_v<decltype(declval<_TyLeft>() = declval<_TyRight>()), _TyLeft>;
};
