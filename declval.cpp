#include <numbers>
#include <string>
#include <type_traits>

// declval is a templated function declaration
// there's no definition since we will not need it.
// this function is only intended to be used in type deduction contexts like decltype() and not in value space operations
template<typename T> [[nodiscard]] constexpr typename std::add_rvalue_reference_t<T> declval() noexcept;
// declval() essentially turns a type into a value (a non materialized one since declval will not return anything because it doesn't have a body)
// it just represents a type to value realization through its signature w/o actually making it happen

// what will happen when we assign two different types? what will be the type of the result given that we did not explicitly specify a type
// e.g the type of the expression std::numbers::pi_v<double> * 2.000L is long double
static_assert(std::is_same_v<decltype(std::numbers::pi_v<double> * 2.000L), long double>); // see!

// since the receiving entity is of type float, the value gets down casted
[[maybe_unused]] constexpr float twopi                       = std::numbers::pi_v<double> * 2.000L;

// this template captures the type of evaluationg the hypothetical assignment expression
template<typename LT, typename RT> using assignment_result_t = decltype(::declval<LT>() = ::declval<RT>());
// LT has to be a modifiable lvalue reference

// static_assert(std::is_same_v<::assignment_result_t<float, double>, float>); //  error C2106: '=': left operand must be l-value
// it has to be a modifiable lvalue reference
// static_assert(std::is_same_v<::assignment_result_t<const float&, double>,float&>); // error C3892: 'declval': you cannot assign to a variable that is const

static_assert(std::is_same_v<::assignment_result_t<float&, double>, float&>);
static_assert(std::is_same_v<::assignment_result_t<float&, double&&>, float&>);
static_assert(std::is_same_v<::assignment_result_t<float&, short&>, float&>);

static_assert(std::is_same_v<::assignment_result_t<short&, short*>, float&>);
// ill formed error C2440: '=': cannot convert from 'short *' to 'short'

template<typename LT, typename RT> using deref_assignment_result_t = decltype(::declval<LT>() = *::declval<RT>());
static_assert(std::is_same_v<::deref_assignment_result_t<float&, short*>, float&>); // that's cool see :)

// instead of this hard errors at compile time we could rework the implementation to handle illformed types more subtly using SFINAE!

template<typename LT, typename RT, bool is_assignment_valid = std::is_same_v<decltype(::declval<LT>() = ::declval<RT>()), LT>>
struct is_assignable {
        // no type aliases
        static constexpr bool value { false };
};

// partial spl for when LT is indeed an lvalue reference & ::declval<LT>() = ::declval<RT>() is well formed
// i.e when ::declval<LT>() = ::declval<RT>() is syntactically and semantically valid
template<typename LT, typename RT> struct is_assignable<LT, RT, true> {
        using left_type  = LT;
        using right_type = RT;
        static constexpr bool value { true };
};

template<typename LT, typename RT> constexpr bool is_assignable_v = ::is_assignable<LT, RT>::value;

template<typename LT, typename RT, bool is_assignable = ::is_assignable_v<LT, RT>> struct assignment_result {
        using result_type = void; // signifies that the assinment is invalid
};

// partial spl when LHT is in fact an lvalue reference and ::declval<LT>() = ::declval<RT>() is well formed
template<typename LT, typename RT> struct assignment_result<LT, RT, true> {
        using result_type = decltype(::declval<LT>() = ::declval<RT>());
};

static_assert(std::is_same_v<typename ::assignment_result<float&, const volatile double>::result_type, float&>);
static_assert(std::is_same_v<
              typename ::assignment_result<const float&, const volatile double>::result_type,
              void>); // const float& cannot be assigned to

static_assert(std::
                  is_same_v<typename ::assignment_result<long&, const double* const>::result_type, void>); // ill formed - invalid type pair
static_assert(std::is_same_v<typename ::assignment_result<long, long>::result_type, void>);   // LHT must be a modifiable lvalue reference
static_assert(std::is_same_v<typename ::assignment_result<long&&, long>::result_type, void>); // LHT must be a modifiable lvalue reference

auto wmain() -> int {
    const auto empty { ::declval<std::wstring>() }; // this will compile but will err at link time!

    // decltype() with expressions do not actually materialize the evaluated results of the expressions
    // decltype() expressions are hypothetically evaluated without realizing the effects of that expression

    int             age       = 27;
    decltype(age++) next_year = 28;
    static_assert(std::is_same_v<int&, decltype(--age)>); // prefixed increment or decrement operators in C++ return an lvalue reference
    // not a value type

    return EXIT_SUCCESS;
}

// instead of using declval we could use something that actually returns an object
template<typename T>
constexpr decltype(auto) materialize() noexcept requires requires {
    T {}; /* T must be default constructible */
} {
    return T {};
}

static_assert(materialize<int>() == 0);
static_assert(materialize<float>() == 0.000000000000);

// or any of the following alternaitive implementations
template<typename T> requires std::is_default_constructible_v<T> consteval T&& realize() noexcept { return T {}; }

template<typename T> consteval typename std::enable_if<std::is_default_constructible<T>::value, T&&>::type realize() noexcept {
    return T {};
}
