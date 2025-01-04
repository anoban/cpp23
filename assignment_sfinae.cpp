#include <cstdlib>
#include <type_traits>

// for an assignment to be valid, the types of right and left operands must be compatible
// left operand must not be const
// and the result of an assignment operation is always the type of the left operand

template<class _TyLeft, class _TyRight, class _TyResult> struct __is_assignment_valid final {
        static constexpr bool value = false; // fallback
};

template<class _TyLeft, class _TyRight>
struct __is_assignment_valid<_TyLeft, _TyRight, decltype(std::declval<_TyLeft>() = std::declval<_TyRight>())> final {
        static constexpr bool value = true; // for all the valid assignments
};

template<class _TyLeft, class _TyRight> static constexpr bool is_assignment_valid =
    ::__is_assignment_valid<_TyLeft, _TyRight, _TyLeft>::value;

static_assert(::is_assignment_valid<double&, long long>);
static_assert(::is_assignment_valid<double&, const long&&>);
static_assert(::is_assignment_valid<double&, const double&>);
static_assert(!::is_assignment_valid<const double, const long&&>);
static_assert(!::is_assignment_valid<const double&, volatile short>);
static_assert(::is_assignment_valid<volatile unsigned&, const long double&&>);
static_assert(!::is_assignment_valid<volatile unsigned* const, volatile unsigned* const>);
static_assert(::is_assignment_valid<const unsigned*&, const unsigned*>);

int main() {
    [[maybe_unused]] constexpr unsigned value { 785776 }, dummy { 7676 };
    auto*                               ptr { &value };
    ptr = &dummy;

    auto&&         rvref { 657.0354 };
    constexpr auto val { 0.767826 };
    rvref = val;

    [[maybe_unused]] constexpr auto isit =
        std::is_same_v<decltype(std::declval<const double*&>() = std::declval<double*>()), const double*&>;
    [[maybe_unused]] constexpr auto _isit = std::is_same_v<decltype(std::declval<double&>() = std::declval<const double&>()), double&>;
    return EXIT_SUCCESS;
}
