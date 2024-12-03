#include <type_traits>

template<typename _Ty> static constexpr typename std::add_rvalue_reference<_Ty>::type            declval() noexcept;

template<typename _TyLeftOperand, typename _TyRightOperand, typename> struct is_valid_assignment final {
        static constexpr bool value = false;
};

template<typename _TyLeftOperand, typename _TyRightOperand>
struct is_valid_assignment<_TyLeftOperand, _TyRightOperand, decltype(declval<_TyLeftOperand>() = declval<_TyRightOperand>())> final {
        static constexpr bool value = true;
};

static_assert(::is_valid_assignment<double&, int>::value);
static_assert(::is_valid_assignment<float&, const long>::value);
static_assert(::is_valid_assignment<long long&, const float&&>::value);
static_assert(::is_valid_assignment<const double, unsigned&&>::value);
static_assert(::is_valid_assignment<double&, int>::value);
static_assert(::is_valid_assignment<double&, int>::value);
static_assert(::is_valid_assignment<double&, int>::value);
static_assert(::is_valid_assignment<double&, int>::value);

template<typename _TyLeftOperand, typename _TyRightOperand, typename = ::is_valid_assignment<_TyLeftOperand, _TyRightOperand, void>::type>
struct is_assignable;

template<typename _TyLeftOperand, typename _TyRightOperand> struct is_assignable<_TyLeftOperand, _TyRightOperand, void> final {
        static constexpr bool value = false;
};

template<typename _TyLeftOperand, typename _TyRightOperand> struct is_assignable<_TyLeftOperand, _TyRightOperand, _TyLeftOperand> final {
        static constexpr bool value = true;
        using result_type           = _TyLeftOperand;
};

static_assert(::is_assignable<double&, float>::value);
