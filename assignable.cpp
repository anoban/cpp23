#include <type_traits>

template<typename _Ty> static constexpr typename std::add_rvalue_reference<_Ty>::type declval() noexcept;

template<typename _TyLeftOperand, typename _TyRightOperand> struct is_valid_assignment final {
        using type = void;
};

template<typename _TyLeftOperand, typename _TyRightOperand>
struct is_valid_assignment<typename decltype(declval<_TyLeftOperand>() = declval<_TyRightOperand>()), _TyRightOperand> final {
        using type = _TyLeftOperand;
};

static_assert(std::is_same_v<::is_valid_assignment<double&, int, void>::type, void>);

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
