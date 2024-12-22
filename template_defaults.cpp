#include <cstdlib>
#include <type_traits>

template<class _Type> constexpr auto declval() noexcept -> typename std::add_rvalue_reference_t<_Type>; // remember reference
                                                                                                        // collapsing
static_assert(std::is_same_v<decltype(::declval<float>()), float&&>);                                   // T + && => T&&
static_assert(std::is_same_v<decltype(::declval<float&>()), float&>);                                   // T& + && => T&
static_assert(std::is_same_v<decltype(::declval<float&&>()), float&&>);                                 // T&& + && => T&&

static_assert(std::is_same_v<decltype(.54564F), float>);
static_assert(std::is_same_v<decltype((.54564F)), float>);
static float global {};
static_assert(std::is_same_v<decltype(global), float>);
static_assert(std::is_same_v<decltype((global)), float&>);

// A DEFAULT TEMPLATE ARGUMENT IN THE PRIMARY TEMPLATE WILL ALWAYS BE EAGERLY EVALUATED DURING TEMPLATE INSTANTIATION
template<class _TyL, class _TyR, class _TySFINAE = bool> struct is_assignable final {
        // we need _TySFINAE to become void when the evaluated assignment operation is valid,
        // so the compiler will opt for the specialization instead of the base template
        static constexpr bool value = false;
};

template<class _TyL, class _TyR> struct is_assignable<_TyL, _TyR, void /* when _TySFINAE = void */> final {
        static constexpr bool value = true;
};

template<typename _TyLeftOperand, typename _TyRightOperand> static constexpr bool is_assignable_v =
    ::is_assignable<_TyLeftOperand, _TyRightOperand>::value;

static_assert(::);

template<typename _TyFrom, typename _TyStripped = typename std::remove_cv_t<_TyFrom>> struct add_rvalue_reference final {
        using type = _TyFrom&&;
};

template<typename _TyFrom> struct add_rvalue_reference<_TyFrom, void> final {
        using type = void;
};

template<typename _TyFrom> using add_rvalue_reference_t = typename ::add_rvalue_reference<_TyFrom>::type;

static_assert(std::is_same_v<::add_rvalue_reference_t<float&>, float&>);
static_assert(!std::is_same_v<::add_rvalue_reference_t<float&&>, float&>);
static_assert(std::is_same_v<::add_rvalue_reference_t<float&&>, float&&>);
static_assert(std::is_same_v<::add_rvalue_reference_t<float>, float&&>);
static_assert(std::is_same_v<::add_rvalue_reference_t<const long&>, const long&>);
static_assert(std::is_same_v<::add_rvalue_reference_t<const void>, void>);
static_assert(std::is_same_v<::add_rvalue_reference_t<const volatile void>, void>);

template<typename _Ty0, typename _Ty1> struct dummy {
        using type = _Ty0;
};

// A PARTIAL SPECILIZATION MUST BE CONSISTENT WITH ITS PRIMARY TEMPLATE
template<typename _Ty0 = bool /* WE CANNOT OVERRIDE A TEMPLATE DEFAULT DEFINED IN THE PRIMARY TEMPLATE IN A SPECILIZATION*/>
struct dummy<_Ty0, int> {
        using type = _Ty0;
};
