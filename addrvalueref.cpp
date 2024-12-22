#include <type_traits>

namespace approach_01 { // using a boolean predicate to choose between the primary template and the partial specialization

    template<class _TyFrom, bool _is_candidate_void> struct add_rvalue_reference final {
            using type = _TyFrom&&;
    };

    template<class _TyFrom> struct add_rvalue_reference<_TyFrom, true> final {
            using type = _TyFrom;
    };

    template<class _TyFrom> using add_rvalue_reference_t =
        typename add_rvalue_reference<_TyFrom, std::is_void_v<_TyFrom>>::type;
}

static_assert(std::is_same_v<approach_01::add_rvalue_reference_t<void>, void>);
static_assert(std::is_same_v<approach_01::add_rvalue_reference_t<const void>, const void>);
static_assert(std::is_same_v<approach_01::add_rvalue_reference_t<volatile void>, volatile void>);
static_assert(std::is_same_v<approach_01::add_rvalue_reference_t<const volatile void>, const volatile void>);
static_assert(std::is_same_v<approach_01::add_rvalue_reference_t<long double&>, long double&>);
static_assert(std::is_same_v<approach_01::add_rvalue_reference_t<float&&>, float&&>);
static_assert(std::is_same_v<approach_01::add_rvalue_reference_t<const short&&>, const short&&>);

namespace approach_02 { // using a template type parameter to redirect void variants to the partial specialization

    template<class _TyFrom, class _TyMayBeVoid> struct add_rvalue_reference final {
            using type = _TyFrom&&;
    };

    template<class _TyFrom> struct add_rvalue_reference<_TyFrom, void> final {
            using type = _TyFrom;
    };

    template<class _TyFrom> using add_rvalue_reference_t =
        typename add_rvalue_reference<_TyFrom, std::remove_cv_t<_TyFrom>>::type;
}

static_assert(std::is_same_v<approach_02::add_rvalue_reference_t<void>, void>);
static_assert(std::is_same_v<approach_02::add_rvalue_reference_t<const void>, const void>);
static_assert(std::is_same_v<approach_02::add_rvalue_reference_t<volatile void>, volatile void>);
static_assert(std::is_same_v<approach_02::add_rvalue_reference_t<const volatile void>, const volatile void>);
static_assert(std::is_same_v<approach_02::add_rvalue_reference_t<long double&>, long double&>);
static_assert(std::is_same_v<approach_02::add_rvalue_reference_t<float&&>, float&&>);
static_assert(std::is_same_v<approach_02::add_rvalue_reference_t<const short&&>, const short&&>);

namespace approach_03 { // using a template type paramater as a predicate

    template<class _TyFrom, class _TyMayBeVoid> struct add_rvalue_reference final {
            using type = _TyFrom;
    };

    template<class _TyFrom> struct add_rvalue_reference<_TyFrom, std::remove_reference_t<_TyFrom&&>> final {
            using type = _TyFrom&&;
    };

    template<class _TyFrom> using add_rvalue_reference_t =
        typename add_rvalue_reference<_TyFrom, std::remove_reference_t<_TyFrom>>::type;

}

static_assert(std::is_same_v<approach_03::add_rvalue_reference_t<void>, void>);
static_assert(std::is_same_v<approach_03::add_rvalue_reference_t<const void>, const void>);
static_assert(std::is_same_v<approach_03::add_rvalue_reference_t<volatile void>, volatile void>);
static_assert(std::is_same_v<approach_03::add_rvalue_reference_t<const volatile void>, const volatile void>);
static_assert(std::is_same_v<approach_03::add_rvalue_reference_t<long double&>, long double&>);
static_assert(std::is_same_v<approach_03::add_rvalue_reference_t<float&&>, float&&>);
static_assert(std::is_same_v<approach_03::add_rvalue_reference_t<const short&&>, const short&&>);
