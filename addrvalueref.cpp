#include <type_traits>

namespace approach_01 { // using a boolean predicate to choose between the primary template and the partial specialization

    template<class _TyCandidate, bool _is_candidate_void> struct add_rvalue_reference final {
            using type = _TyCandidate&&;
    };

    template<class _TyCandidate> struct add_rvalue_reference<_TyCandidate, true> final {
            using type = _TyCandidate;
    };

    template<class _TyCandidate> using add_rvalue_reference_t =
        typename add_rvalue_reference<_TyCandidate, std::is_void_v<_TyCandidate>>::type;
}

static_assert(std::is_same_v<approach_01::add_rvalue_reference_t<void>, void>);
static_assert(std::is_same_v<approach_01::add_rvalue_reference_t<const void>, const void>);
static_assert(std::is_same_v<approach_01::add_rvalue_reference_t<volatile void>, volatile void>);
static_assert(std::is_same_v<approach_01::add_rvalue_reference_t<const volatile void>, const volatile void>);
static_assert(std::is_same_v<approach_01::add_rvalue_reference_t<long double&>, long double&>);
static_assert(std::is_same_v<approach_01::add_rvalue_reference_t<float&&>, float&&>);
static_assert(std::is_same_v<approach_01::add_rvalue_reference_t<const short&&>, const short&&>);

namespace approach_02 {

    template<class _TyCandidate, class _TyMayBeVoid> struct add_rvalue_reference final {
            using type = _TyCandidate&&;
    };

    template<class _TyCandidate> struct add_rvalue_reference<_TyCandidate, void> final {
            using type = _TyCandidate;
    };

    template<class _TyCandidate> using add_rvalue_reference_t =
        typename add_rvalue_reference<_TyCandidate, std::remove_cv_t<_TyCandidate>>::type;
}

static_assert(std::is_same_v<approach_02::add_rvalue_reference_t<void>, void>);
static_assert(std::is_same_v<approach_02::add_rvalue_reference_t<const void>, const void>);
static_assert(std::is_same_v<approach_02::add_rvalue_reference_t<volatile void>, volatile void>);
static_assert(std::is_same_v<approach_02::add_rvalue_reference_t<const volatile void>, const volatile void>);
static_assert(std::is_same_v<approach_02::add_rvalue_reference_t<long double&>, long double&>);
static_assert(std::is_same_v<approach_02::add_rvalue_reference_t<float&&>, float&&>);
static_assert(std::is_same_v<approach_02::add_rvalue_reference_t<const short&&>, const short&&>);

namespace approach_03 {

    template<class _TyCandidate, class _TyMayBeVoid> struct add_rvalue_reference final {
            using type = _TyCandidate;
    };

    template<class _TyCandidate> struct add_rvalue_reference<_TyCandidate, std::remove_reference_t<_TyCandidate&&>> final {
            using type = _TyCandidate&&;
    };

    template<class _TyCandidate> using add_rvalue_reference_t =
        typename add_rvalue_reference<_TyCandidate, std::remove_reference_t<_TyCandidate>>::type;
}

static_assert(std::is_same_v<approach_03::add_rvalue_reference_t<void>, void>);
static_assert(std::is_same_v<approach_03::add_rvalue_reference_t<const void>, const void>);
static_assert(std::is_same_v<approach_03::add_rvalue_reference_t<volatile void>, volatile void>);
static_assert(std::is_same_v<approach_03::add_rvalue_reference_t<const volatile void>, const volatile void>);
static_assert(std::is_same_v<approach_03::add_rvalue_reference_t<long double&>, long double&>);
static_assert(std::is_same_v<approach_03::add_rvalue_reference_t<float&&>, float&&>);
static_assert(std::is_same_v<approach_03::add_rvalue_reference_t<const short&&>, const short&&>);
