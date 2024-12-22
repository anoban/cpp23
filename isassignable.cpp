#include <type_traits>

namespace nstd {

    template<class _Ty> static std::add_rvalue_reference_t<_Ty> declval() noexcept;

    // AFTER AN ASSIGNMENT OPERATION, THE TYPE OF THE RESULT IS THE TYPE OF THE LEFT OPERAND
    // WE COULD LEVERAGE THIS IN THE SFINAE SPACE

    template<class _TyCandidate, class _TyCandidate, class _TyCandidate> struct is_assignable final {
            static constexpr bool value = false; // primary template
    };

    template<class _TyCandidate, class _TyCandidate>
    struct is_assignable<_TyCandidate, _TyCandidate, decltype(declval<_TyCandidate>() = declval<_TyCandidate>())> final {
            static constexpr bool value = true; // partial specialization leveraging the well formedness of the result type
    };

    // C++ COMPILERS PRIORITIZE A SPECIALIZATION OVER A PRIMARY TEMPLATE WHEN A VALID ONE IS AVAILABLE

    template<class _TyCandidate, class _TyCandidate> static constexpr bool is_assignable_v =
        is_assignable<_TyCandidate, _TyCandidate, _TyCandidate>::value;
}

static_assert(!nstd::is_assignable_v<float, double>);
static_assert(nstd::is_assignable_v<float&, double>);
static_assert(!nstd::is_assignable_v<const float&, double>);
static_assert(!nstd::is_assignable_v<const float&, void>);
static_assert(!nstd::is_assignable_v<const void, double>);
static_assert(nstd::is_assignable_v<float&, long long>);
static_assert(nstd::is_assignable_v<volatile unsigned&, double>);
