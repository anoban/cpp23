#include <type_traits>

namespace nstd {

    template<class _Ty> static std::add_rvalue_reference_t<_Ty> declval() noexcept;

    // AFTER AN ASSIGNMENT OPERATION, THE TYPE OF THE RESULT IS THE TYPE OF THE LEFT OPERAND
    // WE COULD LEVERAGE THIS IN THE SFINAE SPACE

    template<class _TyTo, class _TyFrom, class _TyResult> struct is_assignable final {
            static constexpr bool value = false; // primary template
    };

    template<class _TyTo, class _TyFrom> struct is_assignable<_TyTo, _TyFrom, decltype(declval<_TyTo>() = declval<_TyFrom>())> final {
            static constexpr bool value = true; // partial specialization leveraging the well formedness of the result type
    };

    // C++ COMPILERS PRIORITIZE A SPECIALIZATION OVER A PRIMARY TEMPLATE WHEN A VALID ONE IS AVAILABLE

    template<class _TyTo, class _TyFrom> static constexpr bool is_assignable_v = is_assignable<_TyTo, _TyFrom, _TyTo>::value;
}

static_assert(!nstd::is_assignable_v<float, double>);
static_assert(nstd::is_assignable_v<float&, double>);
static_assert(!nstd::is_assignable_v<const float&, double>);
static_assert(!nstd::is_assignable_v<const float&, void>);
static_assert(!nstd::is_assignable_v<const void, double>);
static_assert(nstd::is_assignable_v<float&, long long>);
static_assert(nstd::is_assignable_v<volatile unsigned&, double>);
