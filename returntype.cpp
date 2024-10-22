#include <cstdlib>

template<class _Ty> static inline constexpr _Ty max(_In_ const _Ty& left, _In_ const _Ty& right) noexcept {
    return left < right ? right : left;
}

template<class _TyRet, class _TyL, class _TyR> static inline constexpr _TyRet max(_In_ const _TyL& left, _In_ const _TyR& right) noexcept
    requires requires(_In_ const _TyL& l, _In_ const _TyR& r) { l.operator<(r); } {
    return left < right ? right : left;
}

template<class _Ty> static inline constexpr _Ty vmax(_In_ const _Ty left, _In_ const _Ty right) noexcept {
    return left < right ? right : left;
}

auto wmain() -> int {
    constexpr auto maxx  = ::max<long double>(45, 98.7647624);              // uses the first overload
    constexpr auto max   = ::max<long double, int, double>(45, 98.7647624); // errs when the second overload is explicitly requested
    constexpr auto maxxx = ::vmax(45, 98.7647624);                          // no implicit type conversions during template type deduction
    // not even the trivial ones are allowed
    constexpr auto okay  = ::vmax<double>(45, 98.7647624);

    return EXIT_SUCCESS;
}
