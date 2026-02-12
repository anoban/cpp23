#include <concepts>
#include <cstdio>
#include <cstdlib>

template<template<typename _Ty> class _UnaryPredicate, typename... _TyCandidateList> [[nodiscard]] static consteval bool all_of() noexcept {
    return (_UnaryPredicate<_TyCandidateList>::value && ...);
}

template<template<typename _Ty> class _UnaryPredicate, typename _TyFirst> [[nodiscard]] static consteval bool any_of() noexcept {
    return _UnaryPredicate<_TyFirst>::value;
}

template<template<typename _Ty> class _UnaryPredicate, typename _TyFirst, typename... _TyList>
[[nodiscard]] static consteval typename std::enable_if<sizeof...(_TyList) != 0, bool>::type any_of( // NOLINT(modernize-use-constraints)
) noexcept {
    return _UnaryPredicate<_TyFirst>::value || ::any_of<_UnaryPredicate, _TyList...>();
}

static_assert(::all_of<std::is_integral, int, short, unsigned, long, long long, unsigned char, char>());
static_assert(!::all_of<std::is_integral, int, short, unsigned, long, long long, unsigned char, char, float>());
static_assert(::all_of<std::is_floating_point, double, float, long double>());

// REMEMBER, FOLD EXPRESSIONS ARE NOT RECURSIVE

template<class... _TyList> requires(::all_of<std::is_arithmetic, _TyList...>())
static constexpr long double left_fold(const _TyList&... _arguments) noexcept {
    return (... + _arguments);
}

template<class... _TyList> requires(::all_of<std::is_arithmetic, _TyList...>())
static constexpr long double right_fold(const _TyList&... _arguments) noexcept {
    return (_arguments + ...); // NOLINT(cppcoreguidelines-narrowing-conversions)
}

int main() {
    [[maybe_unused]] constexpr auto left  = ::left_fold(12, 87, 654.98, 0.7467356F, 'A', L'Q');
    constexpr auto                  right = ::right_fold(12, 87, 654.98, 0.7467356F, 'A', L'Q');

    ::printf( // NOLINT(cppcoreguidelines-pro-type-vararg)
        "Sums are %.5Lf, %.5Lf\n",
        ::left_fold(12, 87, 654.98, 0.7467356F, 'A', L'Q'),
        right
    );
    return EXIT_SUCCESS;
}

static_assert(::any_of<std::is_integral, int, short, unsigned, long, long long, unsigned char, char>());
static_assert(::any_of<std::is_floating_point, int, short, unsigned, long, long long, unsigned char, char, float>());
static_assert(::any_of<std::is_floating_point, double, float, long double>());
