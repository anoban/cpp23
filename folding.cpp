#include <concepts>
#include <cstdio>
#include <cstdlib>

template<template<typename _Ty> class _UnaryPredicate, typename... _TyCandidateList> [[nodiscard]] static consteval bool all_of() noexcept {
    return (_UnaryPredicate<_TyCandidateList>::value && ...);
}

static_assert(::all_of<std::is_integral, int, short, unsigned, long, long long, unsigned char, char>());
static_assert(!::all_of<std::is_integral, int, short, unsigned, long, long long, unsigned char, char, float>());
static_assert(::all_of<std::is_floating_point, double, float, long double>());

template<class... _TyList> requires ::all_of<std::is_arithmetic, _TyList...>
static constexpr long double left_fold(const _TyList&... _arguments) noexcept {
    ::puts(__FUNCSIG__); // NOLINT
    return (... + _arguments);
}

template<class... _TyList> requires std::is_arithmetic_v<_TyList...>
static constexpr long double right_fold(const _TyList&... _arguments) noexcept {
    ::puts(__FUNCSIG__); // NOLINT
    return (_arguments + ...);
}

int main() {
    ::left_fold(12, 87, 654.98);
    return EXIT_SUCCESS;
}
