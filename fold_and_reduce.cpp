#include <concepts>
#include <cstdlib>

template<typename _Ty> class add final {
        static consteval std::enable_if<std::is_arithmetic<_Ty>::value, _Ty>::type operator()(
            const _Ty& _first, const _Ty& _next
        ) noexcept {
            return _first + _next;
        }
};

template<template<typename> class _TyOperator, typename _TyFirst, typename... _TyList> static consteval long double reduce() noexcept { }

auto main() -> int {
    //
    return EXIT_SUCCESS;
}
