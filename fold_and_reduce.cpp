#include <concepts>
#include <cstdlib>

template<typename _Ty0, typename _Ty1> class add final {
    public:
        static consteval std::enable_if<std::is_arithmetic<long double>::value, _Ty0>::type operator()(
            const _Ty0& _first, const _Ty1& _next
        ) noexcept {
            return _first + _next;
        }
};

template<typename _Ty0, typename _Ty1> class prod final {
    public:
        static consteval std::enable_if<std::is_arithmetic<long double>::value, _Ty0>::type operator()(
            const _Ty0& _first, const _Ty1& _next
        ) noexcept {
            return _first * _next;
        }
};

template<template<typename, typename> class _TyOperator, typename _TyFirst, typename... _TyList> requires()
static consteval long double reduce(const _TyFirst& _arg, const _TyList&... _arglist) noexcept(noexcept(_TyOperator<typename>)) { }

auto main() -> int {
    //
    return EXIT_SUCCESS;
}
