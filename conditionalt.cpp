#include <type_traits>

namespace nstd {

    template<bool _is_true, typename _TyIfTrue, typename _TyIfFalse> struct conditional final {
            using type = _TyIfFalse;
    };

    template<typename _TyIfTrue, typename _TyIfFalse> struct conditional<true, _TyIfTrue, _TyIfFalse> final {
            using type = _TyIfTrue;
    };

    template<bool is_true, typename _TyIfTrue, typename _TyIfFalse> using conditional_t =
        typename conditional<is_true, _TyIfTrue, _TyIfFalse>::type;
}

static_assert(!std::is_const_v<const long double&>); // WOW

static_assert(std::is_same_v<nstd::conditional_t<std::is_const_v<double&&>, int, volatile float&>, volatile float&>);
static_assert(std::is_same_v<nstd::conditional_t<std::is_const_v<const long double>, int&, volatile float&>, int&>);
static_assert(std::is_same_v<
              nstd::conditional_t<std::is_default_constructible_v<const long double>, volatile long&, volatile float&>,
              volatile long&>);
