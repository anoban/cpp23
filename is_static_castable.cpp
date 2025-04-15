#include <cstdlib>
#include <type_traits>

template<typename _TyFrom, typename _TyTo, typename _TyResult> struct __is_static_castable final {
        static constexpr bool value { false };
};

template<typename _TyFrom, typename _TyTo>
struct __is_static_castable<_TyFrom, _TyTo, decltype(static_cast<_TyTo>(std::declval<_TyFrom>()))> final {
        static constexpr bool value { true };
};

template<typename _TyFrom, typename _TyTo> struct is_static_castable final {
        static constexpr bool value = __is_static_castable<_TyFrom, _TyTo, _TyTo>::value;
};

template<typename _TyFrom, typename _TyTo> static constexpr bool is_static_castable_v = ::is_static_castable<_TyFrom, _TyTo>::value;

static_assert(::is_static_castable_v<float, double>);
static_assert(::is_static_castable_v<float, const double>);
static_assert(::is_static_castable_v<const float, double>);
static_assert(::is_static_castable_v<float*, const float*>);
static_assert(::is_static_castable_v<const float*, float*>);
static_assert(::is_static_castable_v<float&, const float&>);

template<typename... _TyList> requires(std::is_integral_v<_TyList> && ...)
[[nodiscard]] static constexpr long double isum(const _TyList&... args) noexcept {
    return (args + ...);
}

template<typename _TyReal>
static constexpr typename std::enable_if_t<std::is_floating_point_v<_TyReal>, double> rsum( // NOLINT(modernize-use-constraints)
    const _TyReal& _arg
) noexcept {
    return _arg;
}

template<typename _TyFirst, typename... _TyList>
static constexpr typename std::enable_if_t<std::is_floating_point_v<_TyFirst>, double> rsum( // NOLINT(modernize-use-constraints)
    const _TyFirst& _farg,
    const _TyList&... _rarg
) noexcept {
    return _farg + ::rsum(_rarg...);
}

int main() {
    //
    [[maybe_unused]] auto is_it = static_cast<const double>(21567);
    constexpr auto        sum { ::isum(0, 5746L, 847LLU, 8964U, 'A', L'P', 2.5469) };
    constexpr auto        fsum { ::rsum(0.000, 5.746L, 84.080977, 89640, 2.5469) };
    return EXIT_SUCCESS;
}
