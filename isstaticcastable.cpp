#include <string>
#include <type_traits>

namespace nstd {

    template<class _TyTo, class _TyFrom, class _TyResult> struct is_static_castable final {
            static constexpr bool value = false;
    };

    template<class _TyTo, class _TyFrom>
    struct is_static_castable<_TyTo, _TyFrom, decltype(static_cast<_TyTo>(std::declval<_TyFrom>()))> final {
            static constexpr bool value = true;
    };

    template<class _TyTo, class _TyFrom> static constexpr bool is_static_castable_v = is_static_castable<_TyTo, _TyFrom, _TyTo>::value;

}

static_assert(nstd::is_static_castable_v<unsigned, const char>);
static_assert(nstd::is_static_castable_v<double, char>);
static_assert(nstd::is_static_castable_v<float, double&>);
static_assert(nstd::is_static_castable_v<const float&, double&&>);
static_assert(!nstd::is_static_castable_v<std::string&, const std::string&>);
static_assert(!nstd::is_static_castable_v<std::string, double&>);
static_assert(!nstd::is_static_castable_v<std::string, const volatile long&>);
static_assert(!nstd::is_static_castable_v<std::wstring&, const volatile std::wstring&>);
