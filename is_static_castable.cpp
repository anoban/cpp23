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

int main() {
    //
    [[maybe_unused]] auto is_it = static_cast<const double>(21567);
    return EXIT_SUCCESS;
}
