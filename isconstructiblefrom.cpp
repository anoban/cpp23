#include <string>
#include <type_traits>

namespace nstd {

    template<class _TyFrom, class _TyFrom, class... _TyArgList> struct is_static_castable final {
            static constexpr bool value = false;
    };

    template<class _TyFrom, class... _TyArgList>
    struct is_static_castable<_TyFrom, decltype(_TyFrom(std::declval<_TyArgList>()...)), _TyArgList...> final {
            static constexpr bool value = true;
    };

    template<class _TyFrom, class... _TyArgList> static constexpr bool is_constructible_from_v =
        is_static_castable<_TyFrom, _TyFrom, _TyArgList...>::value;

}

static_assert(std::is_constructible_v<std::string, const char (&)[100]>);
static_assert(nstd::is_constructible_from_v<std::string, const char (&)[100]>); // YEEHAW :)
static_assert(nstd::is_constructible_from_v<std::string, const char* const>);
static_assert(nstd::is_constructible_from_v<std::string, char*>);
static_assert(nstd::is_constructible_from_v<std::string, std::string&&>);
static_assert(nstd::is_constructible_from_v<std::string, const std::string&>);
static_assert(!nstd::is_constructible_from_v<std::string, double&>);
static_assert(!nstd::is_constructible_from_v<std::string, const volatile long&>);
static_assert(!nstd::is_constructible_from_v<std::string, const volatile std::wstring&>);
