#include <sstring>
#include <type_traits>

namespace nstd {

    // primary template that will be used when _TyCandidate is not constructible from the provided types
    // or may throw during construction from the provided types
    template<typename _TyCandidate, typename _TyConstructed, typename... TyArgList> struct is_nothrow_constructible_from final {
            static constexpr bool value = false;
    };

    // specialization that will only be chosen when _TyCandidate is no throw constructible from the provided types
    template<typename _TyCandidate, typename... _TyArgList> struct is_nothrow_constructible_from<
        _TyCandidate,
        typename std::
            enable_if_t<noexcept(_TyCandidate(std::declval<_TyArgList>()...)), decltype(_TyCandidate(std::declval<_TyArgList>()...))>,
        _TyArgList...>
        final {
            static constexpr bool value = true;
    };

    template<typename _TyCandidate, typename... _TyArgList> static constexpr bool is_nothrow_constructible_from_v =
        is_nothrow_constructible_from<_TyCandidate, _TyCandidate, _TyArgList...>::value;

}

struct may_throw final {
        double value;

        may_throw() : value() { } // a potentially throwing ctor

        explicit may_throw(const double& v) noexcept : value(v) { } // a non-throwing ctor
};

static_assert(nstd::is_nothrow_constructible_from_v<::sstring>);
static_assert(nstd::is_nothrow_constructible_from_v<::sstring, const char (&)[100]>);
static_assert(nstd::is_nothrow_constructible_from_v<::sstring, std::string>);
static_assert(!nstd::is_nothrow_constructible_from_v<::may_throw>);
static_assert(nstd::is_nothrow_constructible_from_v<::may_throw, double>);
static_assert(nstd::is_nothrow_constructible_from_v<::may_throw, const long&>);
static_assert(!nstd::is_nothrow_constructible_from_v<::may_throw, const ::sstring>);
static_assert(nstd::is_nothrow_constructible_from_v<::may_throw, double&&>);

namespace sstd {

    // primary template that will be chosen when _TyCandidate is not constructible from the provided types
    template<typename _TyCandidate, typename _TyConstructed, typename... TyArgList> struct is_nothrow_constructible_from final {
            static constexpr bool value = false;
    };

    // specialization that will be chosen when _TyCandidate is constructible from the provided types
    template<typename _TyCandidate, typename... _TyArgList>
    struct is_nothrow_constructible_from<_TyCandidate, decltype(_TyCandidate(std::declval<_TyArgList>()...)), _TyArgList...> final {
            static constexpr bool value = noexcept(_TyCandidate(std::declval<_TyArgList>()...));
    };

    template<typename _TyCandidate, typename... _TyArgList> static constexpr bool is_nothrow_constructible_from_v =
        is_nothrow_constructible_from<_TyCandidate, _TyCandidate, _TyArgList...>::value;

}

static_assert(sstd::is_nothrow_constructible_from_v<::sstring>);
static_assert(sstd::is_nothrow_constructible_from_v<::sstring, const char (&)[100]>);
static_assert(sstd::is_nothrow_constructible_from_v<::sstring, std::string>);
static_assert(!sstd::is_nothrow_constructible_from_v<::may_throw>);
static_assert(sstd::is_nothrow_constructible_from_v<::may_throw, double>);
static_assert(sstd::is_nothrow_constructible_from_v<::may_throw, const long&>);
static_assert(!sstd::is_nothrow_constructible_from_v<::may_throw, const ::sstring>);
static_assert(sstd::is_nothrow_constructible_from_v<::may_throw, double&&>);
