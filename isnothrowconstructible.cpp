#include <sstring>
#include <type_traits>

namespace approach_00 {

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

static_assert(approach_00::is_nothrow_constructible_from_v<::sstring>);
static_assert(approach_00::is_nothrow_constructible_from_v<::sstring, const char (&)[100]>);
static_assert(approach_00::is_nothrow_constructible_from_v<::sstring, std::string>);
static_assert(!approach_00::is_nothrow_constructible_from_v<::may_throw>);
static_assert(approach_00::is_nothrow_constructible_from_v<::may_throw, double>);
static_assert(approach_00::is_nothrow_constructible_from_v<::may_throw, const long&>);
static_assert(!approach_00::is_nothrow_constructible_from_v<::may_throw, const ::sstring>);
static_assert(approach_00::is_nothrow_constructible_from_v<::may_throw, double&&>);

namespace approach_01 {

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

static_assert(approach_01::is_nothrow_constructible_from_v<::sstring>);
static_assert(approach_01::is_nothrow_constructible_from_v<::sstring, const char (&)[100]>);
static_assert(approach_01::is_nothrow_constructible_from_v<::sstring, std::string>);
static_assert(!approach_01::is_nothrow_constructible_from_v<::may_throw>);
static_assert(approach_01::is_nothrow_constructible_from_v<::may_throw, double>);
static_assert(approach_01::is_nothrow_constructible_from_v<::may_throw, const long&>);
static_assert(!approach_01::is_nothrow_constructible_from_v<::may_throw, const ::sstring>);
static_assert(approach_01::is_nothrow_constructible_from_v<::may_throw, double&&>);

#ifdef __llvm__

namespace approach_02 {

    template<typename _TyCandidate, bool _is_construction_noexcept, typename... TyArgList> struct is_nothrow_constructible_from final {
            static constexpr bool value = false;
    };

    // this specialization will only be chosen when _TyCandidate is constructible from the provided types AND the construction in non throwing
    template<typename _TyCandidate, typename... _TyArgList>

    // THE PROBLEM WITH MSVC IS THAT IT EVALUATES THE EXPRESSION noexcept(_TyCandidate(std::declval<_TyArgList>()...)) TO RESULT IN A noexcept
    // SPECIFIER INSTEAD OF A BOOLEAN, THIS LEADS TO A TYPE CONFLICT FOR THE SECOND TEMPLATE PARAMETER
    // AND MSVC ERRS COMPLAINING THAT A NON-TYPE TEMPLATE ARGUMENT HAPPENS TO BE DEPENDENT ON A TYPE ARGUMENT OF THE PARTIAL SPECIALIZATION,
    // WHICH IS NOT ACCEPTABLE
    struct is_nothrow_constructible_from<_TyCandidate, noexcept(_TyCandidate(std::declval<_TyArgList>()...)), _TyArgList...> final {
            static constexpr bool value = true;
    };

    template<typename _TyCandidate, typename... _TyArgList> static constexpr bool is_nothrow_constructible_from_v =
        is_nothrow_constructible_from<_TyCandidate, true, _TyArgList...>::value;

} // namespace approach_02

static_assert(approach_02::is_nothrow_constructible_from_v<::sstring>);
static_assert(approach_02::is_nothrow_constructible_from_v<::sstring, const char (&)[100]>);
static_assert(approach_02::is_nothrow_constructible_from_v<::sstring, std::string>);
static_assert(approach_02::is_nothrow_constructible_from_v<::may_throw, double&&>);
static_assert(approach_02::is_nothrow_constructible_from_v<::may_throw, double>);
static_assert(approach_02::is_nothrow_constructible_from_v<::may_throw, const long&>);
static_assert(!approach_02::is_nothrow_constructible_from_v<::may_throw>);
static_assert(!approach_02::is_nothrow_constructible_from_v<::may_throw, const ::sstring>);

#endif
