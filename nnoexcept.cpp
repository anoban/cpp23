#include <cassert>
#include <string>

[[maybe_unused]] static constexpr bool yes = noexcept(true);

template<class _Ty> [[maybe_unused]] static consteval bool is_default_construction_noexcept() noexcept { return noexcept(_Ty {}); }

static_assert(::is_default_construction_noexcept<float>());
static_assert(::is_default_construction_noexcept<std::string>());

struct foo {
        foo() noexcept(false) { }
};

static_assert(!::is_default_construction_noexcept<::foo>());

[[maybe_unused]] static constexpr bool nope          = noexcept(foo());
[[maybe_unused]] static constexpr bool okay          = noexcept(971 + 10);
[[maybe_unused]] static constexpr bool not_evaluated = noexcept(1 / 0);
// true, because operator/(int, int) is implicitly noexcept
// if evaluated, this will be a compile time error

static constexpr auto error { 12 / 0.00 };

// may throw
constexpr bool faalse { noexcept(std::declval<std::wstring>().replace(0, 9, L"..")) };
constexpr bool ffalse { noexcept(std::declval<std::wstring>().clear()) };
constexpr bool yyess { noexcept(std::declval<std::wstring>().empty()) };

// noexcept
constexpr bool uhuh { noexcept(std::declval<std::wstring>().resize(122)) };
constexpr bool noppe { noexcept(std::declval<std::wstring>().reserve(1226)) };

// ways to specify a non-throwing function
// noexcept
// noexcept(true)
// throw()

// exception specification for a potentially throwing  function
// noexcept(false)
// nothing

template<typename _Ty> struct complicated final {
        _Ty _value; // a type that's constructed from multiple arguments of different types

        // this constructor will be noexcept when _Ty can be constructed from _TyList&& without throwing an exception
        template<typename... _TyList> explicit complicated(_TyList&&... args) noexcept(noexcept(_value(std::forward<_TyList>(args)...))) :
            _value(std::forward<_TyList>(args)...) { }
};

template<typename _Ty> requires std::is_arithmetic_v<_Ty> static constexpr double square(const _Ty& _val) noexcept { return _val * _val; }

template<typename _TyFirst, typename... _TyList>
static consteval long double sum(const _TyFirst& _first, const _TyList&... _arguments) noexcept(noexcept(::square(_first))) {
    if constexpr (!sizeof...(_TyList))
        return _first;
    else
        return _first + ::sum(_arguments...);
}

static_assert(::sum(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10) == 55);
static_assert(noexcept(::square(978)));

constexpr bool must_be_true = noexcept(::square(787547)) && noexcept(::square('A')) && noexcept(::square(1254.04654));

template<typename... _TyList> // fold expression inside a noexcept() operator
static consteval long double prod(const _TyList&... _arguments) noexcept((noexcept(::square(_arguments)) && ...)) {
    return (_arguments * ...);
}

static_assert(::prod(1, 2, 3, 4, 5, 6, 7, 8, 9, 10) == 3628800);
static_assert(noexcept(::prod(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)));

// cannot overload the noexcept operator

struct dummy { };

static constexpr unsigned operator+([[maybe_unused]] const dummy& left, [[maybe_unused]] const dummy& right) noexcept { return 122; }

static_assert(dummy {} + dummy {} == 122);

template<typename... _TyList> static consteval bool sqsum(const _TyList&... _arguments) noexcept {
    const auto expand = ::sum(::square(_arguments)...);
    const auto fold   = (::square(_arguments) + ...);
    return expand == fold;
}

static_assert(::sqsum(0, 1, 2, 3, 4, 5));
