#include <string>

[[maybe_unused]] static constexpr bool yes = noexcept(true);

template<class _Ty> static consteval bool is_default_construction_noexcept() noexcept { return noexcept(_Ty {}); }

static_assert(::is_default_construction_noexcept<float>());
static_assert(::is_default_construction_noexcept<std::string>());

struct foo {
        foo() noexcept(false) { }
};

static_assert(!::is_default_construction_noexcept<::foo>());

[[maybe_unused]] static constexpr bool nope          = noexcept(foo());

[[maybe_unused]] static constexpr bool not_evaluated = noexcept(1 / 0); // if evaluated, this will be a compile time error

static constexpr auto error { 12 / 0 };

constexpr bool faalse { noexcept(std::declval<std::wstring>().replace(0, 9, L"..")) };
constexpr bool ffalse { noexcept(std::declval<std::wstring>().clear()) };
constexpr bool yyess { noexcept(std::declval<std::wstring>().empty()) };
