#include <memory>
#include <string>
#include <type_traits>

template<typename _Ty, typename _TyOpt> struct is_nothrow_copy_constructible_helper final {
        static constexpr bool value = false;
};

template<typename _Ty>
struct is_nothrow_copy_constructible_helper<_Ty, decltype(_Ty { std::declval<std::add_lvalue_reference_t<std::add_const_t<_Ty>>>() })>
    final {
        static constexpr bool value = noexcept(_Ty { std::declval<std::add_lvalue_reference_t<std::add_const_t<_Ty>>>() });
};

template<typename _Ty> struct is_nothrow_copy_constructible final {
        static constexpr bool value = ::is_nothrow_copy_constructible_helper<_Ty, _Ty>::value;
};

template<typename _Ty> static constexpr bool is_nothrow_copy_constructible_v = ::is_nothrow_copy_constructible<_Ty>::value;

struct yes final {
        yes(const yes&) noexcept { }
};

struct nope final {
        nope(const nope&) noexcept(false) { } // may throw
};

static_assert(!::is_nothrow_copy_constructible_v<std::string>);
static_assert(!::is_nothrow_copy_constructible_v<std::wstring>);
static_assert(!::is_nothrow_copy_constructible_v<std::wstring>);
static_assert(::is_nothrow_copy_constructible_v<yes>);   // amazing
static_assert(!::is_nothrow_copy_constructible_v<nope>); // amazing
static_assert(::is_nothrow_copy_constructible_v<std::pair<float, float>>);
static_assert(::is_nothrow_copy_constructible_v<std::pair<float, const double&>>);
