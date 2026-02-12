#include <cstdio>
#include <numbers>
#include <string>

template<typename T, typename U> struct is_same final {
        static constexpr bool value { false };
};

template<typename T> struct is_same<T, T> final {
        static constexpr bool value { true };
};

template<typename T, typename U> constexpr bool is_same_v = ::is_same<T, U>::value;

static_assert(!is_same_v<const float&&, volatile float&>);
static_assert(is_same_v<const std::wstring&&, const std::wstring&&>);

template<typename T, typename... TList> struct all_identical final {
    private:
        // MSVC is okay with using pack expansion for "head and rest" style variadic alias templates but all LLVM based compiles aren't
        template<typename _Ty, typename... _TyList> using __first_type = _Ty;

        template<typename _Ty, typename... _TyList> struct __first final {
                using type = _Ty; // capture the first type
        };

    public:
#if defined(__llvm__) && defined(__clang__)
        static constexpr bool value { ::is_same_v<T, typename __first<TList...>::type> && all_identical<TList...>::value };
#elif defined(_MSC_VER) && defined(_MSC_FULL_VER)
        static constexpr bool value { ::is_same_v<T, __first_type<TList...>> && all_identical<TList...>::value };
#endif
        // as usual, FUCK g++
};

template<typename T, typename U> struct all_identical<T, U> final {
        static constexpr bool value { ::is_same_v<T, U> };
};

static_assert(!::all_identical<float, const double, short&&, volatile std ::string>::value);
static_assert(::all_identical<float&, float&, float&, float&>::value);
static_assert(!::all_identical<float&, float&, const float&, float&>::value);

auto wmain() -> int {
    constexpr auto pi { std::numbers::pi_v<typename std::enable_if_t<
        ::all_identical<const volatile double&&, const volatile double&&, const volatile double&&>::value,
        long double>> };
    printf("%2.15Lf\n", pi); // NOLINT(cppcoreguidelines-pro-type-vararg)

    return EXIT_SUCCESS;
}
